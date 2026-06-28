"""
Vision Transformer VAE backend for the cvae package.

This module provides a ViT-based encoder/decoder as an alternative to the
convolutional backend in ``cvae.convolutional``. Both backends expose
``PreTrainedModel``-compatible classes (``VITVAE`` / ``VAE``) so checkpoints
can be saved and loaded via ``save_pretrained`` / ``from_pretrained``.

The only primitive shared with the convolutional backend is ``sample`` (the
reparameterization trick), imported from ``cvae.convolutional.model``.
Everything else — patch embedding, transformer blocks, projection layers — is
independent, so the two backends can be developed without coupling.
"""

import torch
from einops.layers.torch import Rearrange
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from ..convolutional.model import sample


class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and project each to a vector.

    A learnable CLS token is prepended to the patch sequence before positional
    embedding is added.  The CLS token participates in every self-attention
    layer, acting as a global aggregation sink, but ``VITEncoder`` discards it
    after the transformer stack and uses global average pooling over the patch
    tokens instead — see ``VITEncoder`` for the rationale.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_size: int,
        num_patches: int,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.patcher = nn.Sequential(
            # Conv2d with kernel_size == stride produces non-overlapping patches,
            # equivalent to a learned linear projection of each patch.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(
            torch.randn(size=(1, 1, embed_dim)), requires_grad=True
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, num_patches + 1, embed_dim)), requires_grad=True
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x
        x = self.dropout(x)
        return x


class VITEncoder(nn.Module):
    """Transformer encoder that maps an image to a VAE latent distribution (mu, log_var, z).

    After the transformer stack the CLS token (index 0) is discarded and the
    remaining patch tokens are averaged (global average pooling) to form the
    input to the mu/log_var heads.  GAP is preferred over CLS pooling here
    because, without a classification objective, gradients do not specifically
    train the CLS token to summarise global image content.  GAP treats every
    spatial region equally, which is more appropriate when the downstream task
    is reconstruction rather than class discrimination.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_z: int,
        n_encoder_layers: int = 3,
        dropout: float = 0.3,
        in_channels: int = 3,
        num_heads: int = 8,
        n_dim_feedforward: int = 256,
    ):
        super().__init__()
        n_patches = image_size // patch_size

        self.patch_embedding = PatchEmbedding(
            n_dim_feedforward, patch_size, n_patches * n_patches, dropout, in_channels
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_dim_feedforward,
            nhead=num_heads,
            dim_feedforward=n_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_encoder_layers)

        self.mu = nn.Linear(n_dim_feedforward, n_z)
        self.log_var = nn.Linear(n_dim_feedforward, n_z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.patch_embedding(x)
        x = self.transformer(x)

        # Discard CLS token; average over patch tokens for global representation.
        x_inp = x[:, 1:].mean(1)

        mu = self.mu(x_inp)
        log_var = self.log_var(x_inp)
        z = sample(mu, log_var)
        return mu, log_var, z


class VITDecoder(nn.Module):
    """Transformer decoder that maps a latent vector z back to pixel space.

    **Why TransformerEncoderLayer, not TransformerDecoderLayer?**
    ``TransformerDecoderLayer`` adds cross-attention between a target sequence
    and a live encoder memory tensor.  Here the encoder's contribution is
    already fully compressed into ``z`` before the decoder runs — there is no
    encoder memory to cross-attend to.  Self-attention only
    (``TransformerEncoderLayer``) is sufficient and avoids wasted parameters.

    **Factored projection from z to patch sequence.**
    A naive ``Linear(n_z, (n_patches²+1) × D)`` would have ~14.5M parameters
    for the default config (n_z=288, 197 tokens, D=256).  Instead the
    projection first expands ``z`` into 16 seed embeddings of dimension D, then
    applies a shared ``Linear(16, n_patches²+1)`` independently per feature
    dimension, interpolating 16 seed positions to 197 token positions.  This
    factored approach dramatically reduces parameters while still allowing every
    output position to depend on every dimension of ``z``.

    **Token 0 as attention sink.**
    The projection produces ``n_patches²+1`` tokens.  During decoding, the
    extra token at index 0 acts as a global attention sink — patch tokens can
    off-load diffuse, non-spatial attention mass to it rather than forcing it
    onto a neighbouring patch.  It is dropped after the transformer so that
    ``unflatten`` receives exactly ``n_patches²`` tokens to reshape into a
    spatial grid.  This mirrors the encoder's CLS token: present during
    attention, absent from the final representation.

    **Note on standalone defaults.**
    ``VITDecoder.__init__`` defaults ``n_dim_feedforward=2048``.  When
    instantiated through ``VITVAE(config)``, ``config.n_dim_feedforward``
    (default 256) overrides this for both encoder and decoder.  Constructing
    ``VITDecoder`` directly with its own defaults produces a much larger model.

    **Reconstruction design (de_patchify).**
    Pixel-level reconstruction uses two ConvTranspose2d stages rather than
    one, matching the progressive-refinement pattern of the conv decoder:

    - *Stage 1* upsamples by ``patch_size // 2`` (e.g. 14→112 for patch=16),
      then applies GELU + InstanceNorm2d.  This mirrors the per-stage
      normalisation used in ``convolutional.Decoder`` and stabilises gradients
      flowing back through the transformer stack.
    - *Stage 2* applies a final ×2 ConvTranspose2d + GELU, separating the
      spatial upsampling concern from the pixel readout.
    - A 3×3 ``Conv2d`` (padding 1) performs the final projection to RGB.
      The 3×3 kernel lets each boundary pixel attend to one pixel from both
      adjacent patches, providing minimal cross-patch smoothing that a 1×1
      kernel cannot.

    ``patch_size`` must be even for the ``patch_size // 2`` split; enforced by
    ``VITVAEConfig``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        n_z: int,
        n_encoder_layers: int,
        dropout: float = 0.3,
        in_channels: int = 3,
        num_heads: int = 8,
        n_dim_feedforward: int = 2048,
    ):
        super().__init__()
        n_patches = image_size // patch_size

        # Factored projection: z -> 16 seed embeddings -> n_patches²+1 token embeddings.
        # Avoids a single massive linear layer while letting every token position
        # depend on all dimensions of z.
        self.projection = nn.Sequential(
            nn.Linear(n_z, 16 * n_dim_feedforward),
            nn.LayerNorm(16 * n_dim_feedforward),
            nn.GELU(),
            Rearrange("b (p d) -> b d p", p=16, d=n_dim_feedforward),
            nn.Linear(16, (n_patches * n_patches + 1)),
            Rearrange("b d p -> b p d"),
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=n_dim_feedforward,
            nhead=num_heads,
            dim_feedforward=n_dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(decoder_layer, n_encoder_layers)

        self.unflatten = Rearrange("b (y x) d -> b d y x", y=n_patches, x=n_patches)

        self.de_patchify = nn.Sequential(
            nn.ConvTranspose2d(
                n_dim_feedforward, 64, patch_size // 4, patch_size // 4, 0
            ),
            nn.InstanceNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 4, 0),
            nn.GELU(),
            # 3×3 conv for inter-patch consistency
            nn.Conv2d(32, in_channels, 3, 1, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        # Drop token 0 (attention sink) before spatial reconstruction.
        x = self.transformer(x)[:, 1:]
        return self.de_patchify(self.unflatten(x))


class VITVAEConfig(PretrainedConfig):
    model_type = "vitvae"

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        # n_z > n_dim_feedforward by design: the bottleneck is still a massive
        # compression of the full patch sequence (196 × D dims), and the slight
        # expansion from D to n_z avoids information loss at the final projection.
        n_z: int = 288,
        n_encoder_layers: int = 3,
        dropout: float = 0.1,
        in_channels: int = 3,
        num_heads: int = 8,
        n_dim_feedforward: int = 256,
        **kwargs,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = image_size // patch_size

        if (image_size % patch_size) != 0:
            raise ValueError(
                f"Image size ({image_size}) is not divisible by patch size ({patch_size})"
            )

        if patch_size % 4 != 0:
            raise ValueError(
                f"patch_size ({patch_size}) must be divisible by 4: de_patchify splits upsampling "
                f"into patch_size // 4 then ×4 stages"
            )

        self.n_z = n_z
        self.n_encoder_layers = n_encoder_layers
        self.dropout = dropout
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.n_dim_feedforward = n_dim_feedforward
        super().__init__(**kwargs)


class VITVAE(PreTrainedModel):
    """ViT-based VAE.

    The encoder uses learned positional embeddings (inside ``PatchEmbedding``);
    the decoder does not — positional structure in the decoder must emerge from
    the projection layer's training.  This asymmetry is intentional: the
    encoder needs to distinguish spatial locations to build a meaningful latent
    code, while the decoder receives a latent vector that already encodes
    spatial information implicitly through the projection weights.
    """

    config_class = VITVAEConfig

    def __init__(self, config: VITVAEConfig):
        super().__init__(config)

        self.encoder = VITEncoder(
            config.image_size,
            config.patch_size,
            config.n_z,
            config.n_encoder_layers,
            config.dropout,
            config.in_channels,
            config.num_heads,
            config.n_dim_feedforward,
        )

        self.decoder = VITDecoder(
            config.image_size,
            config.patch_size,
            config.n_z,
            config.n_encoder_layers,
            config.dropout,
            config.in_channels,
            config.num_heads,
            config.n_dim_feedforward,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, sig, z = self.encoder(x)
        return self.decoder(z)


class cVITVAEConfig(VITVAEConfig):
    model_type = "cvitvae"

    def __init__(
        self,
        num_classes: int,
        **kwargs,
    ):
        self.num_classes = num_classes
        super().__init__(**kwargs)


class cVITVAE(VITVAE):
    """ViT-based VAE.

    The encoder uses learned positional embeddings (inside ``PatchEmbedding``);
    the decoder does not — positional structure in the decoder must emerge from
    the projection layer's training.  This asymmetry is intentional: the
    encoder needs to distinguish spatial locations to build a meaningful latent
    code, while the decoder receives a latent vector that already encodes
    spatial information implicitly through the projection weights.
    """

    config_class = cVITVAEConfig

    def __init__(self, config: cVITVAEConfig):
        super().__init__(config)
        self.classifier = nn.Linear(config.n_z, config.num_classes)

    def forward(self, x: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        mu, sig, z = self.encoder(x)
        classification = self.classifier(mu)
        return self.decoder(z), classification
