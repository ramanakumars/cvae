import torch
from einops.layers.torch import Rearrange
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel


def sample(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, conv_filt, hidden, input_size, input_channels=3):  # conv_filt
        super().__init__()

        self.layers = []

        conv_filts = [32, 64]  # list of filters to run through later
        for i in range(2):
            conv_filts.append(conv_filt)

        # make layers
        filt_prev = input_channels  # previous number of input channels, starting at 3

        size = input_size
        for i, filt in enumerate(conv_filts):  # prev was padding='valid' (no padding)
            self.layers.append(
                nn.Conv2d(filt_prev, filt, 4, stride=2, padding=1)
            )  # in channels, out channels, kernel size
            self.layers.append(nn.LeakyReLU(0.2))  # alpha parameter
            self.layers.append(nn.InstanceNorm2d(filt))
            filt_prev = filt
            size = int(size / 2)

        self.final_size = size

        assert self.final_size > 1, (
            f"Final size ({self.final_size}) < 1. Use bigger images."
        )

        nconv = len(hidden)  # is hidden [512, 256, 128, encoded_space_dim]?

        # convolutional in bottleneck instead of flattened
        # runs through each filter in hidden and does a 1x1 convolution
        for i in range(nconv):
            self.layers.append(
                nn.Conv2d(filt, hidden[i], 1, 1, padding=0)
            )  # filt is 128, so 128->512
            self.layers.append(nn.LeakyReLU(0.2))  # Tanh?
            # self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]  # ends at batchnorm(encoded_space_dim)

        # self.layers.append(nn.Flatten(start_dim=1))

        self.layers = nn.ModuleList(self.layers)
        # self.conv_mu = nn.Sequential(nn.Conv2d(hidden[-1], hidden[-1], 1), nn.Flatten())
        # self.conv_sig = nn.Sequential(nn.Conv2d(hidden[-1], hidden[-1], 1), nn.Flatten())
        self.conv_mu = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                hidden[-1] * self.final_size * self.final_size,
                hidden[-1] * self.final_size * self.final_size,
                1,
            ),
        )
        self.conv_sig = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                hidden[-1] * self.final_size * self.final_size,
                hidden[-1] * self.final_size * self.final_size,
                1,
            ),
        )

    def forward(self, x):  # list of encoding + hidden layers
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        mu = self.conv_mu(x)
        sig = self.conv_sig(x)
        z = sample(mu, sig)
        return mu, sig, z


class Decoder(nn.Module):
    def __init__(self, conv_filt, hidden, input_size, input_channels, output_channels):
        super().__init__()
        self.layers = []

        self.layers.append(
            nn.Linear(
                input_channels * input_size * input_size,
                input_channels * input_size * input_size,
            )
        )
        self.layers.append(
            Rearrange("b (c h w) -> b c h w", h=input_size, w=input_size)
        )

        filt = input_channels  # last layer of hidden- encoded_space_dim
        # convolutional layers in bottleneck
        nconv = len(hidden)
        for i in range(nconv):
            self.layers.append(nn.Conv2d(filt, hidden[i], 1, 1, padding=0))
            self.layers.append(nn.LeakyReLU(0.2))
            # self.layers.append(nn.BatchNorm2d(hidden[i]))
            filt = hidden[i]

        conv_filts = []
        for i in range(1):
            conv_filts.append(conv_filt)
        conv_filts.extend([64, 32])

        filt_prev = filt  # and filt = final element of hidden = 128
        for i, filt in enumerate(conv_filts):
            self.layers.append(
                nn.ConvTranspose2d(filt_prev, filt, 4, stride=2, padding=1)
            )
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.InstanceNorm2d(filt))
            filt_prev = filt

        self.layers.append(
            nn.ConvTranspose2d(filt, output_channels, 4, stride=2, padding=1)
        )  # try kernel 5 instead of upsample
        self.layers.append(nn.LeakyReLU(0.2))
        self.layers.append(nn.Conv2d(output_channels, output_channels, 3, padding=1))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        # run the input through the layers
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        return x


class VAEConfig(PretrainedConfig):
    model_type = "vae"

    def __init__(
        self,
        input_size: int = 96,
        input_channels: int = 3,
        conv_filts: int = 128,
        hidden: list[int] = [64, 8],
        **kwargs,
    ):
        self.input_size = input_size
        self.input_channels = input_channels
        self.conv_filts = conv_filts
        self.hidden = hidden
        super().__init__(**kwargs)


class VAE(PreTrainedModel):
    config_class = VAEConfig

    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self.encoder = Encoder(
            config.conv_filts, config.hidden, config.input_size, config.input_channels
        )
        self.decoder = Decoder(
            config.conv_filts,
            config.hidden[::-1],
            self.encoder.final_size,
            config.hidden[-1],
            config.input_channels,
        )

    def forward(self, x):
        mu, sig, z = self.encoder(x)
        return self.decoder(z)


class CVAEConfig(VAEConfig):
    model_type = "cvae"

    def __init__(
        self,
        num_classes: int,
        **kwargs,
    ):
        self.num_classes = num_classes

        super().__init__(**kwargs)


class CVAE(VAE):
    config_class = CVAEConfig

    def __init__(self, config: CVAEConfig):
        super().__init__(config)
        self.classifier = nn.Linear(
            self.encoder.final_size * self.encoder.final_size * config.hidden[-1],
            config.num_classes,
        )

    def forward(self, x):
        mu, sig, z = self.encoder(x)
        return self.decoder(z), self.classifier(mu)
