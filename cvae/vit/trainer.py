import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR

from ..losses import (
    KL_loss,
    contractive_loss,
    mae_loss,
    transf_invariant_loss,
    classification_loss
)
from .models import VITVAE, VITVAEConfig, cVITVAE, cVITVAEConfig


class VITVAELit(L.LightningModule):
    """PyTorch Lightning wrapper for ``VITVAE``.

    Uses a two-phase LR schedule: linear warmup followed by exponential decay.
    Warmup is necessary for transformer training because large LRs at
    initialisation cause attention weights to collapse (all mass on a single
    token) before key/query projections have been calibrated.  The convolutional
    ``VAELit`` skips warmup because spatial weight sharing in conv layers
    provides natural gradient stability from the start.
    """

    model_type = VITVAE

    def __init__(
        self,
        config: VITVAEConfig,
        kl_beta=0.5,
        contractive_loss=False,
        rot_inv_loss=False,
        gen_loss_beta: float = 2,
        learning_rate=1.0e-3,
        lr_decay=0.98,
        optim_params={'name': 'adam', 'betas': (0.9, 0.99)},
        decay_freq: int = 5,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])

        self.model = self.model_type(config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        Train the generator and discriminator on a single batch
        '''
        _, _, _, _, mean_loss = self.batch_step(batch)

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

        return mean_loss['loss']

    def validation_step(self, batch, batch_idx):
        if self.hparams.contractive_loss:
            torch.set_grad_enabled(True)
        _, _, _, _, mean_loss = self.batch_step(batch)

        for key, val in mean_loss.items():
            self.log(key, val, prog_bar=True, on_epoch=True, reduce_fx=torch.mean)

        return mean_loss['loss']

    def batch_step(self, batch):
        img = batch['pixel_values']

        if self.hparams.contractive_loss:
            img.requires_grad_(True)
            img.retain_grad()

        mu, sig, z = self.model.encoder(img)
        gen = self.model.decoder(z)

        loss = {}
        gen_img_loss = self.hparams.gen_loss_beta * mae_loss(img, gen)
        kl_loss = self.hparams.kl_beta * KL_loss(mu, sig)

        loss['gen_loss'] = gen_img_loss.item()
        loss['kl_loss'] = kl_loss.item()

        total_loss = gen_img_loss + kl_loss

        if self.hparams.contractive_loss:
            contr_loss = contractive_loss(z, img)
            loss['contr_loss'] = contr_loss.item()
            total_loss += contr_loss

        if self.hparams.rot_inv_loss:
            Linv, Lres = transf_invariant_loss(img, mu, self.model.encoder)
            loss['Linv'] = Linv.item()
            loss['Lres'] = Lres.item()
            total_loss += Linv + Lres

        return mu, sig, z, gen, {'loss': total_loss, **loss}

    def get_model_params(self):
        return list(self.model.parameters())

    def configure_optimizers(self):
        learning_rate = self.hparams.learning_rate

        parameters = self.get_model_params()

        # Copy before popping to avoid mutating the hparams dict on repeated calls.
        optim_params = self.hparams.optim_params.copy()
        optim = optim_params.pop('name')
        if optim == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=learning_rate, **optim_params)
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                parameters, lr=learning_rate, **optim_params
            )
        elif optim == 'nadam':
            optimizer = torch.optim.NAdam(parameters, lr=learning_rate, **optim_params)
        else:
            raise ValueError(f'{optim} not implemented')

        # Phase 1: ramp from 0.1× to 1.0× learning_rate over warmup_epochs.
        # Phase 2: exponential decay at lr_decay per epoch thereafter.
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        exp_scheduler = ExponentialLR(
            optimizer, gamma=(self.hparams.lr_decay ** (1 / self.hparams.decay_freq))
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, exp_scheduler],
            milestones=[self.hparams.warmup_epochs],
        )

        # frequency=1 so Lightning steps the SequentialLR every epoch,
        # matching the warmup cadence.
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class cVITVAELit(VITVAELit):
    model_type = cVITVAE

    def __init__(self, config: cVITVAEConfig, **kwargs):
        super().__init__(config, **kwargs)
    
    def batch_step(self, batch):
        label = batch['labels']

        mu, sig, z, gen, loss = super().batch_step(batch)

        pred_label = self.model.classifier(z)

        class_loss = self.hparams.class_beta * classification_loss(pred_label, label)

        loss['loss'] += class_loss

        return mu, sig, z, gen, {**loss, 'class_loss': class_loss.item()}
