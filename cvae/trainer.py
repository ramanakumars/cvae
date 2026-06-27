import lightning as L
import torch
from torch.optim.lr_scheduler import ExponentialLR

from .losses import (
    KL_loss,
    classification_loss,
    contractive_loss,
    mae_loss,
    transf_invariant_loss,
)
from .model import CVAE, VAE, CVAEConfig, VAEConfig


class VAELit(L.LightningModule):
    model_type = VAE

    def __init__(
        self,
        config: VAEConfig,
        kl_beta=0.5,
        contractive_loss=False,
        rot_inv_loss=False,
        gen_loss_beta: float = 2,
        learning_rate=1.0e-3,
        lr_decay=0.98,
        optim_params={'name': 'adam', 'betas': (0.9, 0.99)},
        decay_freq=5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])

        self.model = self.model_type(config)

    def forward(self, x):
        self.model(x)

    def training_step(self, batch, batch_idx):
        '''
        Train the generator and discriminator on a single batch
        '''
        _, _, _, _, mean_loss = self.batch_step(batch)

        scheduler = self.lr_schedulers()
        if (
            self.trainer.is_last_batch
            and (self.trainer.current_epoch + 1) % self.hparams.decay_freq == 0
        ):
            scheduler.step()

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
        optim = self.hparams.optim_params.pop('name')
        if optim == 'adam':
            optimizer = torch.optim.Adam(
                parameters, lr=learning_rate, **self.hparams.optim_params
            )
        elif optim == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                parameters, lr=learning_rate, **self.hparams.optim_params
            )
        elif optim == 'nadam':
            optimizer = torch.optim.NAdam(
                parameters, lr=learning_rate, **self.hparams.optim_params
            )
        else:
            raise ValueError(f'{self.hparams.optim_params["name"]} not implemented')

        lr_scheduler = ExponentialLR(optimizer, gamma=self.hparams.lr_decay)

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.hparams.decay_freq,
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class ClassVAELit(VAELit):
    model_type = CVAE

    def __init__(self, config: CVAEConfig, class_beta=10.0, **VAE_kwargs):
        super().__init__(config=config, **VAE_kwargs)

    def batch_step(self, batch):
        label = batch['labels']

        mu, sig, z, gen, loss = super().batch_step(batch)

        pred_label = self.model.classifier(z)

        class_loss = self.hparams.class_beta * classification_loss(pred_label, label)

        loss['loss'] += class_loss

        return mu, sig, z, gen, {**loss, 'class_loss': class_loss.item()}
