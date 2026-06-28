import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Generator
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from torchvision import transforms as T

from cvae.vit.models import VITVAEConfig 
from cvae.vit.trainer import VITVAELit 
from datasets import load_dataset

transform = T.Compose(
    [
        T.ToTensor(),
        T.Resize(224),
        T.ColorJitter(0.1, 0.1, 0.05),
    ]
)

torch.set_float32_matmul_precision('medium')


def transform_fn(batch):
    images = batch['image']
    labels = torch.Tensor(
        [
            [vort, ffr, cb]
            for vort, ffr, cb in zip(
                batch['vortex'], batch['ffr'], batch['cloud_bands']
            )
        ]
    )
    return {
        'pixel_values': [transform(img) for img in images],
        'labels': labels,
    }


if __name__ == "__main__":
    batch_size = 64
    checkpoint_path = 'checkpoint_jvh_vit/'
    save_freq = 5

    datagenerator = load_dataset(
        'ramanakumars/JovianVortexHunter', cache_dir='./datasets', split='full'
    )
    datagenerator.set_transform(transform_fn)

    train_datagen, val_datagen = random_split(
        datagenerator, [0.9, 0.1], Generator().manual_seed(1234)
    )

    train_data = DataLoader(
        train_datagen,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,
    )
    val_data = DataLoader(
        val_datagen,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=6,
        persistent_workers=True,
    )

    config = VITVAEConfig()

    vae = VITVAELit(
        config,
        gen_loss_beta=50,
        kl_beta=2,
        rot_inv_loss=False,
        lr_decay=0.99,
        learning_rate=5e-4,
        optim_params={'name': 'adam'},
    )

    summary(vae.model, [1, 3, 224, 224])

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        filename='cvae_vit_{epoch:03d}',
        save_top_k=-1,
        every_n_epochs=save_freq,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        accelerator='cuda', max_epochs=200, callbacks=[checkpoint_callback, lr_monitor]
    )

    trainer.fit(vae, train_data, val_data)
