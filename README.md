# Conditional Variational Auto-Encoder

Implementation of a conditional variational auto-encoder (cVAE) model built in PyTorch. 
The model features an VAE for image feature extraction and a classifier head from the latent space to make label prediction. 
The goal of this model is to learn in a semi-supervised fashion the relation between image features and target classes.


## Installation
Install the dependencies using `requirements.txt`, as follows:
```
python3 -m pip install -r requirements.txt
```

## Usage
The model can be trained and inferred using PyTorch Lightning. Here is a basic implementation:
```python
from cvae.trainer import ClassVAE
from cvae.io import ZooniverseLabelGenerator
from torchinfo import summary
from torch import Generator
from torch.utils.data import DataLoader, random_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

batch_size = 64
checkpoint_path = 'checkpoints/'
save_freq = 5
n_epochs = 200

# create the data loader
num_classes = 3
datagenerator = ZooniverseLabelGenerator('/path/to/images', '/path/to/labels.csv')


# do the training/validation split 
# we will use a Generator to ensure consistent samples
train_datagen, val_datagen = random_split(datagenerator, [0.9, 0.1], Generator().manual_seed(1234))
train_data = DataLoader(train_datagen, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_data = DataLoader(val_datagen, batch_size=batch_size, pin_memory=True, num_workers=8)

# initialize the model
vae = ClassVAE(num_classes, conv_filt=128, hidden=[64, 8], input_size=96, class_beta=150,
               rot_inv_loss=True, lr_decay=0.99, learning_rate=1.e-4, optim_params={'name': 'nadam'})

summary(vae, [1, 3, 96, 96])

# if you want to save every n-epochs
checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path,
                                      filename='vade_{epoch:03d}',
                                      save_top_k=-1,
                                      every_n_epochs=save_freq,
                                      verbose=True)

# for tracking the learning rate
lr_monitor = LearningRateMonitor(logging_interval='epoch')

# train the model
trainer = Trainer(accelerator='cuda', max_epochs=n_epochs, callbacks=[checkpoint_callback, lr_monitor])
trainer.fit(vae, train_data, val_data)
```

## ClassVAE model
The ClassVAE model is a VAE with an MLP attached to the latent variable to do class prediction.
The model parameters are:

* `num_classes`: the number of classes to predict
* `conv_filt`: the convolutional filter size for the final layers of the encoder and first two layers of the decoder
* `hidden`: an array of filter sizes to use in the 1D convolution for the hidden layers
* `input_size`: the size of the input image (assuming square images)

Optional parameters for training:
* `input_channels`: the number of channels in the input image [default: 3]
* `kl_beta`: weight of the KL-divergence loss [default: 3]
* `class_beta`: weight of the classification loss [default: 10]
* `contractive_loss`: toggle for Contractive Loss (between latent variable and input image) [default: false]
* `rot_inv_loss`: toggle for Rotational Invariance Loss (Kurihana++ 2021) [default: false]
* `learning_rate`: model learning_rate [default: 1e-3]
* `lr_decay`: exponential decay rate for learning (applied every `decay_freq` epochs) [default: 0.95]
* `decay_freq`: frequency to apply learning rate decay [default: 5 (epochs)]
* `optim_params`: dictionary of optimizer parameters. Requires `name` keyword describing the optimizer and other keywords are passed into the optimizer.

## Custom DataLoaders
You can create a custom `DataLoader` and import it into the training script. 
The DataLoader must return the image (augmented if needed) and the corresponding label (see `ZooniverseLabelGenerator` for examples in `io.py`)
