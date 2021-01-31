# import sys
# sys.path.append('./model')      #Quick way to bypass
'''
from model.discriminator import Discriminator
from model.generator import Generator
import init_train
'''

from loss import Loss
from trial import Trial
import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
from meter import LossMeters
import matplotlib.pyplot as plt
import typer
from typer import Argument, Option
from typing import List, Optional


def main(batch_size: int = Argument(32),
         G_lr: float = Argument(0.05, help="Learning rate for generator training"),
         D_lr: float = Argument(0.05, help="Learning rate for discriminator training"),
         GAN_G_lr: float = Argument(0.05, help="Learning rate for GAN generator training"),
         GAN_D_lr: float = Argument(0.05, help="Learning rate for GAN discriminator training"),
         G_epoch: int = Argument(10, help="Iteration of generator training"),
         D_epoch: int = Argument(3, help="Iteration of discriminator training"),
         itr: int = Argument(1, help="Iteration of whole NOGAN training"),
         optim_type: str = Argument("ADAB", help="Options of Optimizers for Generator")):
    """
    NOGAN training Trial.
    """
    torch.backends.cudnn.benchmark = True
    assert(itr > 0), "Number must be bigger than 0"
    for _ in range(itr):
        trial = Trial(batch_size=32, G_lr=G_lr, D_lr=D_lr, optim_type=optim_type)
        trial.Generator_NOGAN(epochs=G_epoch, content_weight=3.0, recon_weight=10.,
                              loss=['content_loss', 'recon_loss'],)
        trial.Discriminator_NOGAN(epochs=D_epoch)
        trial.GAN_NOGAN()


if __name__ == '__main__':
    typer.run(main)
"""
arr = np.array((166.2, 144.2, 134.7, 129., 122.5, 117.8, 114.2,
                111.6, 109.1, 108.1, 105.6, 104, 102.7, 101.9, 101.8))
x = np.arange(len(arr))
grad = np.gradient(arr)
thresh = -1.0
plt.axhline(thresh, c='r')
plt.plot(x, grad)
plt.show()

arr = np.array((200.))
grad = np.gradient(arr)
print(grad)
"""
