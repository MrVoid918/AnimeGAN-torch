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


def main(batch_size: int = Option(32, "-b"),
         G_lr: float = Option(0.05, help="Learning rate for generator training"),
         D_lr: float = Option(0.05, help="Learning rate for discriminator training"),
         GAN_G_lr: float = Option(0.00008, help="Learning rate for GAN generator training"),
         GAN_D_lr: float = Option(0.00016, help="Learning rate for GAN discriminator training"),
         G_epoch: int = Option(10, help="Iteration of generator training"),
         D_epoch: int = Option(3, help="Iteration of discriminator training"),
         itr: int = Option(1, help="Iteration of whole NOGAN training"),
         optim_type: str = Option("ADAB", help="Options of Optimizers for Generator")):
    """
    NOGAN training Trial.
    """
    torch.backends.cudnn.benchmark = True
    assert(itr > 0), "Number must be bigger than 0"
    trial = Trial(batch_size=batch_size, G_lr=G_lr, D_lr=D_lr, optim_type=optim_type)
    for _ in range(itr):
        trial.Generator_NOGAN(epochs=G_epoch, content_weight=3.0, recon_weight=10.,
                              loss=['content_loss', 'recon_loss'],)
        trial.Discriminator_NOGAN(epochs=D_epoch)
        trial.GAN_NOGAN(1, GAN_G_lr=GAN_G_lr, GAN_D_lr=GAN_D_lr)


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
