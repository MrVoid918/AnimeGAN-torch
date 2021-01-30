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


if __name__ == '__main__':

    trial = Trial(batch_size=8, G_lr=0.05)
    torch.backends.cudnn.benchmark = True
    trial.Generator_NOGAN(epoch=10)

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
