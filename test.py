#import sys
# sys.path.append('./model')      #Quick way to bypass
'''
from model.discriminator import Discriminator
from model.generator import Generator
import init_train
'''

from trial import Trial
import torch

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    trial = Trial(batch_size=16,)
    trial.init_train()
