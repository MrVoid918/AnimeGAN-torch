#import sys
#sys.path.append('./model')      #Quick way to bypass
'''
from model.discriminator import Discriminator
from model.generator import Generator
import init_train
'''

from optim.omd import OptimisticAdam
from model.discriminator import Discriminator
from model.generator import Generator
from optimizers import GANOptimizer
import torch.optim as optim

G = Generator()

print(op)
print(op1)
