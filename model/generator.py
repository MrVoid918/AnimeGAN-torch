from layers import *
import torch.nn as nn
import torch

class Generator(nn.Module):

  def __init__(self, bias = False):
    super(Generator, self).__init__()
    self.downsample = nn.Sequential(ConvNormLRelu(3, 64, 3, 1, 2, 1, bias = bias),
                          ConvNormLRelu(64, 128, 3, 1, 2, 1, bias),
                          ConvNormLRelu(128, 256, 3, 1, 2, 1, bias),
                          ConvNormLRelu(256, 512, 3, 1, 2, 1, bias))

    self.residual = nn.Sequential(InvertedResidual(512, 512, 1, 1, 2),
                                  InvertedResidual(512, 512, 1, 1, 2),
                                  InvertedResidual(512, 512, 1, 1, 2),
                                  InvertedResidual(512, 512, 1, 1, 2),
                                  InvertedResidual(512, 512, 1, 1, 2))

    self.upsample = nn.Sequential(Upsample(512), #512->256
                                  Upsample(256), #256 -> 128
                                  Upsample(128),    #128 -> 64
                                  Upsample(64),     #64 -> 32
                                  ConvNormLRelu(32, 3, 3, 1, 1, 1, bias))

    self.net = nn.Sequential(self.downsample,
                             self.residual,
                             self.upsample)

  def forward(self, x):
    return torch.tanh(self.net(x))
