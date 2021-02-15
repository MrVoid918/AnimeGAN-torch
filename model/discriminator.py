from model.layers import *  # Import layers from layers.py, quick fix
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, bias=False):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(ConvBNLRelu(3, 32, 3, 1, 2, 1, bias),
                                 ConvBNLRelu(32, 32, 3, 1, 2, 1, bias),
                                 ConvBNLRelu(32, 64, 3, 1, 2, 1, bias),
                                 ConvBNLRelu(64, 64, 3, 1, 2, 1, bias),
                                 ConvBNLRelu(64, 128, 3, 1, 2, 1, bias),
                                 ConvBNLRelu(128, 128, 3, 1, 2, 1, bias),
                                 nn.Conv2d(128, 1, 4, bias=bias))

    def forward(self, x):
        return self.net(x)  # torch.sigmoid(self.net(x))


class PatchDiscriminator(nn.Module):

    def __init__(self, bias=False):
        super(PatchDiscriminator, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(in_channels=64, out_channels=128,
                                           kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(in_channels=128, out_channels=256,
                                           kernel_size=4, stride=2, padding=1, bias=False),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(in_channels=256, out_channels=512,
                                           kernel_size=4, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(0.2, True),
                                 nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,
                                           stride=1, padding=1, bias=False),
                                 )

        self.net2 = nn.Sequential(Depthwise_Separable_Conv(3, 64, 4, 2, 1),
                                  nn.LeakyReLU(0.2, True),
                                  Depthwise_Separable_Conv(64, 128, 4, 2, 1),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(0.2, True),
                                  Depthwise_Separable_Conv(128, 256, 4, 2, 1),
                                  nn.BatchNorm2d(256),
                                  nn.LeakyReLU(0.2, True),
                                  Depthwise_Separable_Conv(256, 512, 4, 1, 1),
                                  nn.BatchNorm2d(512),
                                  nn.LeakyReLU(0.2, True),
                                  Depthwise_Separable_Conv(512, 1, 4, 1, 1),
                                  )

    def forward(self, x):
        return self.net2(x)
