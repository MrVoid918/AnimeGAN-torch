from model.layers import *  # Import layers from layers.py, quick fix
import torch.nn as nn
import torch


class Generator(nn.Module):

    def __init__(self,
                 res_fmap_size=256,
                 n_resblock=4,
                 bias=False):
        """
        args: res_fmap_size: Feature map size of resblock inputs.
        n_resblock: Number of resblocks"""
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(ConvINLRelu(3, 32, 3, 1, 2, 1, bias=bias),
                                        ConvINLRelu(32, 64, 3, 1, 2, 1, bias=bias),
                                        ConvINLRelu(64, 128, 3, 1, 2, 1, bias))
        # ConvINLRelu(128, 256, 3, 1, 2, 1, bias),)
        # ConvINLRelu(256, 512, 3, 1, 2, 1, bias))
        """
        self.residual = nn.Sequential(InvertedResidual(512, 512, 1, 1, 2),
                                      InvertedResidual(512, 512, 1, 1, 2),
                                      InvertedResidual(512, 512, 1, 1, 2),
                                      InvertedResidual(512, 512, 1, 1, 2),
                                      InvertedResidual(512, 512, 1, 1, 2),
                                      InvertedResidual(512, 512, 1, 1, 2))
        """
        self.residual = nn.Sequential(*[InvertedResidual(res_fmap_size, res_fmap_size, 1, 1, 2)
                                        for i in range(n_resblock)])
        """
        self.residual = nn.Sequential(InvertedResidual(256, 256, 1, 1, 2),
                                      InvertedResidual(256, 256, 1, 1, 2),
                                      InvertedResidual(256, 256, 1, 1, 2),
                                      InvertedResidual(256, 256, 1, 1, 2),
                                      InvertedResidual(256, 256, 1, 1, 2),
                                      InvertedResidual(256, 256, 1, 1, 2))"""

        self.upsample = nn.Sequential(  # Upsample(512),  # 512->256
            # Upsample(256),  # 256 -> 128
            Upsample(128),  # 128 -> 64
            Upsample(64),  # 64 -> 32
            nn.Conv2d(32, 3, 3, 1, 1, bias=bias))

        self.net = nn.Sequential(self.downsample,
                                 self.residual,
                                 self.upsample)

    def forward(self, x):
        return torch.tanh(self.net(x))
