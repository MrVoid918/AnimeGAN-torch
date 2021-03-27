from model.layers import *  # Import layers from layers.py, quick fix
import torch.nn as nn
import torch
import math


class Generator(nn.Module):

    def __init__(self,
                 base_fmap=32,
                 res_fmap_size=256,
                 n_resblock=4,
                 bias=False):
        """
        args: res_fmap_size: Feature map size of resblock inputs.
        n_resblock: Number of resblocks"""
        def check_exp_of_2(x): return math.log2(x).is_integer()
        assert check_exp_of_2(base_fmap) and check_exp_of_2(res_fmap_size)
        super(Generator, self).__init__()
        downsample_blocks = [ConvINLRelu(3, base_fmap, 3, 1, 1, 1, bias=bias)]

        for i in range(int(math.log2(base_fmap)), int(math.log2(res_fmap_size))):
            downsample_blocks.append(ConvINLRelu(2 ** i, 2 ** (i+1), 3, 1, 1, 1, bias=bias))
        self.downsample = nn.Sequential(*downsample_blocks)

        """
        self.downsample = nn.Sequential(ConvINLRelu(3, 32, 3, 1, 1, 1, bias=bias),
                                        ConvINLRelu(32, 64, 3, 1, 2, 1, bias=bias),
                                        ConvINLRelu(64, 128, 3, 1, 2, 1, bias))
                                        """
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
        upsample_blocks = [
            Upsample(2 ** i) for i in range(int(math.log2(res_fmap_size)), int(math.log2(base_fmap)), -1)]
        upsample_blocks.append(nn.Conv2d(base_fmap, 3, 3, 1, 1, bias=bias))

        self.upsample = nn.Sequential(*upsample_blocks)

        """
        self.upsample=nn.Sequential(  # Upsample(512),  # 512->256
            # Upsample(256),  # 256 -> 128
            Upsample(128),  # 128 -> 64
            Upsample(64),  # 64 -> 32pytho
            nn.Conv2d(32, 3, 3, 1, 1, bias=bias))
        """
        self.net = nn.Sequential(self.downsample,
                                 self.residual,
                                 self.upsample)

    def forward(self, x):
        return torch.tanh(self.net(x))
