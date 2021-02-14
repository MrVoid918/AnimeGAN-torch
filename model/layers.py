import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F


class Depthwise_Separable_Conv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 bias=False,
                 kernels_per_layer: int = 1):

        super(Depthwise_Separable_Conv, self).__init__()
        self.depthwise = spectral_norm(nn.Conv2d(in_channels,
                                                 in_channels * kernels_per_layer,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 groups=in_channels,
                                                 stride=stride,
                                                 dilation=dilation,
                                                 padding_mode='reflect',
                                                 bias=bias))

        self.pointwise = spectral_norm(nn.Conv2d(in_channels * kernels_per_layer,
                                                 out_channels,
                                                 kernel_size=1,
                                                 bias=bias))

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvNormLRelu(nn.Module):

    def __init__(self, nin, nout, kernel_size, padding, stride, dilation, bias=False):
        super(ConvNormLRelu, self).__init__()
        self.conv = Depthwise_Separable_Conv(
            nin, nout, kernel_size, stride, padding, dilation, bias=bias)
        self.norm = nn.InstanceNorm2d(nout)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class ConvBNLRelu(nn.Module):

    def __init__(self, nin, nout, kernel_size, padding, stride, dilation, bias=False):
        super(ConvNormLRelu, self).__init__()
        self.conv = Depthwise_Separable_Conv(
            nin, nout, kernel_size, stride, padding, dilation, bias=bias)
        self.norm = nn.InstanceNorm2d(nout)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = ConvNormLRelu(hidden_dim, oup, 3, 1, stride, dilation=dilation)
            '''nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )'''
        else:
            self.conv = nn.Sequential(
                # pw
                spectral_norm(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # dw
                spectral_norm(nn.Conv2d(hidden_dim, hidden_dim, 3,
                                        stride, 1, groups=hidden_dim, bias=False)),
                nn.InstanceNorm2d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                # pw-linear
                spectral_norm(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)),
                nn.InstanceNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Upsample(nn.Module):

    def __init__(self, nin):
        super(Upsample, self).__init__()
        self.conv = ConvNormLRelu(nin, nin // 2, 3, 1, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = self.conv(x)
        return x
