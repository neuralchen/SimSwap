import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


### single layers


def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))


def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))


def embedding(*args, **kwargs):
    return spectral_norm(nn.Embedding(*args, **kwargs))


def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))


def NormLayer(c, mode='batch'):
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)


### Activations


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


### Upblocks


class InitLayer(nn.Module):
    def __init__(self, nz, channel, sz=4):
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel*2, sz, 1, 0, bias=False),
            NormLayer(channel*2),
            GLU(),
        )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlockSmall(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NormLayer(out_planes*2), GLU())
    return block


class UpBlockSmallCond(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn = which_bn(2*out_planes)
        self.act = GLU()

    def forward(self, x, c):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c)
        x = self.act(x)
        return x


def UpBlockBig(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU()
        )
    return block


class UpBlockBigCond(nn.Module):
    def __init__(self, in_planes, out_planes, z_dim):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2*out_planes)
        self.bn2 = which_bn(2*out_planes)
        self.act = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c):
        # block 1
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)
        x = self.act(x)

        # block 2
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)
        x = self.act(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        return feat_big * self.main(feat_small)


### Downblocks


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        if not separable:
            self.main = nn.Sequential(
                conv2d(in_planes, out_planes, 4, 2, 1),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.main = nn.Sequential(
                SeparableConv2d(in_planes, out_planes, 3),
                NormLayer(out_planes),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2, 2),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockPatch(nn.Module):
    def __init__(self, in_planes, out_planes, separable=False):
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes, separable),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        return self.main(feat)


### CSM


class ResidualConvUnit(nn.Module):
    def __init__(self, cin, activation, bn):
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        return self.skip_add.add(self.conv(x), x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, lowest=False):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


### Misc


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise


class CCBN(nn.Module):
    ''' conditional batchnorm '''
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        super().__init__()
        self.output_size, self.input_size = output_size, input_size

        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size, mode='bilinear', align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
