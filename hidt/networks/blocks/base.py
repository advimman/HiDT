__all__ = ['Conv2dBlock',
           'LinearBlock',
           'ResBlock',
           'FUNITResBlock',
           'FUNITConv2dBlock'
           ]

import numpy as np
import torch.nn.functional as F
from torch import nn

from hidt.networks.blocks.norm import LayerNorm, AdaptiveInstanceNorm2d
from hidt.networks.blocks.specnorm import SpectralNorm


class FUNITResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None, activation='lrelu', norm='none'):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid

        self.conv_0 = FUNITConv2dBlock(self.fin, self.fhid, 3, 1,
                                       padding=1, pad_type='reflect', norm=norm,
                                       activation=activation, activation_first=True)

        self.conv_1 = FUNITConv2dBlock(self.fhid, self.fout, 3, 1,
                                       padding=1, pad_type='reflect', norm=norm,
                                       activation=activation, activation_first=True)

        if self.learned_shortcut:
            self.conv_s = FUNITConv2dBlock(self.fin, self.fout, 1, 1,
                                           activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x.clone()
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class FUNITConv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(FUNITConv2dBlock, self).__init__()

        self.use_bias = use_bias
        self.activation_first = activation_first

        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln',
                 res_off=False):
        super().__init__()
        self.res_off = res_off
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type,
                              style_dim=style_dim, norm_after_conv=norm_after_conv)]
        self.model = nn.ModuleList(model)

    def forward(self, x, spade_input=None):
        residual = x
        for layer in self.model:
            x = layer(x, spade_input)
        if self.res_off:
            return x
        else:
            return x + residual


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero', style_dim=3, norm_after_conv='ln'):
        super().__init__()
        self.use_bias = True
        self.norm_type = norm
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        self.compute_kernel = True if norm == 'conv_kernel' else False
        self.WCT = True if norm == 'WCT' else False

        norm_dim = output_dim

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'WCT':
            self.norm = nn.InstanceNorm2d(norm_dim)
            self.style_dim = style_dim
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.output_dim = output_dim
            self.stride = stride
            self.mlp_W = nn.Sequential(
                nn.Linear(self.style_dim, output_dim**2),
            )
            self.mlp_bias = nn.Sequential(
                nn.Linear(self.style_dim, output_dim),
            )
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        elif norm == 'conv_kernel':
            self.style_dim = style_dim
            self.norm_after_conv = norm_after_conv
            self._get_norm(self.norm_after_conv, norm_dim)
            self.dim = output_dim, input_dim, kernel_size, kernel_size
            self.stride = stride
            self.mlp_kernel = nn.Linear(self.style_dim, int(np.prod(self.dim)))
            self.mlp_bias = nn.Linear(self.style_dim, output_dim)
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

        self.style = None

    def _get_norm(self, norm, norm_dim):
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

    def forward(self, x, spade_input=None):
        if self.compute_kernel:
            conv_kernel = self.mlp_kernel(self.style)
            conv_bias = self.mlp_bias(self.style)
            x = F.conv2d(self.pad(x), conv_kernel.view(*self.dim), conv_bias.view(-1), self.stride)
        else:
            x = self.conv(self.pad(x))
        if self.WCT:
            x_mean = x.mean(-1).mean(-1)
            x = x.permute(0, 2, 3, 1)
            x = x - x_mean
            W = self.mlp_W(self.style)
            bias = self.mlp_bias(self.style)
            W = W.view(self.output_dim, self.output_dim)
            x = x @ W
            x = x + bias
            x = x.permute(0, 3, 1, 2)
        if self.norm:
            if self.norm_type == 'spade':
                x = self.norm(x, spade_input)
            else:
                x = self.norm(x)
        if self.activation:
            x = self.activation(x)

        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out