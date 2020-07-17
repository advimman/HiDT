__all__ = ['MLP',
           'ResBlocks',
           'NormalizeOutput',
           ]

import torch.nn.functional as F
from torch import nn

from hidt.networks.blocks import ResBlock, LinearBlock


##################################################################################
# Sequential Models
##################################################################################

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero', non_local=False,
                 style_dim=3, norm_after_conv='ln'):
        super(ResBlocks, self).__init__()
        self.model = []
        if isinstance(non_local, (list,)):
            for i in range(num_blocks):
                if i in non_local:
                    raise DeprecationWarning()
                else:
                    self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                            style_dim=style_dim, norm_after_conv=norm_after_conv)]
        else:
            for i in range(num_blocks):
                self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type,
                                        style_dim=style_dim, norm_after_conv=norm_after_conv)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, num_blocks, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(num_blocks - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


class NormalizeOutput(nn.Module):
    """
    Module that scales the input tensor to the unit norm w.r.t. the specified axis.
    Actually, the module analog of `torch.nn.functional.normalize`
    """

    def __init__(self, dim=1, p=2, eps=1e-12):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.p = p

    def forward(self, tensor):
        return F.normalize(tensor, p=self.p, dim=self.dim, eps=self.eps)