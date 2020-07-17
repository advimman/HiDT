__all__ = ['ContentEncoderBase',
           'ContentEncoderBC',
           'ContentEncoderUnet',
           ]

from itertools import chain

from torch import nn
import torch

from hidt.networks.blocks import Conv2dBlock
from hidt.networks.blocks import ResBlocks
from hidt.utils.base import module_list_forward

from typing import List

class ContentEncoderBase(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model_preparation = nn.ModuleList()
        self.model_downsample = nn.ModuleList()
        self.model_postprocess = nn.ModuleList()

        self.output_dim = dim

    def forward(self, tensor, spade_input=None):
        model = chain(self.model_preparation, self.model_downsample, self.model_postprocess)
        return module_list_forward(model, tensor, spade_input)


class ContentEncoderBC(ContentEncoderBase):
    def __init__(self, num_downsamples, num_blocks, input_dim, dim, norm, activ, pad_type, non_local=False, **kwargs):
        super().__init__(dim)
        self.model_preparation += [Conv2dBlock(input_dim, dim, 9, 1, 4, norm=norm, activation=activ, pad_type=pad_type)]
        # downsampling blocks
        for i in range(num_downsamples):
            self.model_downsample += [
                Conv2dBlock(dim, 2 * dim, 6, 2, 2, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # residual blocks
        self.model_postprocess += [
            ResBlocks(num_blocks, dim, norm=norm, activation=activ, pad_type=pad_type, non_local=non_local)]


class ContentEncoderUnet(ContentEncoderBC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_dim = kwargs['skip_dim']
        if isinstance(self.skip_dim, int):
            self.skip_dim = [self.skip_dim] * kwargs['num_downsamples']

    def forward(self, tensor: torch.Tensor):
        output : List[torch.Tensor] = []
        for layer in self.model_preparation:
            tensor = layer(tensor)
        #tensor = module_list_forward(self.model_preparation, tensor, spade_input)

        for layer in self.model_downsample:
            skip_dim = 5
            if skip_dim > 0:
                out = tensor[:, :skip_dim]
            else:
                out = tensor
            output.append(out)
            tensor = layer(tensor)


        for layer in self.model_postprocess:
            tensor = layer(tensor)
        #tensor = module_list_forward(self.model_postprocess, tensor, spade_input)
        output.append(tensor)
        output_reversed: List[torch.Tensor] = [output[2], output[1], output[0]]
        return output_reversed
