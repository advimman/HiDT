__all__ = ['GeneratorBase']

import copy
import logging

from torch import nn

from hidt.networks.generators import gen_parts
from hidt.utils.base import weights_init, get_total_data_dim, split_tensor_to_maps

logger = logging.getLogger(__name__)


class GeneratorBase(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._check_config(self.params)
        self._init_modules(self.params['modules'])
        self.apply(weights_init(self.params['initialization']))

    def _init_modules(self, params):
        for module_name, module_config in params.items():
            module_config_copy = copy.deepcopy(module_config)
            architecture = module_config_copy.pop('architecture')
            frozen = module_config_copy.pop('frozen', False)

            if 'input_data' in module_config_copy:
                module_config_copy['input_dim'] = get_total_data_dim(module_config_copy['input_data'])
                module_config_copy.pop('input_data')
            if 'output_data' in module_config_copy:
                module_config_copy['output_dim'] = get_total_data_dim(module_config_copy['output_data'])
                module_config_copy.pop('output_data')

            logger.debug(f'Building {module_name} with {architecture}')
            setattr(self,
                    module_name,
                    getattr(gen_parts, architecture)(**module_config_copy)
                    )

            if frozen:
                for param in getattr(self, module_name).parameters():
                    param.requires_grad = False
                logger.debug(f'{module_name} was frozen')

    def _check_config(self, params):
        """
        Assure module has all necessary submodules with correct names
        """
        raise NotImplementedError

    def forward(self, data, mode=None):
        if mode is None:
            # reconstruct an image
            decomposition = self.encode(data)
            output = self.decode(decomposition)
            return output
        if mode == 'encode':
            return self.encode(data)
        if mode == 'decode':
            return self.decode(data)
        if mode == 'mapper':
            return self.mapper(data['style'])

    def encode(self, data):
        """
        Define input tensors for encoders here and process those tensors
        """
        raise NotImplementedError

    def decode(self, decomposition):
        """
        Take necessary tensors from the given decomposition and pass through your decoder
        """
        raise NotImplementedError

    def _split_decoded_tensor_to_maps(self, tensor) -> dict:
        return split_tensor_to_maps(tensor, self.params['modules']['decoder']['output_data'])
