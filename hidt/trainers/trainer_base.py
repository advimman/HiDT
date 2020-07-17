__all__ = ['TrainerBase']

import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim

from hidt import networks

logger = logging.getLogger(__name__)


class TrainerBase(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.models_dict = defaultdict(list)
        models_parameters = self._init_models()

    def _init_models(self):
        parameters = defaultdict(list)
        for model_name, model_config in self.params['models'].items():
            architecture = self.params['models'][model_name]['architecture']
            logger.debug(f'Building {model_name} with: {architecture}')
            setattr(self,
                    model_name,
                    getattr(networks, architecture)(model_config).cuda()
                    )
            parameters[model_config['optimizer_group']].extend(
                getattr(self, model_name).parameters())
            self.models_dict[model_config['optimizer_group']].append(
                model_name)
        return parameters

    def forward(self, images_a, images_b):
        raise NotImplementedError

    def encode(self, data):
        if self.params['train_mode'] == 'DataParallel':
            bs = data['images'].shape[0]
            return nn.parallel.data_parallel(self.gen, data,
                                             device_ids=list(
                                                 range(bs)) if bs < torch.cuda.device_count() else None,
                                             module_kwargs={'mode': 'encode'})
        else:
            return self.gen(data, mode='encode')

    def decode(self, decomposition):
        if self.params['train_mode'] == 'DataParallel':
            bs = decomposition['content'].shape[0]
            return nn.parallel.data_parallel(self.gen, decomposition,
                                             device_ids=list(
                                                 range(bs)) if bs < torch.cuda.device_count() else None,
                                             module_kwargs={'mode': 'decode'})
        else:
            return self.gen(decomposition, mode='decode')

    def _unpack_data(self, data):
        x_a = {}
        x_b = {}
        for map_name in data['a']:
            x_a[map_name] = data['a'][map_name].cuda()
            x_b[map_name] = data['b'][map_name].cuda()
        return x_a, x_b

    def _mix_decompositions(self, source, target):
        """
        Swap the `style` fields of source and target
        """
        out = {key: value for key, value in source.items() if key != 'style'}

        if source['content'].shape[0] != target['style'].shape[0]:  # case of style transformer
            assert target['style'].shape[0] == 1
            out['style'] = torch.cat(
                [target['style']] * source['content'].shape[0])
        else:
            out['style'] = target['style']
        return out

    def _swap_decompositions(self, decomposition_a: dict, decomposition_b: dict):
        """
        Do something to swap styles between a and b
        """
        ab = self._mix_decompositions(decomposition_a, decomposition_b)
        ba = self._mix_decompositions(decomposition_b, decomposition_a)
        return ab, ba

    def _mix_and_decode_decompositions(self, decomposition_a: dict, decomposition_b: dict):
        ab = self._mix_decompositions(decomposition_a, decomposition_b)
        return self.decode(ab)

    def _swap_and_decode_decompositions(self, decomposition_a: dict, decomposition_b: dict):
        ab, ba = self._swap_decompositions(decomposition_a, decomposition_b)
        return self.decode(ab), self.decode(ba)
