import logging
import math
import os
import time

import torch
import torch.nn.init as init
import torch.nn as nn
from typing import Iterable

logger = logging.getLogger(__name__)

NONLINEARITIES = dict(
    tanh=torch.tanh,
    sigmoid=torch.sigmoid,
    clamp=lambda x: torch.clamp(x, -1., 1.),
    logsoftmax=lambda x: torch.log_softmax(x, dim=1),
    none=lambda x: x,
)


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        logger.info(self.msg % (time.time() - self.start_time))


def get_latest_model_name(dirname, name):
    if os.path.exists(dirname) is False:
        return None
    files = [os.path.join(dirname, f) for f in os.listdir(dirname) if
             os.path.isfile(os.path.join(dirname, f)) and f.startswith('model.') and f.endswith('.pt') and name in f]
    if not len(files):
        return None
    files.sort()
    return files[-1]


def weights_init(init_type='gaussian'):
    def init_fun(module):
        classname = module.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(module, 'weight'):
            if init_type == 'gaussian':
                init.normal_(module.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(module.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(module.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(module, 'bias') and module.bias is not None:
                init.constant_(module.bias.data, 0.0)

    return init_fun


def stack_list_of_dicts_along_dim(list_of_dicts: list, dim=0):
    result = {}
    for level_1_key in list_of_dicts[0].keys():
        level_2_keys = set.intersection(*[set(el[level_1_key].keys()) for el in list_of_dicts])
        if 'label' in level_2_keys:
            level_2_keys.remove('label')
        result[level_1_key] = {}


        for elem in list_of_dicts:
            for level_2_key in level_2_keys:
                item = elem[level_1_key][level_2_key]
                if isinstance(item, torch.Tensor):
                    if level_2_key not in result[level_1_key]:
                        result[level_1_key][level_2_key] = []
                    result[level_1_key][level_2_key].append(item.unsqueeze(dim))
                else:
                    logger.debug(f'{item} is not a tensor => dont add')

        for level_2_key in level_2_keys:
                tensors = result[level_1_key][level_2_key]
                result[level_1_key][level_2_key] = torch.cat(tensors, dim=dim)

    return result


def get_total_data_dim(data_dict):
    """
    :param data_dict: data dict (input_data/output_data) from config, looks like:
         'output_dims': {
            'images': {'color_space': 'lab', 'dim': 3},
            'depth_map': {'color_space': 'grayscale', 'dim': 1}
            }

    :return: num of dims for example above 3 + 1 = 4
    """
    return sum(data_type['dim'] for data_type in data_dict.values())


def module_list_forward(module_list: nn.ModuleList, tensor: torch.Tensor,
                         spade_input=torch.zeros(1)):
    if spade_input:
        for layer in module_list:
            tensor = layer(tensor, spade_input)
    else:
        for layer in module_list:
            tensor = layer(tensor)

    return tensor


def split_tensor_to_maps(tensor, maps_types: dict) -> dict:
    output = {}
    i = 0
    for name, properties in maps_types.items():
        func = properties.get('func', 'tanh')
        output[name] = NONLINEARITIES[func](tensor[:, i:i + properties['dim']])
        i += properties['dim']
    return output


def atanh(x, eps=1e-5):
    """A poor realization of tanh^-1 (x)"""
    x = x.clamp(-1., 1.)
    out = 0.5 * (torch.log1p(x + eps) - torch.log1p(-x + eps))  # .clamp(-100., 100.)
    return out
