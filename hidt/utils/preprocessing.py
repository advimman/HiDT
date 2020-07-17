import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

RESIZE_BASE = 16


def pytorch_preprocess(batch):
    """
    The scaling procedure for all the pretrained models from torchvision is described in the docs
    https://pytorch.org/docs/stable/torchvision/models.html
    """
    batch = (batch + 1) * 0.5  # [-1, 1] -> [0, 1]

    batch_color_transformed = []
    batch = torch.stack(batch_color_transformed, 0)

    batch = torch.clamp(batch, 0, 1)
    mean = torch.tensor([.485, .456, .406], dtype=batch.dtype, device=batch.device)[None, :, None, None]
    batch = batch.sub(mean)  # subtract mean
    std = torch.tensor([.229, .224, .225], dtype=batch.dtype, device=batch.device)[None, :, None, None]
    batch = batch.div(std)
    return batch


def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt['preprocess'] == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt['preprocess'] == 'scale_width_and_crop':
        new_w = opt['load_size']
        new_h = opt['load_size'] * h // w
    elif opt['preprocess'] == 'scale_shorter_side_and_crop':
        if w < h:
            new_w = opt['load_size']
            new_h = opt['load_size'] * h // w
        else:
            new_w = opt['load_size'] * w // h
            new_h = opt['load_size']

    x = random.randint(0, np.maximum(0, new_w - opt['crop_image_width']))
    y = random.randint(0, np.maximum(0, new_h - opt['crop_image_height']))

    flip = random.random() > 0.5

    return {'preprocess': opt['preprocess'], 'no_flip': opt['no_flip'], 'load_size': opt['load_size'],
            'crop_image_width': opt['crop_image_width'], 'crop_image_height': opt['crop_image_height'],
            'crop_pos': (x, y), 'new_size': (new_w, new_h), 'flip': flip,
            'color_space': opt.get('color_space', None),
            'dequantization': opt['dequantization']}


def get_transform(params, method=Image.BICUBIC, convert=True):
    transform_list = []
    if params['color_space'] == 'grayscale':
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in params['preprocess']:
        osize = [params['load_size'], params['load_size']]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in params['preprocess']:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, params['load_size'], method)))
    elif 'scale_shorter_side' in params['preprocess']:
        transform_list.append(transforms.Lambda(lambda img: __scale(img,
                                                                    params['new_size'][0],
                                                                    params['new_size'][1], method)))
    elif 'scale_load_shorter_side' in params['preprocess']:
        transform_list.append(transforms.Lambda(lambda img: __scale_shorter(img,
                                                                            params['load_size'], method)))

    if 'crop' in params['preprocess']:
        if params is None:
            transform_list.append(transforms.RandomCrop((params['crop_image_width'], params['crop_image_height'])))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'],
                                                                       params['crop_image_width'],
                                                                       params['crop_image_height'])))

    if params['preprocess'] == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=RESIZE_BASE, method=method)))

    if not params['no_flip']:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params.get('flip', False):
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        if params['color_space'] == 'labeled':
            transform_list += [
                transforms.Lambda(lambda x: torch.tensor([np.array(x)])[:, :, :, 0].type(torch.LongTensor))]
        else:
            transform_list += [transforms.ToTensor()]
            if params['dequantization']:
                transform_list += [transforms.Lambda(lambda x: __dequantization(x))]
            if params['color_space'] == 'grayscale':
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def transform_with_color_space(img, transform_params, color_space):
    transform_params['color_space'] = color_space

    transform = get_transform(transform_params)
    return transform(img)


def default_loader(path):
    # with Image.open(path) as img:
    img = Image.open(path)
    if img.mode != 'RGB':
        return img.convert('RGB')
    return img


class GridCrop:
    """
    Make crops from initial image with pad and stride
    """

    def __init__(self, pad=4, stride=1, hires_size=1024):
        self.pad = pad
        self.stride = stride
        self.initial_transform = transforms.Resize(hires_size + 4)
        self.final_transform = transforms.Resize(hires_size // 4)

    def __call__(self, initial_image):
        output = []
        initial_image = self.initial_transform(initial_image)
        w, h = initial_image.size
        for x1 in range(0, self.pad, self.stride):
            for y1 in range(0, self.pad, self.stride):
                coords = (x1, y1, x1 + w, y1 + h)
                output.append(self.final_transform(initial_image.crop(coords)))
        return output


def enhancement_preprocessing(transferred_crops: List[torch.Tensor], normalize: bool = True) -> torch.Tensor:
    enh_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    padded_stack = []
    for idx, transferred_crop in enumerate(transferred_crops):
        hor_idx = idx // 4
        vert_idx = idx % 4
        c, w, h = transferred_crop.shape
        padded_image = __padd_with_idxs(F.interpolate(transferred_crop[None, :, :, :],
                                                      size=(int(w * 4),
                                                            int(h * 4)),
                                                      mode='nearest'),
                                        vert_idx,
                                        hor_idx,
                                        pad=4)
        if normalize:
            padded_image = enh_norm(padded_image[0]).unsqueeze(0)
        padded_image = F.interpolate(padded_image,
                                     (w, h),
                                     mode='nearest')
        padded_stack.append(padded_image)

    return torch.cat(padded_stack, dim=1)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    return img.resize((w, h), method)


def __scale(img, target_width, target_height, method=Image.BICUBIC):
    return img.resize((target_width, target_height), method)


def __scale_shorter(img, target_size, method=Image.BICUBIC):
    if img.size[0] > img.size[1]:
        return __make_power_2(img.resize((int(img.size[0] * (target_size / img.size[1])), target_size), method),
                              RESIZE_BASE)
    else:
        return __make_power_2(img.resize((target_size, int(img.size[1] * (target_size / img.size[0]))), method),
                              RESIZE_BASE)


def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_width:
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


def __crop(img, pos, size_width, size_height):
    ow, oh = img.size
    x1, y1 = pos
    if ow > size_width or oh > size_height:
        return img.crop((x1, y1, x1 + size_width, y1 + size_height))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __dequantization(img):
    return img + torch.rand_like(img) / 255


def __padd_with_idxs(img, vert_idx, hor_idx, pad):
    b, c, h, w = img.shape
    padded = torch.zeros((b, c, h + pad, w + pad), device=img.device)
    padded[:, :, vert_idx:h + vert_idx, hor_idx:w + hor_idx] += img
    return padded
