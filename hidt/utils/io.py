from typing import List

import yaml
from PIL import Image
from tqdm import tqdm


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def save_img(img: Image.Image, path: str) -> None:
    img.save(path, format='JPEG',
             subsampling=0, quality=100)


def extract_images(images_list: List[str]) -> List[Image.Image]:
    output = []
    for im_path in tqdm(images_list):
        img = Image.open(im_path).convert('RGB')
        output.append(img)
    return output
