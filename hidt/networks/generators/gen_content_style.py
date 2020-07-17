__all__ = ['GeneratorContentStyle']

from collections import defaultdict

import torch

from hidt.networks.generators import GeneratorBase


class GeneratorContentStyle(GeneratorBase):
    # AdaIN auto-encoder architecture

    def __init__(self, params):
        super().__init__(params)
        self.style_dim = self.decoder.style_dim

    def _check_config(self, params):
        assert 'content_encoder' in params['modules']
        assert 'style_encoder' in params['modules']
        assert 'decoder' in params['modules']

    def encode_style(self, data, batch_size=None):
        styles = []
        styles.append(self.style_encoder(data))

        return dict(
            style=torch.cat(styles),
        )

    def encode_style_batch(self, data, batch_size=None):
        styles = []
        if batch_size is None:
            batch_size = data['images'].shape[0]

        for images in data['images'].split(batch_size):
            styles.append(self.style_encoder(images))

        return dict(
            style=torch.cat(styles),
        )

    def encode_content(self, data, batch_size=None):
        contents = []
        if batch_size is None:
            batch_size = data['images'].shape[0]

        for images in data['images'].split(batch_size):
            contents.append(self.content_encoder(images))

        return dict(
            content=torch.cat(contents),
        )

    def encode(self, data, batch_size=None):
        output = self.encode_content(data, batch_size=batch_size)
        output.update(self.encode_style(data, batch_size=batch_size))
        return output

    def decode(self, decomposition, batch_size=None):
        if batch_size is None:
            batch_size = decomposition['content'].shape[0]
        output_maps = defaultdict(list)

        for cur_content, cur_style in zip(decomposition['content'].split(batch_size),
                                          decomposition['style'].split(batch_size)):
            cur_tensor = self.decoder(cur_content, cur_style)
            cur_maps = self._split_decoded_tensor_to_maps(cur_tensor)
            for map_name, map_value in cur_maps.items():
                output_maps[map_name].append(map_value)

        output_maps = {map_name: torch.cat(map_value) for map_name, map_value in output_maps.items()}
        return output_maps
