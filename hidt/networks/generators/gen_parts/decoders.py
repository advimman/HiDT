__all__ = ['DecoderBase',
           'DecoderAdaINBase',
           'DecoderUnet',
           'DecoderAdaINConvBase',
           ]

import torch
from torch import nn

import hidt.networks.blocks.modules
from hidt.networks.blocks import Conv2dBlock
from hidt.networks.blocks import ResBlocks
from hidt.utils.base import module_list_forward


class DecoderBase(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.body = nn.ModuleList()
        self.upsample_head = nn.ModuleList()

        self._init_modules(**kwargs)

    def _init_modules(self, **kwargs):
        raise NotImplementedError

    def forward(self, tensor, spade_input=None):
        tensor = module_list_forward(self.body, tensor, spade_input)

        for layer in self.upsample_head:
            tensor = layer(tensor)

        return tensor


class DecoderAdaINBase(DecoderBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        adain_net_config = kwargs['adain_net']
        architecture = adain_net_config.pop('architecture')
        num_adain_params = self._calc_adain_params()
        adain_net_config['output_dim'] = num_adain_params
        self.adain_net = getattr(hidt.networks.blocks.modules, architecture)(**adain_net_config)
        self.style_dim = adain_net_config['input_dim']
        self.pred_adain_params = 'adain' == kwargs['res_norm'] or 'adain' == kwargs['up_norm'] or 'adain' == kwargs[
            'norm_after_conv']

    def _calc_adain_params(self):
        return self.get_num_adain_params(self)

    @staticmethod
    def get_num_adain_params(model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ("AdaptiveInstanceNorm2d", 'AdaLIN'):
                num_adain_params += 2 * m.num_features
        return num_adain_params

    @staticmethod
    def assign_adain_params(adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ('AdaptiveInstanceNorm2d', 'AdaLIN'):
                assert adain_params.shape[1]
                mean = adain_params[:, :m.num_features]
                assert mean.shape[1]
                std = adain_params[:, m.num_features:2 * m.num_features]
                assert std.shape[1]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) >= 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, content_tensor, style_tensor, spade_input=None):
        if self.pred_adain_params:
            adain_params = self.adain_net(style_tensor)
            self.assign_adain_params(adain_params, self)
        return super().forward(content_tensor, spade_input)


class DecoderAdaINConvBase(DecoderAdaINBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pred_conv_kernel = 'conv_kernel' == kwargs['res_norm'] or 'conv_kernel' == kwargs['up_norm'] or 'WCT' == kwargs['res_norm']

    @staticmethod
    def assign_style(style, model):
        # assign a style to the Conv2dBlocks
        for m in model.modules():
            if m.__class__.__name__ == "Conv2dBlock":
                m.style = style

    def forward(self, content_tensor, style_tensor, spade_input=None):
        if self.pred_conv_kernel:
            assert style_tensor.size(0) == 1, 'prediction of convilution does not work with batch size > 1'
            self.assign_style(style_tensor.view(1, -1), self)
        return super().forward(content_tensor, style_tensor, spade_input)


class DecoderUnet(DecoderAdaINConvBase):
    def _init_modules(self, **kwargs):
        self.num_upsamples = kwargs['num_upsamples']
        self.body += [ResBlocks(kwargs['num_blocks'],
                                kwargs['dim'],
                                norm=kwargs['res_norm'],
                                activation=kwargs['activ'],
                                pad_type=kwargs['pad_type'],
                                style_dim=kwargs.get('style_dim', 3))]

        self.upsample_postprocess = nn.ModuleList()
        self.skip_preprocess = nn.ModuleList()

        dim = kwargs['dim']
        skip_dim = kwargs['skip_dim']
        if isinstance(skip_dim, int):
            skip_dim = [skip_dim] * kwargs['num_upsamples']
        skip_dim = skip_dim[::-1]

        for i in range(kwargs['num_upsamples']):
            self.upsample_head += [nn.Upsample(scale_factor=2)]
            current_upsample_postprocess = [
                Conv2dBlock(dim + skip_dim[i],
                            dim // 2, 7, 1, 3,
                            norm=kwargs['up_norm'],
                            activation=kwargs['activ'],
                            pad_type=kwargs['pad_type'],
                            style_dim=kwargs.get('style_dim', 3),
                            norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                            )]
            if kwargs['num_res_conv']:
                current_upsample_postprocess += [ResBlocks(kwargs['num_res_conv'],
                                                           dim // 2,
                                                           norm=kwargs['up_norm'],
                                                           activation=kwargs['activ'],
                                                           pad_type=kwargs['pad_type'],
                                                           style_dim=kwargs.get('style_dim', 3),
                                                           norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                                                           )]

            current_skip_preprocess = [Conv2dBlock(skip_dim[i],
                                                   skip_dim[i], 7, 1, 3,
                                                   norm=kwargs['res_norm'],
                                                   activation=kwargs['activ'],
                                                   pad_type=kwargs['pad_type'],
                                                   style_dim=kwargs.get('style_dim', 3),
                                                   norm_after_conv=kwargs.get('norm_after_conv', 'ln'),
                                                   )]

            self.upsample_postprocess += [nn.Sequential(*current_upsample_postprocess)]
            self.skip_preprocess += [nn.Sequential(*current_skip_preprocess)]
            dim //= 2

        # use reflection padding in the last conv layer
        self.model_postprocess = nn.ModuleList([Conv2dBlock(dim, kwargs['output_dim'], 9, 1, 4,
                                                            norm='none',
                                                            activation='none',
                                                            pad_type=kwargs['pad_type'])])

    def forward(self, content_list, style_tensor, spade_input=None, pure_generation=False):
        if self.pred_adain_params:
            adain_params = self.adain_net(style_tensor)
            self.assign_adain_params(adain_params, self)

        if self.pred_conv_kernel:
            assert style_tensor.size(0) == 1, 'prediction of convilution does not work with batch size > 1'
            self.assign_style(style_tensor.view(1, -1), self)

        tensor = module_list_forward(self.body, content_list[0], spade_input)
        for skip_content, up_layer, up_postprocess_layer, skip_preprocess_layer in zip(content_list[1:],
                                                                                       self.upsample_head,
                                                                                       self.upsample_postprocess,
                                                                                       self.skip_preprocess):
            tensor = up_layer(tensor)
            skip_tensor = skip_preprocess_layer(skip_content)
            tensor = torch.cat([tensor, skip_tensor], 1)
            tensor = up_postprocess_layer(tensor)
        tensor = module_list_forward(self.model_postprocess, tensor, spade_input)
        return tensor
