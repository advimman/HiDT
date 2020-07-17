import functools

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

try:
    from itertools import izip as zip
except ImportError:  # will be 3.x series
    pass


class AdaINHalfGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, input_dim, params):
        super(AdaINHalfGen, self).__init__()
        dim = params['dim']
        style_dim = params['style_dim']
        self.style_content_dim = params['style_content_dim']
        n_downsample = params['n_downsample']
        n_res = params['n_res']
        activ = params['activ']
        pad_type = params['pad_type']
        mlp_dim = params['mlp_dim']
        non_local = False
        if 'non_local' in params:
            non_local = params['non_local']
        # style encoder
        self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)

        # content encoder
        if 'big_conv' in params:
            self.enc_content = ContentEncoderBC(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type,
                                                non_local=non_local)
            self.dec = DecoderBC(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain',
                                 activ=activ,
                                 pad_type=pad_type,
                                 non_local=non_local)
        elif 'mb_conv' in params:
            self.enc_content = ContentEncoderMBconv(n_downsample, n_res, input_dim, dim, 'in', activ,
                                                    expand=params['mb_conv'], pad_type=pad_type, non_local=non_local)
            self.dec = DecoderMBconv(n_downsample, n_res, self.enc_content.output_dim, input_dim,
                                     expand=params['mb_conv'], res_norm='adain',
                                     activ=activ,
                                     pad_type=pad_type,
                                     non_local=non_local)
        elif 'mb_convs' in params:
            self.enc_content = ContentEncoderMBconvS(n_downsample, n_res, input_dim, dim, 'in', activ,
                                                     expand=params['mb_convs'], pad_type=pad_type, non_local=non_local)
            self.dec = DecoderMBconvS(n_downsample, n_res, self.enc_content.output_dim, input_dim,
                                      expand=params['mb_convs'], res_norm='adain',
                                      activ=activ,
                                      pad_type=pad_type,
                                      non_local=non_local)
        else:
            self.enc_content = ContentEncoder(n_downsample, n_res, input_dim, dim, 'in', activ, pad_type=pad_type)
            self.dec = Decoder(n_downsample, n_res, self.enc_content.output_dim, input_dim, res_norm='adain',
                               activ=activ,
                               pad_type=pad_type)

        # MLP to generate AdaIN parameters
        self.mlp = MLP(style_dim, self.get_num_adain_params(self.dec), mlp_dim, 3, norm='none', activ=activ)

    def forward(self, images):
        # reconstruct an image
        content, style_fake = self.encode(images)
        images_recon = self.decode(content, style_fake)
        return images_recon

    def encode(self, images):
        style = -1 * self.style_content_dim
        # encode an image to its content and style codes
        style_fake = self.enc_style(images)
        content = self.enc_content(images)
        style_c = content[:, style:].mean(dim=2).mean(dim=2).unsqueeze(2).unsqueeze(2).repeat(
            (1, 1, content.shape[2], content.shape[3]))
        return content[:, :style], style_fake, style_c

    def decode(self, content, style, content_style):
        # decode content and style codes to an image
        content = torch.cat((content, content_style), dim=1)
        adain_params = self.mlp(style)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2 * m.num_features
        return num_adain_params
