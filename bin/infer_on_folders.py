import argparse
import glob
import os

import torch
from torchvision import transforms
from tqdm import tqdm

from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing
from hidt.utils.io import save_img, extract_images


def main():
    parser = argparse.ArgumentParser(description='Transfer images to styles.')
    parser.add_argument('--content-dir', type=str, dest='data_dir',
                        help='path to content images directory',
                        )
    parser.add_argument('--style-dir', type=str, dest='style_dir',
                        help='path to style images directory',
                        )
    parser.add_argument('--cfg-path', type=str, dest='cfg_path',
                        help='path to model config',
                        )
    parser.add_argument('--chk-path', type=str, dest='weight_path',
                        help='path to model weights',
                        )
    parser.add_argument('--enh-path', type=str, dest='enh_weights_path',
                        help='path to enhancer weights',
                        )
    parser.add_argument('--enhancement', type=str, dest='enhancement',
                        help='enhancement strategy: generator | fullconv',
                        default='generator'
                        )
    parser.add_argument('--inference-size', type=int, dest='inference_size',
                        help='size to rescale model to', default=256,
                        )
    parser.add_argument('--device', type=str,
                        help='type of inference device',
                        default='cuda',
                        )
    parser.add_argument('--batch-size', type=int, dest='batch_size',
                        default=4)
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        help='where to store transferred images',
                        default='.'
                        )

    args = parser.parse_args()
    style_transformer = StyleTransformer(args.cfg_path, args.weight_path,
                                         inference_size=args.inference_size,
                                         device=args.device)

    result_path = os.path.join(args.output_dir, 'results/')
    os.makedirs(result_path, exist_ok=True)
    print("result path : ", result_path)

    source_images_path = glob.glob(os.path.join(args.data_dir, '*'))
    style_images_path = glob.glob(os.path.join(args.style_dir, '*'))

    source_images_pil = extract_images(source_images_path)
    style_images_pil = extract_images(style_images_path)

    with torch.no_grad():
        styles_decomposition = style_transformer.get_style(style_images_pil)
        if args.enhancement == 'generator':
            g_enh = RRDBNet(in_nc=48,
                            out_nc=3,
                            nf=64,
                            nb=5,
                            gc=32).to(torch.device(args.device))
            g_enh.load_state_dict(torch.load(args.enh_weights_path))
            result_images = []
            crop_transform = GridCrop(4, 1, hires_size=args.inference_size * 4)
            for style in styles_decomposition:
                styled_imgs = []
                for source_image in source_images_pil:
                    crops = [img for img in crop_transform(source_image)]
                    out = style_transformer.transfer_images_to_styles(crops,
                                                                      [style],
                                                                      batch_size=args.batch_size,
                                                                      return_pil=False)
                    padded_stack = enhancement_preprocessing(out[0])
                    out = g_enh(padded_stack)
                    styled_imgs.append([transforms.ToPILImage()((out[0].cpu().clamp(-1, 1) + 1.) / 2.)])
                result_images.append(styled_imgs)

        elif args.enhancement == 'fullconv':
            result_images = []
            for style in styles_decomposition:
                one_style_out = style_transformer.transfer_images_to_styles(source_images_pil,
                                                                            [style],
                                                                            batch_size=args.batch_size,
                                                                            return_pil=True)
                result_images.append(one_style_out)

        else:
            raise ValueError('Choose one of the following enhancement schemes: fullconv or generator')

        for i, content_img_path in tqdm(enumerate(source_images_path)):
            source_name = content_img_path.split('/')[-1].split('.')[0]
            for j, style_img_path in tqdm(enumerate(style_images_path)):
                style_name = style_img_path.split('/')[-1].split('.')[0]
                save_img(result_images[j][i][0],
                         os.path.join(result_path,
                                      source_name + '_to_' + style_name + '.jpg')
                         )


if __name__ == '__main__':
    main()
