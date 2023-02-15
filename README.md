# High-Resolution Daytime Translation Without Domain Labels

### [Project Page](https://advimman.github.io/HiDT/) | [Video Explanation](https://youtu.be/DALQYKt-GJc) | [Paper](https://arxiv.org/abs/2003.08791) | [Appendix](https://advimman.github.io/HiDT/paper/High-Resolution_Daytime_Translation_Without_Domain_Labels.pdf) | [TwoMinutePapers](https://www.youtube.com/watch?v=EWKAgwgqXB4)

[![Open HiDT in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/advimman/hidt/blob/master/notebooks/HighResolutionDaytimeTranslation.ipynb)

Official PyTorch implementation (only inference part) for the paper I. Anokhin, P. Solovev, D. Korzhenkov, A. Kharlamov, T. Khakhulin, A. Silvestrov, S. Nikolenko, V. Lempitsky, and G. Sterkin. "High-Resolution Daytime Translation Without Domain Labels." In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
![teaser image](./docs/img/1_intro_grid.jpg)

## Installation
Make sure that you use python >= 3.7. We have tested it with conda package manager. If you are new to conda, proceed to https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

```
conda create -n hidt python=3.7
conda activate hidt
```
#### Clone the repo
```
git clone https://github.com/advimman/HiDT.git
```
#### Install requirenments
```
cd HiDT
pip install -r requirements.txt
```

## Inference
Daytime translation, upsampling with G<sub>enh</sub>
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):${PYTHONPATH} \
python ./bin/infer_on_folders.py \
    --content-dir ./images/daytime/content/ \
    --style-dir ./images/daytime/styles/ \
    --cfg-path ./configs/daytime.yaml \
    --chk-path ./trained_models/generator/daytime.pt \
    --enh-path ./trained_models/enhancer/enhancer.pth \
    --enhancement generator
```
Daytime translation, generator in fully convolutional mode, no postprocessing
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):${PYTHONPATH} \
python ./bin/infer_on_folders.py \
    --content-dir ./images/daytime/content/ \
    --style-dir ./images/daytime/styles/ \
    --cfg-path ./configs/daytime.yaml \
    --chk-path ./trained_models/generator/daytime.pt \
    --enhancement fullconv
```
Model, trained on wikiart, upsampling with G<sub>enh</sub>
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):${PYTHONPATH} \
python ./bin/infer_on_folders.py \
    --content-dir ./images/wikiart/content/ \
    --style-dir ./images/wikiart/styles/ \
    --cfg-path ./configs/wikiart.yaml \
    --chk-path ./trained_models/generator/wikiart.pt \
    --enh-path ./trained_models/enhancer/enhancer.pth \
    --enhancement generator
```
Model, trained on wikiart, generator in fully convolutional mode, no postprocessing
```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$(pwd):${PYTHONPATH} \
python ./bin/infer_on_folders.py \
    --content-dir ./images/wikiart/content/ \
    --style-dir ./images/wikiart/styles/ \
    --cfg-path ./configs/wikiart.yaml \
    --chk-path ./trained_models/generator/wikiart.pt \
    --enhancement fullconv
```

## Citation
If you found our work useful, please don't forget to cite
```
@inproceedings{Anokhin_2020_CVPR,
  author = {Anokhin, Ivan and
            Solovev, Pavel and
            Korzhenkov, Denis and
            Kharlamov, Alexey and
            Khakhulin, Taras and
            Silvestrov, Alexey and
            Nikolenko, Sergey and
            Lempitsky, Victor and
            Sterkin, Gleb
  },
  title = {High-Resolution Daytime Translation Without Domain Labels},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020},
}
```
