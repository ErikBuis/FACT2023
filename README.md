Code DOI: https://zenodo.org/badge/latestdoi/585223762 <br />
OpenReview URL: https://openreview.net/forum?id=nsrHznwHhl <br />
Paper PDF: https://openreview.net/pdf?id=nsrHznwHhl

# LaViSE
This repository is a heavily refactored implementation of [Explaining Deep Convolutional Neural Networks via Latent Visual-Semantic Filter Attention](https://arxiv.org/abs/2204.04601) as part of the [Machine Learning Reproducibility Challange (MLRC)](https://paperswithcode.com/rc2022) 2022. The original code from the authors can be found [here](https://github.com/YuYang0901/LaViSE).

# Requirements
One of our main contributions is making the experiments and code easily reproducible. Therefore, we took the greatest care in writing the following guide to be as concise and easy to follow as possible. If you have any questions or comments, don't hestitate to contact us (preferably via email).

## Environment
To set up the correct environment, we recommend downloading the [Conda](https://docs.conda.io/en/latest/) package manager. Once installed, create a new environment with the following commands:
```commandline
conda config --set channel_priority flexible
conda env create -f fact2023.yml
```

Be aware that this can take a while (depending on the hardware and network speed, around 10 to 40 minutes). Once the environment is created, activate it with:
```commandline
conda activate fact2023
```

## Datasets
The original paper used two datasets: [Common Objects in Context (COCO)](https://cocodataset.org/) and [Visual Genome (VG)](https://visualgenome.org/). We recommend downloading the datasets using the following scripts, which will also preprocess the data. Alternatively, you can download the datasets manually via the aforementioned websites and preprocess them yourself. The datasets will be downloaded to the `data` folder.

### COCO
[Common Objects in Context (COCO)](https://cocodataset.org/) is a large-scale object detection, segmentation, and captioning dataset. To download COCO and preprocess the dataset, run:
```commandline
sh ./setup_coco.sh
```
Around 20 GB of data will be downloaded and processed.

### VG
[Visual Genome (VG)](https://visualgenome.org/) is a large-scale dataset of images annotated with object and region bounding boxes, object and attribute labels, and image-level relationships. To download VG and preprocess the dataset, run:
```commandline
sh ./setup_vg.sh
```
Around 15 GB of data will be downloaded and processed.

## Getting started

### Usage
Train an explainer with:
```commandline
python3 train_explainer.py
```

Explain a target filter of any model with:
```commandline
python3 infer_filter.py
```

### Examples
For training an explainer to explain ResNet1-8's layer 4 using the VG dataset, run:
```commandline
python3 train_explainer.py --model resnet18 --layer-target layer4 --layer-classifier fc --refer vg --epochs 10
```
Each epoch takes around 30 minutes on a single A100 GPU, and around an hour on a Titan RTX GPU. The trained explainer will be saved to the `outputs` folder.

For training an explainer to explain AlexNet's feature layer using the COCO dataset, run:
```commandline
python3 train_explainer.py --model alexnet --layer-target features --layer-classifier classifier --refer coco --epochs 10
```
Each epoch takes around 30 minutes on a single A100 GPU, and around an hour on a Titan RTX GPU. The trained explainer will be saved to the `outputs` folder.

To run inference on the trained ResNet-18 layer 4 explainer and evaluate it with recall@20, use the following command:
```commandline
python3 infer_filter.py --model resnet18 --layer-target layer4 --layer-classifier fc --refer vg --path-model "path-to-trained-model-here" --s 20
```
# Citation of the original paper
```
@inproceedings{yang2022explaining,
  author    = {Yang, Yu and Kim, Seungbae and Joo, Jungseock},
  title     = {Explaining Deep Convolutional Neural Networks via Unsupervised Visual-Semantic Filter Attention},
  booktitle = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022},
  pages     = {8323-8333},
  doi       = {10.1109/CVPR52688.2022.00815}
}
```
