# LaViSE
This repository contains our heavily refactored code for the paper [Explaining Deep Convolutional Neural Networks via Latent Visual-Semantic Filter Attention](https://arxiv.org/abs/2204.04601) which appeared in CVPR 2022. We found the [paper's original code](https://github.com/YuYang0901/LaViSE) to be error-ridden, inefficient, poorly documented and incomplete, with key components such as the implementation of the main quantitative evaluation metric (recall) missing, which is what this version of the code aims to fix.


# FACT at the UvA
This repository was made as part of a project done during the course Fairness, Accountability, Confidentiality and Transparency in AI (FACT) at the University of Amsterdam (UvA) in 2023. It follows the setup of the [Machine Learning Reproducibility Challange (MLRC)](https://paperswithcode.com/rc2022).

The main goal of the project was to assess the reproducibility of an existing state-of-the-art research paper in the field of FACT by reimplementing an algorithm, replicating and/or extending the experiments from the corresponding paper, and detailing findings in a report. Our report can be found [here TODO].


# Requirements
One of our main contributions is making the experiments and code easily reproducible. Therefore, we took the greatest care in writing the following guide to be as concise and easy to follow as possible.

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
For training an explainer to explain ResNet18's layer 4 using the COCO dataset, run:
```commandline
python3 train_explainer.py --model resnet18 --layer-target layer4 --refer coco --epochs 10
```
Note that each epoch takes around 30 minutes on a single A100 GPU, and around an hour on a Titan RTX GPU. The trained explainer will be saved to the `outputs` folder.


# Citation of the original paper
```
@inproceedings{yang2022explaining,
    author    = {Yang, Yu and Kim, Seungbae and Joo, Jungseock},
    title     = {Explaining Deep Convolutional Neural Networks via Unsupervised Visual-Semantic Filter Attention},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2022},
}
```
