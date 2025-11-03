## Overview
This repository accompanies the paper submission titled "Deep Learning for the Detection of Melt Ponds on Sea Ice from Airborne Infrared Images". The repository contains code for segmenting helicopter-borne thermal infrared images into three classes: melt pond, sea ice, and ocean.

![alt text](image.png)

## Available Materials:
- Labeled Training Data: The labeled subset of the thermal infrared (TIR) images and manual annotations used for training the model are available in the `data/training/` directory.
- Fine-Tuned UNet++ Weights: The weights of the final fine-tuned UNet++ architecture, as presented in the paper, are stored in the `final_checkpoints/` directory.
- Surface Fraction Results: The surface fraction results from five helicopter flights, based on UNet++ classifications, can be found in the `runs/` directory. This is accompanied by a README.md file with further documentation.

## Dataset
The dataset used in this project is available in full on [PANGAEA](https://doi.org/10.1594/PANGAEA.971908). Please refer to the dataset for additional details and licensing information.

## Acknowledgments
This project includes code from the following sources:

- [AutoSAM](https://github.com/xhu248/AutoSAM), which is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). The respective code, license, and further information can be found in `models/`.
- [Segmentation Models](https://github.com/qubvel-org/segmentation_models.pytorch), which is licensed under the MIT license.
- [Satlas Pretrain](https://github.com/allenai/satlaspretrain_models), which is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- [SAM2 UNet](https://github.com/WZH0120/SAM2-UNet), which is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- [Pytorch_FID](https://github.com/mseitzer/pytorch-fid), which is licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- [DMT](https://github.com/voldemortX/DST-CBC?tab=readme-ov-file), licensed under [BSD 3-Clause License](https://opensource.org/license/bsd-3-clause).

AutoSAM itself relies on code from [SAM](https://github.com/facebookresearch/segment-anything/) and SAM2 UNet relies on code from [SAM2](https://github.com/facebookresearch/sam2). We gratefully acknowledge all original authors for making their code available.
We further acknowledge the authors of [remote_sensing_pretrained_models](https://github.com/lsh1994/remote_sensing_pretrained_models?tab=readme-ov-file) for making the pre-trained checkpoints for AID and RSD46-WHU publicly available.

## Getting Started

This code requires `python >= 3.10`.

To install additional dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Data and Weights

The data and weights contained in this repository are tracked with Git LFS. To use them, please install [Git LFS](https://git-lfs.com/).

### Pre-training Checkpoints

This repository relies on pre-training checkpoints from the following sources:

1) [remote_sensing_pretrained_models](https://github.com/lsh1994/remote_sensing_pretrained_models?tab=readme-ov-file) (Apache License 2.0):
- Please download the pre-trained weights from [here](https://github.com/lsh1994/remote_sensing_pretrained_models/releases/).
- The checkpoints of interest are:
            `resnet34_224-epoch.9-val_acc.0.966.ckpt` (AID)
            `resnet34-epoch.19-val_acc.0.921.ckpt` (RSD46-WHU)

2) [SAM](https://github.com/facebookresearch/segment-anything/) (Apache License 2.0)
3) [SAM2](https://github.com/facebookresearch/sam2) (Apache License 2.0)

Please store all downloaded checkpoints in the respective folders within the `pretraining_checkpoints/` directory.

### Experiment Tracking

This project uses [Weights & Biases (WandB)](https://wandb.ai) for experiment tracking.
To track fine-tuning experiments, a WandB account is needed.

## How to Use
### 1) Run Inference
To run inference on a helicopter flight of interest, follow these steps:

- Download the IR temperature netCDF file of the flight from [PANGAEA](https://doi.org/10.1594/PANGAEA.971908).
- Store the file in the `data/prediction/temperatures/` directory.
- Execute the following command to run the inference:

```bash
python -m scripts.run_inference --data "[PATH_TO_TEMPERATURE_FILE]"
```

- The prediction results will be stored in the `data/prediction/` directory.

### 2) Fine-Tune the Model
To fine-tune the model on 11 training images, use the following command:

```bash
python -m scripts.run_smp_finetune --pref "[PREFIX_OF_CHOICE]" --wandb_entity "[YOUR_EXISTING_WANDB_ENTITY]"
```

- The final fine-tuned model weights will be stored in the `models/weights/` directory.

### 3) Run Cross-Validation
To perform 3-fold cross-validation for performance evaluation, execute:

```bash
python -m scripts.crossvalidate_smp_torch --final_sweep --wandb_entity "[YOUR_EXISTING_WANDB_ENTITY]"
```

### Additional Notes
- WandB Configuration: To ensure the scripts run correctly, ensure that for fine-tuning and cross-validating, you set the `--wandb_entity` argument to an existing entity of your wandb account. You can also disable wandb tracking by specifying `--disable_wandb` while calling the respective script.

Contact: marlena1@gmx.de
