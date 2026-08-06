# Reproducibility

## Scope

This repository is the TaFD method release. It contains the method code,
ablation entrypoints, attack/evaluation utilities, and vendored dependencies
needed by the TaFD scripts. It does not include datasets, paper result
archives, or checkpoints.

## Default Paper Configuration

- Threat domains: K=2
- Backbones: ResNet and MobileViT
- Datasets: CIFAR-10, CIFAR-100, and Imagenette
- Imagenette preprocessing: RandomResizedCrop(224) for training and
  Resize(256) plus CenterCrop(224) for evaluation
- Main entrypoint: train_tafd.py

## Dataset Root

Always pass --dataset_path explicitly. The code no longer uses private machine
paths. CIFAR datasets can be downloaded automatically by torchvision.
Imagenette must already exist under the dataset root.

## Checkpoints

Checkpoints are not committed. To evaluate an archived model, pass:

    --resume /path/to/latest_model.pth

The checkpoint must match the dataset, backbone, attack configuration, and K
used by the command.

## Smoke Testing

Before training, run:

    python -m py_compile train_tafd.py
    python train_tafd.py --help

For a full run, use a GPU and set --gpu to the intended CUDA device.
