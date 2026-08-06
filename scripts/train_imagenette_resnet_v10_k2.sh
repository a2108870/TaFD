#!/usr/bin/env bash
set -euo pipefail

python train_tafd.py \
  --dataset Imagenette \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_config v10 \
  --domains 2 \
  --batch_size 12 \
  --test_batch_size 16 \
  --end_epoch 76
