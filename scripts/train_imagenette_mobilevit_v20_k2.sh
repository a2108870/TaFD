#!/usr/bin/env bash
set -euo pipefail

python train_tafd.py \
  --dataset Imagenette \
  --dataset_path ./datasets \
  --backbone mobilevit \
  --attack_config v20 \
  --domains 2 \
  --batch_size 12 \
  --test_batch_size 16 \
  --end_epoch 76
