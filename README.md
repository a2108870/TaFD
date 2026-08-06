# TaFD

Official release repository for TaFD, a threat-aware frequency-domain defense
for heterogeneous adversarial training.

This repository contains the TaFD method code used for the paper submission.
Third-party baselines such as PUAT, FACE, TRADES, RAMP, MNG, DAT, and GBN are
not included. The release focuses on reproducing TaFD and its method ablations.

## Method Components

- Threat-domain diagnosis: learns spectral prototypes and assigns heterogeneous
  attacks to threat domains.
- Diagnosis-dispatch: routes each input through threat-dependent frequency
  processing according to the diagnosed domain.
- Frequency-domain experts: FC-Conv modules apply domain-conditioned frequency
  transformations before classification.
- Default setting: K=2 threat domains, matching the main TaFD configuration in
  the paper.

## Repository Layout

- train_tafd.py: paper-aligned training and evaluation entrypoint.
- evaluate_adaptive_diagnosis.py: adaptive diagnosis-attack evaluation entrypoint.
- ablation_no_threat_domain_diagnosis.py: ablation used for removing diagnosis.
- ablation_direct_frequency_mask.py: direct-mask ablation entrypoint.
- ablation_no_assignment_alignment.py: assignment-alignment ablation entrypoint.
- ablation_standard_convolution.py: standard-convolution ablation entrypoint.
- main_train_pgdtrain.py: original implementation retained for compatibility.
- models/: TaFD backbones, threat-domain diagnosis, and FC-Conv implementation.
- attacks/: heterogeneous attack implementations used by TaFD.
- torchattacks/: vendored torchattacks source required by this code snapshot.
- utils/: dataset and training utilities.
- docs/: reproducibility and provenance notes.

## Installation

Create a Python environment with PyTorch and install the remaining dependencies:

    pip install -r requirements.txt

The code expects CUDA for full training and evaluation. CPU import tests are
supported, but paper-scale runs require a GPU.

## Dataset Layout

Place datasets under a user-controlled root and pass it through --dataset_path.
No dataset files are included in this repository.

Expected layouts:

    datasets/
      cifar100/
      tiny-imagenet-200-32x32/
      tiny-imagenet_10class_32/
      imagenette2-320/
        train/
        val/

Imagenette can also be named imagenette2, imagenette2-160, or imagenette.

## Training Examples

CIFAR-100, ResNet, v10, K=2:

    python train_tafd.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --batch_size 128 --test_batch_size 16 --end_epoch 76

Imagenette, ResNet, v10, K=2:

    python train_tafd.py --dataset Imagenette --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --batch_size 12 --test_batch_size 16 --end_epoch 76

Imagenette, MobileViT, v20, K=2:

    python train_tafd.py --dataset Imagenette --dataset_path ./datasets --backbone mobilevit --attack_config v20 --domains 2 --batch_size 12 --test_batch_size 16 --end_epoch 76

## Evaluation

Resume from a checkpoint by passing --resume:

    python train_tafd.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --resume ./checkpoints/latest_model.pth --start_epoch 75 --end_epoch 76

Adaptive diagnosis evaluation:

    python evaluate_adaptive_diagnosis.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --resume ./checkpoints/latest_model.pth

## Reproducibility Notes

- Generated outputs are written to results/ by default and are ignored by git.
- Checkpoints, logs, datasets, and local server scripts are intentionally
  excluded.
- The compatibility scripts keep the original experiment behavior. The clean
  wrapper names are provided so that commands match the paper terminology.
- For ablation results, use the ablation_* entrypoints instead of unrelated
  baseline code.

## License

This project is released under the MIT License. See LICENSE for details.
