# TaFD

Official release repository for TaFD, a threat-aware frequency-domain defense
for heterogeneous adversarial training.

This repository contains the TaFD method code used for the paper submission and
the ablation entrypoints needed to reproduce the reported TaFD variants.

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
No dataset files are included in this repository. The paper experiments use
CIFAR-10, CIFAR-100, and Imagenette.

CIFAR-10 and CIFAR-100 are loaded through torchvision. For Imagenette, place the
dataset under one of the supported folder names:

    datasets/
      imagenette2-320/
        train/
        val/

Imagenette can also be named imagenette2, imagenette2-160, or imagenette.

## Training Examples

CIFAR-10, ResNet, v10, K=2:

    python train_tafd.py --dataset CIFAR10 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --batch_size 128 --test_batch_size 16 --end_epoch 76

CIFAR-100, ResNet, v10, K=2:

    python train_tafd.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --batch_size 128 --test_batch_size 16 --end_epoch 76

Imagenette, ResNet, v10, K=2:

    python train_tafd.py --dataset Imagenette --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --batch_size 12 --test_batch_size 16 --end_epoch 76

Imagenette, MobileViT, v20, K=2:

    python train_tafd.py --dataset Imagenette --dataset_path ./datasets --backbone mobilevit --attack_config v20 --domains 2 --batch_size 12 --test_batch_size 16 --end_epoch 76

The end_epoch argument is exclusive. The paper-style 75-epoch run is therefore
specified as --start_epoch 0 --end_epoch 76.

## Evaluation

Resume from a checkpoint by passing --resume:

    python train_tafd.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --resume ./checkpoints/latest_model.pth --start_epoch 75 --end_epoch 76

Adaptive diagnosis evaluation:

    python evaluate_adaptive_diagnosis.py --dataset CIFAR100 --dataset_path ./datasets --backbone resnet --attack_config v10 --domains 2 --resume ./checkpoints/latest_model.pth

## Reproducibility Notes

- Generated outputs are written to results/ by default and are ignored by git.
- Checkpoints, logs, datasets, and local launch scripts are intentionally
  excluded.
- The compatibility scripts keep the original experiment behavior. The clean
  wrapper names are provided so that commands match the paper terminology.
- For ablation results, use the ablation_* entrypoints listed above.

## License

This project is released under the MIT License. See LICENSE for details.
