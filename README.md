# TaFD

Official implementation of **TaFD**, a threat-aware frequency-domain defense
for heterogeneous adversarial training.

The release provides the TaFD training pipeline, the four component ablations
reported in the paper, the heterogeneous attacks used by the method, and the
supplemental adaptive diagnosis evaluation.

## Method Overview

TaFD contains three paper-aligned components:

- **Threat-domain diagnosis** groups heterogeneous attack sources into learned
  threat domains from their spectral characteristics.
- **Diagnosis-dispatch** selects threat-dependent processing according to the
  diagnosed domain of each input.
- **Frequency-conditional convolution (FC-Conv)** applies a learned spectral
  mask and dispatches features to domain-specialized frequency experts.

The default configuration uses `K=2` threat domains.

## Repository Layout

```text
train_tafd.py                                      Main TaFD entrypoint
train_ablation_without_threat_domain_diagnosis.py Threat-domain diagnosis ablation
train_ablation_without_basis_parameterized_mask.py
train_ablation_without_hungarian_alignment.py
train_ablation_without_frequency_decoupling.py
evaluate_adaptive_diagnosis.py                    Adaptive diagnosis evaluation
models/                                           TaFD and FC-Conv implementations
training/                                         Training and validation pipelines
evaluation/                                       Supplemental evaluation utilities
attacks/                                          Heterogeneous attack implementations
scripts/                                          Paper-configuration examples
configs/                                          Configuration summaries
docs/                                             Reproducibility notes
```

## Installation

Create a Python environment with a CUDA-enabled PyTorch installation, then run:

```bash
pip install -r requirements.txt
```

CPU execution is supported for imports and smoke tests. Full training and
adversarial evaluation require a CUDA GPU.

## Datasets

Pass the dataset root explicitly with `--dataset_path`. CIFAR-10 and CIFAR-100
are loaded through `torchvision`. Imagenette must be placed under the dataset
root in ImageFolder format:

```text
datasets/
  imagenette2-320/
    train/
    val/
```

The loader also recognizes `imagenette2`, `imagenette2-160`, and `imagenette`.
All Imagenette experiments use `RandomResizedCrop(224)` during training and
`Resize(256)` followed by `CenterCrop(224)` during evaluation.

## Attack Unions

The public entrypoints select one of the two complete attack unions evaluated
in the paper. Individual attacks cannot be enabled or disabled from the CLI.

- `canonical`: training uses 10-step PGD-Linf, PGD-L2, ACE, HSVAdv,
  ReColorAdv, ALA, and RetouchUAA. Evaluation uses 100-step APGD-Linf and
  APGD-L2 together with ACE, ALA, HSVAdv, ReColorAdv, and RetouchUAA.
- `broader`: training uses 10-step PGD-Linf, PGD-L2, ACE, GPGD, and StAdv.
  Evaluation uses 100-step APGD-Linf and APGD-L2 together with ACE, StAdv,
  and GPGD.

The remaining attacks use the same configurations during training and
evaluation, as specified in the paper. The internal `APGD_Linf` and
`APGD_L2` source identifiers are retained solely for compatibility with the
released K=2 checkpoints.

GPGD PCA bases are loaded from `--gpgd_basis_path` or generated from the
training set and saved beside the run outputs.

## Training

CIFAR-100 with ResNet and the canonical attack union:

```bash
python train_tafd.py \
  --dataset CIFAR100 \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_union canonical \
  --num_threat_domains 2 \
  --batch_size 128 \
  --test_batch_size 16 \
  --end_epoch 76
```

Imagenette with MobileViT and the broader attack union:

```bash
python train_tafd.py \
  --dataset Imagenette \
  --dataset_path ./datasets \
  --backbone mobilevit \
  --attack_union broader \
  --num_threat_domains 2 \
  --batch_size 12 \
  --test_batch_size 16 \
  --end_epoch 76
```

Equivalent launch examples are provided in `scripts/`. Run directories are
created under `results/` unless `--result_dir` is specified.

## Checkpoints and Evaluation

Resume a run or evaluate an archived checkpoint with `--resume`:

```bash
python train_tafd.py \
  --dataset CIFAR100 \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_union canonical \
  --num_threat_domains 2 \
  --resume ./checkpoints/latest_model.pth \
  --start_epoch 75 \
  --end_epoch 76
```

The paper-aligned refactor preserves the original TaFD state-dict keys. K=2
checkpoints produced by the experiment code can therefore be loaded directly.

Run the adaptive diagnosis evaluation with:

```bash
python evaluate_adaptive_diagnosis.py \
  --dataset CIFAR100 \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_union canonical \
  --num_threat_domains 2 \
  --resume ./checkpoints/latest_model.pth
```

The diagnosis-dispatch stress test can also be run independently:

```bash
python -m evaluation.diagnosis_dispatch_stress_test \
  --checkpoint ./checkpoints/latest_model.pth \
  --output_dir ./results/diagnosis_dispatch_stress_test \
  --dataset CIFAR100 \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_union canonical
```

## Ablations

The four paper ablations use the same command-line interface as `train_tafd.py`:

```text
train_ablation_without_threat_domain_diagnosis.py
train_ablation_without_basis_parameterized_mask.py
train_ablation_without_hungarian_alignment.py
train_ablation_without_frequency_decoupling.py
```

For example:

```bash
python train_ablation_without_threat_domain_diagnosis.py \
  --dataset CIFAR100 \
  --dataset_path ./datasets \
  --backbone resnet \
  --attack_union canonical \
  --num_threat_domains 2
```

## Reproducibility

Detailed dataset, checkpoint, output, and validation notes are provided in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). Generated datasets,
checkpoints, logs, and result directories are excluded from version control.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
