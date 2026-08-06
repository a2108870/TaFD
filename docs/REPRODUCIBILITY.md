# Reproducibility

## Paper Configuration

- Threat domains: `K=2`
- Backbones: ResNet-34 and MobileViT
- Datasets: CIFAR-10, CIFAR-100, and Imagenette
- Main entrypoint: `train_tafd.py`
- Random seed: `0` unless overridden with `--seed`

The complete command-line defaults are available through:

```bash
python train_tafd.py --help
```

## Attack Protocols

TaFD uses the two heterogeneous attack unions reported in the paper:

- `canonical`: APGD-Linf, APGD-L2, ACE, ALA, HSVAdv, ReColorAdv, and
  RetouchUAA.
- `broader`: APGD-Linf, APGD-L2, ACE, StAdv, and GPGD.

All attacks use 10 optimization steps during training. During evaluation,
APGD-Linf and APGD-L2 use 100-step AutoPGD; the other attacks retain their
training-time configurations. The attack source order used internally is kept
compatible with the released K=2 checkpoints.

## Dataset Handling

Always pass `--dataset_path` explicitly. CIFAR datasets are managed by
`torchvision`. Imagenette is loaded from an ImageFolder-compatible directory
named `imagenette2-320`, `imagenette2`, `imagenette2-160`, or `imagenette`.

Imagenette preprocessing is shared across TaFD configurations:

- Training: `RandomResizedCrop(224)` and random horizontal flip.
- Evaluation: `Resize(256)` followed by `CenterCrop(224)`.

## Checkpoint Compatibility

Checkpoints store the model, optimizer, epoch, accuracy history, and
threat-domain diagnosis state. Supply a checkpoint with:

```bash
--resume /path/to/latest_model.pth
```

The command must use the same dataset, backbone, and attack union as the saved
model. The number of threat domains must match the checkpoint. Public naming
changes do not alter the serialized TaFD model keys.

## GPGD Bases

The broader attack union requires class-conditional PCA bases for GPGD. Use
`--gpgd_basis_path` to load an existing basis file. If none is supplied, the
code builds the bases from the training set and stores them with the run.
Reusing the same basis file is recommended when comparing configurations.

## Outputs

Each run records checkpoints, text logs, accuracy histories, and evaluation
figures under `--result_dir`. When this argument is omitted, a directory name
is generated from the dataset, backbone, number of threat domains, attack
union, optimization settings, batch size, epoch count, and seed.

## Verification

Before a full experiment, verify the installation with:

```bash
python -m compileall -q .
python train_tafd.py --help
python evaluate_adaptive_diagnosis.py --help
```

Full adversarial training and evaluation require a CUDA GPU. CPU execution is
intended for imports, parser checks, model construction, and small smoke tests.
