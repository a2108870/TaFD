# Code Provenance

This release was assembled from the curated TaFD working snapshot:

    _v20trad_run

Included:

- TaFD main training/evaluation implementation.
- TaFD method ablation implementations.
- Attack utilities used by the TaFD training and evaluation scripts.
- Local torchattacks snapshot required by the code.

Excluded:

- Third-party baseline entrypoints: PUAT, FACE, TRADES, RAMP, MNG, DAT, GBN,
  and standard adversarial-training scripts.
- Datasets, checkpoints, generated results, logs, pid files, cache files, and
  server-specific launch scripts.
- Private machine paths and hardcoded server paths.

Naming changes:

- train_tafd.py wraps main_train_pgdtrain.py.
- evaluate_adaptive_diagnosis.py wraps main_eval_gateatk_scales.py.
- ablation_no_threat_domain_diagnosis.py wraps
  main_train_pgdtrain_woDomainUniformMix.py.
- ablation_direct_frequency_mask.py wraps main_train_pgdtrain_directMask.py.
- ablation_no_assignment_alignment.py wraps main_train_pgdtrain_woHungary.py.
- ablation_standard_convolution.py wraps main_train_pgdtrain_stdconv.py.

The original implementation filenames are retained for compatibility and to
avoid silently changing the experimental logic.
