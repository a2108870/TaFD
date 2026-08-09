# Code Provenance

This anonymous release is derived from the TaFD experiment snapshot used for
the paper results. Public module, function, argument, log, and output names were
aligned with the terminology used in the manuscript.

The refactor preserves the method's computation and archived checkpoint
compatibility. In particular, the state-dict fields required by trained TaFD
models retain their original serialized keys.

## Included

- TaFD training and validation code.
- ResNet and MobileViT TaFD implementations.
- FC-Conv.
- Heterogeneous attacks used by TaFD.
- The vendored `torchattacks` implementation required by this snapshot.

## Excluded

- Datasets and generated dataset caches.
- Trained checkpoints and experiment result archives.
- Machine-specific launch files, process files, and logs.
- Credentials, server addresses, personal paths, and identifying metadata.
