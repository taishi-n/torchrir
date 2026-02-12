# TorchRIR

## Summary
TorchRIR is a PyTorch-based toolkit for room impulse response (RIR) simulation,
with CPU/CUDA/MPS support, static and dynamic scenes, and dataset utilities.
If you find a bug or have a feature request, please open an issue.
Contributions are welcome.

!!! warning
    TorchRIR is under active development and may contain bugs or breaking changes.
    Please validate results for your use case.

## Installation
```bash
pip install torchrir
```

## Overview
### Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Dynamic convolution via trajectory or hop-based modes.
- Scene visualization (plots, GIFs) and metadata export (JSON).
- Dataset utilities for building small mixtures from speech corpora.

### Limitations
- Ray tracing and FDTD simulators are placeholders.
- Deterministic mode is best-effort and backend-dependent.
- MPS disables the LUT path for fractional delay (slower, minor numerical diffs).
- Experimental status: APIs and outputs may change as the library matures.

### Supported datasets
- CMU ARCTIC
- LibriSpeech
- Experimental template dataset stub under `torchrir.experimental`
- Dataset usage details (options, directory layouts, error handling):
  [Datasets](datasets.md)
- Dataset attribution and redistribution notes:
  [THIRD_PARTY_DATASETS.md](https://github.com/taishi-n/torchrir/blob/main/THIRD_PARTY_DATASETS.md)

### License
TorchRIR is released under the Apache License 2.0. See `LICENSE`.

See the detailed overview: [Overview](overview.md).

## Core Workflows
### Static room acoustic simulation
- Compute static RIRs with `torchrir.sim.simulate_rir`.
- Convolve dry signals with `torchrir.signal.convolve_rir`.

### Dynamic room acoustic simulation
- Compute time-varying RIRs with `torchrir.sim.simulate_dynamic_rir`.
- Convolve with `torchrir.signal.DynamicConvolver(mode="trajectory")`.

### Dataset generation
- Use `torchrir.datasets.load_dataset_sources` to build fixed-length sources.
- Use the dataset example scripts to generate per-scene WAV files and metadata.
- See dataset-specific options and validation behavior:
  [Datasets](datasets.md).

See runnable examples and command-line usage: [Examples](examples.md).

## Documentation Pages
- [Getting started](getting-started.md)
- [Overview](overview.md)
- [Datasets](datasets.md)
- [Examples](examples.md)
- [Library Comparisons](comparisons.md)
- [Changelog](changelog.md)
- [API documentation](api.md)
