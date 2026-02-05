# TorchRIR

[![GitHub stars](https://img.shields.io/github/stars/taishi-n/torchrir?style=social)](https://github.com/taishi-n/torchrir)

## Summary
TorchRIR is a PyTorch-based toolkit for room impulse response (RIR) simulation
with CPU/CUDA/MPS support, static and dynamic scenes, and dataset utilities.
If you find bugs or have feature requests, please open an issue.
Contributions are welcome.

```{warning}
TorchRIR is under active development and may contain bugs or breaking changes.
Please validate results for your use case.
```

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
- Experimental template stub: {py:mod}`torchrir.experimental`

### License
TorchRIR is released under the Apache 2.0 license. See `LICENSE`.

See the detailed overview: {doc}`overview`.

## Main features
### Static room acoustic simulation
- Compute static RIRs with `simulate_rir`.
- Convolve dry signals with `convolve_rir`.

### Dynamic room acoustic simulation
- Compute time-varying RIRs with `simulate_dynamic_rir`.
- Convolve with `DynamicConvolver(mode="trajectory")`.

### Building dataset
- Use `torchrir.datasets.load_dataset_sources` to build fixed-length sources.
- Use dataset examples to generate per-scene WAV + metadata.

See runnable examples and command-line usage: {doc}`examples`.

## Documentation pages
- {doc}`overview`
- {doc}`examples`
- {doc}`changelog`
- {doc}`api` (API documentation)
- {ref}`genindex`

```{toctree}
:maxdepth: 2
:caption: Contents
:hidden:

overview
examples
changelog
API documentation <api>
Index <genindex>
```
