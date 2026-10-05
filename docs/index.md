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

Install optional capabilities explicitly: `torchrir[audio]` for SoundFile I/O,
`torchrir[viz]` for plots/animation, `torchrir[datasets]` for dataset workflows,
`torchrir[hpf]` for RIR high-pass filtering, and `torchrir[cli]` for YAML example
CLI configuration. `torchrir[oobss]` adds oobss integration.
`torchrir[all]` installs every optional Python dependency; MP4 rendering still
needs a system `ffmpeg`.

## Overview
### Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Dynamic convolution with an explicit emission/observation time reference and
  an exact sample-domain `FrameSchedule` carried by the scene or supplied to a
  convolution call.
- Scene visualization (plots, GIFs) and metadata export (JSON).
- Dataset utilities for building small mixtures from speech corpora.

### Limitations
- Ray tracing and FDTD are roadmap items and are not exposed as APIs.
- Deterministic mode is best-effort and backend-dependent.
- RIR simulation supports float32 and float64. Lower-precision geometry is
  rejected before the ISM kernel; convolution separately supports float16 and
  bfloat16 through float32 work buffers. MPS rejects float64 and disables the
  LUT path; CPU disables requested compilation.
- No backward compatibility or data migration is provided during development.
  Regenerate generated data after relevant format or numerical changes.

### Supported datasets
- CMU ARCTIC
- LibriSpeech
- Secure local corpus reads, archive handling, and publication require Linux or
  macOS with the documented POSIX descriptor and atomic-rename primitives.
- Dataset usage details (options, directory layouts, error handling):
  [Datasets](datasets.md)
- Survey of external datasets with moving sources or microphone arrays:
  [Related Dynamic Speech and Acoustic Datasets](related-datasets.md)
- Dataset attribution and redistribution notes:
  [THIRD_PARTY_DATASETS.md](https://github.com/taishi-n/torchrir/blob/main/THIRD_PARTY_DATASETS.md)

### License
TorchRIR is released under the Apache License 2.0. See `LICENSE`.

See the detailed overview: [Overview](overview.md).

## Core Workflows
### Static room acoustic simulation
- Compute static and dynamic RIR results with `torchrir.sim.simulate`.
- Convolve dry signals with `torchrir.signal.convolve_rir`.

### Dynamic room acoustic simulation
- Use `StaticScene` or `DynamicScene` to make geometry and time axes explicit.
- Use `DynamicConvolver(time_reference="emission")` for moving sources with
  fixed microphones.
- Use `DynamicConvolver(time_reference="observation")` for fixed sources with
  moving microphones. Attach exact integer starts as `DynamicScene.schedule`
  to let its `RIRResult` provide them, or omit the scene schedule and pass a
  call-level `FrameSchedule`. Simultaneous motion is not supported without a
  retarded-time propagation model.

### Dataset generation
- Use `torchrir.datasets.load_dataset_sources` to build fixed-length sources.
- Use the dataset example scripts to generate per-scene WAV files and metadata.
- See dataset-specific options and validation behavior:
  [Datasets](datasets.md).

### Audio I/O
- Use `torchrir.io.load_audio`, `save_audio`, and `info_audio` for formats
  supported by soundfile.
- Use the strict `load_wav`, `save_wav`, and `info_wav` entry points when a WAV
  path is required.
- Use `AudioData`, `load_audio_data`, and `save_audio_data` to preserve all
  channels, sample rate, and subtype. The stored format is load metadata; the
  output path selects the saved file format, and a stored subtype is inherited
  only when that container is unchanged. Save functions preserve gain by
  default; a WAV with no subtype uses `FLOAT`, while non-floating subtypes
  reject out-of-range samples. Use `normalize=True` only when independent peak
  normalization is intentional.

See runnable examples and command-line usage: [Examples](examples.md).

## Documentation Pages
- [Getting started](getting-started.md)
- [Overview](overview.md)
- [Datasets](datasets.md)
- [Related Dynamic Speech and Acoustic Datasets](related-datasets.md)
- [Examples](examples.md)
- [Library Comparisons](comparisons.md)
- [Changelog](changelog.md)
- [API documentation](api.md)
