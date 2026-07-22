# TorchRIR

A PyTorch-based room impulse response (RIR) simulation toolkit with a clean API and GPU support.
This project has been developed with substantial assistance from Codex.
> [!WARNING]
> TorchRIR is under active development and may contain bugs or breaking changes.
> Please validate results for your use case.
If you find bugs or have feature requests, please open an issue.
Contributions are welcome.

## Installation

TorchRIR supports Python 3.11, 3.12, and 3.13.

```bash
pip install torchrir
```

Install only the optional features you use:

```bash
pip install "torchrir[audio]"       # SoundFile-backed audio I/O
pip install "torchrir[viz]"         # plots, GIFs, and MP4 rendering
pip install "torchrir[datasets]"    # dataset loaders and builders
pip install "torchrir[oobss]"       # oobss integration
pip install "torchrir[all]"         # all optional features
```

## Library Comparison
| Feature | `torchrir` | `gpuRIR` | `pyroomacoustics` | `rir-generator` |
|---|---|---|---|---|
| 🎯 Dynamic Sources | ✅ | 🟡 Single moving source | 🟡 Manual loop | ❌ |
| 🎤 Dynamic Microphones | ✅ | ❌ | 🟡 Manual loop | ❌ |
| 🖥️ CPU | ✅ | ❌ | ✅ | ✅ |
| 🧮 CUDA | ✅ | ✅ | ❌ | ❌ |
| 🍎 MPS | ✅ | ❌ | ❌ | ❌ |
| 📊 Scene Plot | ✅ | ❌ | ✅ | ❌ |
| 🎞️ Dynamic Scene GIF | ✅ | ❌ | 🟡 Manual animation script | ❌ |
| 🗂️ Dataset Build | ✅ | ❌ | ✅ | ❌ |
| 🎛️ Signal Processing | ❌ Scope out | ❌ | ✅ | ❌ |
| 🧱 Non-shoebox Geometry | 🚧 Candidate | ❌ | ✅ | ❌ |
| 🌐 Geometric Acoustics | 🚧 Candidate | ❌ | ✅ | ❌ |

Legend: `✅` native support, `🟡` manual setup, `🚧` candidate (not yet implemented), `❌` unavailable

For detailed notes and equations, see
[Documentation: Library Comparisons](https://taishi.org/torchrir/comparisons.html).

## CUDA CI (GitHub Actions)
- CUDA tests run in `.github/workflows/cuda-ci.yml` on a self-hosted runner with labels:
  `self-hosted`, `linux`, `x64`, `cuda`.
- The workflow validates installation via `uv sync --group test`, checks `torch.cuda.is_available()`,
  runs `tests/test_device_parity.py` with `-k cuda`, and installs the pinned
  `gpuRIR` reference revision.
- The extended workflow requires `gpuRIR` installation and runs direct-path
  static/dynamic RIR comparisons plus an isolated `simulateTrajectory`
  comparison using identical synthetic RIRs. Known normalization, image-count,
  and fractional-delay conventions are handled explicitly; signals are never
  freely aligned. Installation failure or a skipped comparison fails that
  workflow.

## Examples
- `examples/static.py`: fixed sources and microphones with configurable mic count (default: binaural).  
  `uv run python examples/static.py --plot`
- `examples/dynamic_src.py`: moving sources, fixed microphones.  
  `uv run python examples/dynamic_src.py --plot`
- `examples/dynamic_mic.py`: fixed sources, moving microphones.  
  `uv run python examples/dynamic_mic.py --plot`
- `examples/cli.py`: unified CLI for static/dynamic scenes with JSON/YAML configs.  
  `uv run python examples/cli.py --mode static --plot`
- `examples/build_dynamic_dataset.py`: small dynamic dataset generation script (CMU ARCTIC / LibriSpeech; fixed room/mics, randomized source motion).  
  `uv run python examples/build_dynamic_dataset.py --dataset cmu_arctic --num-scenes 4 --num-sources 2`
- `torchrir.datasets.dynamic_cmu_arctic`: oobss-compatible dynamic CMU ARCTIC builder CLI.  
  `python -m torchrir.datasets.dynamic_cmu_arctic --cmu-root datasets/cmu_arctic --n-scenes 2 --overwrite-dataset`
- `examples/benchmark_device.py`: CPU/GPU benchmark for RIR simulation.  
  `uv run python examples/benchmark_device.py --dynamic`

## Dataset Notices
- For dataset attribution and redistribution notes, see
  [THIRD_PARTY_DATASETS.md](THIRD_PARTY_DATASETS.md).

## Dataset API Quick Guide
- `torchrir.datasets.CmuArcticDataset(root, speaker=..., download=...)`
  - Accepted `speaker`: `aew`, `ahw`, `aup`, `awb`, `axb`, `bdl`, `clb`, `eey`, `fem`, `gka`, `jmk`, `ksp`, `ljm`, `lnh`, `rms`, `rxr`, `slp`, `slt`
  - Invalid `speaker` raises `ValueError`.
  - Missing local files with `download=False` raises `FileNotFoundError`.
- `torchrir.datasets.LibriSpeechDataset(root, subset=..., speaker=..., download=...)`
  - Accepted `subset`: `dev-clean`, `dev-other`, `test-clean`, `test-other`, `train-clean-100`, `train-clean-360`, `train-other-500`
  - Invalid `subset` raises `ValueError`.
  - Missing subset/speaker paths with `download=False` raise `FileNotFoundError`.
- `torchrir.datasets.build_dynamic_cmu_arctic_dataset(...)`
  - Builds oobss-compatible scene folders with `mixture.wav`, `source_XX.wav`, `metadata.json`, and `source_info.json`.
  - Static layout images (`room_layout_2d.png`, `room_layout_3d.png`) and optional layout videos (`room_layout_2d.mp4`, `room_layout_3d.mp4`) are generated, with source-index annotations by default.
  - Default behavior includes `n_sources=3`, moving speed range `0.3-0.8 m/s`, and motion profile ratios `0-35%`, `35-65%`, `65-100%`.
- Local-only (no download) example:
  ```python
  from pathlib import Path
  from torchrir.datasets import CmuArcticDataset, LibriSpeechDataset

  cmu = CmuArcticDataset(Path("datasets/cmu_arctic"), speaker="bdl", download=False)
  libri = LibriSpeechDataset(
      Path("datasets/librispeech"),
      subset="train-clean-100",
      speaker="103",
      download=False,
  )
  ```
- Full dataset usage details, expected directory layout, and invalid-input handling:
  [Documentation: Datasets](https://taishi.org/torchrir/datasets.html)

## Core API Overview
- Geometry: `Room`, `Source`, `MicrophoneArray`
- Scene models: `StaticScene`, `DynamicScene` (`Scene` is deprecated)
- Scene-oriented simulation: `torchrir.sim.simulate(scene, config)`
- Tensor-level compatibility APIs: `torchrir.sim.simulate_rir` and
  `torchrir.sim.simulate_dynamic_rir`
- Simulator backend: `torchrir.sim.ISMSimulator()`
- Dynamic convolution: `torchrir.signal.DynamicConvolver`
- Audio I/O:
  - wav-specific: `torchrir.io.load_wav`, `torchrir.io.save_wav`, `torchrir.io.info_wav`
  - backend-supported formats: `torchrir.io.load_audio`, `torchrir.io.save_audio`, `torchrir.io.info_audio`
  - metadata-preserving: `torchrir.io.AudioData`, `torchrir.io.load_audio_data`
- Metadata export: `torchrir.io.build_result_metadata`,
  `torchrir.io.save_result_metadata`

## Module Layout (for contributors)
- `torchrir.sim`: simulation backends (ISM implementation lives under `torchrir.sim.ism`)
- `torchrir.signal`: convolution utilities and dynamic convolver
- `torchrir.geometry`: array geometries, sampling, trajectories
- `torchrir.viz`: plotting and GIF/MP4 animation helpers
  - Default plot style follows SciencePlots Grid (`science` + `grid`).
- `torchrir.models`: room/scene/result data models
- `torchrir.io`: audio I/O and metadata serialization (`*_wav` for wav-only, `*_audio` for backend-supported formats)
- `torchrir.util`: shared math/tensor/device helpers
- `torchrir.logging`: logging utilities
- `torchrir.config`: simulation configuration objects

## Design Notes
- Scene typing is explicit: use `StaticScene` for fixed geometry and `DynamicScene` for trajectory-based simulation.
- `DynamicScene` accepts tensor-like trajectories (e.g., lists) and normalizes them to tensors internally.
- `Scene` remains as a backward-compatibility wrapper and emits `DeprecationWarning`.
- `Scene.validate()` performs validation without emitting additional deprecation warnings.
- `SimulationConfig` is the single owner of simulation settings.
- `ISMSimulator` constructor settings are deprecated and remain available until 1.0.
- `RIRResult.config` records the resolved sample count, duration, device, dtype, and directivity.
- Model dataclasses are frozen, but tensor payloads remain mutable (shallow immutability).
- `torchrir.load` / `torchrir.save` and `torchrir.io.load` / `save` / `info` are deprecated compatibility aliases.

```python
from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate
from torchrir.signal import DynamicConvolver

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 2.0, 1.5]])
mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])

scene = StaticScene(room=room, sources=sources, mics=mics)
result = simulate(scene, SimulationConfig(max_order=6, tmax=0.3))
rir = result.rirs
# DynamicConvolver also accepts a dynamic RIRResult directly:
# y = DynamicConvolver().convolve(signal, dynamic_result)
```

## Specification

### Geometry and scenes

- Room, source, microphone, and trajectory coordinates are real floating-point
  tensors. Integer inputs are promoted to PyTorch's default floating dtype.
- Room coordinates are 2D or 3D. Source and microphone coordinates must lie in
  the inclusive range `[0, room.size]`.
- Static positions have shape `(entities, dimensions)`.
- Dynamic trajectories have shape `(frames, entities, dimensions)`. A 2D
  `(frames, dimensions)` trajectory is accepted for one entity.
- `DynamicScene.sources.positions` and `mics.positions` equal the first frame of
  their respective trajectories.
- Dynamic timestamps, when present, are finite seconds, begin at zero, are
  strictly increasing, and match the trajectory frame count.
- Entity orientation is constant over a dynamic trajectory in 0.9. Time-varying
  orientation is not yet supported.

### Simulation

- `SimulationConfig` owns all simulation settings. Exactly one of `tmax` and
  `nsample` is required when a simulation is executed.
- Static RIR tensors have shape `(sources, microphones, samples)`.
- Dynamic RIR tensors have shape `(frames, sources, microphones, samples)`.
- `torchrir.sim.simulate` returns `RIRResult`; its config is a
  `ResolvedSimulationConfig` containing the effective settings.
- Explicitly supplied device and dtype values must not conflict with config
  values. Silent precedence is not used.

### Signal and audio shapes

- Signals use `(samples,)` for mono and `(channels, samples)` for multichannel
  data.
- Static convolution RIRs use `(sources, microphones, samples)`; dynamic
  convolution RIRs add the leading frame dimension.
- Signal and RIR tensors must share device and dtype.
- `AudioData` preserves every channel. Its save API does not normalize unless
  `normalize=True` is requested.
- Legacy tuple loaders continue to return channel 0 for multichannel files and
  emit a warning directing callers to `load_audio_data`.

### Numerical verification

- Image-source coordinates, reflection coefficients, path delays, path gains,
  and fractional-delay accumulation are checked against independent analytic
  or direct implementations.
- Reflected source directivity mirrors the source orientation component normal
  to every wall with an odd reflection count.
- Static and dynamic convolution are checked against direct NumPy convolution,
  including segment and timestamp boundaries.
- pyroomacoustics comparisons cover 2D/3D rooms, orders 0/1/3, asymmetric wall
  coefficients, and multiple source/microphone pairs. Comparisons use the
  native sample axis and do not shift or crop RIRs using cross-correlation.
- Unexpected warnings fail the test suite. Reference, CUDA, MPS, numerical, and
  slow tests use explicit pytest markers.
- CI measures branch coverage and requires at least 75% overall coverage.

### Compatibility policy

- `Scene`, top-level `load`/`save`, process-global audio backend selection,
  `ISMSimulator` constructor settings, and per-argument simulation settings are
  deprecated in 0.9 and scheduled for removal in 1.0.
- Deprecated APIs remain compatibility wrappers throughout the 0.9 release
  series and reject conflicting values instead of silently selecting one.

For detailed documentation:
[Documentation](https://taishi.org/torchrir/)

## Documentation Development

The documentation site is configured in [`zensical.toml`](zensical.toml) and
built with [Zensical](https://zensical.org/).

```bash
uv sync --group docs
uv run zensical serve
```

Before submitting documentation changes, run the same strict build used by CI:

```bash
uv run zensical build --strict
```

## Future Work
- Advanced room geometry pipeline beyond shoebox rooms (e.g., irregular polygons/meshes and boundary handling).  
  Motivation: [pyroomacoustics#393](https://github.com/LCAV/pyroomacoustics/issues/393), [pyroomacoustics#405](https://github.com/LCAV/pyroomacoustics/issues/405)
- General reflection/path capping controls (e.g., first-K, strongest-K, or energy-threshold-based path selection).  
  Motivation: [pyroomacoustics#338](https://github.com/LCAV/pyroomacoustics/issues/338)
- Microphone hardware response modeling (frequency response, sensitivity, and self-noise).  
  Motivation: [pyroomacoustics#394](https://github.com/LCAV/pyroomacoustics/issues/394)
- Near-field speech source modeling for more realistic close-talk scenarios.  
  Motivation: [pyroomacoustics#417](https://github.com/LCAV/pyroomacoustics/issues/417)
- Integrated 3D spatial response visualization (e.g., array/directivity beam-pattern rendering).  
  Motivation: [pyroomacoustics#397](https://github.com/LCAV/pyroomacoustics/issues/397)

## Related Libraries
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
- [Cross3D](https://github.com/DavidDiazGuerra/Cross3D)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [rir-generator](https://github.com/audiolabs/rir-generator)
