# Overview

## Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Directivity patterns (`omni`, `cardioid`, `hypercardioid`, `subcardioid`, `bidir`)
  with per-source/mic orientation handling.
- Acoustic parameters via `beta` or `t60` (Sabine), optional diffuse tail via `tdiff`.
- Dynamic convolution via `torchrir.signal.DynamicConvolver` (`trajectory` or `hop` modes).
- Explicit scene models via `torchrir.models.StaticScene` and `torchrir.models.DynamicScene`.
- Scene-oriented simulation via `torchrir.sim.simulate(scene, config)` returning
  an `RIRResult` with resolved settings.
- CPU/CUDA/MPS execution with optional `torch.compile` acceleration for ISM accumulation
  (when enabled; MPS disables LUT).
- Standard array geometries (linear, circular, polyhedron, binaural, Eigenmike)
  and trajectory sampling utilities.
- Dataset utilities (CMU ARCTIC, LibriSpeech, template stubs) plus DataLoader collate helpers.
  See [Datasets](datasets.md) for accepted options, directory layouts, and
  invalid-input handling.
- Plotting utilities for static/dynamic scenes and GIF animation.
- Metadata export helpers for time axis, DOA, array attributes, and trajectories (JSON-ready).
- Channel-preserving audio metadata I/O via `torchrir.io.AudioData`
  (`load_audio_data` / `save_audio_data`).
- Explicit split between WAV-only and backend-format audio I/O:
    - wav-only: `torchrir.io.load_wav` / `save_wav` / `info_wav`
    - backend-supported formats: `torchrir.io.load_audio` / `save_audio` / `info_audio`
- Dataset examples can emit per-source reference audio (RIR-convolved premix) and record it in metadata.
- Unified CLI example with JSON/YAML config and deterministic flag support.

## Module layout
- `torchrir.sim`: Simulation engines and configuration for RIR generation.
- `torchrir.signal`: Signal processing utilities for static and dynamic RIR convolution.
- `torchrir.geometry`: Geometry helpers for arrays, trajectories, and sampling.
- `torchrir.viz`: Visualization helpers for scenes and trajectories.
- `torchrir.models`: Core data models for rooms, sources, microphones, scenes, and results.
- `torchrir.io`: I/O helpers for audio files and metadata serialization
  (wav-only `load_wav`/`save_wav`/`info_wav`, backend-format
  `load_audio`/`save_audio`/`info_audio`; explicit metadata-preserving
  audio I/O via `torchrir.io.AudioData` and `torchrir.io.load_audio_data`).
- `torchrir.util`: General-purpose math, device, and tensor utilities for TorchRIR.
- `torchrir.logging`: Logging configuration and helpers.
- `torchrir.config`: Simulation configuration objects.
- `torchrir.datasets`: Dataset helpers and collate utilities.
  See [Datasets](datasets.md) for practical usage guidance.
- `torchrir.experimental`: Work-in-progress APIs (ray tracing, FDTD, template datasets).

## Device selection
- `device="cpu"`: CPU execution
- `device="mps"`: Apple Silicon GPU via Metal (MPS) if available, otherwise fallback to CPU
- `device="cuda"`: CUDA execution (validated in CI on CUDA runners; requires a CUDA-enabled PyTorch environment)
- `device="auto"`: backend is selected by internal priority

```python
from torchrir.util import DeviceSpec

device, dtype = DeviceSpec(device="auto").resolve()
```

## Limitations and Failure Modes
- Experimental ray tracing and FDTD simulators (`torchrir.experimental`) are placeholders and
  raise `NotImplementedError`.
- Experimental dataset stubs (`torchrir.experimental`) are not implemented and raise
  `NotImplementedError`.
- `torchrir.models.Scene` is deprecated; use `StaticScene`/`DynamicScene`.
- `DynamicScene` normalizes tensor-like trajectories to tensors during initialization.
- `Scene.validate()` does not re-emit deprecation warnings.
- `SimulationConfig` is the single source of simulation settings. Legacy
  `ISMSimulator` constructor settings raise `ValueError` when they conflict.
- `torchrir.load`/`save` and `torchrir.io.load`/`save`/`info` are deprecated aliases.
- `torchrir.sim.simulate` requires `SimulationConfig.max_order` and exactly one
  of `nsample` or `tmax`.
- `simulate_rir` and `simulate_dynamic_rir` remain Tensor-level compatibility
  APIs. Passing individual simulation settings is deprecated until 1.0.
- Non-`omni` directivity requires orientation; mismatched shapes raise `ValueError`.
- `beta` must have 4 (2D) or 6 (3D) elements; invalid sizes raise `ValueError`.
- `simulate_dynamic_rir` requires `src_traj` and `mic_traj` to have matching time steps.
- Audio tensors are `(samples,)` for mono and channel-first
  `(channels, samples)` for multichannel data.
- `torchrir.signal.DynamicConvolver` with 3D dynamic RIR input (`(T, n_mic, rir_len)`) is treated as single-source only; multi-source dynamic convolution must use 4D RIR input (`(T, n_src, n_mic, rir_len)`).
- Dynamic simulation batches trajectory frames, but memory and compute still grow with
  the frame, source, microphone, and image counts.
- MPS disables the sinc LUT path (falls back to direct sinc), which can be slower and slightly different numerically.
- HPF requires SciPy and currently applies filtering via CPU-domain processing, which can add host/device transfer overhead on CUDA/MPS runs.
- Very short RIRs may be too short for zero-phase HPF padding; increase the RIR
  length or disable the high-pass filter.
- Deterministic mode is best-effort; some backends may still be non-deterministic.
- YAML configs require `PyYAML`; otherwise a `ModuleNotFoundError` is raised.
- Downloading CMU ARCTIC requires network access when `download=True`; local
  preloaded datasets are supported with `download=False`.
- Dataset option validation and error behavior are documented in
  [Datasets](datasets.md).
- GIF output requires Pillow (via Matplotlib's animation writer).
- Dataclass models are frozen but hold mutable tensors (shallow immutability).


## Specification (current)
### Purpose
- Provide room impulse response (RIR) simulation on PyTorch with CPU/CUDA/MPS support.
- Support static and dynamic scenes with a maintainable, modern API.

### Room model
- Shoebox (rectangular) room model.
- 2D or 3D.
- Image Source Method (ISM) implementation.

### Inputs
- Room size: `[Lx, Ly, Lz]` (2D uses `[Lx, Ly]`).
- Source positions: `(n_src, dim)`.
- Microphone positions: `(n_mic, dim)`.
- Reflection order: `max_order`.
- Sample rate: `fs`.
- Speed of sound: `c` (default 343.0 m/s).
- Wall reflection coefficients: `beta` (4 faces for 2D, 6 for 3D) or `t60` (Sabine).
- Output length: `nsample` or `tmax`.

### Outputs
- Static RIR shape: `(n_src, n_mic, nsample)`.
- Dynamic RIR shape: `(T, n_src, n_mic, nsample)`.
- Preserves dtype/device.
