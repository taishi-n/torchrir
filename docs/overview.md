# Overview

## Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Directivity patterns (`omni`, `cardioid`, `hypercardioid`, `subcardioid`, `bidir`)
  with per-source/mic orientation handling.
- Acoustic parameters via `beta` or `t60` (Sabine), optional diffuse tail via `tdiff`.
- Dynamic convolution via `torchrir.signal.DynamicConvolver` (`trajectory` or `hop` modes).
- CPU/CUDA/MPS execution with optional `torch.compile` acceleration for ISM accumulation
  (when enabled; MPS disables LUT).
- Standard array geometries (linear, circular, polyhedron, binaural, Eigenmike)
  and trajectory sampling utilities.
- Dataset utilities (CMU ARCTIC, LibriSpeech, template stubs) plus DataLoader collate helpers.
- Plotting utilities for static/dynamic scenes and GIF animation.
- Metadata export helpers for time axis, DOA, array attributes, and trajectories (JSON-ready).
- Explicit audio metadata I/O container via `torchrir.io.AudioData` (`load_audio_data` / `save_audio_data`).
- Dataset examples can emit per-source reference audio (RIR-convolved premix) and record it in metadata.
- Unified CLI example with JSON/YAML config and deterministic flag support.

## Module layout
- {py:mod}`torchrir.sim`: Simulation engines and configuration for RIR generation.
- {py:mod}`torchrir.signal`: Signal processing utilities for static and dynamic RIR convolution.
- {py:mod}`torchrir.geometry`: Geometry helpers for arrays, trajectories, and sampling.
- {py:mod}`torchrir.viz`: Visualization helpers for scenes and trajectories.
- {py:mod}`torchrir.models`: Core data models for rooms, sources, microphones, scenes, and results.
- {py:mod}`torchrir.io`: I/O helpers for audio files and metadata serialization
  (wav-only `load`/`save`/`info` with backend selection; non-wav via
  `torchrir.io.audio.*`; explicit metadata-preserving audio I/O via
  `torchrir.io.AudioData` and `torchrir.io.audio.load_audio_data`).
- {py:mod}`torchrir.util`: General-purpose math, device, and tensor utilities for torchrir.
- {py:mod}`torchrir.logging`: Logging configuration and helpers.
- {py:mod}`torchrir.config`: Simulation configuration objects.
- {py:mod}`torchrir.datasets`: Dataset helpers and collate utilities.
- {py:mod}`torchrir.experimental`: Work-in-progress APIs (ray tracing, FDTD, template datasets).

## Device selection
- `device="cpu"`: CPU execution
- `device="mps"`: Apple Silicon GPU via Metal (MPS) if available, otherwise fallback to CPU
- `device="cuda"`: CUDA execution (validated in CI on CUDA runners; requires a CUDA-enabled PyTorch environment)
- `device="auto"`: backend is selected by internal priority

```python
from torchrir.util import DeviceSpec

device, dtype = DeviceSpec(device="auto").resolve()
```

## Limitations and potential errors
- Experimental ray tracing and FDTD simulators (`torchrir.experimental`) are placeholders and
  raise `NotImplementedError`.
- Experimental dataset stubs (`torchrir.experimental`) are not implemented and raise
  `NotImplementedError`.
- `torchrir.sim.simulate_rir`/`torchrir.sim.simulate_dynamic_rir` require `max_order`
  (or `torchrir.config.SimulationConfig.max_order`) and either `nsample` or `tmax`.
- Non-`omni` directivity requires orientation; mismatched shapes raise `ValueError`.
- `beta` must have 4 (2D) or 6 (3D) elements; invalid sizes raise `ValueError`.
- `simulate_dynamic_rir` requires `src_traj` and `mic_traj` to have matching time steps.
- `torchrir.signal.DynamicConvolver` with 3D dynamic RIR input (`(T, n_mic, rir_len)`) is treated as single-source only; multi-source dynamic convolution must use 4D RIR input (`(T, n_src, n_mic, rir_len)`).
- Dynamic simulation currently loops per time step; very long trajectories can be slow.
- MPS disables the sinc LUT path (falls back to direct sinc), which can be slower and slightly different numerically.
- HPF requires SciPy and currently applies filtering via CPU-domain processing, which can add host/device transfer overhead on CUDA/MPS runs.
- Deterministic mode is best-effort; some backends may still be non-deterministic.
- YAML configs require `PyYAML`; otherwise a `ModuleNotFoundError` is raised.
- CMU ARCTIC downloads require network access.
- GIF animation output requires Pillow (via matplotlib animation writer).


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
