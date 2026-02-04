# TorchRIR

PyTorch-based room impulse response (RIR) simulation toolkit focused on a clean, modern API with GPU support.

## Example Usage
```bash
# CMU ARCTIC + static RIR (fixed sources/mics)
uv run python examples/static.py --plot

# Dynamic RIR demos
uv run python examples/dynamic_mic.py --plot
uv run python examples/dynamic_src.py --plot
```

```python
from torchrir import DynamicConvolver

# Trajectory-mode dynamic convolution
y = DynamicConvolver(mode="trajectory").convolve(signal, rirs)

# Hop-mode dynamic convolution
y = DynamicConvolver(mode="hop", hop=1024).convolve(signal, rirs)
```
`convolve_dynamic_rir(hop=...)` is deprecated; use `DynamicConvolver(mode="hop")`.

### Dataset-agnostic utilities
```python
from torchrir import (
    CmuArcticDataset,
    binaural_mic_positions,
    clamp_positions,
    load_dataset_sources,
    sample_positions,
)

def dataset_factory(speaker: str | None):
    spk = speaker or "bdl"
    return CmuArcticDataset("datasets/cmu_arctic", speaker=spk, download=True)

signals, fs, info = load_dataset_sources(
    dataset_factory=dataset_factory,
    num_sources=2,
    duration_s=10.0,
    rng=random.Random(0),
)
```

### Dataset template (for future extension)
`TemplateDataset` provides a minimal stub to implement new datasets later.

### Logging
```python
from torchrir import LoggingConfig, get_logger, setup_logging

setup_logging(LoggingConfig(level="INFO"))
logger = get_logger("examples")
logger.info("running torchrir example")
```

### Scene container
```python
from torchrir import Scene

scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
scene.validate()
```

## Device Selection
- `device="cpu"`: CPU execution
- `device="cuda"`: NVIDIA GPU (CUDA) if available, otherwise fallback to CPU
- `device="mps"`: Apple Silicon GPU via Metal (MPS) if available, otherwise fallback to CPU
- `device="auto"`: prefer CUDA → MPS → CPU

```python
from torchrir import DeviceSpec

device, dtype = DeviceSpec(device="auto").resolve()
```

## References
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [das-generator](https://github.com/ehabets/das-generator)
- [rir-generator](https://github.com/audiolabs/rir-generator)

## Specification (Current)
### Purpose
- Provide room impulse response (RIR) simulation on PyTorch with CPU/CUDA/MPS support.
- Support static and dynamic scenes with a maintainable, modern API.

### Room Model
- Shoebox (rectangular) room model.
- 2D or 3D.
- Image Source Method (ISM) implementation.

### Inputs
#### Scene Geometry
- Room size: `[Lx, Ly, Lz]` (2D uses `[Lx, Ly]`).
- Source positions: `(n_src, dim)`.
- Microphone positions: `(n_mic, dim)`.
- Reflection order: `max_order`.

#### Acoustic Parameters
- Sample rate: `fs`.
- Speed of sound: `c` (default 343.0 m/s).
- Wall reflection coefficients: `beta` (4 faces for 2D, 6 for 3D) or `t60` (Sabine).

#### Output Length
- Specify `nsample` (samples) or `tmax` (seconds).

#### Directivity
- Patterns: `omni`, `cardioid`, `hypercardioid`, `subcardioid`, `bidir`.
- Orientation specified by vector or angles.

#### Configuration
- `SimulationConfig` controls algorithm settings (e.g., max_order, tmax, directivity, device, seed, fractional delay length, LUT, chunk sizes, compile path).
- Passed explicitly via `simulate_rir(..., config=...)` or `simulate_dynamic_rir(..., config=...)`.

### Outputs
- Static RIR shape: `(n_src, n_mic, nsample)`.
- Dynamic RIR shape: `(T, n_src, n_mic, nsample)`.
- Preserves dtype/device.

### Core APIs
#### Static RIR
```python
room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
sources = Source.positions([[1.0, 2.0, 1.5], [4.5, 1.0, 1.2]])
mics = MicrophoneArray.positions([[2.0, 2.0, 1.5], [3.0, 2.0, 1.5]])

rir = simulate_rir(
    room=room,
    sources=sources,
    mics=mics,
    max_order=8,
    tmax=0.4,
    directivity="omni",
    device="auto",
)
```

#### Dynamic RIRs + Convolution
```python
rirs = simulate_dynamic_rir(
    room=room,
    src_traj=src_traj,   # (T, n_src, dim)
    mic_traj=mic_traj,   # (T, n_mic, dim)
    max_order=8,
    tmax=0.4,
    device="auto",
)

y = DynamicConvolver(mode="trajectory").convolve(signal, rirs)
```

### Device Control
- `device="cpu"`, `"cuda"`, `"mps"`, or `"auto"`; resolves with fallback to CPU.
