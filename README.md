# TorchRIR

PyTorch-based room impulse response (RIR) simulation toolkit focused on a clean, modern API with GPU support.
This project has been substantially assisted by AI using Codex.

## License
Apache-2.0. See `LICENSE` and `NOTICE`.

## Example Usage
```bash
# CMU ARCTIC + static RIR (fixed sources/mics)
uv run python examples/static.py --plot

# Dynamic RIR demos
uv run python examples/dynamic_mic.py --plot
uv run python examples/dynamic_src.py --plot

# Unified CLI
uv run python examples/cli.py --mode static --plot
uv run python examples/cli.py --mode dynamic_mic --plot
uv run python examples/cli.py --mode dynamic_src --plot

# Config + deterministic
uv run python examples/cli.py --mode static --deterministic --seed 123 --config-out outputs/cli.json
uv run python examples/cli.py --config-in outputs/cli.json
```
YAML configs are supported when `PyYAML` is installed.
```bash
# YAML config
uv run python examples/cli.py --mode static --config-out outputs/cli.yaml
uv run python examples/cli.py --config-in outputs/cli.yaml
```
`examples/cli_example.yaml` provides a ready-to-use template.

```python
from torchrir import DynamicConvolver

# Trajectory-mode dynamic convolution
y = DynamicConvolver(mode="trajectory").convolve(signal, rirs)

# Hop-mode dynamic convolution
y = DynamicConvolver(mode="hop", hop=1024).convolve(signal, rirs)
```
Dynamic convolution is exposed via `DynamicConvolver` only (no legacy function wrappers).

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

### Immutable geometry helpers
`Room`, `Source`, and `MicrophoneArray` are immutable; use `.replace()` to update fields.

### Result container
```python
from torchrir import RIRResult

result = RIRResult(rirs=rirs, scene=scene, config=config)
```

### Simulation strategies
```python
from torchrir import ISMSimulator

sim = ISMSimulator()
result = sim.simulate(scene, config)
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

## Future Work
- Ray tracing backend: implement `RayTracingSimulator` with frequency-dependent absorption/scattering.
- FDTD backend: implement `FDTDSimulator` with configurable grid resolution and boundary conditions.
- Dataset expansion: add additional dataset integrations beyond CMU ARCTIC (see `TemplateDataset`).
- Enhanced acoustics: frequency-dependent absorption and more advanced diffuse tail models.
