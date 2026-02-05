# Overview

## Capabilities
- ISM-based static and dynamic RIR simulation (2D/3D shoebox rooms).
- Directivity patterns: `omni`, `cardioid`, `hypercardioid`, `subcardioid`, `bidir` with orientation handling.
- Acoustic parameters: `beta` or `t60` (Sabine), optional diffuse tail via `tdiff`.
- Dynamic convolution via `DynamicConvolver` (`trajectory` or `hop` modes).
- GPU acceleration for ISM accumulation (CUDA/MPS; MPS disables LUT).
- Dataset utilities with CMU ARCTIC support and example pipelines.
- Plotting utilities for static and dynamic scenes.
- Metadata export helpers for time axis, DOA, and array attributes (JSON-ready).
- Unified CLI with JSON/YAML config and deterministic flag support.

## Device selection
- `device="cpu"`: CPU execution
- `device="cuda"`: NVIDIA GPU (CUDA) if available, otherwise fallback to CPU
- `device="mps"`: Apple Silicon GPU via Metal (MPS) if available, otherwise fallback to CPU
- `device="auto"`: prefer CUDA → MPS → CPU

```python
from torchrir import DeviceSpec

device, dtype = DeviceSpec(device="auto").resolve()
```

## Limitations and potential errors
- Ray tracing and FDTD simulators are placeholders and raise `NotImplementedError`.
- `TemplateDataset` methods are not implemented and will raise `NotImplementedError`.
- `simulate_rir`/`simulate_dynamic_rir` require `max_order` (or `SimulationConfig.max_order`) and either `nsample` or `tmax`.
- Non-`omni` directivity requires orientation; mismatched shapes raise `ValueError`.
- `beta` must have 4 (2D) or 6 (3D) elements; invalid sizes raise `ValueError`.
- `simulate_dynamic_rir` requires `src_traj` and `mic_traj` to have matching time steps.
- Dynamic simulation currently loops per time step; very long trajectories can be slow.
- MPS disables the sinc LUT path (falls back to direct sinc), which can be slower and slightly different numerically.
- Deterministic mode is best-effort; some backends may still be non-deterministic.
- YAML configs require `PyYAML`; otherwise a `ModuleNotFoundError` is raised.
- CMU ARCTIC downloads require network access.
- GIF animation output requires Pillow (via matplotlib animation writer).

## Dataset utilities
```python
from torchrir import (
    CmuArcticDataset,
    clamp_positions,
    load_dataset_sources,
    collate_dataset_items,
    sample_positions,
)
import random
from torch.utils.data import DataLoader

def dataset_factory(speaker: str | None):
    spk = speaker or "bdl"
    return CmuArcticDataset("datasets/cmu_arctic", speaker=spk, download=True)

signals, fs, info = load_dataset_sources(
    dataset_factory=dataset_factory,
    num_sources=2,
    duration_s=10.0,
    rng=random.Random(0),
)

dataset = CmuArcticDataset("datasets/cmu_arctic", speaker="bdl", download=True)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_dataset_items)
```

`TemplateDataset` provides a minimal stub for future dataset integrations.

### LibriSpeech example
```python
from pathlib import Path
from torchrir import LibriSpeechDataset

dataset = LibriSpeechDataset(
    Path("datasets/librispeech"),
    subset="train-clean-100",
    download=True,
)
audio, fs = dataset.load_wav("103-1240-0000")
```

## Logging
```python
from torchrir import LoggingConfig, get_logger, setup_logging

setup_logging(LoggingConfig(level="INFO"))
logger = get_logger("examples")
logger.info("running torchrir example")
```

## Scene and results
```python
from torchrir import Scene, RIRResult

scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
scene.validate()
result = RIRResult(rirs=rirs, scene=scene, config=config)
```

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
