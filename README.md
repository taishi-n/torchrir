# TorchRIR

PyTorch-based room impulse response (RIR) simulation toolkit focused on a clean, modern API with GPU support.
This project has been substantially assisted by AI using Codex.

## Installation
```bash
pip install torchrir
```

## Examples
- `examples/static.py`: fixed sources/mics with binaural output.  
  `uv run python examples/static.py --plot`
- `examples/dynamic_src.py`: moving sources, fixed mics.  
  `uv run python examples/dynamic_src.py --plot`
- `examples/dynamic_mic.py`: fixed sources, moving mics.  
  `uv run python examples/dynamic_mic.py --plot`
- `examples/cli.py`: unified CLI for static/dynamic scenes, JSON/YAML configs.  
  `uv run python examples/cli.py --mode static --plot`
- `examples/cmu_arctic_dynamic_dataset.py`: small dynamic dataset generator (fixed room/mics, randomized source motion).  
  `uv run python examples/cmu_arctic_dynamic_dataset.py --num-scenes 4 --num-sources 2`
- `examples/benchmark_device.py`: CPU/GPU benchmark for RIR simulation.  
  `uv run python examples/benchmark_device.py --dynamic`

## Core API Overview
- Geometry: `Room`, `Source`, `MicrophoneArray`
- Static RIR: `simulate_rir`
- Dynamic RIR: `simulate_dynamic_rir`
- Dynamic convolution: `DynamicConvolver`
- Metadata export: `build_metadata`, `save_metadata_json`

```python
from torchrir import DynamicConvolver, MicrophoneArray, Room, Source, simulate_rir

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 2.0, 1.5]])
mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])

rir = simulate_rir(room=room, sources=sources, mics=mics, max_order=6, tmax=0.3)
# For dynamic scenes, compute rirs with simulate_dynamic_rir and convolve:
# y = DynamicConvolver(mode="trajectory").convolve(signal, rirs)
```

For detailed documentation, see the docs under `docs/` and Read the Docs.

## Future Work
- Ray tracing backend: implement `RayTracingSimulator` with frequency-dependent absorption/scattering.
- CUDA-native acceleration: introduce dedicated CUDA kernels for large-scale RIR generation.
- Dataset expansion: add additional dataset integrations beyond CMU ARCTIC (see `TemplateDataset`).
- Add regression tests comparing generated RIRs against gpuRIR outputs.

## Related Libraries
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
- [Cross3D](https://github.com/DavidDiazGuerra/Cross3D)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [das-generator](https://github.com/ehabets/das-generator)
- [rir-generator](https://github.com/audiolabs/rir-generator)
