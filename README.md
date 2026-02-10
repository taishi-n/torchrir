# TorchRIR

PyTorch-based room impulse response (RIR) simulation toolkit focused on a clean, modern API with GPU support.
This project has been substantially assisted by AI using Codex.
> [!WARNING]
> TorchRIR is under active development and may contain bugs or breaking changes.
> Please validate results for your use case.
If you find bugs or have feature requests, please open an issue.
Contributions are welcome.

## Installation
```bash
pip install torchrir
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

Detailed notes and equations:
[Read the Docs: Library Comparisons](https://torchrir.readthedocs.io/en/latest/comparisons.html)

## CUDA CI (GitHub Actions)
- CUDA tests run in `.github/workflows/cuda-ci.yml` on a self-hosted runner with labels:
  `self-hosted`, `linux`, `x64`, `cuda`.
- The workflow validates installation via `uv sync --group test`, checks `torch.cuda.is_available()`,
  runs `tests/test_device_parity.py` with `-k cuda`, and then tries to install
  `gpuRIR` from GitHub.
- If `gpuRIR` installs successfully, the workflow runs `tests/test_compare_gpurir.py`
  (static + dynamic RIR comparisons). If installation fails, those comparison tests
  are skipped without failing the whole CUDA CI job.

## Examples
- `examples/static.py`: fixed sources/mics with configurable mic count (default: binaural).  
  `uv run python examples/static.py --plot`
- `examples/dynamic_src.py`: moving sources, fixed mics.  
  `uv run python examples/dynamic_src.py --plot`
- `examples/dynamic_mic.py`: fixed sources, moving mics.  
  `uv run python examples/dynamic_mic.py --plot`
- `examples/cli.py`: unified CLI for static/dynamic scenes, JSON/YAML configs.  
  `uv run python examples/cli.py --mode static --plot`
- `examples/build_dynamic_dataset.py`: small dynamic dataset generator (CMU ARCTIC / LibriSpeech; fixed room/mics, randomized source motion).  
  `uv run python examples/build_dynamic_dataset.py --dataset cmu_arctic --num-scenes 4 --num-sources 2`
- `examples/benchmark_device.py`: CPU/GPU benchmark for RIR simulation.  
  `uv run python examples/benchmark_device.py --dynamic`

## Core API Overview
- Geometry: `Room`, `Source`, `MicrophoneArray`
- Static RIR: `torchrir.sim.simulate_rir`
- Dynamic RIR: `torchrir.sim.simulate_dynamic_rir`
- Dynamic convolution: `torchrir.signal.DynamicConvolver`
- Audio metadata I/O: `torchrir.io.AudioData`, `torchrir.io.audio.load_audio_data`
- Metadata export: `torchrir.io.build_metadata`, `torchrir.io.save_metadata_json`

## Module Layout (for contributors)
- `torchrir.sim`: simulation backends (ISM implementation lives under `torchrir.sim.ism`)
- `torchrir.signal`: convolution utilities and dynamic convolver
- `torchrir.geometry`: array geometries, sampling, trajectories
- `torchrir.viz`: plotting and animation helpers
- `torchrir.models`: room/scene/result data models
- `torchrir.io`: audio I/O and metadata serialization (wav-only load/save/info with backend selection)
- `torchrir.util`: shared math/tensor/device helpers
- `torchrir.logging`: logging utilities
- `torchrir.config`: simulation configuration objects

```python
from torchrir import MicrophoneArray, Room, Source
from torchrir.sim import simulate_rir
from torchrir.signal import DynamicConvolver

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 2.0, 1.5]])
mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])

rir = simulate_rir(room=room, sources=sources, mics=mics, max_order=6, tmax=0.3)
# For dynamic scenes, compute rirs with torchrir.sim.simulate_dynamic_rir and convolve:
# y = DynamicConvolver(mode="trajectory").convolve(signal, rirs)
```

For detailed documentation:
[Read the Docs](https://torchrir.readthedocs.io/en/latest/)

## Future Work
- Ray tracing backend: implement `torchrir.experimental.RayTracingSimulator` with frequency-dependent absorption/scattering.
- Dataset expansion: add additional dataset integrations beyond CMU ARCTIC (see `torchrir.experimental.TemplateDataset`), including torchaudio datasets (e.g., LibriSpeech, VCTK, LibriTTS, SpeechCommands, CommonVoice, GTZAN, MUSDB-HQ).

## Related Libraries
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
- [Cross3D](https://github.com/DavidDiazGuerra/Cross3D)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [rir-generator](https://github.com/audiolabs/rir-generator)
