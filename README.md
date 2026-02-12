# TorchRIR

A PyTorch-based room impulse response (RIR) simulation toolkit with a clean API and GPU support.
This project has been developed with substantial assistance from Codex.
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
| 🎛️ Signal Processing | ❌ Scope out | ❌ | ✅ | ❌ |
| 🧱 Non-shoebox Geometry | 🚧 Candidate | ❌ | ✅ | ❌ |
| 🌐 Geometric Acoustics | 🚧 Candidate | ❌ | ✅ | ❌ |

Legend: `✅` native support, `🟡` manual setup, `🚧` candidate (not yet implemented), `❌` unavailable

For detailed notes and equations, see
[Read the Docs: Library Comparisons](https://torchrir.readthedocs.io/en/latest/comparisons.html).

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
  [Read the Docs: Datasets](https://torchrir.readthedocs.io/en/latest/datasets.html)

## Core API Overview
- Geometry: `Room`, `Source`, `MicrophoneArray`
- Scene models: `StaticScene`, `DynamicScene` (`Scene` is deprecated)
- Static RIR: `torchrir.sim.simulate_rir`
- Dynamic RIR: `torchrir.sim.simulate_dynamic_rir`
- Simulator object: `torchrir.sim.ISMSimulator(max_order=..., tmax=... | nsample=...)`
- Dynamic convolution: `torchrir.signal.DynamicConvolver`
- Audio I/O:
  - wav-specific: `torchrir.io.load_wav`, `torchrir.io.save_wav`, `torchrir.io.info_wav`
  - backend-supported formats: `torchrir.io.load_audio`, `torchrir.io.save_audio`, `torchrir.io.info_audio`
  - metadata-preserving: `torchrir.io.AudioData`, `torchrir.io.load_audio_data`
- Metadata export: `torchrir.io.build_metadata`, `torchrir.io.save_metadata_json`

## Module Layout (for contributors)
- `torchrir.sim`: simulation backends (ISM implementation lives under `torchrir.sim.ism`)
- `torchrir.signal`: convolution utilities and dynamic convolver
- `torchrir.geometry`: array geometries, sampling, trajectories
- `torchrir.viz`: plotting and animation helpers
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
- `ISMSimulator` fails fast when `max_order` or `tmax` conflicts with the provided `SimulationConfig`.
- Model dataclasses are frozen, but tensor payloads remain mutable (shallow immutability).
- `torchrir.load` / `torchrir.save` and `torchrir.io.load` / `save` / `info` are deprecated compatibility aliases.

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
