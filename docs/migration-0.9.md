# Migrating to the scene-oriented API

TorchRIR 0.9 introduces a scene-oriented simulation API while retaining the
0.8 entry points as deprecated compatibility wrappers until 1.0.

## Simulation

Replace settings stored on `ISMSimulator` or passed separately to
`simulate_rir` with one `SimulationConfig`:

```python
from torchrir import StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate

scene = StaticScene(room=room, sources=sources, mics=mics)
result = simulate(scene, SimulationConfig(max_order=6, tmax=0.3))
rirs = result.rirs
```

`result.config` is a `ResolvedSimulationConfig` containing the effective sample
count, duration, device, dtype, and directivity.

## Dynamic scenes

`DynamicScene` validates that static entity positions equal the first
trajectory frame. Optional timestamps begin at zero and are strictly
increasing. Constant source and microphone orientations are propagated to the
dynamic ISM backend.

## Audio tensors

`AudioData` uses `(samples,)` for mono and `(channels, samples)` for
multichannel audio. `load_audio_data` preserves every channel and
`save_audio_data` no longer normalizes by default.

Legacy tuple loaders continue to select channel 0 from multichannel files until
1.0. Use `AudioData` for lossless channel round trips.

## Dataset builds

Use `DynamicCmuArcticBuildConfig` and `build_dynamic_cmu_arctic` for new code.
The former keyword-based builder remains available until 1.0. Dataset output is
now staged and replaces an existing destination only after a complete build.

## Scheduled for removal in 1.0

- `torchrir.models.Scene`
- top-level `torchrir.load` and `torchrir.save`
- `torchrir.io.load`, `torchrir.io.save`, and `torchrir.io.info`
- process-global `set_audio_backend`
- simulation settings on the `ISMSimulator` constructor
- individual simulation-setting arguments on Tensor-level RIR functions
- `SimulationConfig.fs` and `SimulationConfig.mixed_precision`
