# Examples

## Static CMU ARCTIC (fixed sources + fixed mic)

This example mixes multiple CMU ARCTIC utterances using a static ISM RIR and
produces a binaural output.

### Key arguments

- `--num-sources`: number of source speakers to mix.
- `--duration`: length (seconds) of each source signal.
- `--order`: ISM reflection order.
- `--tmax`: RIR length in seconds.
- `--room`: room size (Lx Ly Lz).
- `--plot`: save layout plots.
- `--out-dir`: output directory for WAV/metadata/plots.

### Example runs

```bash
uv run python examples/static.py --num-sources 1 --duration 5 --plot
```

Expected outputs:
- `static_binaural.wav`
- `static_binaural_metadata.json`
- `static_static_2d.png` (and `static_static_3d.png` if 3D)

```bash
uv run python examples/static.py --order 12 --tmax 0.6 --device auto
```

Expected outputs:
- `static_binaural.wav`
- `static_binaural_metadata.json`

## Dynamic CMU ARCTIC (moving sources, fixed mic)

This example generates moving source trajectories and convolves with dynamic
RIRs (trajectory mode).

Plotting utilities are provided by `save_scene_plots` and `save_scene_gifs`.

### Key arguments

- `--steps`: number of RIR time steps for the trajectory.
- `--order`: ISM reflection order.
- `--tmax`: RIR length in seconds.
- `--gif`: save trajectory animation GIF.
- `--gif-fps`: override GIF FPS (<=0 uses auto).
- `--out-dir`: output directory for WAV/metadata/plots/GIFs.

### Example runs

```bash
uv run python examples/dynamic_src.py --steps 24 --gif --plot
```

Expected outputs:
- `dynamic_src_binaural.wav`
- `dynamic_src_binaural_metadata.json`
- `dynamic_src_static_2d.png` / `dynamic_src_dynamic_2d.png`
- `dynamic_src.gif` (and `dynamic_src_3d.gif` if 3D)

```bash
uv run python examples/dynamic_src.py --num-sources 3 --duration 8 --order 10
```

Expected outputs:
- `dynamic_src_binaural.wav`
- `dynamic_src_binaural_metadata.json`

## Dynamic CMU ARCTIC (fixed sources, moving mic)

This example keeps sources fixed and moves the binaural mic along a linear path.

### Key arguments

- `--steps`: number of RIR time steps for the trajectory.
- `--gif`: save trajectory animation GIF.
- `--plot`: save layout plots.
- `--out-dir`: output directory for WAV/metadata/plots/GIFs.

### Example runs

```bash
uv run python examples/dynamic_mic.py --steps 20 --gif --plot
```

Expected outputs:
- `dynamic_mic_binaural.wav`
- `dynamic_mic_binaural_metadata.json`
- `dynamic_mic_static_2d.png` / `dynamic_mic_dynamic_2d.png`
- `dynamic_mic.gif` (and `dynamic_mic_3d.gif` if 3D)

```bash
uv run python examples/dynamic_mic.py --order 12 --tmax 0.6 --device auto
```

Expected outputs:
- `dynamic_mic_binaural.wav`
- `dynamic_mic_binaural_metadata.json`

## Unified CLI (static/dynamic)

The unified CLI wraps the three scenarios above and supports JSON/YAML configs.

### Key arguments

- `--mode`: `static`, `dynamic_src`, or `dynamic_mic`.
- `--config-in`: load settings from JSON/YAML.
- `--config-out`: write current settings to JSON/YAML.
- `--deterministic`: enable deterministic kernels (best-effort).
- `--out-dir`: output directory for WAV/metadata/plots/GIFs.

### Example runs

```bash
uv run python examples/cli.py --mode static --plot
```

Expected outputs:
- `static_binaural.wav`
- `static_binaural_metadata.json`
- `static_static_2d.png` (and 3D variant if room is 3D)

```bash
uv run python examples/cli.py --mode dynamic_src --gif --steps 24
```

Expected outputs:
- `dynamic_src_binaural.wav`
- `dynamic_src_binaural_metadata.json`
- `dynamic_src.gif` (and 3D variant if room is 3D)

## Benchmark (CPU vs GPU)

This script times static ISM and optional dynamic trajectory simulation.

### Key arguments

- `--repeats`: number of iterations to average.
- `--gpu`: `cuda`, `mps`, or `auto`.
- `--dynamic`: benchmark dynamic trajectory path as well.

### Example runs

```bash
uv run python examples/benchmark_device.py --repeats 10 --gpu auto
```

Expected output (logs):
- `cpu avg: ... ms`
- `<device> avg: ... ms`
- `speedup: ...x`

```bash
uv run python examples/benchmark_device.py --dynamic --repeats 5 --gpu mps
```

Expected output (logs):
- `cpu dynamic avg: ... ms`
- `mps dynamic avg: ... ms`
- `speedup: ...x`

## Dynamic dataset builder (fixed room, fixed mic, moving sources)

This example generates a small dynamic dataset inspired by Cross3D: the room
and binaural microphone are fixed, while source positions and trajectories are
randomized per scene. Each scene produces a convolved mixture and metadata.
You can choose CMU ARCTIC or LibriSpeech from the command line.

### What it does

- Uses CMU ARCTIC or LibriSpeech utterances as source signals.
- Samples random source trajectories (linear or zigzag) within a fixed room.
- Keeps the microphone array fixed across all scenes.
- Simulates dynamic RIRs and convolves the sources.
- Saves one WAV and one metadata JSON per scene.

### Output files

For each scene index `k`:

- `scene_k.wav` — binaural mixture
- `scene_k_metadata.json` — room size, trajectories, DOA, array attributes, etc.
- `scene_k_static_2d.png` / `scene_k_dynamic_2d.png` — layout plots
  (3D variants are saved when the room is 3D)

### Run (CMU ARCTIC)

```bash
uv run python examples/build_dynamic_dataset.py \
  --dataset cmu_arctic \
  --num-scenes 4 \
  --num-sources 2 \
  --duration 6
```

### Run (LibriSpeech)

```bash
uv run python examples/build_dynamic_dataset.py \
  --dataset librispeech \
  --subset train-clean-100 \
  --num-scenes 4 \
  --num-sources 2 \
  --duration 6
```

### Additional examples

```bash
# CMU ARCTIC: only 1 moving source, plotting enabled
uv run python examples/build_dynamic_dataset.py \
  --dataset cmu_arctic \
  --num-scenes 2 \
  --num-sources 3 \
  --num-moving-sources 1 \
  --plot
```

```bash
# LibriSpeech: more steps, fewer scenes
uv run python examples/build_dynamic_dataset.py \
  --dataset librispeech \
  --subset dev-clean \
  --num-scenes 2 \
  --num-sources 2 \
  --steps 96
```

### Key arguments

- `--dataset`: dataset backend (`cmu_arctic` / `librispeech`).
- `--subset`: LibriSpeech subset (e.g., `train-clean-100`).
- `--num-scenes`: number of scenes to generate.
- `--num-sources`: number of sources per scene.
- `--num-moving-sources`: number of sources that move (others stay fixed).
- `--num-mics`: number of microphones in the fixed array.
- `--duration`: length (seconds) of each source mixture.
- `--steps`: number of RIR steps (trajectory resolution).
- `--order`: ISM reflection order.
- `--tmax`: RIR length in seconds.
- `--seed`: RNG seed for reproducibility.
- `--dataset-dir`: dataset root path.
- `--out-dir`: output directory for per-scene WAV/JSON/plots.
- `--plot`: enable plotting (default: off).
- `--download`: download the dataset if missing (default: off; auto-downloads when data is absent).
- `--device`: cpu/cuda/mps/auto.

### Implementation notes

The example is implemented in `examples/build_dynamic_dataset.py` and uses:

- `torchrir.datasets.load_dataset_sources` to build fixed-length signals from multiple utterances.
- `torchrir.sim.simulate_dynamic_rir` to generate the dynamic RIR sequence.
- `torchrir.signal.DynamicConvolver(mode="trajectory")` to produce the final mixture.
- `save_scene_audio` + `save_scene_metadata` to store scene metadata (kept as separate calls).

### Additional example

```bash
uv run python examples/build_dynamic_dataset.py --dataset cmu_arctic --num-scenes 2 --out-dir outputs/ds_small
```

Expected outputs:
- `outputs/ds_small/scene_000.wav`, `scene_001.wav`
- `outputs/ds_small/scene_000_metadata.json`, `scene_001_metadata.json`
