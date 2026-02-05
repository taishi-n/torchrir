# Examples

## Dynamic CMU ARCTIC dataset (fixed room, fixed mic, moving sources)

This example generates a small dynamic dataset inspired by Cross3D: the room
and binaural microphone are fixed, while source positions and trajectories are
randomized per scene. Each scene produces a convolved mixture and metadata.

### What it does

- Uses CMU ARCTIC utterances as source signals.
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

### Run

```bash
uv run python examples/cmu_arctic_dynamic_dataset.py \
  --num-scenes 4 \
  --num-sources 2 \
  --duration 6
```

### Key arguments

- `--num-scenes`: number of scenes to generate.
- `--num-sources`: number of sources per scene.
- `--duration`: length (seconds) of each source mixture.
- `--steps`: number of RIR steps (trajectory resolution).
- `--order`: ISM reflection order.
- `--tmax`: RIR length in seconds.
- `--seed`: RNG seed for reproducibility.

### Implementation notes

The example is implemented in `examples/cmu_arctic_dynamic_dataset.py` and uses:

- `load_dataset_sources` to build fixed-length signals from multiple utterances.
- `simulate_dynamic_rir` to generate the dynamic RIR sequence.
- `DynamicConvolver(mode="trajectory")` to produce the final mixture.
- `build_metadata` + `save_metadata_json` to store scene metadata.
