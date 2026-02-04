# TODO

This TODO list is ordered by priority (higher priority items appear first).

## GPU/CPU Parity (gpuRIR-style acceleration)
- [x] Add unified device selection (cpu/cuda/mps/auto) with availability checks and fallbacks.
- [x] Introduce a `DeviceSpec` helper to resolve device + dtype defaults consistently.
- [x] Replace Python loops in RIR accumulation with batched GPU kernels (torch.compile path), and add MPS-safe fallbacks.
- [x] Implement gpuRIR-like convolution kernel for trajectory mode (segment-wise GPU conv).
- [x] Add a GPU-optimized image source generator (vectorized across sources/mics/images).
- [x] Provide benchmarking + correctness tests comparing CPU vs GPU outputs.

## Dynamic Convolution API
- [x] Introduce `DynamicConvolver` class:
  - `mode="trajectory"` (gpuRIR-style timestamps/segments)
  - `mode="hop"` (legacy fixed hop)
- [x] Deprecate direct `hop` argument on `convolve_dynamic_rir` in favor of the class.

## Dataset Abstraction
- [x] Define `BaseDataset` protocol:
  - `list_speakers()`, `available_sentences()`, `load_wav()`
- [x] Make `CmuArcticDataset` implement `BaseDataset`.
- [x] Add dataset-agnostic utilities (dataset loading + scene helpers).

## Configuration & Logging
- [x] Replace global config (`config.py`) with explicit config passed into simulation.
- [x] Add `LoggingConfig` and standardize log levels across examples and library.

## Domain Model (stabilize inputs/outputs)
- [x] Add `Scene` dataclass to encapsulate room, sources, mics, and optional trajectories.
- [x] Add `SimulationConfig` dataclass for algorithm settings (fs, max_order, tmax, directivity, device, seed).
- [x] Add `RIRResult` dataclass to return RIR plus metadata (scene/config/timestamps/seed).
- [x] Make `Room`, `Source`, `MicrophoneArray` immutable and add `.replace()` helpers.

## Strategy-based Simulation
- [ ] Define a `RIRSimulator` interface with `simulate(scene, config) -> RIRResult`.
- [ ] Implement `ISMSimulator` using current core logic.
- [ ] Add placeholder `RayTracingSimulator` and `FDTDSimulator` shells to clarify extension points.
- [ ] Ensure `simulate_rir`/`simulate_dynamic_rir` remain as thin wrappers for backward compatibility.

## CLI / Recipes
- [ ] Consolidate examples into a single CLI with `--mode` (static/dynamic_src/dynamic_mic).
- [ ] Add deterministic flags, seed control, and config serialization (YAML/JSON).
