# TODO

## Phase 1: Domain Model (stabilize inputs/outputs)
- Add `Scene` dataclass to encapsulate room, sources, mics, and optional trajectories.
- Add `SimulationConfig` dataclass for algorithm settings (fs, max_order, tmax, directivity, device, seed).
- Add `RIRResult` dataclass to return RIR plus metadata (scene/config/timestamps/seed).
- Make `Room`, `Source`, `MicrophoneArray` immutable and add `.replace()` helpers.

## Phase 2: Strategy-based Simulation
- Define a `RIRSimulator` interface with `simulate(scene, config) -> RIRResult`.
- Implement `ISMSimulator` using current core logic.
- Add placeholder `RayTracingSimulator` and `FDTDSimulator` shells to clarify extension points.
- Ensure `simulate_rir`/`simulate_dynamic_rir` remain as thin wrappers for backward compatibility.

## Phase 3: Dynamic Convolution API
- Introduce `DynamicConvolver` class:
  - `mode="trajectory"` (gpuRIR-style timestamps/segments)
  - `mode="hop"` (legacy fixed hop)
- Deprecate direct `hop` argument on `convolve_dynamic_rir` in favor of the class.

## Phase 4: Dataset Abstraction
- Define `BaseDataset` protocol:
  - `list_speakers()`, `available_sentences()`, `load_wav()`
- Make `CmuArcticDataset` implement `BaseDataset`.
- Add dataset-agnostic example utilities (remove CMU-specific assumptions where possible).

## Phase 5: Configuration & Logging
- Replace global config (`config.py`) with explicit config passed into simulation.
- Add `LoggingConfig` and standardize log levels across examples and library.

## Phase 6: CLI / Recipes
- Consolidate examples into a single CLI with `--mode` (static/dynamic_src/dynamic_mic).
- Add deterministic flags, seed control, and config serialization (YAML/JSON).

## GPU/CPU Parity (gpuRIR-style acceleration)
- Add unified device selection (cpu/cuda/mps/auto) with availability checks and fallbacks.
- Introduce a `DeviceSpec` helper to resolve device + dtype defaults consistently.
- Replace Python loops in RIR accumulation with batched GPU kernels (custom CUDA extension or torch.compile), and add MPS-safe fallbacks.
- Implement gpuRIR-like convolution kernel for trajectory mode (segment-wise GPU conv).
- Add a GPU-optimized image source generator (vectorized across sources/mics/images).
- Provide benchmarking + correctness tests comparing CPU vs GPU outputs.
