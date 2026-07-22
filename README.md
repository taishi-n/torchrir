# TorchRIR

A PyTorch-based room impulse response (RIR) simulation toolkit with a clean API and GPU support.
This project has been developed with substantial assistance from Codex.
> [!WARNING]
> TorchRIR is under active development and may contain bugs or breaking changes.
> Please validate results for your use case.
If you find bugs or have feature requests, please open an issue.
Contributions are welcome.

## Installation

TorchRIR supports Python 3.11.4 through 3.13. Python 3.11.4 is the minimum
because dataset extraction requires the standard-library
`tarfile.data_filter` security policy; Python 3.14 is not yet supported.

```bash
pip install torchrir
```

Install only the optional features you use:

```bash
pip install "torchrir[audio]"       # SoundFile-backed audio I/O
pip install "torchrir[cli]"         # YAML configuration for the example CLI
pip install "torchrir[hpf]"         # optional RIR high-pass filtering
pip install "torchrir[viz]"         # plots, GIFs, and MP4 rendering
pip install "torchrir[datasets]"    # dataset loaders and builders
pip install "torchrir[oobss]"       # oobss integration
pip install "torchrir[all]"         # all optional features
```

## Library Comparison
| Feature | `torchrir` | `gpuRIR` | `pyroomacoustics` | `rir-generator` |
|---|---|---|---|---|
| 🎯 Dynamic Sources | ✅ Emission-time | 🟡 Single moving source | 🟡 Manual loop | ❌ |
| 🎤 Dynamic Microphones | ✅ Observation-time | ❌ | 🟡 Manual loop | ❌ |
| 🖥️ CPU | ✅ | ❌ | ✅ | ✅ |
| 🧮 CUDA | ✅ | ✅ | ❌ | ❌ |
| 🍎 MPS | ✅ | ❌ | ❌ | ❌ |
| 📊 Scene Plot | ✅ | ❌ | ✅ | ❌ |
| 🎞️ Dynamic Scene GIF | ✅ | ❌ | 🟡 Manual animation script | ❌ |
| 🗂️ Dataset Build | ✅ | ❌ | ✅ | ❌ |
| 🎛️ RIR Convolution | ✅ Static/dynamic | 🟡 Dynamic helper | ✅ | ❌ |
| 🧱 Non-shoebox Geometry | 🚧 Candidate | ❌ | ✅ | ❌ |
| 🌐 Geometric Acoustics | 🚧 Candidate | ❌ | ✅ | ❌ |

Legend: `✅` native support, `🟡` manual setup, `🚧` candidate (not yet implemented), `❌` unavailable

For detailed notes and equations, see
[Documentation: Library Comparisons](https://torchrir.readthedocs.io/en/latest/comparisons.html).

## CUDA CI (GitHub Actions)

- CUDA tests run in `.github/workflows/cuda-ci.yml` on a self-hosted runner with labels:
  `self-hosted`, `linux`, `x64`, `cuda`.
- The workflow validates the lockfile-based installation via
  `uv sync --locked --group test`, checks `torch.cuda.is_available()`, runs
  `tests/test_device_parity.py` with `-k cuda`, and executes the dynamic
  emission-time convolution autograd regression on CUDA.
- The extended job installs the pinned `gpuRIR` reference revision.
- The extended workflow requires `gpuRIR` installation and runs direct-path
  static/dynamic RIR comparisons plus an isolated `simulateTrajectory`
  comparison using identical synthetic RIRs. Known normalization, image-count,
  and fractional-delay conventions are handled explicitly; signals are never
  freely aligned. Installation failure or a skipped comparison fails that
  workflow.

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
- `torchrir.datasets.dynamic_cmu_arctic`: oobss-compatible dynamic CMU ARCTIC builder CLI.  
  `python -m torchrir.datasets.dynamic_cmu_arctic --cmu-root datasets/cmu_arctic --n-scenes 2 --overwrite-dataset`
- `examples/benchmark_device.py`: CPU/GPU benchmark for RIR simulation.  
  `uv run python examples/benchmark_device.py --dynamic`

## Dataset Notices
- For dataset attribution and redistribution notes, see
  [THIRD_PARTY_DATASETS.md](https://github.com/taishi-n/torchrir/blob/main/THIRD_PARTY_DATASETS.md).

## Dataset API Quick Guide
- `torchrir.datasets.CmuArcticDataset(root, speaker=..., download=...)`
  - Accepted `speaker`: `aew`, `ahw`, `aup`, `awb`, `axb`, `bdl`, `clb`, `eey`, `fem`, `gka`, `jmk`, `ksp`, `ljm`, `lnh`, `rms`, `rxr`, `slp`, `slt`
  - A non-string `speaker` raises `TypeError`; an unsupported string raises
    `ValueError`.
  - Every non-empty transcript entry must parse and contain a canonical
    `arctic_<letter><four-digits>` ID, and at least one such ID must have a
    same-named regular WAV below the dataset tree. A missing or incomplete tree
    raises `FileNotFoundError` with `download=False`.
  - The speaker dataset root, every traversed parent directory, and final
    transcript/audio entries must not be symlinks. Files are opened by walking
    directory descriptors component by component with `O_NOFOLLOW`; the same
    opened descriptor is consumed through parsing or SoundFile decoding.
  - Downloads use HTTPS, verify the pinned per-speaker SHA-256 table also
    published by
    [torchaudio](https://docs.pytorch.org/audio/2.0.0/_modules/torchaudio/datasets/cmuarctic.html),
    and extract only regular files/directories below the dataset root.
    Extraction is validated in staging before publication, and an already
    usable tree skips archive and network access even with `download=True`.
  - `cmu_arctic_speakers()` lists every supported speaker;
    `cmu_arctic_speakers(root)` lists only usable local speaker directories.
- `torchrir.datasets.LibriSpeechDataset(root, subset=..., speaker=..., download=...)`
  - Accepted `subset`: `dev-clean`, `dev-other`, `test-clean`, `test-other`, `train-clean-100`, `train-clean-360`, `train-other-500`
  - Invalid `subset` raises `ValueError`.
  - A numeric speaker tree is ready only when, in every transcript discovered
    directly below one of its numeric chapters, each non-empty entry uses a
    canonical ID that agrees with its speaker and chapter directories. At least
    one entry must have a same-named regular FLAC. Missing or incomplete
    subset/speaker trees raise `FileNotFoundError` with `download=False`.
  - The subset root, numeric speaker/chapter directories, and final
    transcript/audio entries must not be symlinks. Final entries are opened
    with the same descriptor walk and consumed through that descriptor rather
    than reopening a validated pathname.
  - Downloads verify the published subset MD5 and reject unsafe or special
    archive members. Validated staged publication preserves the previous tree
    if replacement fails; an already usable subset skips archive/network work.
    With `speaker=...`, the fast path checks that requested speaker rather than
    accepting an unrelated usable speaker in a partial subset.
- Shared download and publication contract:
  - Secure dataset filesystem operations are supported on Linux and macOS only.
    They require POSIX `dir_fd`, `O_DIRECTORY`, `O_NOFOLLOW`, and the platform's
    atomic no-replace/exchange rename primitives; unsupported platforms or
    filesystems raise `NotImplementedError` before an operation starts.
  - Network requests use a 60-second per-connect/per-read timeout and a six-hour
    transfer deadline. The remaining total time is applied to every connection
    and response-body read; this is a network-transfer deadline, not a hard
    real-time bound on hashing or disk I/O. A response is capped at 64 GiB and
    streamed into a fresh, uniquely named, TorchRIR-owned sibling workspace.
    If supplied, `Content-Length` must be a valid non-negative value at most
    64 GiB and must match the complete body exactly. The final archive path is
    changed only after the mandatory pinned digest has also been verified.
  - One retry is made only for HTTP 408/429/5xx, transport, or transfer-integrity
    failures, including a malformed `Content-Length`. Local filesystem,
    extraction, validation, publication, and 64 GiB safety-limit errors are
    propagated without retrying or re-downloading an already verified archive.
  - A cached archive may be absent, a regular file, or a symlink; other entry
    types are rejected. An absent destination is published with atomic
    no-replace. An existing entry is exchanged with the verified temporary file
    only if its device, inode, and type are unchanged; a racing third-party entry
    is restored and retained. A symlink target is never followed. Digest
    verification and tar parsing share one `O_NOFOLLOW` descriptor and inode.
  - Archives are fully prevalidated before extraction. Only regular files and
    directories are accepted, with limits of 1,000,000 members, 8 GiB per file,
    and 512 GiB total declared file data.
  - A per-speaker CMU or per-subset LibriSpeech writer lock covers stale owned
    workspace cleanup, the readiness check, cache handling, download,
    extraction, validation, and publication. Unsafe or unowned stale paths are
    retained with a warning; cleanup failures do not hide the primary result.
    A separate per-target publication lock covers only transaction recovery and
    the final target check/rename/rollback/cleanup sequence.
  - Replacement transactions persist a fsynced manifest containing the staged
    and previous tree identities plus the publication phase. Recovery restores
    only an identity recorded by that manifest. Unknown transaction entries,
    unsafe symlinks, and an unrelated target stop recovery without deleting the
    retained backup.
  - Dynamic builds for one target acquire a separate build lock before recovery,
    target checks, stale owned-build-workspace cleanup, or scene generation.
    They are therefore serialized for the complete expensive build. Publication
    still uses its narrower nested target lock for the final transaction.
- `torchrir.datasets.build_dynamic_cmu_arctic(DynamicCmuArcticBuildConfig(...))`
  - Builds oobss-compatible scene folders with `mixture.wav`, `source_XX.wav`, `metadata.json`, and `source_info.json`.
  - Returns `DynamicDatasetBuildResult` with the resolved output root, sample
    rate, microphone count, and scene paths.
  - Static layout images (`room_layout_2d.png`, `room_layout_3d.png`) and optional layout videos (`room_layout_2d.mp4`, `room_layout_3d.mp4`) are generated, with source-index annotations by default.
  - `simulation=SimulationConfig(...)` owns every RIR algorithm setting. The
    default is `max_order=6`, `nsample=4096`, and `dtype=torch.float32`, with
    `trajectory_steps=256`.
  - The corpus root and output root must not overlap. Related stems and their
    mixture receive one common anti-clipping scale and are stored as
    floating-point WAV, preserving relative gain and additivity.
  - Requested duration is converted with `ceil(duration_sec * sample_rate)`;
    metadata records both requested and effective sample-aligned durations.
  - Other defaults include `n_sources=3`, moving speed range `0.3-0.8 m/s`, and motion profile ratios `0-35%`, `35-65%`, `65-100%`.
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
  [Documentation: Datasets](https://torchrir.readthedocs.io/en/latest/datasets.html)

## Core API Overview
- Geometry: `Room`, `Source`, `MicrophoneArray`
- Scene models: `StaticScene`, `DynamicScene`
- Scene-oriented simulation: `torchrir.sim.simulate(scene, config)`
- Dynamic convolution: `torchrir.signal.DynamicConvolver` with an explicit
  emission/observation time reference and a scene-carried or call-level
  `FrameSchedule`
- Audio I/O:
  - wav-specific: `torchrir.io.load_wav`, `torchrir.io.save_wav`, `torchrir.io.info_wav`
  - SoundFile-supported formats: `torchrir.io.load_audio`, `torchrir.io.save_audio`, `torchrir.io.info_audio`
  - channel/metadata-aware: `torchrir.io.AudioData`,
    `torchrir.io.load_audio_data`, `torchrir.io.save_audio_data`
- Metadata export: `torchrir.io.build_result_metadata`,
  `torchrir.io.save_result_metadata`

## Module Layout (for contributors)
- `torchrir.sim`: the scene-oriented simulation entry point (the private ISM
  implementation lives under `torchrir.sim.ism`)
- `torchrir.signal`: convolution utilities and dynamic convolver
- `torchrir.geometry`: array geometries, sampling, trajectories
- `torchrir.viz`: plotting and GIF/MP4 animation helpers
  - Default plot style follows SciencePlots Grid (`science` + `grid`).
- `torchrir.models`: room/scene/result data models
- `torchrir.io`: audio I/O and metadata serialization (`*_wav` for wav-only,
  `*_audio` for SoundFile-supported formats)
- `torchrir.util`: shared math/tensor/device helpers
- `torchrir.logging`: logging utilities
- `torchrir.config`: simulation configuration objects

## Design Notes
- Scene typing is explicit: use `StaticScene` for fixed geometry and `DynamicScene` for trajectory-based simulation.
- `DynamicScene` accepts tensor-like trajectories (e.g., lists) and normalizes them to tensors internally.
- `DynamicScene.schedule` optionally stores the authoritative frame axis as
  exact CPU `int64` sample starts. A schedule made with `from_seconds`
  remembers the sample rate used for conversion and is rejected if it
  conflicts with `Room.fs`.
- `FrameSchedule` keeps immutable starts internally; its `starts` property
  returns a fresh CPU `int64` snapshot.
- `DynamicConvolver` requires `time_reference="emission"` for moving sources
  with fixed microphones and `time_reference="observation"` for fixed sources
  with moving microphones. Simultaneous motion is rejected because it requires
  a two-time retarded-propagation model.
- `SimulationConfig` owns algorithm settings and must be complete when
  constructed. Room sampling rate and endpoint directivity live in the scene.
- `RIRResult.config` is the single resolved config used by the kernel and
  records the effective sample count, duration, device, dtype, LUT, and compile
  settings.
- Model dataclasses are frozen but tensor payloads remain mutable. `RIRResult`
  snapshots every scene tensor and compares its identity and value at consumer
  boundaries, so mutation through PyTorch or a shared NumPy array is rejected.
  RIR finiteness is also rescanned on every validation.
- A scene requires all geometry Tensors to share one device and dtype. Sequence
  trajectories inherit their endpoint layout; Tensor trajectories are
  preserved and mismatches are rejected instead of silently converted.
- Endpoint orientations are stored as unit vectors with shape
  `(entities, dimensions)`. Angles use radians. In 2D, use a scalar or `(n, 1)`
  for angles and `(2,)` or `(n, 2)` for vectors; a length-two value is one
  shared vector. In 3D, use `(azimuth, elevation)` or `(n, 2)` angles and
  `(3,)` or `(n, 3)` vectors. A single row is broadcast to all entities.
- Audio I/O uses one SoundFile implementation; there is no process-global
  backend selector or format override.

```python
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate
from torchrir.signal import DynamicConvolver, FrameSchedule

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 2.0, 1.5]])
mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])

scene = StaticScene(room=room, sources=sources, mics=mics)
result = simulate(scene, SimulationConfig(max_order=6, tmax=0.3))
rir = result.rirs

# A scene-owned schedule is reused by the resulting RIRResult.
signal = torch.ones(1, 8000)
src_traj = torch.tensor([[[1.0, 2.0, 1.5]], [[1.5, 2.0, 1.5]]])
mic_traj = mics.positions.unsqueeze(0).expand(2, -1, -1).clone()
schedule = FrameSchedule.from_samples([0, 4000])
dynamic_scene = DynamicScene(
    room=room,
    sources=sources,
    mics=mics,
    src_traj=src_traj,
    mic_traj=mic_traj,
    schedule=schedule,
)
dynamic_result = simulate(
    dynamic_scene, SimulationConfig(max_order=6, tmax=0.3)
)
y = DynamicConvolver(time_reference="emission").convolve(signal, dynamic_result)
```

## Specification

### Geometry and scenes

- Room, source, microphone, and geometry-factory coordinates are real
  floating-point tensors. Their integer inputs are promoted to PyTorch's
  default floating dtype. Sequence trajectories inherit the endpoint dtype;
  Tensor trajectories must already match it.
- Room coordinates are 2D or 3D. Source, microphone, and trajectory coordinates
  must lie strictly inside the room: `0 < position < room.size`. Wall positions
  are rejected because they create duplicate images and singular paths.
- Every source--microphone pair must remain at least
  `SimulationConfig.min_source_mic_distance` apart (default: `1e-6` m).
- Static positions have shape `(entities, dimensions)`.
- Dynamic trajectories have shape `(frames, entities, dimensions)`. A 2D
  `(frames, dimensions)` trajectory is accepted for one entity.
- `DynamicScene.sources.positions` and `mics.positions` equal the first frame of
  their respective trajectories.
- `DynamicScene.schedule` is optional. When present, it contains one exact
  integer start per trajectory frame, begins at zero, and increases strictly.
  `from_seconds` floors once and retains conversion-sample-rate provenance;
  a mismatch with `Room.fs` is rejected. Emission-time starts must precede the
  dry-signal endpoint; observation-time starts may extend into the convolution
  tail.
- Continuous geometry must be sampled on that same frame axis. Use
  `progress = schedule.normalized_progress(stop_sample=..., dtype=...,
  device=...)` and pass it as the keyword-only `progress` argument to
  `linear_trajectory`. Frame `i` corresponds to
  `schedule.starts[i] / stop_sample`; the division is evaluated in float64
  before the requested dtype cast. If that dtype cannot distinguish the final
  ratio from one, it is capped at the greatest representable value below one;
  if two starts would collapse to the same cast value, conversion fails
  explicitly. The nominal endpoint is therefore never a frame start, and the
  last sampled frame remains active until the next boundary or timeline end.
- `Source.directivity` and `MicrophoneArray.directivity` default to `"omni"`.
  Non-omnidirectional endpoints require an orientation on the same entity.
- Supported canonical patterns are `omni`, `halfomni`, `subcardioid`,
  `cardioid`, `hypercardioid`, and `bidir`; common aliases are normalized at
  construction.
- Entity orientation is constant over a dynamic trajectory. Time-varying
  orientation is not yet supported.

### Simulation

- `SimulationConfig` is keyword-only and validates a complete request at
  construction. Exactly one of `tmax`/`nsample` and exactly one of
  `max_order`/`nb_img` are required.
- `nb_img` is normalized to an immutable tuple whose length must match the room
  dimension.
- Image-source indices are counted with checked integer arithmetic and streamed
  in `image_chunk_size` batches; the simulation path does not materialize the
  complete L1/rectangular index grid at once.
- `Room.beta` wall order is `[x-low, x-high, y-low, y-high]` in 2D and adds
  `[z-low, z-high]` in 3D.
- With `tmax`, the sample count is `ceil(tmax * fs)`. The resolved duration in
  `RIRResult.config.tmax` is `nsample / fs`, while the final returned sample is
  located at `(nsample - 1) / fs`.
- Returned RIRs always use the physical propagation-time sample axis.
  Fractional-delay FIR group delay is an internal implementation detail and
  late arrivals retain their available support at the requested horizon.
- Sabine conversion uses `12 ln(10) / c` in 2D and `24 ln(10) / c` in 3D.
  Targets requiring an absorption coefficient above one are rejected.
- A diffuse handoff requires `tdiff > 0`. Its level is estimated from the
  preceding 10 ms RMS and joined with a 5 ms power-complementary crossfade. A
  zero-energy handoff window raises an actionable error; seeded tails are
  reproducible across output horizons and unrelated batch composition, use
  `Room.c` for decay estimation, and preserve infinite Sabine T60 as a
  non-decaying carrier.
- RIR high-pass filtering is disabled by default, avoiding hidden CPU
  post-processing in accelerator simulations. Enable it explicitly with
  `high_pass=RIRHighPassConfig(...)`; `phase="causal"` is prefix invariant,
  while `phase="zero_phase"` is non-causal and depends on the finite endpoint.
  Supported filter families are `bessel`, `butter`, `cheby1`, `cheby2`, and
  `ellip`. This optional SciPy post-process detaches autograd and transfers
  accelerator results through CPU memory. Chebyshev-I and elliptic filters
  require positive passband ripple; Chebyshev-II and elliptic filters require
  positive stopband attenuation; elliptic attenuation must exceed its ripple.
- Static RIR tensors have shape `(sources, microphones, samples)`.
- Dynamic RIR tensors have shape `(frames, sources, microphones, samples)`.
- `torchrir.sim.simulate` returns `RIRResult`; its config is a
  `ResolvedSimulationConfig` containing the effective settings.
- `device=None` follows scene tensors. `device="auto"` explicitly chooses CUDA,
  then MPS, then CPU, even when the scene was constructed on CPU.
- RIR simulation supports `torch.float32` and `torch.float64`; inherited or
  explicit lower-precision geometry is rejected because image positions,
  distances, delays, and gains are not numerically safe in float16/bfloat16.
  MPS rejects `float64`; MPS resolves `use_lut=False`, and CPU resolves
  `use_compile=False`. These effective values are stored in `RIRResult.config`.

### Dynamic convolution time conventions

For RIR frame selector `f`, source `s`, microphone `m`, output sample `n`, and
delay sample `k`, emission-time convolution selects the RIR from the input
sample's emission time:

```text
y_m[n] = sum_s sum_k h_{f_e(n-k),s,m}[k] x_s[n-k]
```

`DynamicConvolver(time_reference="emission")` implements this convention for
moving sources and fixed microphones. Schedule starts delimit input samples.

Observation-time convolution instead selects the RIR from the output sample's
observation time:

```text
y_m[n] = sum_s sum_k h_{f_o(n),s,m}[k] x_s[n-k]
```

`DynamicConvolver(time_reference="observation")` implements this convention for
fixed sources and moving microphones. Schedule starts delimit output samples,
may occur after the dry signal ends, and the final RIR frame remains active
through the convolution tail.

`FrameSchedule` is independent of the time reference and stores one strictly
increasing CPU `int64` start sample per RIR frame. Construct it with
`from_samples`, `from_seconds`, `uniform`, or `fixed_hop`. Raw RIR tensors and
results whose `DynamicScene.schedule` is `None` require an explicit call-level
schedule. When the dynamic scene already owns a schedule, its `RIRResult`
supplies that schedule automatically and rejects a second explicit schedule to
prevent competing frame axes.

An `RIRResult` carries enough scene metadata to reject the wrong
motion/time-reference
combination. A raw RIR tensor does not; callers using raw tensors must ensure
that the selected convention matches the simulated motion. Simultaneous source
and microphone motion requires a two-time retarded propagation kernel and is
not implemented.

### Signal and audio shapes

- Dry convolution signals use `(samples,)`, `(1, samples)` for source-broadcast,
  or `(sources, samples)`. Static convolution accepts RIRs shaped
  `(rir_samples,)`, `(microphones, rir_samples)`, or
  `(sources, microphones, rir_samples)`; dynamic RIRs add the leading frame
  dimension.
- Static and dynamic convolution always return
  `(microphones, output_samples)`, including a single microphone.
- Dynamic convolution uses a required emission-time or observation-time
  reference and a frame schedule with exactly as many starts as RIR frames.
- Signal and RIR tensors must share device and dtype.
- Audio, dataset, and convolution Tensor payloads accept only `float16`,
  `bfloat16`, `float32`, or `float64`; float8 and non-floating payloads are
  rejected before backend kernels. Geometry constructors may promote integer
  coordinates as described above. RIR simulation itself accepts only
  `float32` and `float64` geometry.
- Public finite-real parameters in simulation configuration, room/acoustics,
  geometry construction/sampling, frame-time conversion, audio normalization,
  and dataset utilities reject booleans, numeric strings, scalar Tensors, NaN,
  and infinity instead of coercing them. Integers too large to represent as a
  finite float raise `ValueError` at the API boundary.
- Public count, index, sample-rate, and seed parameters accept non-boolean
  integer scalars, including NumPy integer scalars, and normalize them to Python
  `int`. Fractional values and booleans raise `TypeError`; operation-specific
  bounds are checked before tensor allocation or arithmetic. Audio and dataset
  sample rates are limited to `1..2**31-1`, simulation sample counts and frame
  starts fit positive/non-negative `int64`, and random seeds fit non-negative
  `int64`.
- Source and microphone FFT convolution is batched. `float16` and `bfloat16`
  inputs use `float32` FFT, source-sum, and overlap-add work buffers, then cast
  once on return; the implementation remains differentiable, including dynamic
  emission-time convolution on CUDA.
- Audio-file tensors separately use `(samples,)` or `(channels, samples)`.
  `AudioData` is keyword-only and validates non-empty, finite, floating-point
  audio plus a positive integer sample rate. It preserves all channels, sample
  rate, and subtype; `format` is retained as load metadata, while the output
  path selects the saved format. Its save API does not normalize unless
  `normalize=True` is requested. Normalization computes the scale in float64
  before narrowing to the destination storage dtype; an unrepresentable peak
  or unnormalized sample is rejected before a file is created.
  `save_audio_data` reuses `AudioData.subtype` only when the destination
  container matches the loaded format; otherwise a WAV with no subtype uses
  `FLOAT`, preserving values outside `[-1, 1]`. An integer or companded subtype
  rejects out-of-range audio before it can be clipped silently. Common suffix
  aliases such as `.wave`, `.aif`, `.aifc`, `.oga`, and `.snd` are resolved to
  their SoundFile container names. Every pathname argument must be a
  `pathlib.Path`; string paths are rejected instead of being coerced.
- `AudioInfo` normalizes Python/NumPy integer metadata. Its sample rate is in
  `1..2**31-1`, frame count in non-negative `int64`, and channel count in
  positive `int32`; invalid types and out-of-range values fail separately.
- `DatasetItem` is keyword-only and requires finite, non-empty mono
  floating-point audio, a positive integer sample rate, and a non-empty
  utterance ID. `collate_dataset_items` revalidates shallow-mutable items,
  rejects mixed sample rates/dtypes/devices, and preserves requested item
  metadata as immutable tuples in `CollateBatch`.
- `BaseDataset` is abstract. Corpus sentence and attribution records are frozen,
  slotted, keyword-only values that validate canonical IDs and required fields
  when constructed.
- `load_dataset_sources` requires an explicit `speakers=` catalog and a
  `dataset_factory` that accepts one selected string ID. It never calls the
  factory with a hidden `None` sentinel; target length is
  `ceil(duration_s * sample_rate)`.
- Tuple-based `load_audio`/`load_wav` return channel 0 for multichannel files
  and emit a warning directing callers to `load_audio_data`. The format-generic
  `load_audio` and `load_audio_data` accept either a `Path` or a caller-owned
  open, seekable binary stream, read metadata and samples through one SoundFile
  handle, and leave that stream open. Other objects raise `TypeError`; closed
  or non-seekable streams raise `ValueError` before SoundFile is opened. The
  WAV-only wrapper requires a `.wav`/`.wave` `Path`.

### Logging

`LoggingConfig` is a frozen, slotted, keyword-only record whose level and
formatting fields are validated at construction. `setup_logging(config)`
configures only the `torchrir` namespace root, owns at most one marked handler,
and updates that handler's level and formatter on every call. Propagation to
Python's process-wide root logger is always disabled, preventing duplicate
records when the host application has its own logging configuration.
`get_logger(name)` qualifies child names below `torchrir`; only `torchrir`
itself and names beginning with `torchrir.` are already qualified.

### Metadata schema

- Metadata uses `{"name": "torchrir.scene", "version": 1}` and canonical
  top-level fields `schema`, `generator`, `room`, `sources`, `mics`,
  `trajectories`, `rir`, `doa`, `frame_schedule`, `signal`, `convolution`, and
  `dynamic`. `generator` identifies the TorchRIR distribution and PyTorch
  versions that produced the file.
- `mics.layout` records `kind` (`single` or `custom`), `center`, and
  `minimum_pair_distance`. `rir` compresses its physical sample axis into
  `origin_sample`, `sample_count`, and `sample_rate` rather than serializing a
  complete `time_axis` array.
- `frame_schedule`, when present, contains only exact `starts_samples` and
  `sample_rate`; seconds are not duplicated. `signal` records its sample count
  and rate, while `convolution` records the time reference and output count.
- `build_metadata`, `build_result_metadata`, `save_scene_metadata`, and
  `save_result_metadata` accept `schedule` and `time_reference`; there is no
  timestamp argument. When convolution metadata is requested, the reference
  must match the source/microphone motion and `signal_len` is required.
- Result-based metadata additionally includes the resolved `simulation`
  object. `source_info` is an optional application payload; `extra` is an
  optional mapping. Their recursive containers must be acyclic, mapping keys
  must be strings, numeric values must be real, and floating values must be
  finite. NumPy values are normalized to JSON values. Passing an explicit empty
  `extra={}` preserves that field; omitting `extra` leaves it absent.
- Scene-output helpers require cross-platform-safe single filenames, validate
  optional logger objects before creating directories, and validate
  attribution/modification text as non-empty single lines. Attribution notices
  render the dataset's explicit `attribution_required` flag instead of always
  claiming redistribution attribution is mandatory.
- Tensor values in these payloads must be materializable dense-strided,
  non-nested, non-quantized, non-complex tensors on CPU, CUDA, or MPS. Floating
  tensors additionally require a supported dtype and finite values; sparse,
  nested, quantized, complex, and unsupported-device tensors are rejected.
- JSON writes are atomic, reject NaN and infinity, and preserve an existing
  destination file's permission mode.

### Numerical verification

- Image-source coordinates, reflection coefficients, path delays, path gains,
  and fractional-delay accumulation are checked against independent analytic
  or direct implementations.
- Finite `float64` geometry near the dtype limits is evaluated with scaled
  affine image coordinates, overflow-resistant vector norms and direction
  normalization, and exponent-scaled `fs / c` delay conversion. An image,
  attenuation, diffuse sample, or metadata value that is not representable
  fails explicitly; a delay beyond the usable sample domain is excluded before
  any `int64` index cast instead of wrapping into the RIR.
- Diffuse handoff RMS uses max-absolute-value scaling, and metadata uses stable
  microphone centers, pair distances, horizontal distances, and angles. These
  paths therefore preserve finite extreme and subnormal values without
  overflowing an avoidable square, sum, or coordinate difference.
- Reflected source directivity mirrors the source orientation component normal
  to every wall with an odd reflection count.
- Static and dynamic convolution are checked against direct NumPy convolution,
  including exact frame-schedule boundaries.
- pyroomacoustics comparisons cover 2D/3D rooms, orders 0/1/3, asymmetric wall
  coefficients, and multiple source/microphone pairs. Its documented fixed
  40-sample fractional-delay offset is removed before comparison;
  cross-correlation is never used to estimate alignment.
- rir-generator 0.3.0 comparisons cover asymmetric walls, orders 0/1/3, and
  analytic microphone directivity after only the predetermined `4*pi`
  amplitude conversion; both outputs use their physical time axes directly.
- gpuRIR CUDA comparisons restrict generated-RIR parity to unambiguous direct
  paths and compare `simulateTrajectory` with identical synthetic RIR tensors,
  so known gpuRIR issues are not promoted to correctness oracles.
- Unexpected warnings fail the test suite. Reference, CUDA, MPS, numerical, and
  slow tests use explicit pytest markers.
- CI measures branch coverage and requires at least 75% overall coverage.

### API stability

- TorchRIR is under active development. Public contracts are intentionally
  small, and obsolete compatibility wrappers are removed rather than retained
  beside canonical APIs.
- Breaking API and numerical changes are recorded in
  [CHANGELOG.md](https://github.com/taishi-n/torchrir/blob/main/CHANGELOG.md).
- Cached RIRs must be regenerated when the changelog reports a timing,
  interpolation, geometry, directivity, diffuse-tail, or filtering change.

For detailed documentation:
[Documentation](https://torchrir.readthedocs.io/en/latest/)

## Development Verification

Install exactly the committed dependency set, then run the same core checks as
CI:

```bash
uv sync --locked --all-groups --all-extras
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run zensical build --strict
uv build
```

CI also inspects the built artifacts: the sdist must contain `CHANGELOG.md`,
and the wheel must declare `License-Expression: Apache-2.0`.

## Documentation Development

The documentation site is configured in
[`zensical.toml`](https://github.com/taishi-n/torchrir/blob/main/zensical.toml)
and built with [Zensical](https://zensical.org/).

```bash
uv sync --locked --group docs
uv run zensical serve
```

Before submitting documentation changes, run the same strict build used by CI:

```bash
uv run zensical build --strict
```

Pushes to `main` are published automatically by Read the Docs using the custom
Zensical build in
[`.readthedocs.yaml`](https://github.com/taishi-n/torchrir/blob/main/.readthedocs.yaml).
GitHub Actions validates the documentation but does not publish a GitHub Pages
site.

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
