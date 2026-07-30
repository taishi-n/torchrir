# Overview

## Capabilities
- ISM-based static and dynamic RIR simulation for 2D/3D shoebox rooms.
- Directivity patterns (`omni`, `halfomni`, `subcardioid`, `cardioid`,
  `hypercardioid`, `bidir`) with per-source/mic orientation handling.
- Acoustic parameters via `beta` or `t60` (dimension- and sound-speed-aware
  Sabine), optional diffuse tail via a strictly positive `tdiff`.
- Dynamic convolution via `torchrir.signal.DynamicConvolver`, with an explicit
  emission/observation time reference and a scene-carried or call-level
  `FrameSchedule`.
- Explicit scene models via `torchrir.models.StaticScene` and `torchrir.models.DynamicScene`.
- Scene-oriented simulation via `torchrir.sim.simulate(scene, config)` returning
  an `RIRResult` with resolved settings.
- CPU/CUDA/MPS execution with optional `torch.compile` acceleration for ISM accumulation
  (when enabled; MPS disables LUT and CPU disables compilation).
- Standard array geometries (linear, circular, polyhedron, binaural, Eigenmike)
  and trajectory sampling utilities.
- Dataset utilities (CMU ARCTIC and LibriSpeech) plus DataLoader collate helpers.
  See [Datasets](datasets.md) for accepted options, directory layouts, and
  invalid-input handling.
- Plotting utilities for static/dynamic scenes and GIF animation.
- Versioned metadata export for scene geometry, compact RIR/sample axes, exact
  frame schedules, DOA, signal dimensions, and convolution semantics.
- Channel-preserving audio I/O via `torchrir.io.AudioData`
  (`load_audio_data` / `save_audio_data`): sample rate and subtype round-trip,
  while stored format is descriptive and the output path selects the container.
- Soundfile-backed audio I/O with explicit entry points:
    - format-generic: `torchrir.io.load_audio` / `save_audio` / `info_audio`
    - strict WAV-only: `torchrir.io.load_wav` / `save_wav` / `info_wav`
- Dataset examples can emit per-source reference audio (RIR-convolved premix) and record it in metadata.
- Unified CLI example with JSON/YAML config and deterministic flag support.

## Module layout
- `torchrir.sim`: Scene-oriented RIR simulation; the private ISM kernel lives
  under `torchrir.sim.ism`.
- `torchrir.signal`: Signal processing utilities for static and dynamic RIR convolution.
- `torchrir.geometry`: Geometry helpers for arrays, trajectories, and sampling.
- `torchrir.viz`: Visualization helpers for scenes and trajectories.
- `torchrir.models`: Core data models for rooms, sources, microphones, scenes, and results.
- `torchrir.io`: I/O helpers for audio files and metadata serialization
  (format-generic `load_audio`/`save_audio`/`info_audio`, strict WAV-only
  `load_wav`/`save_wav`/`info_wav`, and metadata-preserving `AudioData` I/O).
- `torchrir.util`: General-purpose math, device, and tensor utilities for TorchRIR.
- `torchrir.logging`: Logging configuration and helpers.
- `torchrir.config`: Simulation configuration objects.
- `torchrir.datasets`: Dataset helpers and collate utilities.
  See [Datasets](datasets.md) for practical usage guidance.

## Device selection
- `device="cpu"`: CPU execution
- `device="mps"`: Apple Silicon GPU via Metal (MPS) if available, otherwise fallback to CPU
- `device="cuda"`: CUDA execution (validated in CI on CUDA runners; requires a CUDA-enabled PyTorch environment)
- `device="auto"`: CUDA, then MPS, then CPU; MPS is skipped for `float64`
- `device=None`: inherit the common scene-tensor device

All scene geometry Tensors must share one device and dtype. Explicit config
overrides perform one conversion immediately before the kernel. An explicit
MPS + `float64` request is rejected before execution because MPS does not
support that dtype. RIR simulation supports `torch.float32` and `torch.float64`;
explicit or inherited float16/bfloat16 geometry is rejected before a kernel
starts because ISM position, distance, delay, and gain calculations are not
numerically safe at those precisions. Generic tensor and convolution utilities
still support float16/bfloat16 using float32 work buffers. `RIRResult.config`
records effective settings, so `use_lut` is false on MPS and `use_compile` is
false on CPU even if requested.

Finite-real parameters used by simulation configuration, rooms/acoustics,
geometry constructors and samplers, frame-time conversion, audio normalization,
and dataset utilities follow one strict boundary contract. They accept real
scalar values but reject booleans, numeric strings, scalar Tensors, NaN, and
infinity rather than coercing them. An integer too large to convert to a finite
float raises `ValueError` with the public parameter name.

Count, index, sample-rate, and seed fields use a separate integer contract.
Non-boolean integer scalars, including NumPy integers, are normalized to Python
`int`; fractional values and booleans are rejected before allocation or index
arithmetic. Audio and dataset sample rates must be in `1..2**31-1`, sample
counts and frame starts fit their documented positive/non-negative `int64`
domains, and random seeds fit non-negative `int64`.

```python
from torchrir.util import DeviceSpec

device, dtype = DeviceSpec(device="auto").resolve()
```

## Logging contract

`LoggingConfig` is frozen, slotted, and keyword-only. It rejects invalid level
names and malformed field types at construction. Repeated `setup_logging`
calls update TorchRIR's managed handler instead of retaining a stale level or
formatter, while leaving unrelated handlers untouched. `setup_logging` always
targets the `torchrir` namespace root and disables propagation to Python's
process-wide root logger, so an application-level root handler cannot duplicate
TorchRIR records. `get_logger` treats only `torchrir` and the `torchrir.`
namespace as already qualified.

## Dynamic convolution time conventions

A dynamic RIR frame can describe the geometry when a sample is emitted or the
geometry when it is observed. These conventions are equivalent for a static
scene but differ as soon as an endpoint moves. Let \(x_s[q]\) be source \(s\)
at emission sample \(q\), \(k\) a propagation delay, and \(y_m[n]\) microphone
\(m\) at observation sample \(n=q+k\).

### Emission-time selection: moving source, fixed microphone

`DynamicConvolver(time_reference="emission")` selects the RIR from the time at
which an input sample is emitted:

$$
y_m[n] = \sum_s \sum_k h_{f_e(n-k),s,m}[k]x_s[n-k],
$$

where \(f_e(q)\) selects the frame containing input sample \(q\). Each input
segment is convolved with the RIR for the source geometry at its emission time,
and its convolution tail continues into later output samples. Use this
reference for moving sources only when the microphones are fixed.

### Observation-time selection: fixed source, moving microphone

`DynamicConvolver(time_reference="observation")` selects one piecewise RIR for
each output sample:

$$
y_m[n] = \sum_s \sum_k h_{f_o(n),s,m}[k]x_s[n-k],
$$

where \(f_o(n)\) selects the frame containing output sample \(n\). Use this
reference for fixed sources and moving microphones. The final RIR frame remains active
through the complete convolution tail, whose length is
`signal_length + rir_length - 1`.

### Frame schedules

`FrameSchedule` keeps one immutable integer start per RIR frame. Its `starts`
property returns a fresh CPU `int64` snapshot, so external mutation cannot
invalidate the schedule. Starts must be non-empty, begin at zero, and increase
strictly. Construct schedules with:

- `FrameSchedule.from_samples(starts)` for exact sample starts;
- `FrameSchedule.from_seconds(times, sample_rate=...)`, which performs the only
  seconds-to-samples conversion with `floor(time * sample_rate)` and retains
  that sample rate as provenance;
- `FrameSchedule.uniform(frame_count=..., stop_sample=...)` for exact integer
  partitioning;
- `FrameSchedule.fixed_hop(stop_sample=..., hop_size=...)` for a fixed hop.

For geometry interpolation, call
`schedule.normalized_progress(stop_sample=..., dtype=..., device=...)` and pass
the returned tensor to `linear_trajectory(..., progress=progress)`. Frame `i`
corresponds to `starts[i] / stop_sample`; the division is evaluated in float64
before the requested cast, so large sample indices do not overflow half
precision. Values are capped at the greatest representable value below one if
the requested dtype would round the final ratio to one. If two different frame
starts would collapse to the same value in the requested dtype, the conversion
raises instead of silently creating duplicate geometry frames. `stop_sample`
is the nominal endpoint, not a frame start. The final sampled geometry
therefore remains active until the next boundary or the selected timeline ends.

The schedule length must equal the RIR frame count. Emission-time starts must
precede the dry-signal endpoint. Observation-time starts may extend into the
convolution tail, allowing the microphone geometry to continue changing after
the source stops. An optional `DynamicScene.schedule` is the scene's
authoritative sample-domain frame axis.
If that schedule came from seconds, its conversion sample rate must match
`Room.fs`. Its `RIRResult` supplies the schedule automatically, and passing
another schedule is rejected as a competing frame axis. Raw RIR tensors and
results whose scene has no schedule require a call-level schedule.

```python
from torchrir import DynamicScene
from torchrir.config import SimulationConfig
from torchrir.signal import DynamicConvolver, FrameSchedule
from torchrir.sim import simulate

schedule = FrameSchedule.uniform(
    frame_count=src_traj.shape[0],
    stop_sample=dry.shape[-1],
)
scene = DynamicScene(
    room=room,
    sources=sources,
    mics=mics,
    src_traj=src_traj,
    mic_traj=mic_traj,
    schedule=schedule,
)
result = simulate(scene, SimulationConfig(max_order=6, nsample=4096))
wet = DynamicConvolver(time_reference="emission").convolve(dry, result)
```

To keep scheduling outside the reusable scene, omit `schedule` from
`DynamicScene` and pass the same object to `convolve(..., schedule=schedule)`.

Passing an `RIRResult` lets `DynamicConvolver` inspect its scene metadata and
reject a moving microphone with emission time, a moving source with
observation time, or simultaneous source and microphone motion. A raw
RIR tensor has no motion metadata, so the caller must choose the correct
convention. Simultaneous motion requires a two-time, retarded propagation
kernel \(g_{s,m}[n,q]\); selecting frames from only \(q\) or only \(n\) is not
sufficient and is not implemented.

## Simulation conventions

### Fractional-delay time origin

TorchRIR exposes only the physical propagation-time axis. It centers the
fractional-delay taps directly around each physical arrival sample and keeps
the part of that support inside the requested RIR interval. Returned sample
`n` therefore represents physical time `n / fs`; the interpolation kernel does
not introduce a public group delay.

### Geometry validation

Sources, microphones, and every trajectory frame must satisfy
`0 < position < room.size`. Wall positions are rejected because they can make
real and image sources coincide. Every source--microphone pair must also remain
at least `min_source_mic_distance` apart; its default is `1e-6` m. TorchRIR
raises an indexed validation error instead of silently clamping the singular
`1 / distance` gain.

Endpoint angles use radians and are normalized at construction to unit vectors
with shape `(entities, dimensions)`. In 2D, a scalar is one shared angle,
`(n, 1)` contains per-entity angles, `(2,)` is one shared vector, and `(n, 2)`
contains per-entity vectors. In 3D, use `(azimuth, elevation)` or `(n, 2)`
angle pairs and `(3,)` or `(n, 3)` vectors. A one-row representation is
broadcast; a flat 2D length-two value is never interpreted as two angles.

### Finite extreme values

Finite `float64` input is not assumed to make every naive intermediate finite.
The ISM kernel scales image-source affine geometry before multiplication, uses
max-absolute-value-scaled Euclidean norms for displacement and directivity, and
evaluates the `fs / c` delay factor in bounded exponent steps. Static, batched,
and dynamic contribution paths share this implementation. Image indices are
counted with checked integer arithmetic and streamed in bounded
`image_chunk_size` batches rather than materializing a complete grid. Arrivals
outside the usable sample domain are masked before conversion to `int64`, so
they cannot wrap to a negative or unrelated RIR index. A final image position or
attenuation that cannot be represented in the selected simulation dtype raises
an explicit `ValueError`.

This behavior prevents avoidable intermediate overflow; it does not promise
that every mathematically finite real-world scene is representable in a chosen
floating dtype. Use a physically scaled coordinate system when possible.

### Diffuse tail

Setting `tdiff` replaces the late ISM field after a strictly positive handoff
time. The handoff must precede `tmax`, and the preceding 10 ms must contain
usable early energy. If it does not, simulation raises an error identifying the
affected RIRs; increase `max_order`/`nb_img` or choose an earlier `tdiff`. A
5 ms power-complementary crossfade joins the early and diffuse fields.

Set `seed=...` in a complete `SimulationConfig` to reproduce the stochastic
carrier. The decay uses the room's configured speed of sound. A carrier is
derived independently for each source--microphone pair, making a seeded prefix
stable when the requested horizon changes and when unrelated pairs are added
to a batch. Dynamic frames share the pair carrier while retaining their own RMS
scale, which avoids unrelated noise realizations at adjacent frames. An
infinite Sabine T60 (perfect reflection) produces a non-decaying carrier rather
than an arbitrary fallback decay. This is a coherent frame-wise tail model,
not a complete spatial diffuse-field model.

Handoff RMS is evaluated after max-absolute-value scaling rather than by
squaring raw extreme samples. Sample-index arithmetic checks the available RIR
range before multiplication, and the completed diffuse field is rescanned for
representability. Consequently, finite extreme or subnormal early levels are
preserved when possible, while an unrepresentable generated tail fails instead
of silently becoming NaN or infinity.

### RIR high-pass filter

High-pass filtering is disabled by default. This avoids an implicit SciPy CPU
round trip for CUDA/MPS simulations. Install `torchrir[hpf]` and enable it
explicitly when the application needs DC or very-low-frequency suppression:

```python
from torchrir.config import RIRHighPassConfig, SimulationConfig

config = SimulationConfig(
    max_order=6,
    tmax=0.3,
    high_pass=RIRHighPassConfig(
        cutoff_hz=10.0,
        order=2,
        filter_family="butter",
        phase="zero_phase",
    ),
)
```

The default `phase="zero_phase"` performs pyroomacoustics-compatible
forward-backward filtering. For every source/microphone pair without a diffuse
tail, TorchRIR filters only through the natural ISM horizon
`min(nsample, ceil(max_arrival_sample) + (frac_delay_length - 1) // 2 + 2)` and
zero-fills the remaining requested samples. A diffuse tail uses the complete
requested horizon. Zero-phase filtering can pre-ring and needs enough samples
for padding. Explicit `phase="causal"` uses a zero initial state, filters the
complete requested RIR, and is prefix invariant. `filter_family` accepts `bessel`,
`butter`, `cheby1`, `cheby2`, or `ellip`. `cheby1` and `ellip` require positive
`passband_ripple_db`; `cheby2` and `ellip` require positive
`stopband_attenuation_db`; for `ellip`, attenuation must exceed ripple.
Enabling either phase requires SciPy, performs a CPU round trip, and detaches
the result from autograd.

## Metadata schema version 1

`build_metadata` and `build_result_metadata` produce schema
`torchrir.scene`, version 1. The canonical top-level fields are:

| Field | Contents |
| --- | --- |
| `schema` | Schema name and integer version. |
| `generator` | TorchRIR distribution version and PyTorch version. |
| `room` | Size, speed of sound, wall coefficients or T60, and sample rate. |
| `sources` | Initial positions, canonical orientations, and directivity. |
| `mics` | Initial positions, orientation, directivity, and `layout`. |
| `trajectories` | Dynamic source/microphone positions, or null entries for a static scene. |
| `rir` | Shape plus the compact sample axis: `origin_sample`, `sample_count`, and `sample_rate`. |
| `doa` | World-frame azimuth/elevation arrays in radians. |
| `frame_schedule` | Exact `starts_samples` and sample rate, or null. |
| `signal` | Input sample count and sample rate when supplied, or null. |
| `convolution` | Emission/observation reference and output sample count when supplied, or null. |
| `dynamic` | Boolean scene classification, including one-frame dynamic scenes. |

`mics.layout.kind` is `single` for one microphone and `custom` otherwise; the
layout also records its center and minimum pair distance (`null` for a single
microphone). The schema does not serialize a full RIR `time_axis`, duplicate an
array representation, or store derived `starts_seconds`. A result-based export
adds the resolved `simulation` record. `source_info` remains an optional
application payload; `extra` must be a mapping when supplied. Both payloads are
normalized recursively: containers must be acyclic, mapping keys must be
strings, numeric values must be real, and floating values must be finite. NumPy
scalars and arrays are converted to ordinary JSON values. An explicitly empty
`extra={}` is retained in the document; omitting the argument leaves the field
absent.

Tensor payloads must be materializable dense-strided tensors on CPU, CUDA, or
MPS. Nested, sparse, quantized, complex, and unsupported-device tensors are
rejected; floating tensors additionally require a supported dtype and finite
values. These checks happen while building metadata rather than being deferred
to `json.dump`. `build_metadata`, `build_result_metadata`,
`save_scene_metadata`, and `save_result_metadata` use explicit `schedule` and
`time_reference` arguments; a requested reference must agree with scene motion
and requires `signal_len`.
`save_metadata_json` rejects NaN and infinity, publishes through an atomic
replacement, and preserves an existing file's permission mode.

Metadata geometry is calculated on CPU in float64 with overflow-resistant
operations: the microphone center uses coordinate scaling, minimum pair
distance uses a scaled vector norm, horizontal DOA distance uses `hypot`, and
angles use `atan2`. This avoids overflow caused solely by a raw sum, square, or
pairwise-distance intermediate for finite extreme coordinates.

## Limitations and Failure Modes
- `StaticScene` and `DynamicScene` are the public scene types; callers must
  choose the type that matches the simulated geometry.
- `DynamicScene` normalizes tensor-like trajectories to tensors during initialization.
- Sequence trajectories inherit the endpoint device/dtype. Tensor trajectories
  preserve their input layout, and a mismatch with the room/endpoints is
  rejected rather than silently cast.
- Sources, microphones, and every trajectory frame must be strictly inside the
  room. Positions on walls are rejected to avoid duplicate images and singular
  paths.
- Dynamic schedule and time-reference semantics are described in
  [Dynamic convolution time conventions](#dynamic-convolution-time-conventions).
- `SimulationConfig` is the single source of algorithm settings; sampling rate,
  endpoint directivity, positions, and orientations belong to the scene.
- Audio I/O uses soundfile directly. `load_audio`/`save_audio`/`info_audio`
  accept soundfile-supported formats; pathname arguments must be
  `pathlib.Path`, and the `*_wav` entry points accept only `.wav` and `.wave`
  paths. `load_audio`/`load_audio_data` additionally accept caller-owned open,
  seekable binary streams and leave them open. Other objects raise `TypeError`;
  closed or non-seekable streams raise `ValueError`. Save paths resolve
  `.wave`, `.aif`, `.aifc`, `.oga`, and `.snd` to their canonical SoundFile
  container names.
- Audio saves preserve gain by default. `save_audio_data` reuses the stored
  subtype only when the destination container matches the loaded format;
  otherwise WAV output without an explicit subtype uses floating-point storage.
  Selecting an integer or companded subtype with samples outside `[-1, 1]`
  raises before SoundFile can clip them. Pass `normalize=True` only when
  independent peak normalization is intended.
- `torchrir.sim.simulate` requires exactly one image-enumeration strategy
  (`max_order` or `nb_img`) and exactly one of `nsample` or `tmax`.
- Non-`omni` directivity requires orientation; mismatched shapes raise `ValueError`.
- `beta` must have 4 (2D) or 6 (3D) elements; invalid sizes raise `ValueError`.
- `DynamicScene` requires `src_traj` and `mic_traj` to have matching time steps.
- Audio tensors are `(samples,)` for mono and channel-first
  `(channels, samples)` for multichannel data.
- `torchrir.signal.DynamicConvolver` with 3D dynamic RIR input (`(T, n_mic, rir_len)`) is treated as single-source only; multi-source dynamic convolution must use 4D RIR input (`(T, n_src, n_mic, rir_len)`).
- Static and dynamic convolution outputs always keep the microphone axis,
  including one-microphone results.
- Static and dynamic source/microphone convolution is batched. Half and
  bfloat16 inputs use float32 FFT, source-sum, and overlap-add work buffers and
  are cast once on return. Autograd is tested for static and both dynamic
  conventions on CPU, with dynamic-emission output/gradient parity also tested
  on accelerator paths.
- Simultaneous source and microphone motion requires a retarded-time model and
  is not implemented.
- Dynamic simulation batches trajectory frames, but memory and compute still grow with
  the frame, source, microphone, and image counts.
- MPS disables the sinc LUT path (falls back to direct sinc), which can be slower and slightly different numerically.
- HPF requires SciPy and currently applies filtering via CPU-domain processing,
  which can add host/device transfer overhead on CUDA/MPS runs. Filtering is
  opt-in; causal filtering is prefix invariant, while explicit zero-phase
  filtering can pre-ring and requires enough samples for forward-backward
  padding.
- Deterministic mode is best-effort; some backends may still be non-deterministic.
- YAML example-CLI configs require `torchrir[cli]`; JSON configs need no CLI
  extra.
- Downloading CMU ARCTIC or LibriSpeech requires network access only when
  `download=True` and the requested local tree is not ready. Constructors may
  still restore or clean an interrupted local publication transaction with
  `download=False`; they do not start a network request in that mode. Transfers,
  cache validation, writer-lock scopes, and crash recovery are specified in
  [Datasets](datasets.md#shared-archive-concurrency-and-recovery-contract).
- Secure local dataset reads, archive transfer/extraction, locking, and staged
  publication currently require Linux or macOS with POSIX descriptor walking
  and atomic no-replace/exchange rename support. Unsupported platforms or
  filesystems raise `NotImplementedError` before those operations start.
- Dataset option validation and error behavior are documented in
  [Datasets](datasets.md).
- GIF output requires Pillow (via Matplotlib's animation writer).
- MP4 output additionally requires a system `ffmpeg`; muxing audio also requires
  the `audio` extra.
- Dataclass models are frozen but hold mutable tensors (shallow immutability).
  `RIRResult` snapshots every scene tensor, compares both identity and value,
  rescans RIR finiteness, and revalidates scene state, RIR shape, and resolved
  config at consumer boundaries. This also detects mutation through shared
  NumPy storage and inference tensors without version counters.
