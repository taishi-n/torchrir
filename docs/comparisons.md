# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries within this comparison scope:

- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`
- `dynamic-sound`
- `das-generator`
- `TASCAR`

## Feature Comparison

| Feature | `torchrir` | `gpuRIR` | `pyroomacoustics` | `rir-generator` | `dynamic-sound` | `das-generator` | `TASCAR` |
|---|---|---|---|---|---|---|---|
| 🎯 Dynamic Sources | ✅ Emission-time | 🟡 Single moving source | 🟡 Manual loop | ❌ | ✅ Retarded-time | ✅ Emission-time | ✅ Delay lines |
| 🎤 Dynamic Microphones | ✅ Observation-time | ❌ | 🟡 Manual loop | ❌ | ✅ Observation-time | ✅ Observation-time* | ✅ Delay lines |
| Source + Microphone Motion | ❌ Signal synthesis | ❌ | 🟡 Custom propagation | ❌ | ✅ Direct sound | ✅ Two-time kernel* | ✅ Block geometry |
| Shoebox ISM | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ Polygon faces |
| 🖥️ CPU | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ C++ |
| 🧮 CUDA | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 🍎 MPS | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 📊 Scene Plot | ✅ | ❌ | ✅ | ❌ | ✅ Paths/arrays | ❌ | ✅ GUI/SVG |
| 🎞️ Dynamic Scene GIF | ✅ | ❌ | 🟡 Manual animation script | ❌ | 🟡 Manual animation script | ❌ | ❌ GIF API |
| 🗂️ Dataset Build | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ Scene rendering only |
| 🎛️ RIR Convolution | ✅ Static/dynamic | 🟡 Dynamic helper | ✅ | ❌ | ❌ Signal time warping | ✅ Internal dynamic RIR | ✅ Static IR plugins |
| 🧱 Non-shoebox Geometry | 🚧 Candidate | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ Polygon reflectors |
| 🌐 Ray Tracing | 🚧 Candidate | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |

Legend:

- `✅` native support
- `🟡` manual setup
- `🚧` candidate (not yet implemented)
- `❌` unavailable

Notes:

- `RIR Convolution` means applying static or time-varying RIRs to source signals.
- Motion rows refer to signal synthesis, not merely generating RIR snapshots.
  TorchRIR rejects simultaneous source and receiver motion during convolution.
- Feature markers describe API availability, not verified numerical correctness.
  `dynamic-sound` has no automatic room-reflection generator and its multiple-source
  output failed a superposition check. `das-generator` has moving-receiver reuse
  and startup errors (`*`); see the [implementation audit](#dynamic-sound-and-das-generator-implementation-audit).
- TorchRIR's candidate markers refer to future work, not available APIs.
  The [acoustic model roadmap](acoustic-roadmap.md) defines the initial
  non-shoebox scope and ray-tracing evaluation gate; FDTD remains deferred.
- `das-generator` applies internally generated RIRs; it does not expose a public
  convolver for caller-supplied RIRs. `dynamic-sound` directly samples a source at
  its retarded emission time rather than generating and convolving room RIRs.
- TASCAR motion entries describe its block-based path renderer. Its fixed-IR
  convolution plugins are separate from that renderer. GUI animation is not a
  dedicated GIF-export API, and offline scene rendering is not a corpus-building
  API. The [TASCAR comparison](#tascar-implementation-comparison) records the
  inspected revision and limitations.
- Broader signal processing such as beamforming, DOA, BSS, adaptive filtering, STFT, and denoising remains out of scope for `torchrir`.

## TASCAR Implementation Comparison

This source audit was performed on 2026-10-06. TASCAR was cloned into
`tmp/tascar/` and inspected at [release 0.239.2][tascar-release], commit
[`4471e181a7c3a25e36a0cea9254c11edf9a9652c`][tascar-revision].
The TorchRIR implementation was at `d5e80a0`. TASCAR was not built or executed;
this comparison does not establish waveform parity or a runtime ranking.
Candidate counts below are calculations from the inspected enumeration rules.

TASCAR targets interactive acoustic scenes for hearing research and spatial
audio reproduction. It combines a C++ propagation engine with JACK, XML scene
definitions, OSC control, and receiver plugins. TorchRIR exposes RIR tensors
and convolution through Python/PyTorch. See the [TASCAR project description][tascar-readme]
and [real-time scene interface][tascar-jack].

### Capabilities and output contracts

| Aspect | TorchRIR | TASCAR at the inspected revision |
|---|---|---|
| Main output | Static `(source, microphone, sample)` or dynamic `(frame, source, microphone, sample)` RIR tensors; convolved signals | Live multichannel audio, offline WAV, or one input channel's scene IR through `tascar_renderir` |
| Scene/API | Python scene models and `SimulationConfig` | XML `.tsc` scenes, C++ engine, CLI tools, and MATLAB helpers |
| Moving endpoints | Emission-time convolution for moving sources or observation-time convolution for moving microphones; joint motion rejected | Both endpoints can move in the block-based renderer; no retarded emission-time solve in the inspected propagation loop |
| Orientation | Fixed per-entity analytic directivity orientation | Scheduled or interactive orientation; Euler-angle interpolation in the inspected trajectory class |
| Room geometry | 2D/3D shoebox ISM | Polygon reflectors, including a shoebox helper and imported face groups |
| Reflection response | One frequency-independent amplitude coefficient per wall | Cascaded first-order reflection filters; material absorption bands are fitted to filter parameters |
| Air absorption | Not implemented | Optional distance-dependent first-order low-pass filter |
| Late reverberation | Seeded statistical diffuse RIR tail | FOA diffuse fields and optional FDN or recorded-IR convolution reverberation |
| Binaural/microphone model | Point-microphone geometry and analytic directivity; no head/HRTF model | Parametric HRTF and hierarchical microphone/head models; SOFA HRIR convolution via virtual-speaker layouts |
| Ambisonics | No native spherical-harmonic output | HOA encoding/decoding and loudspeaker rendering plugins |
| Execution | PyTorch CPU/CUDA/MPS | C++ CPU audio processing; no CUDA/MPS propagation backend in the inspected core |
| Differentiation | PyTorch autograd on supported simulation/convolution paths | No PyTorch/autograd interface in the inspected engine |
| Interactive audio/control | No JACK/OSC scene renderer | JACK audio and OSC/module-based scene interaction |
| Dataset workflow | Source loaders and builders with mixtures, stems, and metadata | Scene rendering and measurement scripts; no equivalent corpus-builder contract |

Sources: [block renderer][tascar-render], [propagation and air filter][tascar-acoustics],
[orientation tracks][tascar-tracks], [reflection/material and reverb definitions][tascar-manual],
[parametric HRTF][tascar-hrtf], [microphone model][tascar-micarray],
[SOFA convolution][tascar-sofa], and [HOA encoder][tascar-hoa].

The `hrtf` receiver is a parametric filter model. The SOFA path loads measured
HRIRs for virtual loudspeaker directions and convolves those speaker feeds;
these are different binaural rendering paths. A two-point microphone layout in
TorchRIR does not reproduce either head model.

### Dynamic propagation and IR measurement

The inspected TASCAR signal path is:

```text
for each audio block at session time t:
    update source, receiver, and reflector geometry at t
    for each receiver and primary/image-source path:
        evaluate current image geometry, direction, delay, and gain
        apply source directivity and cascaded reflection filters
        ramp distance and gain from previous to current values over the block
        read the source history through the variable delay line
        apply the air filter and receiver rendering
    process diffuse fields and reverberation
```

Changing the delay-line read position produces time warping and Doppler effects.
The geometry update, however, uses both endpoints at the same block time. It
does not solve `te = t - distance(source(te), receiver(t)) / c`.
Native simultaneous motion therefore describes operational support, not an
exact two-time propagation kernel. Its approximation and block interpolation
must be considered when comparing against a retarded-time reference.
See [geometry update][tascar-render], [path processing][tascar-acoustics],
and [delay-line reads][tascar-delay].

The propagation delay line defaults to `sincorder=0`: it reads an integer
delay obtained by truncating the positive sample delay. A nonzero order uses
a finite, lookup-table sinc sum. This differs from TorchRIR's default 81-tap
Hann-windowed fractional-delay deposition. Shared use of ISM or `1/r` gain
does not imply identical RIR samples.
See [source defaults][tascar-source] and [delay interpolation][tascar-delay].

`tascar_renderir` freezes the scene at `starttime`, runs a zero-input
initialization block, then injects a unit impulse into the selected input
channel and writes the receiver outputs. It uses the requested IR length as
the block size and keeps session time unchanged. It measures the configured
signal graph, including its filters and reverberation; it is not a dynamic RIR
tensor export. See [IR rendering][tascar-ir].

For offline moving scenes, `tascar_renderfile` defaults to dynamic time updates,
but its default fragment covers the entire input/duration. An explicit
`--fragsize` is needed for repeated geometry updates; `--static` freezes time.
Also, the inspected MATLAB `tascar_renderscene` helper appends `-d` when its
`dynamic` option is true, while the inspected CLI does not accept `-d`.
This is a source-level interface mismatch; the helper was not executed.
See [CLI options][tascar-file] and [MATLAB helper][tascar-matlab].

### Computational structure and comparison boundary

TASCAR builds a path graph for every receiver. With `W` reflectors and maximum
order `q >= 1`, its constructor creates `W * (W-1)^(k-1)` candidate paths
of order `k` per source/receiver, excluding consecutive reflection at the same
surface. Visibility and activity are evaluated during processing. This is
different from TorchRIR's unique shoebox L1 image-index enumeration.

For six reflector faces and one source/receiver pair:

| Maximum order | TASCAR candidate path nodes, including direct sound | TorchRIR 3D L1 image indices |
|---|---|---|
| 3 | 187 | 63 |
| 5 | 4,687 | 231 |

These counts are not equivalent sets of valid physical reflections and do not
give a speed ratio. TASCAR's low-order early-reflection model plus separate
late reverberation must not be equated to a high-order shoebox RIR calculation.
See [path-graph construction][tascar-images] and TorchRIR's
`src/torchrir/sim/ism/images.py::_image_source_count`.

TASCAR retains delay-line, filter, and audio-block state per path. Its streaming
renderer avoids storing a full frame/source/microphone/RIR cube, but memory
still depends on path count, maximum delay, and fragment size. The IR command
sets the maximum delay from the requested IR length. Consequently, CPU
real-time design alone does not establish faster or smaller RIR generation.
See [per-path state][tascar-path-state] and [IR allocation][tascar-ir].

A numerical comparison would first need stationary omni endpoints, matched
faces/orders, disabled air and reflection filtering, explicit gains/calibration,
matched interpolation, and matched output support. Moving tests additionally
need explicit block sizes and a stated propagation-time approximation.
No TASCAR parity test or performance result is claimed here, and this comparison
does not add a backend or an implementation milestone to TorchRIR.

[tascar-release]: https://github.com/gisogrimm/tascar/releases/tag/release_0.239.2
[tascar-revision]: https://github.com/gisogrimm/tascar/tree/4471e181a7c3a25e36a0cea9254c11edf9a9652c
[tascar-readme]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/README.md
[tascar-jack]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/include/jackrender.h#L33-L55
[tascar-render]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/render.cc#L235-L330
[tascar-acoustics]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/acousticmodel.cc#L127-L278
[tascar-tracks]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/dynamicobjects.cc#L156-L193
[tascar-manual]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/manual/manual.tex#L1395-L1557
[tascar-hrtf]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/manual/recgenhrtf.tex
[tascar-micarray]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/manual/recgenmicarray.tex
[tascar-sofa]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/speakerarray.cc#L607-L645
[tascar-hoa]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/plugins/src/receivermod_hoa3d_enc.cc#L52-L103
[tascar-delay]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/include/delayline.h#L90-L144
[tascar-source]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/acousticmodel.cc#L1000-L1034
[tascar-ir]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/irrender.cc#L282-L355
[tascar-file]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/apps/src/tascar_renderfile.cc#L43-L145
[tascar-matlab]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/scripts/tascar_renderscene.m#L33-L62
[tascar-images]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/acousticmodel.cc#L315-L354
[tascar-path-state]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/acousticmodel.cc#L90-L106
[tascar-gain]: https://github.com/gisogrimm/tascar/blob/4471e181a7c3a25e36a0cea9254c11edf9a9652c/libtascar/src/acousticmodel.cc#L810-L860

## Dynamic-Sound and DAS-Generator Implementation Audit

The following source audit and CPU checks were performed on 2026-10-05.
The observations apply to these revisions, not to all past or future releases:

| Repository | Inspected commit |
|---|---|
| [dynamic-sound](https://github.com/vlsi-nanocomputing/dynamic-sound) | [`f3aceae2bd483d7f347b3e2e842b1a72bb57ebce`](https://github.com/vlsi-nanocomputing/dynamic-sound/tree/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce) |
| [das-generator](https://github.com/ehabets/das-generator) | [`6f2cd6d22872edd6264ca0c60e9f58477ddaeb48`](https://github.com/ehabets/das-generator/tree/6f2cd6d22872edd6264ca0c60e9f58477ddaeb48) |
| TorchRIR | `c1101277aacc86422819c7ac895f6f5d6fab03c0` |

### Processing and time coordinates

```text
TorchRIR:
    H[frame] = ISM(source[frame], receiver[frame])
    emission:    y[n] = sum_k H[frame(n-k), k] * x[n-k]
    observation: y[n] = sum_k H[frame(n),   k] * x[n-k]

dynamic-sound:
    for each observation time t and microphone:
        evaluate receiver position r(t)
        solve ||r(t) - s(te)|| = c * (t - te) on source path segments
        sample source at te, apply distance/directivity gain and air FIR
        accumulate sources and write microphone WAV

das-generator (intended kernel, subject to the cache errors below):
    for each observation sample n and microphone:
        update RIR rows for past emission samples e, using s[e] and r[n]
        y[n] = sum_k ISM(s[n-k], r[n])[k] * x[n-k]
```

`dynamic-sound` solves a quadratic for each piecewise-linear source segment;
receiver position is evaluated at observation time and source position and
orientation at emission time. This time warping represents direct-path Doppler
without a separate frequency-shift stage. It does not enumerate room images:
the reflection integration test manually adds a mirrored source, and
`material_reflection` is a stub returning `1.0`. Neither is an automatic ISM
implementation. See the [emission-time solver and renderer][ds-simulation],
[reflection stub][ds-attenuations], and [manual reflection test][ds-reflection-test].

`das-generator` evaluates each image using a past source position and the current
receiver position. Its response matrix retains emission time and delay as two
independent axes. With a fixed receiver this reduces to emission-time
convolution; with a fixed source it should reduce to observation-time
convolution. With both moving, it needs both positions at different times.
TorchRIR's simultaneous-position RIR snapshots and either single frame-selection
rule cannot represent that general kernel. The current TorchRIR scene-aware
convolver explicitly rejects that case. See the [DAS update and convolution loops][das-core]
and [TorchRIR motion validation](https://github.com/taishi-n/torchrir/blob/c1101277aacc86422819c7ac895f6f5d6fab03c0/src/torchrir/models/scene.py#L264-L287).

### Geometry, interpolation, and output

| Aspect | `torchrir` | `dynamic-sound` | `das-generator` |
|---|---|---|---|
| Propagation | Shoebox ISM snapshots | Direct path with retarded-time solve | Shoebox ISM with two-time geometry |
| Motion input | Frame trajectories and explicit sample schedule | Time/position/quaternion knots; linear position and SLERP | Source `(3, N)`, receivers `(M, 3, N)`; position at every sample |
| Multiple sources | Source axis and summation | Source list; shared FIR-history defect below | One source signal per call; sum separate calls manually |
| Orientation | Source/receiver directivity; per-entity orientation fixed over the trajectory | Source/array quaternion paths; local microphone quaternion is not used by renderer | Fixed receiver azimuth/elevation per call; omnidirectional source |
| Geometric gain | `1/r` | `1/r`, plus air absorption | `1/(4πr)` |
| Interpolation | Configurable odd-length Hann sinc; default 81 taps | Source-signal integer indexing/linear/sinc; linear by default | Hann sinc with `Tw = 2*round(0.004*fs)` and `Tw+1` taps |
| Filtering | Optional configured RIR HPF | Per-distance 33-tap air-absorption FIR | 100 Hz Allen–Berkley RIR HPF, enabled by default |
| Public output | RIR tensors and full convolution output | Clipped integer WAV over microphone-path duration | Floating waveform `(N, M)`, not an RIR tensor |
| Tail | Convolver returns `N + L - 1` samples | No automatic FIR/propagation tail extension | Caller must zero-pad input and extend trajectories |
| Execution | Batched PyTorch; CPU/CUDA/MPS | Python sample/channel/source loops; NumPy/SciPy CPU | C++ loops via CFFI; CPU, double precision |

TorchRIR supports time-varying source and microphone positions, while each
source and microphone keeps one fixed orientation throughout the trajectory.
Time-varying orientation is not supported.

The DAS image on each coordinate axis is `(1-2*q)*s + 2*m*room_size`.
Its two wall gains are `beta_low**abs(m-q) * beta_high**abs(m)`;
an explicit order keeps images whose sum of `abs(2*m-q)` is within that order.
This can be matched to TorchRIR's L1-order enumeration. `order=-1` instead
uses the room/RIR-length search bounds without an explicit order cutoff.
The DAS Hann window is centered at the integer floor of the arrival; its sinc
is shifted by the fractional part. At 16 kHz it has 129 taps. Thus matching
TorchRIR requires setting `frac_delay_length=129`, disabling its LUT for the
analytic comparison, disabling the DAS HPF, and applying the known `4*pi`
gain conversion. No global delay crop is appropriate. See [DAS core][das-core]
and [TorchRIR accumulation](https://github.com/taishi-n/torchrir/blob/c1101277aacc86422819c7ac895f6f5d6fab03c0/src/torchrir/sim/ism/accumulate.py#L32-L102).

The DAS response buffer alone contains `L*L` doubles: `L=16000` requires
2.048 GB (about 1.91 GiB). Once the history is full, a moving receiver visits
`L` past emission rows per output sample. Source-only motion needs one new row;
both stationary positions allow row reuse. These are structural costs, not
measured runtime comparisons. `dynamic-sound` searches source segments and
designs a 33-tap FIR for each valid sample/source/channel contribution.
Its symmetric FIR adds 16 samples of group delay in the constant-distance
single-source case; this is separate from physical propagation delay.
See [DAS allocation/update loops][das-core], [DS rendering][ds-simulation],
[source interpolation][ds-audio-signal], and [DAS API][das-api].

### CPU validation and reproduced discrepancies

The DAS checks compiled the unchanged upstream C++ core with Clang and called
the unchanged Python `generate` through CFFI ABI loading. The DynamicSound
checks extracted the unchanged `Simulation` class from source, loaded upstream
path/air/attenuation functions, supplied small signal/microphone probe objects,
and replaced only the progress display. They did not test package installation,
CUDA execution, or benchmark performance.

| Check | Observed maximum absolute error | Interpretation |
|---|---|---|
| DynamicSound emission time vs constant-velocity analytic solution | `2.16e-15` seconds | Agrees in the tested subsonic, non-crossing case |
| DynamicSound combined sources vs sum of separate renders | `6.39e-2` | Superposition fails |
| DynamicSound reversed source registration order | `3.97e-5` | Order dependence reproduced |
| DAS static direct path vs analytic kernel | `3.47e-18` | Agrees |
| DAS moving source, fixed receiver vs analytic kernel | `6.25e-17` | Agrees |
| DAS fixed source, moving receiver vs analytic kernel | `8.38e-3` | RIR reuse fails |
| DAS both moving, with first input sample zero | `7.81e-18` | Agrees when the tested path avoids reuse/startup defects |
| DAS both moving, impulse at first input sample | `1.67e-3` | Startup error reproduced |
| DAS static direct plus six first-order images vs analytic kernel | `6.07e-18` | Geometry, wall gains, and fractional-delay deposition agree |
| DAS static first-order output vs TorchRIR, after `4*pi` conversion | `5.42e-18` | Agrees with matched 9-tap interpolation at 1 kHz |
| DAS moving-source output vs TorchRIR emission convolution | `6.25e-17` | Agrees with one RIR frame per input sample |

DynamicSound's time probe uses `c=343.2`, source velocity `20 m/s`, receiver
velocity `5 m/s`, and initial separation `10 m`. For observation times
`0.1..0.5 s`, the analytic emission time is
`te = ((c-5)*t - 10)/(c-20)`. The implied frequency ratio is
`(c-5)/(c-20) = 1.046410891`; this check validates the solver, not rendered
audio-frequency accuracy. The superposition probe uses a stationary source
and receiver 1 m apart, 512 samples at 8 kHz, and two tones with frequencies
660/1100 Hz and amplitudes 0.03/0.02. All output magnitudes stay below 0.05;
the discrepancy is neither clipping nor 32-bit PCM quantization (`4.66e-10`).
The cause is a single FIR history per microphone channel: each source appends
to that same history during one output sample. History advances per source
instead of maintaining each source's time sequence. See
[buffer creation and source accumulation][ds-simulation].

The DAS motion probes use `fs=1000`, `c=343`, `N=96`, `L=32`, a
`10×10×10 m` room, omnidirectional endpoints, zero wall coefficients,
`order=0`, and `hp_filter=False`. Initial source/receiver positions are
`(1,1,1)` and `(2.11475,1,1)`. Moving coordinates increase along x by
`0.01 m/sample` for the source and `0.02 m/sample` for the receiver.
The input is `0.1*sin(0.37*n)`, or a `0.1` impulse at sample zero for
the startup probe. The independent reference evaluates each path with
`s[n-k]`, `r[n]`, `1/(4*pi*r)`, and the same finite Hann sinc.
The first-order check instead uses a `4×5×3 m` room, receiver y=2,
`L=64`, and wall coefficients `[[.4,.5],[.6,.7],[.8,.9]]`.

Two distinct errors occur in the [DAS row-update logic][das-core]:

1. When the receiver moves and a past source position repeats, rows are visited
   newest to oldest, but `copy_previous_rir` copies the older row that has not
   yet been updated for the current receiver. The moving-receiver probe fails
   even after filling the `L`-sample history. A separate receiver step of
   `-0.343 m` at sample 20 produces error `5.49e-3`, then agrees again after
   the stale history has passed.
2. For `0 < n < L`, `no_rows_to_update = n` omits emission sample zero;
   there are actually `n+1` valid emission rows. That first RIR retains the
   initial receiver position while the receiver moves. With both positions
   changing on every sample, a nonzero first input sample still exposes this
   separate startup defect.

These counterexamples prevent using either library's affected output as an
unconditional correctness oracle. They do not invalidate the supported APIs
or establish errors for every scene. Direct-path time warping and a sampled
two-time convolution are also different signal models; full cross-model
waveform equivalence has not been established by these checks.

[ds-simulation]: https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/src/dynamic_sound/_simulation.py#L47-L154
[ds-attenuations]: https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/src/dynamic_sound/acoustics/attenuations.py#L12-L18
[ds-reflection-test]: https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/tests/integration/test_simulations.py#L10-L33
[ds-audio-signal]: https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/src/dynamic_sound/sources/_audio_signal.py#L6-L52
[das-core]: https://github.com/ehabets/das-generator/blob/6f2cd6d22872edd6264ca0c60e9f58477ddaeb48/das_generator/_cffi/core.cpp#L80-L358
[das-api]: https://github.com/ehabets/das-generator/blob/6f2cd6d22872edd6264ca0c60e9f58477ddaeb48/das_generator/__init__.py#L22-L312

## Visualization, Dynamic GIF, and Dataset Build (Source-Level)

Scoring criteria in this section:

- Mark as `✅` when the functionality is provided as a library API/submodule (not only in examples).
- Mark as `🟡` when possible only via manual composition without a dedicated library feature surface.
- Mark as `❌` when no corresponding library functionality exists.

### Visualization

- `torchrir` (`✅`):
    - Dedicated visualization submodule and public functions are provided.
    - Source files: [`src/torchrir/viz/__init__.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/__init__.py), [`src/torchrir/viz/scene.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/scene.py), [`src/torchrir/viz/io.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py)
- `gpuRIR` (`❌`):
    - Package exports simulation/control functions only; plotting appears in example scripts.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L11), [`examples/example.py#L8-L9`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/example.py#L8-L9), [`examples/example.py#L35-L36`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/example.py#L35-L36), [`examples/polar_plots.py#L3`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/polar_plots.py#L3), [`examples/polar_plots.py#L9-L10`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/polar_plots.py#L9-L10), [`examples/polar_plots.py#L66-L75`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/polar_plots.py#L66-L75)
- `pyroomacoustics` (`✅`):
    - Library-level plotting APIs exist (`Room.plot`, `Room.plot_rir`), with optional plotting helpers in other submodules.
    - Source lines: [`pyroomacoustics/room.py#L1535-L1547`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1535-L1547), [`pyroomacoustics/room.py#L1827-L1843`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1827-L1843), [`pyroomacoustics/__init__.py#L123-L134`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123-L134)
- `rir-generator` (`❌`):
    - The package API is focused on RIR generation (`generate`) and does not include plotting APIs.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

- `dynamic-sound` (`✅ Paths/arrays`):
    - Provides [path and orientation plots](https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/src/dynamic_sound/environment/_path.py#L94-L190)
      and [array geometry plots](https://github.com/vlsi-nanocomputing/dynamic-sound/blob/f3aceae2bd483d7f347b3e2e842b1a72bb57ebce/src/dynamic_sound/microphones/_hedraphone.py#L99-L134).
- `das-generator` (`❌`):
    - The [package API][das-api] provides signal generation without plotting.

### Dynamic Scene GIF

`dynamic-sound` has no dedicated scene-GIF API; its plotting functions require
manual animation composition (`🟡`). `das-generator` has neither plotting
nor GIF APIs in its [package interface][das-api] (`❌`).

- `torchrir` (`✅`):
    - Dedicated GIF APIs are provided for dynamic trajectories.
    - Source files: [`src/torchrir/viz/__init__.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/__init__.py), [`src/torchrir/viz/animation.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/animation.py), [`src/torchrir/viz/io.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py)
- `gpuRIR` (`❌`):
    - No GIF/animation API is exposed in the package interface.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L11)
- `pyroomacoustics` (`🟡`):
    - Plotting APIs are provided, but no dedicated dynamic-scene GIF API; animation must be manually composed by users.
    - Source lines: [`pyroomacoustics/room.py#L1535-L1547`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1535-L1547), [`pyroomacoustics/room.py#L1827-L1843`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1827-L1843), [`pyroomacoustics/__init__.py#L123-L134`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123-L134)
- `rir-generator` (`❌`):
    - No GIF/animation API is provided.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

### Dataset Build

Neither `dynamic-sound` nor `das-generator` provides a dataset/corpus-building
API in the inspected package. DynamicSound's [WAV renderer][ds-simulation]
and DAS's [returned signal array][das-api] are signal outputs, which do not
meet the dataset API criterion here.

- `torchrir` (`✅`):
    - Dataset utilities are provided as library modules (`torchrir.datasets`) including dataset wrappers and source-loading utilities used by dataset generation workflows.
    - Source files: [`src/torchrir/datasets/__init__.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/datasets/__init__.py), [`src/torchrir/datasets/utils.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/datasets/utils.py)
- `gpuRIR` (`❌`):
    - No dataset submodule or dataset-building API is exposed.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L11)
- `pyroomacoustics` (`✅`):
    - Dataset functionality is provided in-library via `pyroomacoustics.datasets` with corpus classes and `build_corpus` methods.
    - Source lines: [`pyroomacoustics/__init__.py#L98-L99`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L98-L99), [`pyroomacoustics/__init__.py#L123`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123), [`pyroomacoustics/datasets/__init__.py#L1-L4`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/__init__.py#L1-L4), [`pyroomacoustics/datasets/cmu_arctic.py#L114-L117`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/cmu_arctic.py#L114-L117), [`pyroomacoustics/datasets/cmu_arctic.py#L196-L202`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/cmu_arctic.py#L196-L202), [`pyroomacoustics/datasets/google_speech_commands.py#L72-L73`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/google_speech_commands.py#L72-L73), [`pyroomacoustics/datasets/google_speech_commands.py#L99-L105`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/google_speech_commands.py#L99-L105)
- `rir-generator` (`❌`):
    - No dataset module or dataset-building API is present.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

## ISM High-Pass Filter (HPF) Implementations

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, `das-generator`, and `pyroomacoustics`.

### `torchrir`: opt-in parameterization and equations

TorchRIR does not filter generated RIRs by default. Filtering is enabled only
by assigning an `RIRHighPassConfig` to `SimulationConfig.high_pass`. Install
the optional SciPy dependency before enabling it:

```bash
pip install "torchrir[hpf]"
```

```python
from torchrir.config import RIRHighPassConfig, SimulationConfig

config = SimulationConfig(
    max_order=3,
    nsample=4096,
    high_pass=RIRHighPassConfig(
        cutoff_hz=10.0,
        order=2,
        passband_ripple_db=5.0,
        stopband_attenuation_db=60.0,
        filter_family="butter",
        phase="zero_phase",
    ),
)
```

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

$$
w_c = \frac{2f_c}{f_s}
$$

Second-order sections are designed from the explicit configuration fields:

$$
\mathrm{SOS} = \mathrm{iirfilter}\left(
\mathrm{order},\; W_n=w_c,\;
rp=\mathrm{passband\_ripple\_db},\;
rs=\mathrm{stopband\_attenuation\_db},\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{filter\_family},\;
\text{output}=\text{"sos"}
\right)
$$

The default `phase="zero_phase"` matches pyroomacoustics and applies:

$$
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x_{:N_\mathrm{natural}})
$$

where

$$
N_\mathrm{natural}
= \min\left(
N_\mathrm{requested},
\left\lceil \tau_{\max} f_s \right\rceil
+ \frac{L_\mathrm{FDL}-1}{2} + 2
\right).
$$

The filtered prefix is zero-filled to the requested output length. This
reproduces pyroomacoustics' per-source/microphone finite ISM endpoint without
exposing its fixed fractional-delay offset. A diffuse tail is filtered through
the complete requested horizon. The forward-backward result has zero phase but
can pre-ring. Explicit `phase="causal"` instead applies `sosfilt` to the
complete requested RIR; it has no pre-ringing and is prefix invariant.

### `rir-generator` and `das-generator`: parameterization and equations

Default: `hp_filter=True`.

The filter coefficients are derived from sampling frequency `f_s`:

$$
W = \frac{2\pi f_c}{f_s} = \frac{2\pi \cdot 100}{f_s}
$$

$$
R_1 = e^{-W}, \quad
B_1 = 2R_1\cos(W), \quad
B_2 = -R_1^2, \quad
A_1 = -(1+R_1)
$$

With input sample `x[n]`, internal state `v[n]`, and output `y[n]`:

$$
v[n] = x[n] + B_1 v[n-1] + B_2 v[n-2]
$$

$$
y[n] = v[n] + A_1 v[n-1] + R_1 v[n-2]
$$

State is initialized to zero (`v[-1] = v[-2] = 0`).

`das-generator` uses these same coefficients and applies this causal filter
independently to each newly computed RIR row. Its public `hp_filter` default
is also `True`; see [DAS coefficients and filtering][das-core] and
[API defaults][das-api]. DynamicSound's air-absorption FIR is a different
operation and is not an ISM high-pass filter.

### `pyroomacoustics`: parameterization and equations

Pyroomacoustics enables its process-global 10 Hz, second-order Butterworth HPF
by default. Its passband-ripple and stopband-attenuation parameters default to
5 dB and 60 dB, respectively.

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

$$
w_c = \frac{2f_c}{f_s}
$$

Second-order sections are designed as:

$$
\mathrm{SOS} = \mathrm{iirfilter}\left(
n,\; W_n=w_c,\; rp,\; rs,\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{type},\;
\text{output}=\text{"sos"}
\right)
$$

For each generated RIR `x`, the library applies:

$$
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
$$

This is forward-backward filtering (zero-phase response).

## ISM Image-Source Amplitude Scaling

In ISM implementations, a common per-image gain form is:

$$
a_i \propto \frac{g_i}{d_i}
$$

where `g_i` aggregates reflection/directivity terms and `d_i` is propagation distance.
Some libraries additionally include free-field normalization by `4\pi`:

$$
a_i \propto \frac{g_i}{4\pi d_i}
$$

### Quick comparison

| Library | Typical distance scaling | Notes |
|---|---|---|
| `torchrir` | `1/r` | Reflection/directivity gains are multiplied, then divided by distance. |
| `gpuRIR` | `1/(4πr)` | CUDA core uses explicit `4π` factor in image-source amplitude. |
| `rir-generator` | `1/(4πr)` | Core C++ implementation uses `4π` free-field normalization. |
| `pyroomacoustics` | Usually `1/r` in room ISM path | `build_rir_matrix` uses `1/(4πr)`, so scale depends on API path. |
| `das-generator` | `1/(4πr)` | Internal dynamic ISM uses the C++ path gain in [DAS core][das-core]. |
| `dynamic-sound` | `1/r` | [Direct-path scaling][ds-attenuations], not an ISM implementation; air absorption is applied separately. |
| `TASCAR` | `1/max(nearfieldlimit, r)` by default | [Normal non-volumetric receiver][tascar-gain] with `gainmodel="1/r"`; filters and calibration add their own gains. |

### Practical implication for cross-library tests

Even with matched geometry, `beta`, image limits, and interpolation settings, direct waveform-level comparisons can show an almost constant gain ratio near `4π` between `1/r` and `1/(4πr)` conventions. Normalize this global factor before enforcing strict amplitude-matching thresholds.

## Cross-Implementation Oracle Policy

Agreement with another simulator is supporting evidence, not an unconditional
definition of correctness. TorchRIR's independent analytic and metamorphic
tests take precedence when a reference has a documented convention difference
or a relevant known issue.

- Comparisons apply only predetermined transforms for amplitude normalization,
  wall coefficients, image counts, a documented reference-side delay, and HPF
  state. Cross-correlation may assert the remaining lag, but it is never used
  to shift or crop a waveform into agreement.
- TorchRIR directivity and orientation are properties of `Source` and
  `MicrophoneArray`, not algorithm settings. Reference scenes attach the
  corresponding patterns to the same physical endpoint.
- gpuRIR has multiple reported
  [known issues](https://github.com/DavidDiazGuerra/gpuRIR/issues) and an
  internal signed reflection-coefficient convention. Consequently, raw signed
  reflected gpuRIR waveforms are not treated as an oracle. RIR comparisons use
  direct paths on the physical sample axis without a delay crop. Dynamic
  convolution is isolated by passing identical synthetic RIR tensors and
  predetermined frame starts to both convolution implementations.
- pyroomacoustics comparisons use the `ShoeBox.compute_rir` path, convert
  pressure reflection coefficients with `alpha = 1 - beta**2`, and restore its
  process-global HPF setting after each call. Pyroomacoustics 0.9.0 exposes the
  known 40-sample delay of its 81-tap fractional-delay filter, so exactly 40
  samples are cropped from the reference before comparison. Analytic
  directivity in the PyPI 0.9.0 oracle uses `DirectionVector` to avoid that
  release's broken ndarray orientation constructor.
- rir-generator comparisons require version 0.3.0, disable its distinct 100 Hz
  HPF, apply the predetermined `4*pi` normalization, and use matched
  fractional-delay support. Both outputs already use a physical sample axis,
  so they are compared directly without a delay crop.

Any remaining mismatch is first localized to geometry, path gain, interpolation,
or convolution before assigning it to either implementation.

The additional source audit does not add either library as a CI dependency.
DynamicSound's emission-time solver can be checked against analytic trajectories,
but its multiple-source output is not a superposition oracle at the pinned
revision. DAS's fixed-receiver output agrees in the tested static and
moving-source cases after the documented gain/window/HPF alignment;
moving-receiver and first-emission cases need independent analytic checks
because of the reproduced cache errors above.

## Known Differences

In addition to the `4π` scaling gap, the following implementation differences affect cross-library RIR waveform comparisons.
Line references below were checked against:

- `torchrir` (this repository, current branch)
- `gpuRIR` (`fd8af43a4a113d3c2c05f0085a0119ecb1f1a484` snapshot)
- `pyroomacoustics` `0.9.0`
- `rir-generator` `0.3.0`
- `dynamic-sound` `f3aceae2bd483d7f347b3e2e842b1a72bb57ebce`
- `das-generator` `6f2cd6d22872edd6264ca0c60e9f58477ddaeb48`

### 1) Image-source enumeration rules (`max_order` / `nb_img`)

- `torchrir`: with `max_order`, it truncates by L1 norm (diamond); with `nb_img`, it enumerates a rectangular index range.
- `rir-generator`: loops over a rectangular range, then filters by the same L1
  reflection-order condition when an explicit non-negative order is supplied.
- `gpuRIR`: maps `nb_img` directly to CUDA-side index expansion, so the enumeration path differs.

Practical note:

- Explicit `max_order=N` can be matched between TorchRIR and rir-generator.
  rir-generator's `order=-1` instead means an automatically selected maximum
  based on RIR length and has no identical TorchRIR parameter. gpuRIR's
  `nb_img` must be converted separately.

Source lines:

- `torchrir`: [`src/torchrir/sim/ism/images.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/images.py)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L177-L207`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L177-L207)
- `gpuRIR`: [`src/gpuRIR_cuda.cu#L337-L341`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L337-L341), [`src/gpuRIR_cuda.cu#L804-L806`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L804-L806), [`src/gpuRIR_cuda.cu#L839`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L839)

### 2) Fractional-delay interpolation kernel

- `torchrir`: Hann-windowed sinc (default tap length 81), with selectable LUT
  on/off. It centers the taps directly around each physical arrival sample and
  retains only support inside the requested interval, so the public RIR has no
  interpolation-filter group delay.
- `pyroomacoustics` 0.9.0: its 81-tap interpolation path exposes a fixed
  40-sample offset. Tests remove exactly those 40 samples from the reference;
  the offset is never estimated from either waveform.
- `gpuRIR`: `Tw`-based implementation with separate LUT and mixed-precision controls.
- `rir-generator`: a different LP interpolation implementation. At 16 kHz its
  `Tw=128` support is closest to a 129-tap TorchRIR filter, but rir-generator
  evaluates its cosine window at the fractional delay and does not add a global
  delay.

Practical note:

- Local waveform shape around sample positions can differ, causing mismatches
  in peak amplitude and fine temporal detail. gpuRIR and rir-generator are
  compared directly on the physical axis; only the documented pyroomacoustics
  reference offset is cropped.

Source lines:

- `torchrir`: [`src/torchrir/config.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/config.py), [`src/torchrir/sim/ism/accumulate.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/accumulate.py)
- `gpuRIR`: [`src/gpuRIR_cuda.cu#L629-L637`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L629-L637), [`src/gpuRIR_cuda.cu#L644`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L644), [`src/gpuRIR_cuda.cu#L676-L696`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L676-L696), [`gpuRIR/__init__.py#L223-L243`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L223-L243)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L144-L145`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L144-L145), [`src/rir_generator/_cffi/rir_generator_core.cpp#L214-L218`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L214-L218)

### 3) HPF implementation and defaults

- `torchrir`: no HPF by default; a caller may opt in with
  `SimulationConfig.high_pass`. The default high-pass configuration uses
  pyroomacoustics-compatible zero-phase filtering and natural ISM horizons.
- `pyroomacoustics`: has HPF-enabled paths and a process-global setting.
- `rir-generator`: uses an Allen-Berkley style HPF.
- `gpuRIR`: does not assume an equivalent built-in HPF path, and project discussion indicates the low-frequency attenuation behavior is an intentional design choice (not just a missing toggle).

Practical note:

- HPF presence and coefficient differences change waveform and energy,
  especially in low-frequency bands. Baseline parity tests leave TorchRIR
  unfiltered and explicitly disable reference HPFs where possible. Filtered
  comparisons must opt in with fully specified coefficients and phase.

Source lines:

- `torchrir`: [`src/torchrir/config.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/config.py), [`src/torchrir/sim/ism/hpf.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/hpf.py)
- `pyroomacoustics`: [`pyroomacoustics/parameters.py#L192-L194`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/parameters.py#L192-L194), [`pyroomacoustics/room.py#L2292-L2295`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2292-L2295), [`pyroomacoustics/room.py#L2356-L2357`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2356-L2357)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L135-L139`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L135-L139), [`src/rir_generator/_cffi/rir_generator_core.cpp#L232-L243`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L232-L243)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L117`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L95-L117) (no HPF parameter in the public `simulateRIR` API)
- `gpuRIR` discussion: [Issue #15](https://github.com/DavidDiazGuerra/gpuRIR/issues/15)

### 4) Late-reverb / diffuse-tail modeling

- `torchrir`: estimates the handoff level from the preceding 10 ms RMS, uses a
  5 ms power-complementary crossfade, uses the configured room speed of sound,
  and shares a seeded source--microphone carrier across adjacent dynamic
  frames. Its prefix is invariant to a longer requested horizon and unrelated
  batch pairs. A zero-time handoff is rejected because it has no early field
  from which to estimate the level.
- `gpuRIR`: also separates early reflections and diffuse components, but not with the same method.

Practical note:

- If `tmax` or tail-related settings are not aligned, late-part waveform error can grow significantly.

Source lines:

- `torchrir`: [`src/torchrir/sim/ism/api.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/api.py), [`src/torchrir/sim/ism/diffuse.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/diffuse.py)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L117`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L95-L117), [`gpuRIR/__init__.py#L166`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L166), [`src/gpuRIR_cuda.cu#L831-L835`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L831-L835), [`src/gpuRIR_cuda.cu#L852-L865`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L852-L865), [`src/gpuRIR_cuda.cu#L872-L883`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L872-L883), [`src/gpuRIR_cuda.cu#L445-L461`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/src/gpuRIR_cuda.cu#L445-L461)

### 5) Dynamic API assumptions

- `torchrir`: constructs a `DynamicScene`, generates its RIR snapshots with
  `simulate(scene, config)`, and applies them with
  `DynamicConvolver(time_reference="emission")` for a moving source or
  `DynamicConvolver(time_reference="observation")` for a moving receiver.
  A `FrameSchedule` supplies exact frame-start samples independently of that
  physical time reference.
- `gpuRIR`: designed around static RIR generation plus path-based convolution;
  for fair comparison, one moving source and fixed microphones are the cleanest
  setup.
- `rir-generator`: no dynamic motion API (static-oriented).
- `dynamic-sound`: observation-time evaluation of receiver trajectories and
  retarded emission-time solution for source trajectories; direct sound only.
- `das-generator`: per-sample source/receiver trajectories and an internal
  two-time ISM response matrix. The [audit](#dynamic-sound-and-das-generator-implementation-audit)
  distinguishes the intended kernel from its moving-receiver reuse defects.

A moving-source scene can carry its exact sample schedule:

```python
from torchrir import DynamicScene
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
result = simulate(scene, config)
wet = DynamicConvolver(time_reference="emission").convolve(dry, result)
```

If `DynamicScene.schedule` is omitted, pass the schedule to
`convolve(..., schedule=schedule)` instead. A result whose scene already owns a
schedule rejects an additional call-level schedule.

Practical note:

- Without matching scene constraints, a dynamic comparison may reflect API assumptions rather than core algorithm differences.
- gpuRIR-style input-segment overlap-add must not be used as a reference for a
  moving receiver. Simultaneous source and receiver motion additionally needs a
  retarded-time propagation model. Convolution-only parity tests pass identical
  synthetic RIR tensors and an identical `FrameSchedule` to isolate scheduling
  logic from RIR generation.

Source lines:

- `torchrir`: [`src/torchrir/sim/simulators.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/simulators.py), [`src/torchrir/signal/dynamic.py`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/signal/dynamic.py)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L175`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L95-L175), [`gpuRIR/__init__.py#L177-L220`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/gpuRIR/__init__.py#L177-L220), [`examples/simulate_trajectory.py#L18-L33`](https://github.com/DavidDiazGuerra/gpuRIR/blob/fd8af43a4a113d3c2c05f0085a0119ecb1f1a484/examples/simulate_trajectory.py#L18-L33)
- gpuRIR moving-receiver limitation: [Issue #76](https://github.com/DavidDiazGuerra/gpuRIR/issues/76)
- `rir-generator`: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50) (static `generate(...)` API)

### 6) API-path differences inside `pyroomacoustics`

- The main room ISM path typically uses `1/r`.
- The `build_rir_matrix` path uses `1/(4πr)`.

Practical note:

- Even within one library, amplitude convention can vary by API path, so fix the call path during comparisons.

Source lines:

- `pyroomacoustics` room ISM path (`1/r`): [`pyroomacoustics/room.py#L2317-L2328`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2317-L2328) -> [`pyroomacoustics/simulation/ism.py#L187`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/simulation/ism.py#L187)
- `pyroomacoustics` `build_rir_matrix` path (`1/(4πr)`): [`pyroomacoustics/soundsource.py#L326-L328`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/soundsource.py#L326-L328)
