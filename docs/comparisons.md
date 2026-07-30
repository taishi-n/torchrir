# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries within this comparison scope:

- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`

## Feature Comparison

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

Legend:

- `✅` native support
- `🟡` manual setup
- `🚧` candidate (not yet implemented)
- `❌` unavailable

Notes:

- `RIR Convolution` means applying static or time-varying RIRs to source signals.
- Broader signal processing such as beamforming, DOA, BSS, adaptive filtering, STFT, and denoising remains out of scope for `torchrir`.

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

### Dynamic Scene GIF

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

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, and `pyroomacoustics`.

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

### `rir-generator`: parameterization and equations

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

## Known Differences

In addition to the `4π` scaling gap, the following implementation differences affect cross-library RIR waveform comparisons.
Line references below were checked against:

- `torchrir` (this repository, current branch)
- `gpuRIR` (`fd8af43a4a113d3c2c05f0085a0119ecb1f1a484` snapshot)
- `pyroomacoustics` `0.9.0`
- `rir-generator` `0.3.0`

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
