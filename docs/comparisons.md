# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries in this scope:

- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`

## Feature Comparison

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

Legend:

- `✅` native support
- `🟡` manual setup
- `🚧` candidate (not yet implemented)
- `❌` unavailable

Notes:

- `Signal Processing` includes beamforming, DOA, BSS, adaptive filtering, STFT, and denoising.
- In `torchrir`, this row is comparison-only and marked as scope out.

## Visualization, Dynamic GIF, and Dataset Build (Source-Level)

Marking criterion in this section:

- Mark as `✅` when the functionality is provided as a library API/submodule (not only in examples).
- Mark as `🟡` when possible only via manual composition without a dedicated library feature surface.
- Mark as `❌` when no corresponding library functionality exists.

### Visualization

- `torchrir` (`✅`):
    - Dedicated visualization submodule and public functions are provided.
    - Source lines: [`src/torchrir/viz/__init__.py#L6-L17`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/__init__.py#L6-L17), [`src/torchrir/viz/scene.py#L14-L23`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/scene.py#L14-L23), [`src/torchrir/viz/scene.py#L51-L63`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/scene.py#L51-L63), [`src/torchrir/viz/io.py#L22-L36`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py#L22-L36), [`src/torchrir/viz/io.py#L93-L107`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py#L93-L107), [`src/torchrir/viz/io.py#L127-L141`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py#L127-L141)
- `gpuRIR` (`❌`):
    - Package exports simulation/control functions only; plotting appears in example scripts.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L11), [`examples/example.py#L8-L9`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py#L8-L9), [`examples/example.py#L35-L36`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py#L35-L36), [`examples/polar_plots.py#L3`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/polar_plots.py#L3), [`examples/polar_plots.py#L9-L10`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/polar_plots.py#L9-L10), [`examples/polar_plots.py#L66-L75`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/polar_plots.py#L66-L75)
- `pyroomacoustics` (`✅`):
    - Library-level plotting APIs exist (`Room.plot`, `Room.plot_rir`), with optional plotting helpers in other submodules.
    - Source lines: [`pyroomacoustics/room.py#L1535-L1547`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1535-L1547), [`pyroomacoustics/room.py#L1827-L1843`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1827-L1843), [`pyroomacoustics/__init__.py#L123-L134`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123-L134)
- `rir-generator` (`❌`):
    - The package API is focused on RIR generation (`generate`) and does not include plotting APIs.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

### Dynamic Scene GIF

- `torchrir` (`✅`):
    - Dedicated GIF APIs are provided for dynamic trajectories.
    - Source lines: [`src/torchrir/viz/__init__.py#L3-L17`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/__init__.py#L3-L17), [`src/torchrir/viz/animation.py#L13-L29`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/animation.py#L13-L29), [`src/torchrir/viz/io.py#L127-L141`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/viz/io.py#L127-L141)
- `gpuRIR` (`❌`):
    - No GIF/animation API is exposed in the package interface.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L11)
- `pyroomacoustics` (`🟡`):
    - Plotting APIs are provided, but no dedicated dynamic-scene GIF API; animation must be manually composed by users.
    - Source lines: [`pyroomacoustics/room.py#L1535-L1547`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1535-L1547), [`pyroomacoustics/room.py#L1827-L1843`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L1827-L1843), [`pyroomacoustics/__init__.py#L123-L134`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123-L134)
- `rir-generator` (`❌`):
    - No GIF/animation API is provided.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

### Dataset Build

- `torchrir` (`✅`):
    - Dataset utilities are provided as library modules (`torchrir.datasets`) including dataset wrappers and source-loading utilities used by dataset generation workflows.
    - Source lines: [`src/torchrir/datasets/__init__.py#L1-L7`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/datasets/__init__.py#L1-L7), [`src/torchrir/datasets/__init__.py#L19-L42`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/datasets/__init__.py#L19-L42), [`src/torchrir/datasets/utils.py#L30-L37`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/datasets/utils.py#L30-L37)
- `gpuRIR` (`❌`):
    - No dataset submodule or dataset-building API is exposed.
    - Source lines: [`gpuRIR/__init__.py#L11`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L11)
- `pyroomacoustics` (`✅`):
    - Dataset functionality is provided in-library via `pyroomacoustics.datasets` with corpus classes and `build_corpus` methods.
    - Source lines: [`pyroomacoustics/__init__.py#L98-L99`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L98-L99), [`pyroomacoustics/__init__.py#L123`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/__init__.py#L123), [`pyroomacoustics/datasets/__init__.py#L1-L4`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/__init__.py#L1-L4), [`pyroomacoustics/datasets/cmu_arctic.py#L114-L117`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/cmu_arctic.py#L114-L117), [`pyroomacoustics/datasets/cmu_arctic.py#L196-L202`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/cmu_arctic.py#L196-L202), [`pyroomacoustics/datasets/google_speech_commands.py#L72-L73`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/google_speech_commands.py#L72-L73), [`pyroomacoustics/datasets/google_speech_commands.py#L99-L105`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/datasets/google_speech_commands.py#L99-L105)
- `rir-generator` (`❌`):
    - No dataset module or dataset-building API is present.
    - Source lines: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50)

## ISM High-Pass Filter (HPF) Implementations

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, and `pyroomacoustics`.

### `torchrir`: parameterization and equations

Defaults:

- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

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

For each generated RIR tensor `x`, TorchRIR applies:

$$
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
$$

along the time axis (last dimension), i.e. static `(n_src, n_mic, nsample)` and dynamic `(T, n_src, n_mic, nsample)` outputs are filtered in-place along `nsample`.

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

Defaults:

- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

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

## Known Differences

In addition to the `4π` scaling gap, the following implementation differences affect cross-library RIR waveform comparisons.
Line references below were checked against:

- `torchrir` (this repository, current branch)
- `gpuRIR` (`master` snapshot used for this comparison)
- `pyroomacoustics` `0.9.0`
- `rir-generator` `0.3.0`

### 1) Image-source enumeration rules (`max_order` / `nb_img`)

- `torchrir`: with `max_order`, it truncates by L1 norm (diamond); with `nb_img`, it enumerates a rectangular index range.
- `rir-generator`: loops over a rectangular range, then filters by an order condition.
- `gpuRIR`: maps `nb_img` directly to CUDA-side index expansion, so the enumeration path differs.

Practical note:

- Even when parameters look identical, the included image-source set may not match exactly.

Source lines:

- `torchrir`: [`src/torchrir/sim/ism/images.py#L21-L32`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/images.py#L21-L32)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L177-L207`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L177-L207)
- `gpuRIR`: [`src/gpuRIR_cuda.cu#L337-L341`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L337-L341), [`src/gpuRIR_cuda.cu#L804-L806`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L804-L806), [`src/gpuRIR_cuda.cu#L839`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L839)

### 2) Fractional-delay interpolation kernel

- `torchrir`: Hann-windowed sinc (default tap length 81), with selectable LUT on/off.
- `gpuRIR`: `Tw`-based implementation with separate LUT and mixed-precision controls.
- `rir-generator`: a different LP interpolation implementation (with different `Tw` definition).

Practical note:

- Local waveform shape around sample positions can differ, causing mismatches in peak amplitude and fine temporal detail.

Source lines:

- `torchrir`: [`src/torchrir/config.py#L26-L30`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/config.py#L26-L30), [`src/torchrir/sim/ism/accumulate.py#L212-L229`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/accumulate.py#L212-L229)
- `gpuRIR`: [`src/gpuRIR_cuda.cu#L629-L637`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L629-L637), [`src/gpuRIR_cuda.cu#L644`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L644), [`src/gpuRIR_cuda.cu#L676-L696`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L676-L696), [`gpuRIR/__init__.py#L223-L243`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L223-L243)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L144-L145`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L144-L145), [`src/rir_generator/_cffi/rir_generator_core.cpp#L214-L218`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L214-L218)

### 3) HPF implementation and defaults

- `torchrir`: IIR HPF (enabled by default).
- `pyroomacoustics`: also has HPF-enabled paths.
- `rir-generator`: uses an Allen-Berkley style HPF.
- `gpuRIR`: does not assume an equivalent built-in HPF path, and project discussion indicates the low-frequency attenuation behavior is an intentional design choice (not just a missing toggle).

Practical note:

- HPF presence and coefficient differences change waveform and energy, especially in low-frequency bands.
- In this repository, disabling HPF in `torchrir` for `gpuRIR` parity tests is only a comparison tactic. It does not imply that HPF should be disabled for normal ISM usage.

Source lines:

- `torchrir`: [`src/torchrir/config.py#L33-L37`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/config.py#L33-L37), [`src/torchrir/sim/ism/hpf.py#L23-L37`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/hpf.py#L23-L37), [`src/torchrir/sim/ism/hpf.py#L40-L56`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/hpf.py#L40-L56)
- `pyroomacoustics`: [`pyroomacoustics/parameters.py#L192-L194`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/parameters.py#L192-L194), [`pyroomacoustics/room.py#L2292-L2295`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2292-L2295), [`pyroomacoustics/room.py#L2356-L2357`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2356-L2357)
- `rir-generator`: [`src/rir_generator/_cffi/rir_generator_core.cpp#L135-L139`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L135-L139), [`src/rir_generator/_cffi/rir_generator_core.cpp#L232-L243`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/_cffi/rir_generator_core.cpp#L232-L243)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L117`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L95-L117) (no HPF parameter in the public `simulateRIR` API)
- `gpuRIR` discussion: [Issue #15](https://github.com/DavidDiazGuerra/gpuRIR/issues/15)

### 4) Late-reverb / diffuse-tail modeling

- `torchrir`: can add a diffuse tail conditionally.
- `gpuRIR`: also separates early reflections and diffuse components, but not with the same method.

Practical note:

- If `tmax` or tail-related settings are not aligned, late-part waveform error can grow significantly.

Source lines:

- `torchrir`: [`src/torchrir/sim/ism/api.py#L130-L132`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/api.py#L130-L132), [`src/torchrir/sim/ism/diffuse.py#L15-L49`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/diffuse.py#L15-L49)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L117`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L95-L117), [`gpuRIR/__init__.py#L166`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L166), [`src/gpuRIR_cuda.cu#L831-L835`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L831-L835), [`src/gpuRIR_cuda.cu#L852-L865`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L852-L865), [`src/gpuRIR_cuda.cu#L872-L883`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L872-L883), [`src/gpuRIR_cuda.cu#L445-L461`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/src/gpuRIR_cuda.cu#L445-L461)

### 5) Dynamic API assumptions

- `torchrir`: explicitly models trajectories via `simulate_dynamic_rir(src_traj, mic_traj, ...)`.
- `gpuRIR`: designed around static RIR generation plus trajectory convolution; for fair comparison, fixed mic + single moving source is the cleanest setup.
- `rir-generator`: no dynamic trajectory API (static-oriented).

Practical note:

- Without matching scene constraints, a dynamic comparison may reflect API assumptions rather than core algorithm differences.

Source lines:

- `torchrir`: [`src/torchrir/sim/ism/api.py#L136-L150`](https://github.com/taishi-n/torchrir/blob/main/src/torchrir/sim/ism/api.py#L136-L150)
- `gpuRIR`: [`gpuRIR/__init__.py#L95-L175`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L95-L175), [`gpuRIR/__init__.py#L177-L220`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/gpuRIR/__init__.py#L177-L220), [`examples/simulate_trajectory.py#L18-L33`](https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/simulate_trajectory.py#L18-L33)
- `rir-generator`: [`src/rir_generator/__init__.py#L36-L50`](https://github.com/audiolabs/rir-generator/blob/v0.3.0/src/rir_generator/__init__.py#L36-L50) (static `generate(...)` API)

### 6) API-path differences inside `pyroomacoustics`

- The main room ISM path typically uses `1/r`.
- The `build_rir_matrix` path uses `1/(4πr)`.

Practical note:

- Even within one library, amplitude convention can vary by API path, so fix the call path during comparisons.

Source lines:

- `pyroomacoustics` room ISM path (`1/r`): [`pyroomacoustics/room.py#L2317-L2328`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/room.py#L2317-L2328) -> [`pyroomacoustics/simulation/ism.py#L187`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/simulation/ism.py#L187)
- `pyroomacoustics` `build_rir_matrix` path (`1/(4πr)`): [`pyroomacoustics/soundsource.py#L326-L328`](https://github.com/LCAV/pyroomacoustics/blob/v0.9.0/pyroomacoustics/soundsource.py#L326-L328)
