# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries in this scope:
- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`

## Dynamic Simulation Feature Comparison

| Feature | `torchrir` | `gpuRIR` | `pyroomacoustics` | `rir-generator` |
|---|---|---|---|---|
| 🎯 Dynamic Sources | ✅ | 🟡 Single moving source | 🟡 Manual loop | ❌ |
| 🎤 Dynamic Microphones | ✅ | ❌ | 🟡 Manual loop | ❌ |
| 🖥️ CPU | ✅ | ❌ | ✅ | ✅ |
| 🧮 CUDA | ✅ | ✅ | ❌ | ❌ |
| 🍎 MPS | ✅ | ❌ | ❌ | ❌ |
| 📊 Visualization | ✅ | ❌ | ✅ | ❌ |
| 🗂️ Dataset Build | ✅ | ❌ | ✅ | ❌ |

Legend:
- `✅` native support
- `🟡` manual setup
- `❌` unavailable

## Visualization and Dataset Build (Source-Level)

Marking criterion in this section:
- Mark as `✅` when the functionality is provided as a library API/submodule (not only in examples).
- Mark as `🟡` when possible only via manual composition without a dedicated library feature surface.
- Mark as `❌` when no corresponding library functionality exists.

### Visualization

- `torchrir` (`✅`):
  - Dedicated visualization submodule and public functions are provided.
  - Source lines: `src/torchrir/viz/__init__.py:6-17`, `src/torchrir/viz/scene.py:14-23`, `src/torchrir/viz/scene.py:51-63`, `src/torchrir/viz/io.py:22-36`, `src/torchrir/viz/io.py:93-107`, `src/torchrir/viz/io.py:127-141`
- `gpuRIR` (`❌`):
  - Package exports simulation/control functions only; plotting appears in example scripts.
  - Source lines: `gpuRIR/__init__.py:11`, `examples/example.py:8-9`, `examples/example.py:35-36`, `examples/polar_plots.py:3`, `examples/polar_plots.py:9-10`, `examples/polar_plots.py:66-75`
- `pyroomacoustics` (`✅`):
  - Library-level plotting APIs exist (`Room.plot`, `Room.plot_rir`), with optional plotting helpers in other submodules.
  - Source lines: `pyroomacoustics/room.py:1535-1547`, `pyroomacoustics/room.py:1827-1843`, `pyroomacoustics/__init__.py:123-134`
- `rir-generator` (`❌`):
  - The package API is focused on RIR generation (`generate`) and does not include plotting APIs.
  - Source lines: `src/rir_generator/__init__.py:36-50`

### Dataset Build

- `torchrir` (`✅`):
  - Dataset utilities are provided as library modules (`torchrir.datasets`) including dataset wrappers and source-loading utilities used by dataset generation workflows.
  - Source lines: `src/torchrir/datasets/__init__.py:1-7`, `src/torchrir/datasets/__init__.py:19-42`, `src/torchrir/datasets/utils.py:30-37`
- `gpuRIR` (`❌`):
  - No dataset submodule or dataset-building API is exposed.
  - Source lines: `gpuRIR/__init__.py:11`
- `pyroomacoustics` (`✅`):
  - Dataset functionality is provided in-library via `pyroomacoustics.datasets` with corpus classes and `build_corpus` methods.
  - Source lines: `pyroomacoustics/__init__.py:98-99`, `pyroomacoustics/__init__.py:123`, `pyroomacoustics/datasets/__init__.py:1-4`, `pyroomacoustics/datasets/cmu_arctic.py:114-117`, `pyroomacoustics/datasets/cmu_arctic.py:196-202`, `pyroomacoustics/datasets/google_speech_commands.py:72-73`, `pyroomacoustics/datasets/google_speech_commands.py:99-105`
- `rir-generator` (`❌`):
  - No dataset module or dataset-building API is present.
  - Source lines: `src/rir_generator/__init__.py:36-50`

## ISM High-Pass Filter (HPF) Implementations

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, and `pyroomacoustics`.

### `torchrir`: parameterization and equations

Defaults:
- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

```{math}
w_c = \frac{2f_c}{f_s}
```

Second-order sections are designed as:

```{math}
\mathrm{SOS} = \mathrm{iirfilter}\left(
n,\; W_n=w_c,\; rp,\; rs,\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{type},\;
\text{output}=\text{"sos"}
\right)
```

For each generated RIR tensor `x`, TorchRIR applies:

```{math}
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
```

along the time axis (last dimension), i.e. static `(n_src, n_mic, nsample)` and dynamic `(T, n_src, n_mic, nsample)` outputs are filtered in-place along `nsample`.

### `rir-generator`: parameterization and equations

Default: `hp_filter=True`.

The filter coefficients are derived from sampling frequency `f_s`:

```{math}
W = \frac{2\pi f_c}{f_s} = \frac{2\pi \cdot 100}{f_s}
```

```{math}
R_1 = e^{-W}, \quad
B_1 = 2R_1\cos(W), \quad
B_2 = -R_1^2, \quad
A_1 = -(1+R_1)
```

With input sample `x[n]`, internal state `v[n]`, and output `y[n]`:

```{math}
v[n] = x[n] + B_1 v[n-1] + B_2 v[n-2]
```

```{math}
y[n] = v[n] + A_1 v[n-1] + R_1 v[n-2]
```

State is initialized to zero (`v[-1] = v[-2] = 0`).

### `pyroomacoustics`: parameterization and equations

Defaults:
- `rir_hpf_enable=True`
- `rir_hpf_fc=10.0` (Hz)
- `rir_hpf_kwargs={"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}`

For sampling frequency `f_s` and cutoff `f_c`, the normalized digital cutoff is:

```{math}
w_c = \frac{2f_c}{f_s}
```

Second-order sections are designed as:

```{math}
\mathrm{SOS} = \mathrm{iirfilter}\left(
n,\; W_n=w_c,\; rp,\; rs,\;
\text{btype}=\text{"highpass"},\;
\text{ftype}=\mathrm{type},\;
\text{output}=\text{"sos"}
\right)
```

For each generated RIR `x`, the library applies:

```{math}
y = \mathrm{sosfiltfilt}(\mathrm{SOS}, x)
```

This is forward-backward filtering (zero-phase response).

## ISM Image-Source Amplitude Scaling

In ISM implementations, a common per-image gain form is:

```{math}
a_i \propto \frac{g_i}{d_i}
```

where `g_i` aggregates reflection/directivity terms and `d_i` is propagation distance.
Some libraries additionally include free-field normalization by `4\pi`:

```{math}
a_i \propto \frac{g_i}{4\pi d_i}
```

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
- `torchrir`: `src/torchrir/sim/ism/images.py:21-32`
- `rir-generator`: `src/rir_generator/_cffi/rir_generator_core.cpp:177-207`
- `gpuRIR`: `src/gpuRIR_cuda.cu:337-341`, `src/gpuRIR_cuda.cu:804-806`, `src/gpuRIR_cuda.cu:839`

### 2) Fractional-delay interpolation kernel

- `torchrir`: Hann-windowed sinc (default tap length 81), with selectable LUT on/off.
- `gpuRIR`: `Tw`-based implementation with separate LUT and mixed-precision controls.
- `rir-generator`: a different LP interpolation implementation (with different `Tw` definition).

Practical note:
- Local waveform shape around sample positions can differ, causing mismatches in peak amplitude and fine temporal detail.

Source lines:
- `torchrir`: `src/torchrir/config.py:26-30`, `src/torchrir/sim/ism/accumulate.py:212-229`
- `gpuRIR`: `src/gpuRIR_cuda.cu:629-637`, `src/gpuRIR_cuda.cu:644`, `src/gpuRIR_cuda.cu:676-696`, `gpuRIR/__init__.py:223-243`
- `rir-generator`: `src/rir_generator/_cffi/rir_generator_core.cpp:144-145`, `src/rir_generator/_cffi/rir_generator_core.cpp:214-218`

### 3) HPF implementation and defaults

- `torchrir`: IIR HPF (enabled by default).
- `pyroomacoustics`: also has HPF-enabled paths.
- `rir-generator`: uses an Allen-Berkley style HPF.
- `gpuRIR`: does not assume an equivalent built-in HPF path.

Practical note:
- HPF presence and coefficient differences change waveform and energy, especially in low-frequency bands.

Source lines:
- `torchrir`: `src/torchrir/config.py:33-37`, `src/torchrir/sim/ism/hpf.py:23-37`, `src/torchrir/sim/ism/hpf.py:40-56`
- `pyroomacoustics`: `pyroomacoustics/parameters.py:192-194`, `pyroomacoustics/room.py:2292-2295`, `pyroomacoustics/room.py:2356-2357`
- `rir-generator`: `src/rir_generator/_cffi/rir_generator_core.cpp:135-139`, `src/rir_generator/_cffi/rir_generator_core.cpp:232-243`
- `gpuRIR`: `gpuRIR/__init__.py:95-117` (no HPF parameter in the public `simulateRIR` API)

### 4) Late-reverb / diffuse-tail modeling

- `torchrir`: can add a diffuse tail conditionally.
- `gpuRIR`: also separates early reflections and diffuse components, but not with the same method.

Practical note:
- If `tmax` or tail-related settings are not aligned, late-part waveform error can grow significantly.

Source lines:
- `torchrir`: `src/torchrir/sim/ism/api.py:130-132`, `src/torchrir/sim/ism/diffuse.py:15-49`
- `gpuRIR`: `gpuRIR/__init__.py:95-117`, `gpuRIR/__init__.py:166`, `src/gpuRIR_cuda.cu:831-835`, `src/gpuRIR_cuda.cu:852-865`, `src/gpuRIR_cuda.cu:872-883`, `src/gpuRIR_cuda.cu:445-461`

### 5) Dynamic API assumptions

- `torchrir`: explicitly models trajectories via `simulate_dynamic_rir(src_traj, mic_traj, ...)`.
- `gpuRIR`: designed around static RIR generation plus trajectory convolution; for fair comparison, fixed mic + single moving source is the cleanest setup.
- `rir-generator`: no dynamic trajectory API (static-oriented).

Practical note:
- Without matching scene constraints, a dynamic comparison may reflect API assumptions rather than core algorithm differences.

Source lines:
- `torchrir`: `src/torchrir/sim/ism/api.py:136-150`
- `gpuRIR`: `gpuRIR/__init__.py:95-175`, `gpuRIR/__init__.py:177-220`, `examples/simulate_trajectory.py:18-33`
- `rir-generator`: `src/rir_generator/__init__.py:36-50` (static `generate(...)` API)

### 6) API-path differences inside `pyroomacoustics`

- The main room ISM path typically uses `1/r`.
- The `build_rir_matrix` path uses `1/(4πr)`.

Practical note:
- Even within one library, amplitude convention can vary by API path, so fix the call path during comparisons.

Source lines:
- `pyroomacoustics` room ISM path (`1/r`): `pyroomacoustics/room.py:2317-2328` -> `pyroomacoustics/simulation/ism.py:187`
- `pyroomacoustics` `build_rir_matrix` path (`1/(4πr)`): `pyroomacoustics/soundsource.py:326-328`
