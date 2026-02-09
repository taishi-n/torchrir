# Library Comparisons

This page summarizes implementation-level differences between TorchRIR and related libraries in this scope:
- `torchrir`
- `gpuRIR`
- `rir-generator`
- `pyroomacoustics`

## Dynamic Simulation Feature Comparison

| Library | Dynamic RIR generation | Many dynamic sources at once | GPU/CPU support | Visualization |
|---|---|---|---|---|
| `torchrir` | Yes (`simulate_dynamic_rir`) | Yes (`(T, n_src, n_mic, nsample)` + multi-source dynamic convolution) | CPU / CUDA / MPS | Built-in static/dynamic plotting and GIF helpers (`torchrir.viz`) |
| `gpuRIR` | Yes (trajectory points via `simulateRIR`) | Partial (multi-source RIRs are supported, but trajectory filtering is centered on one moving source signal per call) | CUDA GPU (no general CPU backend for core RIR simulation) | No built-in visualization API (examples use external plotting) |
| `rir-generator` | No (static RIR only) | No | CPU (C/C++ core via CFFI) | No built-in visualization API |
| `pyroomacoustics` | Partial (static RIR pipeline; dynamic motion typically requires manual chunking/recomputing) | No native API for many dynamic trajectories simultaneously | CPU (Python + C++ acceleration) | Built-in room/RIR plotting (`Room.plot`, `Room.plot_rir`) |

Notes:
- The table focuses on native APIs for moving-source/moving-microphone simulation workflows.
- "Partial" means feasible with extra user-side orchestration rather than a dedicated end-to-end dynamic API.

## ISM High-Pass Filter (HPF) Implementations

This section focuses on libraries (in this comparison scope) that implement a built-in HPF for ISM-generated RIRs: `torchrir`, `rir-generator`, and `pyroomacoustics`.

| Library | HPF design | Cutoff / parameter decision | Application method |
|---|---|---|---|
| `torchrir` | Configurable IIR high-pass designed by `scipy.signal.iirfilter` (SOS) | Default parameters are aligned with `pyroomacoustics`: `f_c=10.0` Hz, `n=2`, `rp=5.0`, `rs=60.0`, `type="butter"`; normalized cutoff `w_c = 2 f_c / f_s` | Zero-phase `sosfiltfilt` on generated RIR tensors (last axis) |
| `rir-generator` | Allen and Berkley-style recursive HPF | Fixed cutoff `f_c = 100` Hz, mapped with sampling frequency `f_s` | One-pass recursive filtering on generated RIR samples |
| `pyroomacoustics` | Configurable IIR high-pass designed by `scipy.signal.iirfilter` (SOS) | User/global parameters `(f_c, n, rp, rs, type)`, with normalized cutoff `w_c = 2 f_c / f_s` | Zero-phase `sosfiltfilt` on each generated RIR |

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
