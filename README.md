# TorchRIR

PyTorch-based room impulse response (RIR) simulation toolkit focused on a clean, modern API with GPU support.

## Example Usage
```bash
# CMU ARCTIC + static RIR (fixed sources/mics)
uv run python examples/static.py --plot

# Dynamic RIR demos
uv run python examples/dynamic_mic.py --plot
uv run python examples/dynamic_src.py --plot
```

## References
- [gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)
- [pyroomacoustics](https://github.com/LCAV/pyroomacoustics)
- [das-generator](https://github.com/ehabets/das-generator)
- [rir-generator](https://github.com/audiolabs/rir-generator)

## Specification (Draft)
### Purpose
- Provide room impulse response (RIR) simulation running on PyTorch.
- Meet the minimum common features of existing tools (gpuRIR, das_generator, pyroomacoustics, rir-generator), while making **dynamic scenes** and **CPU/GPU switching** mandatory.

### Room Model
- Standard: shoebox (rectangular) room model.
- Support both 2D and 3D.
- Use the Image Source Method (ISM) as the baseline algorithm.

### Required Inputs
#### Scene Geometry
- Room size: `L = [Lx, Ly, Lz]` (2D uses `Lx, Ly`).
- Source positions: `src` (N×3 or N×2).
- Microphone positions: `mic` (M×3 or M×2).
- Reflection order: `order` (maximum reflection count).

#### Acoustic Parameters
- Sample rate: `fs`.
- Speed of sound: `c` (defaults allowed).
- Wall reflection coefficients: `beta` (6 faces) **or** `T60` (derived via Sabine).

#### Output Length
- Specify `nsample` (samples) **or** `Tmax` (seconds).

#### Directivity
- Minimum support for `omni`.
- Provide `cardioid / hypercardioid / subcardioid / bidir` for compatibility.
- Orientation specified by vector or angles.

### Outputs
- RIR as a Torch Tensor.
- Shape explicitly defined (e.g., `(n_src, n_mic, n_sample)`).
- Preserve dtype/device (float32/float16, CPU/GPU).

### Core APIs (Proposed)
#### Static RIR
```python
room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9, 0.9, 0.8, 0.8, 0.7, 0.7])
sources = Source.positions([[1.0, 2.0, 1.5], [4.5, 1.0, 1.2]])
mics = MicrophoneArray.positions([[2.0, 2.0, 1.5], [3.0, 2.0, 1.5]])

rir = simulate_rir(
    room=room,
    sources=sources,
    mics=mics,
    max_order=order,
    nsample=nsample_or_Tmax,
    directivity=directivity,
    orientation=orientation,
    device=device,
)
```

#### Dynamic Scenes (Required)
- Provide functionality equivalent to gpuRIR for time-varying RIRs.
- When source/mic positions change over time, generate time-varying RIRs.
```python
rir_t = simulate_dynamic_rir(
    room=room,
    src_traj=src_traj,   # (T, N, 3)
    mic_traj=mic_traj,   # (T, M, 3)
    max_order=order,
    nsample=nsample_or_Tmax,
    directivity=directivity,
    orientation=orientation,
    device=device,
)
```
- Consider an extension to convolve time-varying RIRs with anechoic signals to synthesize dynamic microphone signals.

### Device Control (Required)
- Allow explicit CPU/GPU switching.
- `device` must accept `"cpu"` or `"cuda"`.
- Follow PyTorch device/dtype semantics and support batching.

### Non-Functional Requirements
- Reproducibility: same inputs yield same outputs.
- Batch processing for multiple sources/mics.
- Performance: GPU acceleration where available.

### Future Compatibility Extensions
- Diffuse reverberation tail (ISM → diffuse model switch).
- Frequency-dependent absorption.
- ISM + ray tracing hybrid approach.
