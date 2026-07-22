import importlib
import os
from typing import Any, cast

import numpy as np
import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.signal import DynamicConvolver, FrameSchedule
from torchrir.sim import simulate


def _import_gpurir() -> Any | None:
    try:
        return importlib.import_module("gpuRIR")
    except Exception:
        return None


gpurir = _import_gpurir()
if gpurir is None and os.environ.get("TORCHRIR_REQUIRE_COMPARISON") == "1":
    raise ImportError("gpuRIR is required for this comparison job")

# NOTE:
# torchrir and gpuRIR use different free-field amplitude normalization conventions.
# torchrir uses gain proportional to 1/r, while gpuRIR is typically interpreted as
# using 1/(4*pi*r). With matched geometry and reflections, raw waveform amplitudes
# can therefore differ by an almost constant factor close to 4*pi.
# Keep this in mind when interpreting direct waveform-L2 comparison results.
_GPURIR_TO_TORCHRIR_AMP_SCALE = float(4.0 * np.pi)
_GPURIR_FRACTIONAL_DELAY_LENGTH = 129


def _configure_gpurir_for_comparison() -> None:
    if gpurir is None:
        return
    if hasattr(gpurir, "activateLUT"):
        gpurir.activateLUT(False)
    if hasattr(gpurir, "activateMixedPrecision"):
        gpurir.activateMixedPrecision(False)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.linalg.norm(a - b).item() / (torch.linalg.norm(b).item() + 1e-8)


def _lag_samples(a: torch.Tensor, b: torch.Tensor) -> int:
    correlation = np.correlate(
        a.detach().cpu().numpy(), b.detach().cpu().numpy(), "full"
    )
    return int(np.argmax(np.abs(correlation)) - (b.numel() - 1))


def _to_tensor_rir(rir: np.ndarray, *, n_src: int, n_mic: int) -> torch.Tensor:
    out = np.asarray(rir, dtype=np.float32)
    if out.ndim != 3:
        raise AssertionError(f"gpuRIR returned unexpected shape: {out.shape}")
    if out.shape[0] == n_src and out.shape[1] == n_mic:
        pass
    elif out.shape[0] == n_mic and out.shape[1] == n_src:
        out = np.transpose(out, (1, 0, 2))
    else:
        raise AssertionError(f"gpuRIR returned unexpected shape: {out.shape}")
    return torch.from_numpy(out)


def _simulate_gpurir_static(
    *,
    room_dim: list[float],
    beta: list[float],
    src: list[float],
    mic: list[float],
    nb_img: list[int],
    tmax: float,
    fs: int,
) -> torch.Tensor:
    if gpurir is None:
        raise RuntimeError("gpuRIR is not installed")
    gpurir_mod = cast(Any, gpurir)
    room_sz = np.asarray(room_dim, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    pos_src = np.asarray(src, dtype=np.float32).reshape(1, 3)
    pos_rcv = np.asarray(mic, dtype=np.float32).reshape(1, 3)
    nb_img_np = np.asarray(nb_img, dtype=np.int32)
    rir = gpurir_mod.simulateRIR(
        room_sz,
        beta_np,
        pos_src,
        pos_rcv,
        nb_img_np,
        float(tmax),
        float(fs),
    )
    return _to_tensor_rir(rir, n_src=1, n_mic=1)


def _simulate_gpurir_dynamic(
    *,
    room_dim: list[float],
    beta: list[float],
    src_traj: torch.Tensor,
    mic: list[float],
    nb_img: list[int],
    tmax: float,
    fs: int,
) -> torch.Tensor:
    frames: list[torch.Tensor] = []
    for src_pos in src_traj:
        frame = _simulate_gpurir_static(
            room_dim=room_dim,
            beta=beta,
            src=src_pos.tolist(),
            mic=mic,
            nb_img=nb_img,
            tmax=tmax,
            fs=fs,
        )
        frames.append(frame)
    return torch.stack(frames, dim=0)


def _convert_gpurir_amplitude_to_torchrir(rir: torch.Tensor) -> torch.Tensor:
    # Convert from gpuRIR's typical 1/(4*pi*r) convention to torchrir's 1/r.
    return rir * _GPURIR_TO_TORCHRIR_AMP_SCALE


@pytest.mark.comparison
@pytest.mark.cuda
@pytest.mark.numerical
def test_static_direct_path_matches_gpurir_with_explicit_conventions():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if gpurir is None:
        pytest.skip("gpuRIR not installed")
    _configure_gpurir_for_comparison()

    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    src = [1.0, 1.5, 1.2]
    direct_delay_samples = 100
    mic = [src[0] + direct_delay_samples * 343.0 / fs, src[1], src[2]]
    beta = [0.9] * 6
    tmax = 0.025

    # gpuRIR's nb_img is the total number of images per axis, whereas TorchRIR's
    # value is the half-width. One image and zero half-width both mean direct-only.
    gpurir_nb_img = [1, 1, 1]
    torchrir_nb_img = (0, 0, 0)

    gpurir_rir = _simulate_gpurir_static(
        room_dim=room_dim,
        beta=beta,
        src=src,
        mic=mic,
        nb_img=gpurir_nb_img,
        tmax=tmax,
        fs=fs,
    )
    gpurir_rir = _convert_gpurir_amplitude_to_torchrir(gpurir_rir)

    room = Room.shoebox(size=room_dim, fs=fs, beta=beta)
    sources = Source.from_positions([src])
    mics = MicrophoneArray.from_positions([mic])
    torch_rir = simulate(
        StaticScene(room=room, sources=sources, mics=mics),
        SimulationConfig(
            nb_img=torchrir_nb_img,
            nsample=gpurir_rir.shape[-1],
            device="cuda",
            use_lut=False,
            frac_delay_length=_GPURIR_FRACTIONAL_DELAY_LENGTH,
        ),
    ).rirs

    actual = torch_rir[0, 0].cpu()
    expected = gpurir_rir[0, 0]
    assert _lag_samples(actual, expected) == 0
    assert int(torch.argmax(torch.abs(actual)).item()) == direct_delay_samples
    assert _rel_l2(actual, expected) < 5e-3


@pytest.mark.comparison
@pytest.mark.cuda
@pytest.mark.numerical
def test_dynamic_direct_path_frames_match_gpurir():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if gpurir is None:
        pytest.skip("gpuRIR not installed")
    _configure_gpurir_for_comparison()

    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    beta = [0.9] * 6
    tmax = 0.025
    gpurir_nb_img = [1, 1, 1]
    torchrir_nb_img = (0, 0, 0)
    steps = 5
    direct_delay_samples = torch.arange(90, 140, 10, dtype=torch.float32)
    fixed_mic_position = [4.0, 2.0, 1.2]
    src_path = torch.stack(
        [
            fixed_mic_position[0] - direct_delay_samples * 343.0 / fs,
            torch.full((steps,), fixed_mic_position[1]),
            torch.full((steps,), fixed_mic_position[2]),
        ],
        dim=1,
    )
    moving_source_traj_for_torchrir = src_path.unsqueeze(1)
    fixed_mic_traj_for_torchrir = (
        torch.tensor(fixed_mic_position, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(steps, 1)
        .unsqueeze(1)
    )

    # gpuRIR dynamic comparison is intentionally limited to one moving source
    # with a fixed microphone, which is the supported/straightforward setup.
    gpurir_rirs = _simulate_gpurir_dynamic(
        room_dim=room_dim,
        beta=beta,
        src_traj=src_path,
        mic=fixed_mic_position,
        nb_img=gpurir_nb_img,
        tmax=tmax,
        fs=fs,
    )
    gpurir_rirs = _convert_gpurir_amplitude_to_torchrir(gpurir_rirs)

    room = Room.shoebox(size=room_dim, fs=fs, beta=beta)
    scene = DynamicScene(
        room=room,
        sources=Source.from_positions(moving_source_traj_for_torchrir[0]),
        mics=MicrophoneArray.from_positions(fixed_mic_traj_for_torchrir[0]),
        src_traj=moving_source_traj_for_torchrir,
        mic_traj=fixed_mic_traj_for_torchrir,
    )
    torch_rirs = simulate(
        scene,
        SimulationConfig(
            nb_img=torchrir_nb_img,
            nsample=gpurir_rirs.shape[-1],
            device="cuda",
            use_lut=False,
            frac_delay_length=_GPURIR_FRACTIONAL_DELAY_LENGTH,
        ),
    ).rirs

    errs: list[float] = []
    for t in range(steps):
        actual = torch_rirs[t, 0, 0].cpu()
        expected = gpurir_rirs[t, 0, 0]
        assert _lag_samples(actual, expected) == 0
        assert int(torch.argmax(torch.abs(actual)).item()) == int(
            direct_delay_samples[t].item()
        )
        errs.append(_rel_l2(actual, expected))
    mean_err = float(np.mean(errs))
    assert mean_err < 5e-3


@pytest.mark.comparison
@pytest.mark.cuda
@pytest.mark.numerical
@pytest.mark.parametrize("custom_timestamps", [False, True])
def test_trajectory_convolution_matches_gpurir_for_identical_rirs(
    custom_timestamps: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if gpurir is None:
        pytest.skip("gpuRIR not installed")
    _configure_gpurir_for_comparison()

    steps = 5
    rir_length = 33
    n_mics = 2
    synthetic_rirs = np.zeros((steps, n_mics, rir_length), dtype=np.float32)
    for step in range(steps):
        synthetic_rirs[step, 0, 2 + step] = 1.0 - 0.1 * step
        synthetic_rirs[step, 0, 20 - step] = -0.25
        synthetic_rirs[step, 1, 5 + step] = -0.4 + 0.05 * step
        synthetic_rirs[step, 1, 24 - step] = 0.125

    source_signal = np.random.default_rng(123).standard_normal(1024).astype(np.float32)
    gpurir_mod = cast(Any, gpurir)
    fs = 8000
    timestamps = (
        np.asarray([0.0, 0.013, 0.031, 0.057, 0.091], dtype=np.float64)
        if custom_timestamps
        else None
    )
    expected_signal = gpurir_mod.simulateTrajectory(
        source_signal,
        synthetic_rirs,
        timestamps=timestamps,
        fs=fs if timestamps is not None else None,
    )
    torch_rirs = torch.from_numpy(synthetic_rirs[:, None, :, :]).cuda()
    schedule = (
        FrameSchedule.uniform(
            frame_count=steps,
            stop_sample=source_signal.size,
        )
        if timestamps is None
        else FrameSchedule.from_seconds(torch.from_numpy(timestamps), sample_rate=fs)
    )
    actual_signal = (
        DynamicConvolver(time_reference="emission")
        .convolve(
            torch.from_numpy(source_signal).cuda(),
            torch_rirs,
            schedule=schedule,
        )
        .cpu()
    )
    expected_signal_t = torch.from_numpy(
        np.asarray(expected_signal, dtype=np.float32).T.copy()
    )
    for microphone in range(n_mics):
        assert (
            _lag_samples(actual_signal[microphone], expected_signal_t[microphone]) == 0
        )
        assert _rel_l2(actual_signal[microphone], expected_signal_t[microphone]) < 2e-4
