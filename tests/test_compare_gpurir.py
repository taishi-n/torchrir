import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchrir import MicrophoneArray, Room, Source
from torchrir.config import SimulationConfig
from torchrir.sim import simulate_dynamic_rir, simulate_rir

try:
    import gpuRIR as gpurir
except Exception:
    gpurir = None

# NOTE:
# torchrir and gpuRIR use different free-field amplitude normalization conventions.
# torchrir uses gain proportional to 1/r, while gpuRIR is typically interpreted as
# using 1/(4*pi*r). With matched geometry and reflections, raw waveform amplitudes
# can therefore differ by an almost constant factor close to 4*pi.
# Keep this in mind when interpreting direct waveform-L2 comparison results.
_GPURIR_TO_TORCHRIR_AMP_SCALE = float(4.0 * np.pi)


def _configure_gpurir_for_comparison() -> None:
    if gpurir is None:
        return
    if hasattr(gpurir, "activateLUT"):
        gpurir.activateLUT(False)
    if hasattr(gpurir, "activateMixedPrecision"):
        gpurir.activateMixedPrecision(False)


def _align_by_xcorr(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    a = a.detach().cpu()
    b = b.detach().cpu()
    corr = F.conv1d(a.view(1, 1, -1), b.view(1, 1, -1), padding=b.numel() - 1)
    corr = corr.view(-1)
    idx = torch.argmax(torch.abs(corr)).item()
    lag = idx - (b.numel() - 1)
    if lag >= 0:
        a = a[lag:]
    else:
        b = b[-lag:]
    min_len = min(a.numel(), b.numel())
    return a[:min_len], b[:min_len]


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.linalg.norm(a - b).item() / (torch.linalg.norm(b).item() + 1e-8)


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
    room_sz = np.asarray(room_dim, dtype=np.float32)
    beta_np = np.asarray(beta, dtype=np.float32)
    pos_src = np.asarray(src, dtype=np.float32).reshape(1, 3)
    pos_rcv = np.asarray(mic, dtype=np.float32).reshape(1, 3)
    nb_img_np = np.asarray(nb_img, dtype=np.int32)
    rir = gpurir.simulateRIR(
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


def test_static_rir_close_to_gpurir():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if gpurir is None:
        pytest.skip("gpuRIR not installed")
    _configure_gpurir_for_comparison()

    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    src = [1.0, 1.5, 1.2]
    mic = [3.0, 2.0, 1.2]
    beta = [0.9] * 6
    tmax = 0.1
    nb_img = [3, 3, 3]

    gpurir_rir = _simulate_gpurir_static(
        room_dim=room_dim,
        beta=beta,
        src=src,
        mic=mic,
        nb_img=nb_img,
        tmax=tmax,
        fs=fs,
    )
    gpurir_rir = _convert_gpurir_amplitude_to_torchrir(gpurir_rir)

    room = Room.shoebox(size=room_dim, fs=fs, beta=beta)
    sources = Source.from_positions([src])
    mics = MicrophoneArray.from_positions([mic])
    # NOTE:
    # HPF is disabled here only to isolate core ISM parity against gpuRIR in this test.
    # This does NOT imply HPF should be disabled in normal usage.
    # In classic ISM practice, enabling HPF is commonly recommended, and many libraries
    # (e.g., pyroomacoustics, rir-generator) enable/use HPF in typical workflows.
    torch_rir = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=0,
        nb_img=tuple(nb_img),
        nsample=gpurir_rir.shape[-1],
        directivity="omni",
        device="cuda",
        config=SimulationConfig(rir_hpf_enable=False, use_lut=False),
    )

    a, b = _align_by_xcorr(torch_rir[0, 0], gpurir_rir[0, 0])
    rir_err = _rel_l2(a, b)
    assert rir_err < 0.5


def test_dynamic_rir_close_to_gpurir():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if gpurir is None:
        pytest.skip("gpuRIR not installed")
    _configure_gpurir_for_comparison()

    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    beta = [0.9] * 6
    tmax = 0.1
    nb_img = [3, 3, 3]
    steps = 5
    src_start = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float32)
    src_end = torch.tensor([2.5, 2.0, 1.2], dtype=torch.float32)
    alpha = torch.linspace(0.0, 1.0, steps, dtype=torch.float32).unsqueeze(1)
    src_path = src_start + (src_end - src_start) * alpha
    fixed_mic_position = [3.0, 2.0, 1.2]
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
        nb_img=nb_img,
        tmax=tmax,
        fs=fs,
    )
    gpurir_rirs = _convert_gpurir_amplitude_to_torchrir(gpurir_rirs)

    room = Room.shoebox(size=room_dim, fs=fs, beta=beta)
    # NOTE:
    # HPF is disabled here only to isolate core ISM parity against gpuRIR in this test.
    # This does NOT imply HPF should be disabled in normal usage.
    # In classic ISM practice, enabling HPF is commonly recommended, and many libraries
    # (e.g., pyroomacoustics, rir-generator) enable/use HPF in typical workflows.
    torch_rirs = simulate_dynamic_rir(
        room=room,
        src_traj=moving_source_traj_for_torchrir,
        mic_traj=fixed_mic_traj_for_torchrir,
        max_order=0,
        nb_img=tuple(nb_img),
        nsample=gpurir_rirs.shape[-1],
        directivity="omni",
        device="cuda",
        config=SimulationConfig(rir_hpf_enable=False, use_lut=False),
    )

    errs: list[float] = []
    for t in range(steps):
        a, b = _align_by_xcorr(torch_rirs[t, 0, 0], gpurir_rirs[t, 0, 0])
        errs.append(_rel_l2(a, b))
    mean_err = float(np.mean(errs))
    assert mean_err < 0.5
