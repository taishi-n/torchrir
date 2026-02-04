import pytest
import torch

from torchrir import MicrophoneArray, Room, Source, simulate_rir
from torchrir.config import activate_lut, get_config


def _compute(device: str) -> torch.Tensor:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.2, 1.0]])
    mics = MicrophoneArray.positions([[2.5, 2.0, 1.0]])
    rir = simulate_rir(room=room, sources=sources, mics=mics, max_order=2, tmax=0.05, device=device)
    return rir.detach().cpu()


def _compute_dynamic(device: str) -> torch.Tensor:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.2, 1.0]])
    mics = MicrophoneArray.positions([[2.5, 2.0, 1.0]])
    steps = 4
    src_traj = sources.positions.unsqueeze(0).repeat(steps, 1, 1)
    mic_start = torch.tensor([2.5, 2.0, 1.0])
    mic_end = torch.tensor([3.0, 2.2, 1.0])
    mic_traj = torch.stack(
        [mic_start + (mic_end - mic_start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    ).unsqueeze(1)
    from torchrir import simulate_dynamic_rir

    drir = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=2,
        tmax=0.05,
        device=device,
    )
    return drir.detach().cpu()


def test_rir_cpu_vs_cuda_close():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev = get_config()["use_lut"]
    activate_lut(False)
    try:
        cpu = _compute("cpu")
        gpu = _compute("cuda")
        assert torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-5)
    finally:
        activate_lut(prev)


def test_rir_cpu_vs_mps_close():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    prev = get_config()["use_lut"]
    activate_lut(False)
    try:
        cpu = _compute("cpu")
        mps = _compute("mps")
        assert torch.allclose(cpu, mps, rtol=1e-3, atol=1e-4)
    finally:
        activate_lut(prev)


def test_dynamic_rir_cpu_vs_cuda_close():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    prev = get_config()["use_lut"]
    activate_lut(False)
    try:
        cpu = _compute_dynamic("cpu")
        gpu = _compute_dynamic("cuda")
        assert torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-5)
    finally:
        activate_lut(prev)


def test_dynamic_rir_cpu_vs_mps_close():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    prev = get_config()["use_lut"]
    activate_lut(False)
    try:
        cpu = _compute_dynamic("cpu")
        mps = _compute_dynamic("mps")
        assert torch.allclose(cpu, mps, rtol=1e-3, atol=1e-4)
    finally:
        activate_lut(prev)
