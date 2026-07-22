import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate


def _compute(device: str, cfg: SimulationConfig) -> torch.Tensor:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.2, 1.0]])
    mics = MicrophoneArray.from_positions([[2.5, 2.0, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    result = simulate(scene, cfg.replace(device=device))
    return result.rirs.detach().cpu()


def _compute_dynamic(device: str, cfg: SimulationConfig) -> torch.Tensor:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.2, 1.0]])
    steps = 4
    src_traj = sources.positions.unsqueeze(0).repeat(steps, 1, 1)
    mic_start = torch.tensor([2.5, 2.0, 1.0])
    mic_end = torch.tensor([3.0, 2.2, 1.0])
    mic_traj = torch.stack(
        [mic_start + (mic_end - mic_start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    ).unsqueeze(1)
    microphones = MicrophoneArray.from_positions(mic_traj[0])
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=microphones,
        src_traj=src_traj,
        mic_traj=mic_traj,
    )
    result = simulate(scene, cfg.replace(device=device))
    return result.rirs.detach().cpu()


@pytest.mark.cuda
@pytest.mark.numerical
def test_rir_cpu_vs_cuda_close():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = SimulationConfig(max_order=2, tmax=0.05, use_lut=False)
    cpu = _compute("cpu", cfg)
    gpu = _compute("cuda", cfg)
    assert torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-5)


@pytest.mark.mps
@pytest.mark.numerical
def test_rir_cpu_vs_mps_close():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cfg = SimulationConfig(max_order=2, tmax=0.05, use_lut=False)
    cpu = _compute("cpu", cfg)
    mps = _compute("mps", cfg)
    assert torch.allclose(cpu, mps, rtol=1e-3, atol=1e-4)


@pytest.mark.cuda
@pytest.mark.numerical
def test_dynamic_rir_cpu_vs_cuda_close():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    cfg = SimulationConfig(max_order=2, tmax=0.05, use_lut=False)
    cpu = _compute_dynamic("cpu", cfg)
    gpu = _compute_dynamic("cuda", cfg)
    assert torch.allclose(cpu, gpu, rtol=1e-4, atol=1e-5)


@pytest.mark.mps
@pytest.mark.numerical
def test_dynamic_rir_cpu_vs_mps_close():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    cfg = SimulationConfig(max_order=2, tmax=0.05, use_lut=False)
    cpu = _compute_dynamic("cpu", cfg)
    mps = _compute_dynamic("mps", cfg)
    assert torch.allclose(cpu, mps, rtol=1e-3, atol=1e-4)
