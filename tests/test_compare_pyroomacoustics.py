import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torchrir import MicrophoneArray, Room, Source
from torchrir.config import SimulationConfig
from torchrir.sim import simulate_rir
from torchrir.signal import fft_convolve

pra = pytest.importorskip("pyroomacoustics")


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


def test_rir_and_convolved_signal_close():
    fs = 16000
    room_dim = [6.0, 4.0, 3.0]
    src = [1.0, 1.5, 1.2]
    mic = [3.0, 2.0, 1.2]
    max_order = 3
    beta = 0.9

    # pyroomacoustics RIR
    pra.constants.set("rir_hpf_enable", False)
    absorption = 1.0 - beta**2
    material = pra.Material(absorption) if hasattr(pra, "Material") else None
    pra_room = pra.ShoeBox(room_dim, fs=fs, max_order=max_order, materials=material)
    pra_room.add_source(src)
    mic_locs = torch.tensor(mic).view(3, 1).numpy()
    mic_array = pra.MicrophoneArray(mic_locs, fs)
    pra_room.add_microphone_array(mic_array)
    pra_room.compute_rir()
    pra_rir_np = np.array(pra_room.rir[0][0], dtype=np.float32, copy=True)
    pra_rir = torch.from_numpy(pra_rir_np)

    # torchrir RIR
    room = Room.shoebox(size=room_dim, fs=fs, beta=[beta] * 6)
    sources = Source.from_positions([src])
    mics = MicrophoneArray.from_positions([mic])
    torch_rir = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=max_order,
        nsample=pra_rir.numel(),
        directivity="omni",
        config=SimulationConfig(rir_hpf_enable=False),
    )[0, 0]

    # Align and compare RIRs
    a, b = _align_by_xcorr(torch_rir, pra_rir)
    rir_err = _rel_l2(a, b)
    assert rir_err < 0.5

    # Convolve a test signal and compare
    t = torch.linspace(0.0, 1.0, fs, dtype=torch.float32)
    test_sig = torch.sin(2.0 * math.pi * 440.0 * t)

    y_torch = fft_convolve(test_sig, a)
    y_pra = fft_convolve(test_sig, b)

    y_a, y_b = _align_by_xcorr(y_torch, y_pra)
    sig_err = _rel_l2(y_a, y_b)
    assert sig_err < 0.5
