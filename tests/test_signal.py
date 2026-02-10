import torch
import pytest

from torchrir.signal import DynamicConvolver, convolve_rir, fft_convolve


def test_fft_convolve_length():
    signal = torch.randn(128)
    rir = torch.randn(64)
    out = fft_convolve(signal, rir)
    assert out.shape[0] == signal.numel() + rir.numel() - 1


def test_dynamic_convolve_length():
    signal = torch.randn(1024)
    rirs = torch.randn(8, 64)
    out = DynamicConvolver(mode="hop", hop=256).convolve(signal, rirs)
    assert out.numel() >= signal.numel()


def test_convolve_rir_multi_mic():
    signal = torch.randn(2, 256)
    rirs = torch.randn(2, 3, 64)
    out = convolve_rir(signal, rirs)
    assert out.shape[0] == 3


def test_dynamic_convolver_multi_mic():
    signal = torch.randn(2, 512)
    rirs = torch.randn(6, 2, 3, 64)
    out = DynamicConvolver(mode="hop", hop=128).convolve(signal, rirs)
    assert out.shape[0] == 3


def test_dynamic_convolver_rejects_ambiguous_3d_rirs_for_multi_source():
    signal = torch.randn(2, 256)
    ambiguous_rirs = torch.randn(5, 2, 64)
    with pytest.raises(ValueError, match="Use 4D"):
        DynamicConvolver(mode="hop", hop=64).convolve(signal, ambiguous_rirs)
