from __future__ import annotations

"""Static convolution utilities."""

import torch
from torch import Tensor

from .internal import _ensure_signal, _ensure_static_rirs


def fft_convolve(signal: Tensor, rir: Tensor) -> Tensor:
    """Convolve a 1D signal with a 1D RIR using FFT.

    Args:
        signal: 1D signal tensor.
        rir: 1D impulse response.

    Returns:
        1D tensor of length len(signal) + len(rir) - 1.

    Example:
        >>> y = fft_convolve(signal, rir)
    """
    if signal.ndim != 1 or rir.ndim != 1:
        raise ValueError("fft_convolve expects 1D tensors")
    n = signal.numel() + rir.numel() - 1
    fft_len = 1 << (n - 1).bit_length()
    sig_f = torch.fft.rfft(signal, n=fft_len)
    rir_f = torch.fft.rfft(rir, n=fft_len)
    out = torch.fft.irfft(sig_f * rir_f, n=fft_len)
    return out[:n]


def convolve_rir(signal: Tensor, rirs: Tensor) -> Tensor:
    """Convolve signals with static RIRs (supports multi-source/mic).

    Args:
        signal: (n_src, n_samples) or (n_samples,) tensor.
        rirs: (n_src, n_mic, rir_len) or compatible shape.

    Returns:
        (n_mic, n_samples + rir_len - 1) tensor or 1D for single mic.

    Example:
        >>> y = convolve_rir(signal, rirs)
    """
    signal = _ensure_signal(signal)
    rirs = _ensure_static_rirs(rirs)
    n_src, n_mic, rir_len = rirs.shape

    if signal.shape[0] not in (1, n_src):
        raise ValueError("signal source count does not match rirs")
    if signal.shape[0] == 1 and n_src > 1:
        signal = signal.expand(n_src, -1)

    out_len = signal.shape[1] + rir_len - 1
    out = torch.zeros((n_mic, out_len), dtype=signal.dtype, device=signal.device)

    for s in range(n_src):
        for m in range(n_mic):
            out[m] += fft_convolve(signal[s], rirs[s, m])

    return out.squeeze(0) if n_mic == 1 else out
