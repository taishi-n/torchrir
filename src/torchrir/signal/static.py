"""Static convolution utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from .internal import (
    _ensure_signal,
    _ensure_static_rirs,
    _fft_convolve_sources,
    _fft_work_dtype,
    _validate_convolution_dtypes,
)


def fft_convolve(signal: Tensor, rir: Tensor) -> Tensor:
    """Convolve a 1D signal with a 1D RIR using FFT.

    Args:
        signal: 1D signal tensor.
        rir: 1D impulse response.

    Returns:
        1D tensor of length len(signal) + len(rir) - 1.

    Examples:
        ```python
        y = fft_convolve(signal, rir)
        ```
    """
    if not torch.is_tensor(signal) or not torch.is_tensor(rir):
        raise TypeError("signal and rir must be Tensors")
    if signal.ndim != 1 or rir.ndim != 1:
        raise ValueError("fft_convolve expects 1D tensors")
    if signal.numel() == 0 or rir.numel() == 0:
        raise ValueError("signal and rir must be non-empty")
    _validate_convolution_dtypes(signal, rir)
    if signal.device != rir.device:
        raise ValueError("signal and rir must be on the same device")
    if signal.dtype != rir.dtype:
        raise ValueError("signal and rir must use the same dtype")
    n = signal.numel() + rir.numel() - 1
    fft_len = 1 << (n - 1).bit_length()
    work_dtype = _fft_work_dtype(signal.dtype)
    sig_f = torch.fft.rfft(signal.to(dtype=work_dtype), n=fft_len)
    rir_f = torch.fft.rfft(rir.to(dtype=work_dtype), n=fft_len)
    out = torch.fft.irfft(sig_f * rir_f, n=fft_len)
    return out[:n].to(dtype=signal.dtype)


def convolve_rir(signal: Tensor, rirs: Tensor) -> Tensor:
    """Convolve signals with static RIRs (supports multi-source/mic).

    Args:
        signal: (n_src, n_samples) or (n_samples,) tensor.
        rirs: ``(rir_len,)``, ``(n_mic, rir_len)``, or
            ``(n_src, n_mic, rir_len)`` tensor.

    Returns:
        ``(n_mic, n_samples + rir_len - 1)`` tensor. The microphone axis is
        retained when there is only one microphone.

    Examples:
        ```python
        y = convolve_rir(signal, rirs)
        ```
    """
    signal = _ensure_signal(signal)
    rirs = _ensure_static_rirs(rirs)
    n_src, n_mic, rir_len = rirs.shape

    if signal.numel() == 0 or rir_len == 0:
        raise ValueError("signal and rirs must be non-empty")
    if n_src == 0 or n_mic == 0:
        raise ValueError("rirs must contain at least one source and microphone")
    _validate_convolution_dtypes(signal, rirs)
    if signal.device != rirs.device:
        raise ValueError("signal and rirs must be on the same device")
    if signal.dtype != rirs.dtype:
        raise ValueError("signal and rirs must use the same dtype")

    if signal.shape[0] not in (1, n_src):
        raise ValueError("signal source count does not match rirs")
    if signal.shape[0] == 1 and n_src > 1:
        signal = signal.expand(n_src, -1)

    return _fft_convolve_sources(signal, rirs)
