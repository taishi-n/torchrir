"""Internal tensor-shape helpers for convolution."""

from __future__ import annotations

import torch
from torch import Tensor

from ..util._dtypes import (
    validate_materialized_tensor,
    validate_supported_float_dtype,
    validate_supported_float_tensor,
)


def _fft_work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return a dtype supported by FFT on every advertised backend."""

    validate_supported_float_dtype(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _fft_convolve_sources(signal: Tensor, rirs: Tensor) -> Tensor:
    """Convolve ``(sources, samples)`` with ``(sources, mics, taps)``."""

    return _fft_convolve_sources_work(signal, rirs).to(dtype=signal.dtype)


def _fft_convolve_sources_work(signal: Tensor, rirs: Tensor) -> Tensor:
    """Convolve sources and retain the backend-safe accumulation dtype."""

    output_length = int(signal.shape[-1] + rirs.shape[-1] - 1)
    fft_length = 1 << (output_length - 1).bit_length()
    work_dtype = _fft_work_dtype(signal.dtype)
    signal_spectrum = torch.fft.rfft(
        signal.to(dtype=work_dtype),
        n=fft_length,
        dim=-1,
    )
    rir_spectrum = torch.fft.rfft(
        rirs.to(dtype=work_dtype),
        n=fft_length,
        dim=-1,
    )
    convolution = torch.fft.irfft(
        signal_spectrum[:, None, :] * rir_spectrum,
        n=fft_length,
        dim=-1,
    )[..., :output_length]
    return convolution.sum(dim=0)


def _ensure_signal(signal: Tensor) -> Tensor:
    """Ensure signal has shape (n_src, n_samples)."""
    if not torch.is_tensor(signal):
        raise TypeError("signal must be a Tensor")
    validate_materialized_tensor(signal, name="signal")
    if signal.ndim == 1:
        return signal.unsqueeze(0)
    if signal.ndim == 2:
        return signal
    raise ValueError("signal must have shape (n_samples,) or (n_src, n_samples)")


def _validate_convolution_dtypes(signal: Tensor, rirs: Tensor) -> None:
    validate_supported_float_tensor(signal, name="signal")
    validate_supported_float_tensor(rirs, name="rirs")


def _ensure_static_rirs(rirs: Tensor) -> Tensor:
    """Normalize static RIR shapes to (n_src, n_mic, rir_len)."""
    if not torch.is_tensor(rirs):
        raise TypeError("rirs must be a Tensor")
    validate_materialized_tensor(rirs, name="rirs")
    if rirs.ndim == 1:
        return rirs.view(1, 1, -1)
    if rirs.ndim == 2:
        return rirs.view(1, rirs.shape[0], rirs.shape[1])
    if rirs.ndim == 3:
        return rirs
    raise ValueError(
        "rirs must have shape (rir_len,), (n_mic, rir_len), or (n_src, n_mic, rir_len)"
    )


def _ensure_dynamic_rirs(rirs: Tensor, signal: Tensor) -> Tensor:
    """Normalize dynamic RIR shapes to (T, n_src, n_mic, rir_len)."""
    if not torch.is_tensor(rirs):
        raise TypeError("rirs must be a Tensor or RIRResult")
    validate_materialized_tensor(rirs, name="rirs")
    if rirs.ndim == 2:
        return rirs.view(rirs.shape[0], 1, 1, rirs.shape[1])
    if rirs.ndim == 3:
        if signal.ndim == 2 and signal.shape[0] != 1:
            raise ValueError(
                "3D dynamic RIRs are only supported for single-source signals "
                "and are interpreted as (T, n_mic, rir_len). "
                "Use 4D (T, n_src, n_mic, rir_len) for multi-source inputs."
            )
        return rirs.view(rirs.shape[0], 1, rirs.shape[1], rirs.shape[2])
    if rirs.ndim == 4:
        return rirs
    raise ValueError(
        "rirs must have shape (T, rir_len), (T, n_mic, rir_len), "
        "or (T, n_src, n_mic, rir_len)"
    )
