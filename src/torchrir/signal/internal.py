"""Internal tensor-shape helpers for convolution."""

from __future__ import annotations

from torch import Tensor


def _ensure_signal(signal: Tensor) -> Tensor:
    """Ensure signal has shape (n_src, n_samples)."""
    if signal.ndim == 1:
        return signal.unsqueeze(0)
    if signal.ndim == 2:
        return signal
    raise ValueError("signal must have shape (n_samples,) or (n_src, n_samples)")


def _ensure_static_rirs(rirs: Tensor) -> Tensor:
    """Normalize static RIR shapes to (n_src, n_mic, rir_len)."""
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
    if rirs.ndim == 2:
        return rirs.view(rirs.shape[0], 1, 1, rirs.shape[1])
    if rirs.ndim == 3:
        if signal.ndim == 2 and rirs.shape[1] == signal.shape[0]:
            return rirs.view(rirs.shape[0], rirs.shape[1], 1, rirs.shape[2])
        return rirs.view(rirs.shape[0], 1, rirs.shape[1], rirs.shape[2])
    if rirs.ndim == 4:
        return rirs
    raise ValueError(
        "rirs must have shape (T, rir_len), (T, n_mic, rir_len), "
        "(T, n_src, rir_len), or (T, n_src, n_mic, rir_len)"
    )
