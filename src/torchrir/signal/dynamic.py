from __future__ import annotations

"""Dynamic convolution utilities.

DynamicConvolver is the public API for time-varying convolution. Lower-level
helpers live in internal modules and are not part of the stable surface.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .internal import _ensure_dynamic_rirs, _ensure_signal
from .static import fft_convolve


@dataclass(frozen=True)
class DynamicConvolver:
    """Convolver for time-varying RIRs.

    Example:
        >>> convolver = DynamicConvolver(mode="trajectory")
        >>> y = convolver.convolve(signal, rirs)
    """

    mode: str = "trajectory"
    hop: Optional[int] = None
    timestamps: Optional[Tensor] = None
    fs: Optional[float] = None

    def __call__(self, signal: Tensor, rirs: Tensor) -> Tensor:
        return self.convolve(signal, rirs)

    def convolve(self, signal: Tensor, rirs: Tensor) -> Tensor:
        """Convolve signals with time-varying RIRs.

        Example:
            >>> y = DynamicConvolver(mode="hop", hop=1024).convolve(signal, rirs)
        """
        if self.mode not in ("trajectory", "hop"):
            raise ValueError("mode must be 'trajectory' or 'hop'")
        if self.mode == "hop":
            if self.hop is None:
                raise ValueError("hop must be provided for hop mode")
            return _convolve_dynamic_hop(signal, rirs, self.hop)
        return _convolve_dynamic_trajectory(
            signal, rirs, timestamps=self.timestamps, fs=self.fs
        )


def _convolve_dynamic_hop(signal: Tensor, rirs: Tensor, hop: int) -> Tensor:
    signal = _ensure_signal(signal)
    rirs = _ensure_dynamic_rirs(rirs, signal)
    return _convolve_dynamic_rir_hop(signal, rirs, hop)


def _convolve_dynamic_trajectory(
    signal: Tensor,
    rirs: Tensor,
    *,
    timestamps: Optional[Tensor],
    fs: Optional[float],
) -> Tensor:
    signal = _ensure_signal(signal)
    rirs = _ensure_dynamic_rirs(rirs, signal)
    return _convolve_dynamic_rir_trajectory(signal, rirs, timestamps=timestamps, fs=fs)


def _convolve_dynamic_rir_hop(signal: Tensor, rirs: Tensor, hop: int) -> Tensor:
    """Dynamic convolution using fixed hop-size segments."""
    t_steps, n_src, n_mic, rir_len = rirs.shape

    frames = math.ceil(signal.shape[1] / hop)
    frames = min(frames, t_steps)

    out_len = hop * (frames - 1) + hop + rir_len - 1
    out = torch.zeros((n_mic, out_len), dtype=signal.dtype, device=signal.device)

    for t in range(frames):
        start = t * hop
        for s in range(n_src):
            frame = signal[s, start : start + hop]
            if frame.numel() == 0:
                continue
            for m in range(n_mic):
                seg = fft_convolve(frame, rirs[t, s, m])
                out[m, start : start + seg.numel()] += seg

    return out.squeeze(0) if n_mic == 1 else out


def _convolve_dynamic_rir_trajectory(
    signal: Tensor,
    rirs: Tensor,
    *,
    timestamps: Tensor | None,
    fs: float | None,
) -> Tensor:
    """Dynamic convolution using variable segments like gpuRIR simulateTrajectory."""
    n_samples = signal.shape[1]
    t_steps, n_src, n_mic, rir_len = rirs.shape

    if timestamps is not None:
        if fs is None:
            raise ValueError("fs must be provided when timestamps are used")
        ts = torch.as_tensor(
            timestamps,
            device=signal.device,
            dtype=torch.float32 if signal.device.type == "mps" else torch.float64,
        )
        if ts.ndim != 1 or ts.numel() != t_steps:
            raise ValueError("timestamps must be 1D and match number of RIR steps")
        if ts[0].item() != 0:
            raise ValueError("first timestamp must be 0")
        w_ini = (ts * fs).to(torch.long)
    else:
        step_fs = n_samples / t_steps
        ts_dtype = torch.float32 if signal.device.type == "mps" else torch.float64
        w_ini = (
            torch.arange(t_steps, device=signal.device, dtype=ts_dtype) * step_fs
        ).to(torch.long)

    w_ini = torch.cat(
        [w_ini, torch.tensor([n_samples], device=signal.device, dtype=torch.long)]
    )
    w_len = w_ini[1:] - w_ini[:-1]

    if signal.device.type in ("cuda", "mps"):
        return _convolve_dynamic_rir_trajectory_batched(
            signal, rirs, w_ini=w_ini, w_len=w_len
        )

    max_len = int(w_len.max().item())
    segments = torch.zeros(
        (t_steps, n_src, max_len), dtype=signal.dtype, device=signal.device
    )
    for t in range(t_steps):
        start = int(w_ini[t].item())
        end = int(w_ini[t + 1].item())
        if end > start:
            segments[t, :, : end - start] = signal[:, start:end]

    out = torch.zeros(
        (n_mic, n_samples + rir_len - 1), dtype=signal.dtype, device=signal.device
    )

    for t in range(t_steps):
        seg_len = int(w_len[t].item())
        if seg_len == 0:
            continue
        start = int(w_ini[t].item())
        for s in range(n_src):
            frame = segments[t, s, :seg_len]
            for m in range(n_mic):
                conv = fft_convolve(frame, rirs[t, s, m])
                out[m, start : start + seg_len + rir_len - 1] += conv

    return out.squeeze(0) if n_mic == 1 else out


def _convolve_dynamic_rir_trajectory_batched(
    signal: Tensor,
    rirs: Tensor,
    *,
    w_ini: Tensor,
    w_len: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    """GPU-friendly batched trajectory convolution using FFT."""
    n_samples = signal.shape[1]
    t_steps, n_src, n_mic, rir_len = rirs.shape
    out = torch.zeros(
        (n_mic, n_samples + rir_len - 1), dtype=signal.dtype, device=signal.device
    )

    for t0 in range(0, t_steps, chunk_size):
        t1 = min(t0 + chunk_size, t_steps)
        lengths = w_len[t0:t1]
        max_len = int(lengths.max().item())
        if max_len == 0:
            continue
        segments = torch.zeros(
            (t1 - t0, n_src, max_len), dtype=signal.dtype, device=signal.device
        )
        for idx, t in enumerate(range(t0, t1)):
            start = int(w_ini[t].item())
            end = int(w_ini[t + 1].item())
            if end > start:
                segments[idx, :, : end - start] = signal[:, start:end]

        conv_len = max_len + rir_len - 1
        fft_len = 1 << (conv_len - 1).bit_length()
        seg_f = torch.fft.rfft(segments, n=fft_len, dim=-1)
        rir_f = torch.fft.rfft(rirs[t0:t1], n=fft_len, dim=-1)
        conv_out = torch.empty(
            (t1 - t0, n_src, n_mic, fft_len),
            dtype=signal.dtype,
            device=signal.device,
        )
        conv = torch.fft.irfft(
            seg_f[:, :, None, :] * rir_f, n=fft_len, dim=-1, out=conv_out
        )
        conv = conv[..., :conv_len]
        conv_sum = conv.sum(dim=1)

        for idx, t in enumerate(range(t0, t1)):
            seg_len = int(lengths[idx].item())
            if seg_len == 0:
                continue
            start = int(w_ini[t].item())
            out[:, start : start + seg_len + rir_len - 1] += conv_sum[
                idx, :, : seg_len + rir_len - 1
            ]

    return out.squeeze(0) if n_mic == 1 else out
