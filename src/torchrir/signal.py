from __future__ import annotations

import math

import torch
from torch import Tensor


def fft_convolve(signal: Tensor, rir: Tensor) -> Tensor:
    if signal.ndim != 1 or rir.ndim != 1:
        raise ValueError("fft_convolve expects 1D tensors")
    n = signal.numel() + rir.numel() - 1
    fft_len = 1 << (n - 1).bit_length()
    sig_f = torch.fft.rfft(signal, n=fft_len)
    rir_f = torch.fft.rfft(rir, n=fft_len)
    out = torch.fft.irfft(sig_f * rir_f, n=fft_len)
    return out[:n]


def convolve_rir(signal: Tensor, rirs: Tensor) -> Tensor:
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


def convolve_dynamic_rir(
    signal: Tensor,
    rirs: Tensor,
    hop: int | None = None,
    *,
    timestamps: Tensor | None = None,
    fs: float | None = None,
) -> Tensor:
    if hop is not None and hop <= 0:
        raise ValueError("hop must be positive")
    if hop is not None and timestamps is not None:
        raise ValueError("use either hop or timestamps, not both")

    signal = _ensure_signal(signal)
    rirs = _ensure_dynamic_rirs(rirs, signal)
    t_steps, n_src, n_mic, rir_len = rirs.shape

    if signal.shape[0] not in (1, n_src):
        raise ValueError("signal source count does not match rirs")
    if signal.shape[0] == 1 and n_src > 1:
        signal = signal.expand(n_src, -1)

    if hop is not None:
        return _convolve_dynamic_rir_hop(signal, rirs, hop)

    return _convolve_dynamic_rir_trajectory(signal, rirs, timestamps=timestamps, fs=fs)


def dynamic_convolve(
    signal: Tensor,
    rirs: Tensor,
    hop: int | None = None,
    *,
    timestamps: Tensor | None = None,
    fs: float | None = None,
) -> Tensor:
    return convolve_dynamic_rir(signal, rirs, hop, timestamps=timestamps, fs=fs)


def _convolve_dynamic_rir_hop(signal: Tensor, rirs: Tensor, hop: int) -> Tensor:
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
    n_samples = signal.shape[1]
    t_steps, n_src, n_mic, rir_len = rirs.shape

    if timestamps is not None:
        if fs is None:
            raise ValueError("fs must be provided when timestamps are used")
        ts = torch.as_tensor(timestamps, device=signal.device, dtype=torch.float64)
        if ts.ndim != 1 or ts.numel() != t_steps:
            raise ValueError("timestamps must be 1D and match number of RIR steps")
        if ts[0].item() != 0:
            raise ValueError("first timestamp must be 0")
        w_ini = (ts * fs).to(torch.long)
    else:
        step_fs = n_samples / t_steps
        w_ini = (torch.arange(t_steps, device=signal.device, dtype=torch.float64) * step_fs).to(
            torch.long
        )

    w_ini = torch.cat(
        [w_ini, torch.tensor([n_samples], device=signal.device, dtype=torch.long)]
    )
    w_len = w_ini[1:] - w_ini[:-1]
    max_len = int(w_len.max().item())

    segments = torch.zeros((t_steps, n_src, max_len), dtype=signal.dtype, device=signal.device)
    for t in range(t_steps):
        start = int(w_ini[t].item())
        end = int(w_ini[t + 1].item())
        if end > start:
            segments[t, :, : end - start] = signal[:, start:end]

    out = torch.zeros((n_mic, n_samples + rir_len - 1), dtype=signal.dtype, device=signal.device)

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


def _ensure_signal(signal: Tensor) -> Tensor:
    if signal.ndim == 1:
        return signal.unsqueeze(0)
    if signal.ndim == 2:
        return signal
    raise ValueError("signal must have shape (n_samples,) or (n_src, n_samples)")


def _ensure_static_rirs(rirs: Tensor) -> Tensor:
    if rirs.ndim == 1:
        return rirs.view(1, 1, -1)
    if rirs.ndim == 2:
        return rirs.view(1, rirs.shape[0], rirs.shape[1])
    if rirs.ndim == 3:
        return rirs
    raise ValueError("rirs must have shape (rir_len,), (n_mic, rir_len), or (n_src, n_mic, rir_len)")


def _ensure_dynamic_rirs(rirs: Tensor, signal: Tensor) -> Tensor:
    if rirs.ndim == 2:
        return rirs.view(rirs.shape[0], 1, 1, rirs.shape[1])
    if rirs.ndim == 3:
        if signal.ndim == 2 and rirs.shape[1] == signal.shape[0]:
            return rirs.view(rirs.shape[0], rirs.shape[1], 1, rirs.shape[2])
        return rirs.view(rirs.shape[0], 1, rirs.shape[1], rirs.shape[2])
    if rirs.ndim == 4:
        return rirs
    raise ValueError(
        "rirs must have shape (T, rir_len), (T, n_mic, rir_len), (T, n_src, rir_len), or (T, n_src, n_mic, rir_len)"
    )
