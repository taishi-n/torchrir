"""Fractional-delay accumulation for ISM."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from ...config import SimulationConfig

_SINC_LUT_CACHE: dict[tuple[int, int, str, torch.dtype], Tensor] = {}
_FDL_GRID_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}
_FDL_OFFSETS_CACHE: dict[tuple[int, str], Tensor] = {}
_FDL_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}
_AccumFn = Callable[[Tensor, Tensor, Tensor], None]
_ACCUM_BATCH_COMPILED: dict[tuple[str, torch.dtype, int, int, bool, int], _AccumFn] = {}


def _accumulate_rir(
    rir: Tensor, sample: Tensor, amplitude: Tensor, cfg: SimulationConfig
) -> None:
    """Accumulate fractional-delay contributions into the RIR tensor."""
    idx0 = torch.floor(sample).to(torch.int64)
    frac = sample - idx0.to(sample.dtype)

    n_mic, nsample = rir.shape
    fdl = cfg.frac_delay_length
    lut_gran = cfg.sinc_lut_granularity
    use_lut = cfg.use_lut and rir.device.type != "mps"

    dtype = amplitude.dtype
    n = _get_fdl_grid(fdl, device=rir.device, dtype=dtype)
    offsets = _get_fdl_offsets(fdl, device=rir.device)
    window = _get_fdl_window(fdl, device=rir.device, dtype=dtype)

    if use_lut:
        sinc_lut = _get_sinc_lut(fdl, lut_gran, device=rir.device, dtype=dtype)

    mic_offsets = (
        torch.arange(n_mic, device=rir.device, dtype=torch.int64) * nsample
    ).view(n_mic, 1, 1)
    rir_flat = rir.view(-1)

    chunk_size = cfg.accumulate_chunk_size
    n_img = idx0.shape[1]
    for start in range(0, n_img, chunk_size):
        end = min(start + chunk_size, n_img)
        idx = idx0[:, start:end]
        amp = amplitude[:, start:end]
        frac_m = frac[:, start:end]

        if use_lut:
            x_off_frac = (1.0 - frac_m) * lut_gran
            lut_gran_off = torch.floor(x_off_frac).to(torch.int64)
            x_off = x_off_frac - lut_gran_off.to(dtype)
            lut_pos = lut_gran_off[..., None] + (
                n[None, None, :].to(torch.int64) * lut_gran
            )

            s0 = torch.take(sinc_lut, lut_pos)
            s1 = torch.take(sinc_lut, lut_pos + 1)
            interp = s0 + x_off[..., None] * (s1 - s0)
            filt = interp * window[None, None, :]
        else:
            fdl2 = (fdl - 1) // 2
            t = n[None, None, :] - fdl2 - frac_m[..., None]
            filt = torch.sinc(t) * window[None, None, :]

        contrib = amp[..., None] * filt
        target = idx[..., None] + offsets[None, None, :]
        valid = (target >= 0) & (target < nsample)
        if not valid.any():
            continue

        target = target + mic_offsets
        target_flat = target[valid].to(torch.int64)
        values_flat = contrib[valid]
        rir_flat.scatter_add_(0, target_flat, values_flat)


def _accumulate_rir_batch(
    rir: Tensor, sample: Tensor, amplitude: Tensor, cfg: SimulationConfig
) -> None:
    """Accumulate fractional-delay contributions for all sources/mics."""
    fn = _get_accumulate_fn(cfg, rir.device, amplitude.dtype)
    return fn(rir, sample, amplitude)


def _accumulate_rir_batch_impl(
    rir: Tensor,
    sample: Tensor,
    amplitude: Tensor,
    *,
    fdl: int,
    lut_gran: int,
    use_lut: bool,
    chunk_size: int,
) -> None:
    """Implementation for batch accumulation (optionally compiled)."""
    idx0 = torch.floor(sample).to(torch.int64)
    frac = sample - idx0.to(sample.dtype)

    n_src, n_mic, nsample = rir.shape
    n_sm = n_src * n_mic
    idx0 = idx0.view(n_sm, -1)
    frac = frac.view(n_sm, -1)
    amplitude = amplitude.view(n_sm, -1)

    fdl2 = (fdl - 1) // 2

    n = _get_fdl_grid(fdl, device=rir.device, dtype=sample.dtype)
    offsets = _get_fdl_offsets(fdl, device=rir.device)
    window = _get_fdl_window(fdl, device=rir.device, dtype=sample.dtype)

    if use_lut:
        sinc_lut = _get_sinc_lut(fdl, lut_gran, device=rir.device, dtype=sample.dtype)

    sm_offsets = (
        torch.arange(n_sm, device=rir.device, dtype=torch.int64) * nsample
    ).view(n_sm, 1, 1)
    rir_flat = rir.view(-1)

    n_img = idx0.shape[1]
    for start in range(0, n_img, chunk_size):
        end = min(start + chunk_size, n_img)
        idx = idx0[:, start:end]
        amp = amplitude[:, start:end]
        frac_m = frac[:, start:end]

        if use_lut:
            x_off_frac = (1.0 - frac_m) * lut_gran
            lut_gran_off = torch.floor(x_off_frac).to(torch.int64)
            x_off = x_off_frac - lut_gran_off.to(sample.dtype)
            lut_pos = lut_gran_off[..., None] + (
                n[None, None, :].to(torch.int64) * lut_gran
            )

            s0 = torch.take(sinc_lut, lut_pos)
            s1 = torch.take(sinc_lut, lut_pos + 1)
            interp = s0 + x_off[..., None] * (s1 - s0)
            filt = interp * window[None, None, :]
        else:
            t = n[None, None, :] - fdl2 - frac_m[..., None]
            filt = torch.sinc(t) * window[None, None, :]

        contrib = amp[..., None] * filt
        target = idx[..., None] + offsets[None, None, :]
        valid = (target >= 0) & (target < nsample)
        if not valid.any():
            continue

        target = target + sm_offsets
        target_flat = target[valid].to(torch.int64)
        values_flat = contrib[valid]
        rir_flat.scatter_add_(0, target_flat, values_flat)


def _get_accumulate_fn(
    cfg: SimulationConfig, device: torch.device, dtype: torch.dtype
) -> _AccumFn:
    """Return an accumulation function with config-bound constants."""
    use_lut = cfg.use_lut and device.type != "mps"
    fdl = cfg.frac_delay_length
    lut_gran = cfg.sinc_lut_granularity
    chunk_size = cfg.accumulate_chunk_size

    def _fn(rir: Tensor, sample: Tensor, amplitude: Tensor) -> None:
        _accumulate_rir_batch_impl(
            rir,
            sample,
            amplitude,
            fdl=fdl,
            lut_gran=lut_gran,
            use_lut=use_lut,
            chunk_size=chunk_size,
        )

    if device.type not in ("cuda", "mps") or not cfg.use_compile:
        return _fn
    key = (str(device), dtype, fdl, lut_gran, use_lut, chunk_size)
    compiled = _ACCUM_BATCH_COMPILED.get(key)
    if compiled is None:
        compiled = torch.compile(_fn, dynamic=True)
        _ACCUM_BATCH_COMPILED[key] = compiled
    return compiled


def _get_fdl_grid(fdl: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (fdl, str(device), dtype)
    cached = _FDL_GRID_CACHE.get(key)
    if cached is None:
        cached = torch.arange(fdl, device=device, dtype=dtype)
        _FDL_GRID_CACHE[key] = cached
    return cached


def _get_fdl_offsets(fdl: int, *, device: torch.device) -> Tensor:
    key = (fdl, str(device))
    cached = _FDL_OFFSETS_CACHE.get(key)
    if cached is None:
        fdl2 = (fdl - 1) // 2
        cached = torch.arange(fdl, device=device, dtype=torch.int64) - fdl2
        _FDL_OFFSETS_CACHE[key] = cached
    return cached


def _get_fdl_window(fdl: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (fdl, str(device), dtype)
    cached = _FDL_WINDOW_CACHE.get(key)
    if cached is None:
        cached = torch.hann_window(fdl, periodic=False, device=device, dtype=dtype)
        _FDL_WINDOW_CACHE[key] = cached
    return cached


def _get_sinc_lut(
    fdl: int, lut_gran: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create a sinc lookup table for fractional delays."""
    key = (fdl, lut_gran, str(device), dtype)
    cached = _SINC_LUT_CACHE.get(key)
    if cached is not None:
        return cached
    fdl2 = (fdl - 1) // 2
    lut_size = (fdl + 1) * lut_gran + 1
    n = torch.linspace(-fdl2 - 1, fdl2 + 1, lut_size, device=device, dtype=dtype)
    cached = torch.sinc(n)
    _SINC_LUT_CACHE[key] = cached
    return cached
