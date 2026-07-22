"""Diffuse tail modeling for ISM."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from ...util.acoustics import estimate_t60_from_beta
from ...util._scalars import normalize_finite_real

_RMS_WINDOW_SECONDS = 0.010
_CROSSFADE_SECONDS = 0.005
_AMPLITUDE_DECAY_60_DB = 3.0 * math.log(10.0)


def _apply_diffuse_tail(
    rir: Tensor,
    room_size: Tensor,
    beta: Tensor,
    tdiff: float,
    tmax: float,
    *,
    fs: float,
    c: float,
    seed: Optional[int] = None,
) -> Tensor:
    """Replace late ISM energy with a statistically continuous diffuse tail.

    Samples strictly before ``ceil(tdiff * fs)`` remain deterministic ISM. The
    tail level is the RMS energy in the preceding 10 ms, and a 5 ms
    power-complementary crossfade joins the two models. Dynamic RIR frames share
    one stochastic carrier per source/microphone pair; frame-specific RMS levels
    still follow the simulated geometry. This makes repeated geometry exactly
    repeatable and avoids unrelated late fields at adjacent trajectory frames.

    ``tdiff`` must be strictly positive because a zero-time handoff has no early
    field from which to estimate its level. Infinite Sabine T60 produces a
    non-decaying tail rather than an arbitrary fallback decay. ``tmax`` defines
    the modeled tail endpoint and must not exceed the supplied RIR tensor.
    """
    if not math.isfinite(tdiff) or tdiff <= 0:
        raise ValueError("tdiff must be positive")
    if not math.isfinite(tmax) or tmax <= tdiff:
        raise ValueError("tmax must be finite and greater than tdiff")
    fs = normalize_finite_real(fs, name="fs", positive=True)
    c = normalize_finite_real(c, name="c", positive=True)
    if not torch.all(torch.isfinite(rir)):
        raise ValueError("rir must contain finite values")

    nsample = rir.shape[-1]
    tmax_idx = _checked_ceil_sample_index(
        tmax,
        fs,
        maximum=nsample,
        name="tmax",
    )
    tdiff_idx = _checked_ceil_sample_index(
        tdiff,
        fs,
        maximum=tmax_idx,
        name="tdiff",
    )
    if tdiff_idx == 0:
        raise ValueError("tdiff must include at least one early-field sample")
    if tdiff_idx >= tmax_idx:
        raise ValueError("tdiff must leave at least one diffuse-tail sample")

    rms_samples = _capped_rounded_sample_count(
        _RMS_WINDOW_SECONDS,
        fs,
        maximum=tdiff_idx,
    )
    rms_start = max(0, tdiff_idx - rms_samples)
    early_window = rir[..., rms_start:tdiff_idx]
    window_scale = torch.amax(torch.abs(early_window), dim=-1, keepdim=True)
    safe_window_scale = torch.where(
        window_scale == 0,
        torch.ones_like(window_scale),
        window_scale,
    )
    scale = window_scale * torch.sqrt(
        torch.mean((early_window / safe_window_scale).square(), dim=-1, keepdim=True)
    )
    scale_is_zero = scale.squeeze(-1) == 0
    if torch.any(scale_is_zero):
        bad_indices = torch.nonzero(scale_is_zero, as_tuple=False).cpu().tolist()
        preview = bad_indices[:8]
        suffix = "..." if len(bad_indices) > len(preview) else ""
        raise ValueError(
            "diffuse handoff RMS has no usable energy at RIR indices "
            f"{preview}{suffix}; increase max_order/nb_img or choose an earlier tdiff"
        )

    tail_len = tmax_idx - tdiff_idx
    t = torch.arange(tail_len, device=rir.device, dtype=rir.dtype) / fs

    t60 = estimate_t60_from_beta(room_size, beta, c=c)
    if math.isinf(t60):
        decay = torch.ones_like(t)
    else:
        decay = torch.exp(-t * (_AMPLITUDE_DECAY_60_DB / t60))

    pair_shape = tuple(rir.shape[1:-1] if rir.ndim == 4 else rir.shape[:-1])
    noise = _pairwise_noise(
        pair_shape,
        tail_len,
        device=rir.device,
        dtype=rir.dtype,
        seed=0 if seed is None else seed,
    )
    if rir.ndim == 4:
        noise = noise.unsqueeze(0).expand(rir.shape[0], -1, -1, -1)
    diffuse = noise * decay * scale
    if not torch.all(torch.isfinite(diffuse)):
        raise ValueError("diffuse tail is not representable in the RIR dtype")

    output = rir.clone()
    blend_len = _capped_rounded_sample_count(
        _CROSSFADE_SECONDS,
        fs,
        maximum=tail_len,
    )
    if blend_len == 1:
        early_weight = torch.zeros(1, device=rir.device, dtype=rir.dtype)
        diffuse_weight = torch.ones(1, device=rir.device, dtype=rir.dtype)
    else:
        phase = torch.linspace(
            0.0,
            math.pi / 2.0,
            blend_len,
            device=rir.device,
            dtype=rir.dtype,
        )
        early_weight = torch.cos(phase)
        diffuse_weight = torch.sin(phase)
    output[..., tdiff_idx : tdiff_idx + blend_len] = (
        rir[..., tdiff_idx : tdiff_idx + blend_len] * early_weight
        + diffuse[..., :blend_len] * diffuse_weight
    )
    output[..., tdiff_idx + blend_len : tmax_idx] = diffuse[..., blend_len:]
    if not torch.all(torch.isfinite(output)):
        raise ValueError("diffuse RIR is not representable in its dtype")
    return output


def _checked_ceil_sample_index(
    seconds: float,
    fs: float,
    *,
    maximum: int,
    name: str,
) -> int:
    limit_seconds = maximum / fs
    if seconds > limit_seconds:
        raise ValueError(f"{name} must not exceed the supplied RIR sample range")
    product = seconds * fs
    if not math.isfinite(product):
        raise ValueError(f"{name} and fs must produce a finite sample index")
    index = math.ceil(product)
    if index > maximum:
        raise ValueError(f"{name} must not exceed the supplied RIR sample range")
    return index


def _capped_rounded_sample_count(
    seconds: float,
    fs: float,
    *,
    maximum: int,
) -> int:
    if maximum <= 1 or seconds >= maximum / fs:
        return max(1, maximum)
    return max(1, min(maximum, round(seconds * fs)))


def _pairwise_noise(
    pair_shape: tuple[int, ...],
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> Tensor:
    """Generate carriers stable across output horizons and batch composition."""

    if len(pair_shape) != 2:
        raise ValueError("diffuse RIRs must have source and microphone axes")
    n_sources, n_microphones = pair_shape
    noise = torch.empty(
        (n_sources, n_microphones, length),
        device=device,
        dtype=dtype,
    )
    modulus = (1 << 63) - 1
    for source_index in range(n_sources):
        for microphone_index in range(n_microphones):
            pair_seed = (
                seed + 1_000_003 * source_index + 97_409 * microphone_index
            ) % modulus
            generator = torch.Generator(device=device)
            generator.manual_seed(pair_seed)
            noise[source_index, microphone_index] = torch.randn(
                length,
                device=device,
                dtype=dtype,
                generator=generator,
            )
    return noise
