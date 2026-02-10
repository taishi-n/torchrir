"""Diffuse tail modeling for ISM."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from ...models import Room
from ...util.acoustics import estimate_t60_from_beta


def _apply_diffuse_tail(
    rir: Tensor,
    room: Room,
    beta: Tensor,
    tdiff: float,
    tmax: float,
    *,
    seed: Optional[int] = None,
) -> Tensor:
    """Apply a diffuse reverberation tail after tdiff."""
    nsample = rir.shape[-1]
    tdiff_idx = min(nsample - 1, int(math.floor(tdiff * room.fs)))
    if tdiff_idx <= 0:
        return rir
    tail_len = nsample - tdiff_idx
    t = torch.arange(tail_len, device=rir.device, dtype=rir.dtype) / room.fs

    t60 = estimate_t60_from_beta(room.size, beta)
    if math.isinf(t60) or t60 <= 0:
        decay = torch.exp(-t * 3.0)
    else:
        tau = t60 / 6.9078
        decay = torch.exp(-t / tau)

    gen = torch.Generator(device=rir.device)
    gen.manual_seed(0 if seed is None else seed)
    noise = torch.randn(
        rir[..., tdiff_idx:].shape, device=rir.device, dtype=rir.dtype, generator=gen
    )
    scale = (
        torch.linalg.norm(rir[..., tdiff_idx - 1 : tdiff_idx], dim=-1, keepdim=True)
        + 1e-8
    )
    rir[..., tdiff_idx:] = noise * decay * scale
    return rir
