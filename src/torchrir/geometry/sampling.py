"""Sampling helpers for scene geometry."""

from __future__ import annotations

import random
import sys
from typing import List

import torch
from torch import Tensor

from ..util.tensor import as_float_tensor
from ..util._scalars import normalize_finite_real, normalize_integer


_MAX_TENSOR_COUNT = min(sys.maxsize, torch.iinfo(torch.int64).max)


def sample_positions(
    *,
    num: int,
    room_size: Tensor,
    rng: random.Random,
    margin: float = 0.5,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Sample random positions within a room with a safety margin."""
    _validate_rng(rng)
    num = normalize_integer(
        num,
        name="num",
        minimum=0,
        maximum=_MAX_TENSOR_COUNT,
    )
    margin = normalize_finite_real(margin, name="margin", non_negative=True)
    room_size = as_float_tensor(
        room_size, device=device, dtype=dtype, name="room_size"
    ).reshape(-1)
    _validate_sampling_bounds(room_size=room_size, margin=margin)
    dim = room_size.numel()
    low = [margin] * dim
    high = [float(room_size[i].item()) - margin for i in range(dim)]
    coords: List[List[float]] = []
    for _ in range(num):
        point = [rng.uniform(low[i], high[i]) for i in range(dim)]
        coords.append(point)
    if not coords:
        return torch.empty((0, dim), device=room_size.device, dtype=room_size.dtype)
    return torch.tensor(coords, device=room_size.device, dtype=room_size.dtype)


def sample_positions_with_z_range(
    *,
    num: int,
    room_size: Tensor,
    rng: random.Random,
    z_range: tuple[float, float] = (1.5, 1.8),
    margin: float = 0.5,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Sample random positions with an explicit z-range constraint."""
    _validate_rng(rng)
    num = normalize_integer(
        num,
        name="num",
        minimum=0,
        maximum=_MAX_TENSOR_COUNT,
    )
    margin = normalize_finite_real(margin, name="margin", non_negative=True)
    room_size = as_float_tensor(
        room_size, device=device, dtype=dtype, name="room_size"
    ).reshape(-1)
    positions = sample_positions(
        num=num,
        room_size=room_size,
        rng=rng,
        margin=margin,
        device=room_size.device,
        dtype=room_size.dtype,
    )
    if room_size.numel() < 3:
        return positions
    z_min, z_max = _normalize_z_range(z_range)
    z_low = max(margin, z_min)
    z_high = min(float(room_size[2].item()) - margin, z_max)
    if z_high <= z_low:
        raise ValueError("z_range has no feasible values inside the room margin")
    z_vals = [rng.uniform(z_low, z_high) for _ in range(num)]
    positions[:, 2] = torch.tensor(
        z_vals, device=positions.device, dtype=positions.dtype
    )
    return positions


def sample_positions_min_distance(
    *,
    num: int,
    room_size: Tensor,
    rng: random.Random,
    center: Tensor,
    min_distance: float,
    z_range: tuple[float, float] | None = (1.5, 1.8),
    margin: float = 0.5,
    max_attempts: int = 1000,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Sample random positions with a minimum distance from a center point."""
    _validate_rng(rng)
    num = normalize_integer(
        num,
        name="num",
        minimum=0,
        maximum=_MAX_TENSOR_COUNT,
    )
    margin = normalize_finite_real(margin, name="margin", non_negative=True)
    min_distance = normalize_finite_real(
        min_distance,
        name="min_distance",
        non_negative=True,
    )
    max_attempts = normalize_integer(
        max_attempts,
        name="max_attempts",
        minimum=1,
        maximum=sys.maxsize,
    )
    room_size = as_float_tensor(
        room_size, device=device, dtype=dtype, name="room_size"
    ).reshape(-1)
    _validate_sampling_bounds(room_size=room_size, margin=margin)
    dim = room_size.numel()
    center = as_float_tensor(
        center, device=room_size.device, dtype=room_size.dtype, name="center"
    ).reshape(-1)
    if center.numel() != dim:
        raise ValueError("center dimension must match room_size.")
    if not torch.all(torch.isfinite(center)):
        raise ValueError("center must contain finite values")
    low = [margin] * dim
    high = [float(room_size[i].item()) - margin for i in range(dim)]
    z_bounds: tuple[float, float] | None = None
    if z_range is not None and dim >= 3:
        z_min, z_max = _normalize_z_range(z_range)
        z_low = max(margin, z_min)
        z_high = min(float(room_size[2].item()) - margin, z_max)
        if z_high <= z_low:
            raise ValueError("z_range has no feasible values inside the room margin")
        z_bounds = (z_low, z_high)
    coords: List[List[float]] = []
    attempts = 0
    while len(coords) < num and attempts < max_attempts:
        attempts += 1
        point = [rng.uniform(low[i], high[i]) for i in range(dim)]
        if z_bounds is not None:
            point[2] = rng.uniform(*z_bounds)
        point_t = torch.tensor(point, device=center.device, dtype=center.dtype)
        dist = torch.linalg.vector_norm(point_t - center).item()
        if dist >= min_distance:
            coords.append(point)
    if len(coords) < num:
        raise RuntimeError("failed to sample positions with requested minimum distance")
    if not coords:
        return torch.empty((0, dim), device=room_size.device, dtype=room_size.dtype)
    return torch.tensor(coords, device=room_size.device, dtype=room_size.dtype)


def clamp_positions(
    positions: Tensor, room_size: Tensor, margin: float = 0.1
) -> Tensor:
    """Clamp positions to remain inside the room with a margin."""
    margin = normalize_finite_real(margin, name="margin", non_negative=True)
    positions = as_float_tensor(positions, name="positions")
    room_size = as_float_tensor(
        room_size,
        device=positions.device,
        dtype=positions.dtype,
        name="room_size",
    )
    if room_size.ndim != 1 or room_size.numel() not in (2, 3):
        raise ValueError("room_size must be a 1D tensor of length 2 or 3")
    if positions.ndim == 0 or positions.shape[-1] != room_size.numel():
        raise ValueError("positions last dimension must match room_size")
    if not torch.all(torch.isfinite(positions)):
        raise ValueError("positions must contain finite values")
    if not torch.all(torch.isfinite(room_size)) or torch.any(room_size <= 0):
        raise ValueError("room_size must contain finite positive values")
    if torch.any(room_size <= 2 * margin):
        raise ValueError("margin leaves no feasible space inside the room")
    min_v = torch.full_like(room_size, margin)
    max_v = room_size - margin
    return torch.max(torch.min(positions, max_v), min_v)


def _validate_sampling_bounds(*, room_size: Tensor, margin: float) -> None:
    if room_size.ndim != 1 or room_size.numel() not in (2, 3):
        raise ValueError("room_size must be a 1D tensor of length 2 or 3")
    if not torch.all(torch.isfinite(room_size)) or torch.any(room_size <= 0):
        raise ValueError("room_size must contain finite positive values")
    if torch.any(room_size <= 2 * margin):
        raise ValueError("margin leaves no feasible sampling range")


def _normalize_z_range(value: object) -> tuple[float, float]:
    if isinstance(value, (str, bytes)):
        raise TypeError("z_range must be a pair of real numbers")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError("z_range must be a pair of real numbers") from exc
    if len(values) != 2:
        raise ValueError("z_range must contain exactly two values")
    return (
        normalize_finite_real(values[0], name="z_range lower bound"),
        normalize_finite_real(values[1], name="z_range upper bound"),
    )


def _validate_rng(value: object) -> None:
    if not isinstance(value, random.Random):
        raise TypeError("rng must be a random.Random instance")
