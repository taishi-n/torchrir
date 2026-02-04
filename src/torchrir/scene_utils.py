from __future__ import annotations

"""Scene-agnostic utilities for example setups."""

import random
from typing import List

import torch


def sample_positions(
    *,
    num: int,
    room_size: torch.Tensor,
    rng: random.Random,
    margin: float = 0.5,
) -> torch.Tensor:
    """Sample random positions within a room with a safety margin.

    Example:
        >>> rng = random.Random(0)
        >>> room = torch.tensor([6.0, 4.0, 3.0])
        >>> positions = sample_positions(num=2, room_size=room, rng=rng)
    """
    dim = room_size.numel()
    low = [margin] * dim
    high = [float(room_size[i].item()) - margin for i in range(dim)]
    coords: List[List[float]] = []
    for _ in range(num):
        point = [rng.uniform(low[i], high[i]) for i in range(dim)]
        coords.append(point)
    return torch.tensor(coords, dtype=torch.float32)


def linear_trajectory(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    """Create a linear trajectory between start and end.

    Example:
        >>> traj = linear_trajectory(torch.tensor([1.0, 1.0, 1.0]), torch.tensor([4.0, 2.0, 1.0]), 8)
    """
    return torch.stack(
        [start + (end - start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    )


def binaural_mic_positions(center: torch.Tensor, offset: float = 0.08) -> torch.Tensor:
    """Create a two-mic binaural layout around a center point.

    Example:
        >>> mics = binaural_mic_positions(torch.tensor([2.0, 2.0, 1.5]))
    """
    dim = center.numel()
    offset_vec = torch.zeros((dim,), dtype=torch.float32)
    offset_vec[0] = offset
    left = center - offset_vec
    right = center + offset_vec
    return torch.stack([left, right], dim=0)


def clamp_positions(positions: torch.Tensor, room_size: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Clamp positions to remain inside the room with a margin.

    Example:
        >>> clamped = clamp_positions(positions, torch.tensor([6.0, 4.0, 3.0]))
    """
    min_v = torch.full_like(room_size, margin)
    max_v = room_size - margin
    return torch.max(torch.min(positions, max_v), min_v)
