"""Trajectory helpers for dynamic scenes."""

from __future__ import annotations

import torch
from torch import Tensor


def linear_trajectory(start: Tensor, end: Tensor, steps: int) -> Tensor:
    """Create a linear trajectory between start and end."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if start.shape != end.shape:
        raise ValueError("start and end must have matching shapes")
    if not start.is_floating_point() or not end.is_floating_point():
        start = start.to(torch.get_default_dtype())
        end = end.to(torch.get_default_dtype())
    if start.device != end.device:
        raise ValueError("start and end must be on the same device")
    if start.dtype != end.dtype:
        raise ValueError("start and end must use the same dtype")
    if not torch.all(torch.isfinite(start)) or not torch.all(torch.isfinite(end)):
        raise ValueError("start and end must contain finite values")
    weights = torch.linspace(0.0, 1.0, steps, device=start.device, dtype=start.dtype)
    return start.unsqueeze(0) + weights.reshape((-1,) + (1,) * start.ndim) * (
        end - start
    ).unsqueeze(0)
