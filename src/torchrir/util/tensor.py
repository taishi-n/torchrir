"""Tensor helpers."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
from torch import Tensor

from .device import resolve_device


def as_tensor(
    value: Tensor | Iterable[float] | Iterable[Iterable[float]] | float | int,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a value to a tensor while preserving device/dtype when possible."""
    if isinstance(device, str):
        device = resolve_device(device)
    if torch.is_tensor(value):
        out = value
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch.as_tensor(value, device=device, dtype=dtype)


def ensure_dim(size: Tensor) -> Tensor:
    """Validate room size dimensionality (2D or 3D)."""
    if size.ndim != 1 or size.numel() not in (2, 3):
        raise ValueError("room size must be a 1D tensor of length 2 or 3")
    return size


def extend_size(size: Tensor, dim: int) -> Tensor:
    """Extend 2D room size to 3D by adding a dummy z dimension."""
    if size.numel() == dim:
        return size
    if size.numel() == 2 and dim == 3:
        pad = torch.tensor([1.0], device=size.device, dtype=size.dtype)
        return torch.cat([size, pad])
    raise ValueError("unsupported room dimension")
