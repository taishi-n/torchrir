"""Helper routines for ISM simulations."""

from __future__ import annotations

import torch
from torch import Tensor

from ...models import Room
from ...util.acoustics import estimate_beta_from_t60
from ...util.orientation import normalize_orientation
from ...util.tensor import as_float_tensor, stable_vector_norm


def _resolve_beta(
    room: Room, room_size: Tensor, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Resolve reflection coefficients from beta/t60/defaults."""
    if room.beta is not None:
        return as_float_tensor(
            room.beta, device=device, dtype=dtype, name="reflection coefficients"
        )
    if room.t60 is not None:
        return estimate_beta_from_t60(
            room_size,
            room.t60,
            c=room.c,
            device=device,
            dtype=dtype,
        )
    dim = room_size.numel()
    default_faces = 4 if dim == 2 else 6
    return torch.ones((default_faces,), device=device, dtype=dtype)


def _validate_beta(beta: Tensor, dim: int) -> Tensor:
    """Validate beta size against room dimension."""
    expected = 4 if dim == 2 else 6
    if beta.numel() != expected:
        raise ValueError(f"beta must have {expected} elements for {dim}D")
    return beta


def _cos_between(vec: Tensor, orientation: Tensor) -> Tensor:
    """Compute cosine between direction vectors and orientation."""
    orientation = normalize_orientation(orientation)
    scale = torch.amax(torch.abs(vec), dim=-1, keepdim=True)
    if torch.any(scale == 0) or not torch.all(torch.isfinite(scale)):
        raise ValueError("direction vectors must be finite and non-zero")
    scaled = vec / scale
    unit = scaled / stable_vector_norm(scaled, dim=-1, keepdim=True)
    return torch.clamp(torch.sum(unit * orientation, dim=-1), -1.0, 1.0)
