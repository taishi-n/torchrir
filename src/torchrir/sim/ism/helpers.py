"""Helper routines for ISM simulations."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ...models import MicrophoneArray, Room, Source
from ...util.acoustics import estimate_beta_from_t60
from ...util.orientation import normalize_orientation, orientation_to_unit
from ...util.tensor import as_tensor


def _prepare_entities(
    entities: Source | MicrophoneArray | Tensor,
    orientation: Optional[Tensor | tuple[Tensor, Tensor]],
    *,
    which: str,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> Tuple[Tensor, Optional[Tensor]]:
    """Extract positions and orientations from entities or raw tensors."""
    if isinstance(entities, (Source, MicrophoneArray)):
        pos = entities.positions
        ori = entities.orientation
    else:
        pos = entities
        ori = None
    if orientation is not None:
        if isinstance(orientation, (list, tuple)):
            if len(orientation) != 2:
                raise ValueError("orientation tuple must have length 2")
            ori = orientation[0] if which == "source" else orientation[1]
        else:
            ori = orientation
    pos = as_tensor(pos, device=device, dtype=dtype)
    if ori is not None:
        ori = as_tensor(ori, device=device, dtype=dtype)
    return pos, ori


def _resolve_beta(
    room: Room, room_size: Tensor, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Resolve reflection coefficients from beta/t60/defaults."""
    if room.beta is not None:
        return as_tensor(room.beta, device=device, dtype=dtype)
    if room.t60 is not None:
        return estimate_beta_from_t60(room_size, room.t60, device=device, dtype=dtype)
    dim = room_size.numel()
    default_faces = 4 if dim == 2 else 6
    return torch.ones((default_faces,), device=device, dtype=dtype)


def _validate_beta(beta: Tensor, dim: int) -> Tensor:
    """Validate beta size against room dimension."""
    expected = 4 if dim == 2 else 6
    if beta.numel() != expected:
        raise ValueError(f"beta must have {expected} elements for {dim}D")
    return beta


def _select_orientation(orientation: Tensor, idx: int, count: int, dim: int) -> Tensor:
    """Pick the correct orientation vector for a given entity index."""
    if orientation.ndim == 0:
        return orientation_to_unit(orientation, dim)
    if orientation.ndim == 1:
        return orientation_to_unit(orientation, dim)
    if orientation.ndim == 2 and orientation.shape[0] == count:
        return orientation_to_unit(orientation[idx], dim)
    raise ValueError("orientation must be shape (dim,), (count, dim), or angles")


def _cos_between(vec: Tensor, orientation: Tensor) -> Tensor:
    """Compute cosine between direction vectors and orientation."""
    orientation = normalize_orientation(orientation)
    unit = vec / torch.linalg.norm(vec, dim=-1, keepdim=True)
    return torch.sum(unit * orientation, dim=-1)
