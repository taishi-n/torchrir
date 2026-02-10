"""Orientation helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_orientation(orientation: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize orientation vectors with numerical stability."""
    norm = torch.linalg.norm(orientation, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return orientation / norm


def orientation_to_unit(orientation: Tensor, dim: int) -> Tensor:
    """Convert orientation representation to unit vectors in 2D/3D."""
    if dim == 2:
        if orientation.ndim == 0:
            angle = orientation
            vec = torch.stack([torch.cos(angle), torch.sin(angle)])
            return normalize_orientation(vec)
        if orientation.shape[-1] == 1:
            angle = orientation.squeeze(-1)
            vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            return normalize_orientation(vec)
        if orientation.ndim == 1 and orientation.numel() != 2:
            angle = orientation
            vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            return normalize_orientation(vec)
        if orientation.shape[-1] == 2:
            return normalize_orientation(orientation)
        raise ValueError("2D orientation must be angle or 2D vector")
    if dim == 3:
        if orientation.shape[-1] == 3:
            return normalize_orientation(orientation)
        if orientation.shape[-1] == 2:
            azimuth = orientation[..., 0]
            elevation = orientation[..., 1]
            x = torch.cos(elevation) * torch.cos(azimuth)
            y = torch.cos(elevation) * torch.sin(azimuth)
            z = torch.sin(elevation)
            vec = torch.stack([x, y, z], dim=-1)
            return normalize_orientation(vec)
        raise ValueError("3D orientation must be vector or (azimuth, elevation)")
    raise ValueError("unsupported dimension for orientation")
