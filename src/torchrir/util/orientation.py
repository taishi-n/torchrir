"""Orientation helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ._dtypes import validate_supported_float_tensor
from ._scalars import normalize_finite_real


def normalize_orientation(orientation: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize non-zero orientation vectors."""
    eps = normalize_finite_real(eps, name="eps", positive=True)
    if not torch.is_tensor(orientation):
        raise TypeError("orientation must be a Tensor")
    validate_supported_float_tensor(orientation, name="orientation")
    if orientation.ndim == 0 or orientation.shape[-1] == 0:
        raise ValueError("orientation vectors must be non-empty")
    if not torch.all(torch.isfinite(orientation)):
        raise ValueError("orientation must contain finite values")
    work_dtype = torch.float64 if orientation.dtype == torch.float64 else torch.float32
    work = orientation.to(dtype=work_dtype)
    scale = torch.amax(torch.abs(work), dim=-1, keepdim=True)
    zero_scale = scale == 0
    safe_scale = torch.where(zero_scale, torch.ones_like(scale), scale)
    scaled = work / safe_scale
    scaled_norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True)
    if torch.any(zero_scale | (scale <= eps / scaled_norm)):
        raise ValueError("orientation vectors must be non-zero")
    normalized = (scaled / scaled_norm).to(dtype=orientation.dtype)
    if not torch.all(torch.isfinite(normalized)):
        raise ValueError("normalized orientation must contain finite values")
    return normalized


def orientation_to_unit(orientation: Tensor, dim: int) -> Tensor:
    """Convert unambiguous angle/vector representations to unit vectors.

    In 2D, angles are scalar or have shape ``(..., 1)`` and vectors have shape
    ``(..., 2)``. In 3D, azimuth/elevation pairs have shape ``(..., 2)`` and
    vectors have shape ``(..., 3)``. A one-dimensional 2D tensor with two
    elements is therefore always one vector, never two per-entity angles.
    """
    if dim == 2:
        if orientation.ndim == 0:
            angle = orientation
            vec = torch.stack([torch.cos(angle), torch.sin(angle)])
            return normalize_orientation(vec)
        if orientation.ndim >= 1 and orientation.shape[-1] == 1:
            angle = orientation.squeeze(-1)
            vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            return normalize_orientation(vec)
        if orientation.ndim >= 1 and orientation.shape[-1] == 2:
            return normalize_orientation(orientation)
        raise ValueError(
            "2D orientation must be a scalar/(..., 1) angle or (..., 2) vector"
        )
    if dim == 3:
        if orientation.ndim >= 1 and orientation.shape[-1] == 3:
            return normalize_orientation(orientation)
        if orientation.ndim >= 1 and orientation.shape[-1] == 2:
            azimuth = orientation[..., 0]
            elevation = orientation[..., 1]
            x = torch.cos(elevation) * torch.cos(azimuth)
            y = torch.cos(elevation) * torch.sin(azimuth)
            z = torch.sin(elevation)
            vec = torch.stack([x, y, z], dim=-1)
            return normalize_orientation(vec)
        raise ValueError("3D orientation must be vector or (azimuth, elevation)")
    raise ValueError("unsupported dimension for orientation")
