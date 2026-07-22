"""Tensor helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Optional, cast

import numpy as np
import torch
from torch import Tensor

from ._dtypes import validate_supported_float_tensor
from .device import resolve_device


def as_tensor(
    value: (
        Tensor
        | Iterable[float]
        | Iterable[Iterable[float]]
        | Iterable[Iterable[Iterable[float]]]
        | float
        | int
    ),
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a value to a tensor while preserving device/dtype when possible."""
    if device is not None:
        if isinstance(device, str) and device.strip().lower() == "auto":
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            effective_dtype = value.dtype if dtype is None else dtype
            prefer = (
                ("cuda", "cpu")
                if effective_dtype == torch.float64
                else ("cuda", "mps", "cpu")
            )
            device = resolve_device("auto", prefer=prefer)
        else:
            device = resolve_device(device)
    if torch.is_tensor(value):
        out = value
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch.as_tensor(value, device=device, dtype=dtype)


def as_float_tensor(
    value: (
        Tensor
        | Iterable[float]
        | Iterable[Iterable[float]]
        | Iterable[Iterable[Iterable[float]]]
        | float
        | int
    ),
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    name: str = "value",
) -> Tensor:
    """Convert numeric input to a real floating-point tensor.

    Integer inputs are promoted to PyTorch's default floating dtype when no
    dtype is requested. Explicit non-floating and complex dtypes are rejected
    because the geometry and acoustic kernels require real floating values.
    """

    normalized_value = (
        value
        if torch.is_tensor(value)
        else _snapshot_and_reject_boolean_values(value, name=name)
    )
    raw = (
        normalized_value
        if torch.is_tensor(normalized_value)
        else torch.as_tensor(cast(Any, normalized_value))
    )
    if raw.dtype == torch.bool:
        raise TypeError(f"{name} must not use a boolean dtype")
    if raw.is_complex():
        raise TypeError(f"{name} must use a real floating-point dtype")
    out = as_tensor(cast(Any, normalized_value), device=device, dtype=dtype)
    if not out.is_floating_point():
        if dtype is not None:
            raise TypeError(f"{name} dtype must be floating-point")
        out = out.to(dtype=torch.get_default_dtype())
    validate_supported_float_tensor(out, name=name)
    return out


def _snapshot_and_reject_boolean_values(value: object, *, name: str) -> object:
    """Materialize non-Tensor iterables without losing mixed boolean values."""

    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not contain boolean values")
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_ or (
            value.dtype == object
            and any(isinstance(item, (bool, np.bool_)) for item in value.flat)
        ):
            raise TypeError(f"{name} must not contain boolean values")
        return value
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        return value
    return [_snapshot_and_reject_boolean_values(item, name=name) for item in value]


def ensure_dim(size: Tensor) -> Tensor:
    """Validate room size dimensionality (2D or 3D)."""
    if size.ndim != 1 or size.numel() not in (2, 3):
        raise ValueError("room size must be a 1D tensor of length 2 or 3")
    return size


def stable_vector_norm(
    value: Tensor,
    *,
    dim: int = -1,
    keepdim: bool = False,
) -> Tensor:
    """Compute a Euclidean norm without squaring unscaled extreme values."""

    scale = torch.amax(torch.abs(value), dim=dim, keepdim=True)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    norm = scale * torch.linalg.vector_norm(
        value / safe_scale,
        dim=dim,
        keepdim=True,
    )
    return norm if keepdim else norm.squeeze(dim)


def extend_size(size: Tensor, dim: int) -> Tensor:
    """Extend 2D room size to 3D by adding a dummy z dimension."""
    if size.numel() == dim:
        return size
    if size.numel() == 2 and dim == 3:
        pad = torch.tensor([1.0], device=size.device, dtype=size.dtype)
        return torch.cat([size, pad])
    raise ValueError("unsupported room dimension")
