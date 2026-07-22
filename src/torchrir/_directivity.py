"""Shared directivity names and analytic gain functions."""

from __future__ import annotations

import torch
from torch import Tensor

from .util._dtypes import validate_supported_float_tensor


_DIRECTIVITY_ALIASES = {
    "omni": "omni",
    "omnidirectional": "omni",
    "homni": "halfomni",
    "halfomni": "halfomni",
    "half-omni": "halfomni",
    "subcardioid": "subcardioid",
    "subcard": "subcardioid",
    "cardioid": "cardioid",
    "card": "cardioid",
    "hypercardioid": "hypercardioid",
    "hypcard": "hypercardioid",
    "bidir": "bidir",
    "bidirectional": "bidir",
    "figure8": "bidir",
    "figure-8": "bidir",
}


def canonicalize_directivity(pattern: str, *, endpoint: str) -> str:
    """Return the canonical name for a supported directivity pattern."""

    label = f"{endpoint} directivity" if endpoint else "directivity"
    if not isinstance(pattern, str):
        raise TypeError(f"{label} must be a string")
    normalized = _DIRECTIVITY_ALIASES.get(pattern.strip().lower())
    if normalized is None:
        raise ValueError(f"unsupported {label} pattern: {pattern}")
    return normalized


def directivity_gain(pattern: str, cos_theta: Tensor) -> Tensor:
    """Compute analytic directivity gain from the path-angle cosine."""

    pattern = canonicalize_directivity(pattern, endpoint="")
    if not torch.is_tensor(cos_theta):
        raise TypeError("cos_theta must be a Tensor")
    validate_supported_float_tensor(cos_theta, name="cos_theta")
    if not torch.all(torch.isfinite(cos_theta)):
        raise ValueError("cos_theta must contain finite values")
    if torch.any((cos_theta < -1) | (cos_theta > 1)):
        raise ValueError("cos_theta values must lie in [-1, 1]")
    if pattern == "omni":
        return torch.ones_like(cos_theta)
    if pattern == "halfomni":
        return (cos_theta > 0).to(cos_theta.dtype)
    if pattern == "subcardioid":
        return 0.75 + 0.25 * cos_theta
    if pattern == "cardioid":
        return 0.5 + 0.5 * cos_theta
    if pattern == "hypercardioid":
        return 0.25 + 0.75 * cos_theta
    if pattern == "bidir":
        return cos_theta
    raise AssertionError(f"unhandled canonical directivity: {pattern}")
