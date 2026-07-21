"""Directivity pattern utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def directivity_gain(pattern: str, cos_theta: Tensor) -> Tensor:
    """Compute directivity gain for a pattern given cos(theta)."""
    pattern = pattern.lower()
    if pattern in ("omni", "omnidirectional"):
        return torch.ones_like(cos_theta)
    if pattern in ("homni", "halfomni", "half-omni"):
        return (cos_theta > 0).to(cos_theta.dtype)
    if pattern in ("subcardioid", "subcard"):
        return 0.75 + 0.25 * cos_theta
    if pattern in ("cardioid", "card"):
        return 0.5 + 0.5 * cos_theta
    if pattern in ("hypercardioid", "hypcard"):
        return 0.25 + 0.75 * cos_theta
    if pattern in ("bidir", "bidirectional", "figure8", "figure-8"):
        return cos_theta
    raise ValueError(f"unsupported directivity pattern: {pattern}")


def split_directivity(directivity: str | tuple[str, str]) -> tuple[str, str]:
    """Normalize directivity specification into (source, mic)."""
    if isinstance(directivity, (list, tuple)):
        if len(directivity) != 2:
            raise ValueError("directivity tuple must have length 2")
        source_pattern, microphone_pattern = directivity
    else:
        source_pattern = microphone_pattern = directivity
    for pattern in (source_pattern, microphone_pattern):
        if not isinstance(pattern, str):
            raise TypeError("directivity patterns must be strings")
        directivity_gain(pattern, torch.tensor(1.0))
    return source_pattern, microphone_pattern
