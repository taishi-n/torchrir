from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def directivity_gain(pattern: str, cos_theta: Tensor) -> Tensor:
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
    if isinstance(directivity, (list, tuple)):
        if len(directivity) != 2:
            raise ValueError("directivity tuple must have length 2")
        return directivity[0], directivity[1]
    return directivity, directivity
