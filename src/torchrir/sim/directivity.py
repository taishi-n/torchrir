"""Public directivity pattern utility."""

from __future__ import annotations

from torch import Tensor

from .._directivity import directivity_gain as _directivity_gain


def directivity_gain(pattern: str, cos_theta: Tensor) -> Tensor:
    """Compute directivity gain for a pattern given cos(theta)."""

    return _directivity_gain(pattern, cos_theta)
