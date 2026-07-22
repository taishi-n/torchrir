"""Internal normalization for frame times and discrete sample starts."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np
import torch
from torch import Tensor

from ._dtypes import validate_materialized_tensor
from ._scalars import normalize_finite_real


_MAX_SAMPLE_INDEX = torch.iinfo(torch.int64).max


def frame_times_to_samples(
    times: Tensor | Sequence[float],
    *,
    sample_rate: float,
    name: str = "frame times",
) -> Tensor:
    """Map finite second-based frame times once to CPU ``int64`` samples."""

    normalized_sample_rate = normalize_finite_real(
        sample_rate,
        name="sample_rate",
        positive=True,
    )

    normalized = _as_cpu_float64(times, name=name)
    if normalized.ndim != 1 or normalized.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence")
    if not torch.all(torch.isfinite(normalized)):
        raise ValueError(f"{name} must contain finite real values")
    if normalized[0].item() != 0.0:
        raise ValueError(f"first {name.removesuffix('s')} must be 0")
    if normalized.numel() > 1 and torch.any(normalized[1:] <= normalized[:-1]):
        raise ValueError(f"{name} must be strictly increasing")

    scaled = normalized * normalized_sample_rate
    if not torch.all(torch.isfinite(scaled)) or torch.any(scaled >= _MAX_SAMPLE_INDEX):
        raise ValueError(f"{name} must map within the non-negative int64 range")
    starts = torch.floor(scaled).to(torch.int64)
    if starts.numel() > 1 and torch.any(starts[1:] <= starts[:-1]):
        raise ValueError(
            f"{name} must map to strictly increasing sample indices at "
            f"sample_rate={normalized_sample_rate:g}"
        )
    return starts


def _as_cpu_float64(times: Tensor | Sequence[float], *, name: str) -> Tensor:
    if torch.is_tensor(times):
        validate_materialized_tensor(times, name=name)
        if times.is_complex():
            raise ValueError(f"{name} must contain finite real values")
        if times.dtype == torch.bool:
            raise TypeError(f"{name} must contain real numbers, not booleans")
        return times.detach().cpu().to(dtype=torch.float64)

    try:
        values = list(times)
    except TypeError as exc:
        raise TypeError(f"{name} must be a sequence of real numbers") from exc
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise TypeError(f"{name} must contain real numbers, not booleans")
    if any(isinstance(value, (complex, np.complexfloating)) for value in values):
        raise ValueError(f"{name} must contain finite real values")
    if any(not isinstance(value, Real) for value in values):
        raise TypeError(f"{name} must contain real numbers")
    try:
        normalized = [float(value) for value in values]
    except OverflowError as exc:
        raise ValueError(
            f"{name} must contain finite values representable as float64"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    return torch.tensor(normalized, device="cpu", dtype=torch.float64)
