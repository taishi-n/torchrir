"""Shared normalization for public scalar parameters."""

from __future__ import annotations

import math
from numbers import Integral, Real


_MAX_SAMPLE_RATE = 2**31 - 1


def normalize_finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    """Return a finite real scalar with one optional sign constraint."""

    if positive and non_negative:
        raise ValueError("positive and non_negative cannot both be requested")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except OverflowError as exc:
        qualifier = (
            "positive and finite"
            if positive
            else "non-negative and finite"
            if non_negative
            else "finite"
        )
        raise ValueError(f"{name} must be {qualifier}") from exc
    qualifier = "finite"
    invalid = not math.isfinite(normalized)
    if positive:
        qualifier = "positive and finite"
        invalid = invalid or normalized <= 0
    elif non_negative:
        qualifier = "non-negative and finite"
        invalid = invalid or normalized < 0
    if invalid:
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


def normalize_integer(
    value: object,
    *,
    name: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Return a non-boolean integer with optional inclusive bounds."""

    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")

    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if minimum is not None and normalized < minimum:
        qualifier = (
            "positive"
            if minimum == 1
            else "non-negative"
            if minimum == 0
            else f">= {minimum}"
        )
        raise ValueError(f"{name} must be {qualifier}")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return normalized


def normalize_sample_rate(value: object, *, name: str = "sample_rate") -> int:
    """Return a positive sample rate representable by audio backends."""

    normalized = normalize_integer(value, name=name, minimum=1)
    if normalized > _MAX_SAMPLE_RATE:
        raise ValueError(f"{name} must be at most {_MAX_SAMPLE_RATE}")
    return normalized


__all__: list[str] = []
