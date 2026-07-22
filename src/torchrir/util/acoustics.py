"""Acoustic utility formulas."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from .tensor import as_float_tensor, ensure_dim
from ._scalars import normalize_finite_real

_DEF_SPEED_OF_SOUND = 343.0
_INT64_MAX = torch.iinfo(torch.int64).max


def _finite_product(left: float, right: float, *, name: str) -> float:
    """Multiply two validated scalars without leaking a non-finite result."""

    result = left * right
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_sum(values: tuple[float, ...], *, name: str) -> float:
    """Sum validated scalars and normalize ``math.fsum`` overflow."""

    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_ratio(numerator: float, denominator: float, *, name: str) -> float:
    """Divide positive validated scalars without returning zero or infinity."""

    if (
        not math.isfinite(numerator)
        or numerator <= 0.0
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    result = numerator / denominator
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _sabine_geometry(size: Tensor) -> tuple[float, float, tuple[float, ...]]:
    """Return finite room measure, boundary, and per-face measures."""

    dimensions = tuple(float(value) for value in size.detach().cpu().tolist())
    if len(dimensions) == 2:
        lx, ly = dimensions
        measure = _finite_product(lx, ly, name="room size measure")
        boundary_sum = _finite_sum((lx, ly), name="room size boundary")
        boundary = _finite_product(
            2.0,
            boundary_sum,
            name="room size boundary",
        )
        surfaces = (ly, ly, lx, lx)
    else:
        lx, ly, lz = dimensions
        measure_xy = _finite_product(lx, ly, name="room size measure")
        measure = _finite_product(measure_xy, lz, name="room size measure")
        surface_xy = measure_xy
        surface_yz = _finite_product(ly, lz, name="room size boundary")
        surface_xz = _finite_product(lx, lz, name="room size boundary")
        boundary_sum = _finite_sum(
            (surface_xy, surface_yz, surface_xz),
            name="room size boundary",
        )
        boundary = _finite_product(
            2.0,
            boundary_sum,
            name="room size boundary",
        )
        surfaces = (
            surface_yz,
            surface_yz,
            surface_xz,
            surface_xz,
            surface_xy,
            surface_xy,
        )
    return measure, boundary, surfaces


def _sabine_coefficient(dim: int, c: float) -> float:
    multiplier = 12.0 if dim == 2 else 24.0
    coefficient = multiplier * math.log(10.0) / c
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("c must produce a positive finite Sabine coefficient")
    return coefficient


def estimate_beta_from_t60(
    size: Tensor,
    t60: float,
    *,
    c: float = _DEF_SPEED_OF_SOUND,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Estimate uniform wall reflection coefficients with Sabine's formula.

    The 2D model uses ``12 ln(10) / c`` with room area and perimeter. The 3D
    model uses ``24 ln(10) / c`` with room volume and surface area. A requested
    T60 that would require an energy absorption coefficient greater than one is
    physically infeasible and raises ``ValueError`` instead of being clipped.

    Note:
        This function corresponds to gpuRIR's ``beta_SabineEstimation``. TorchRIR
        uses snake_case naming for consistency.

    Examples:
        ```python
        beta = estimate_beta_from_t60(torch.tensor([6.0, 4.0, 3.0]), t60=0.4)
        ```
    """
    t60 = normalize_finite_real(t60, name="t60", positive=True)
    c = normalize_finite_real(c, name="c", positive=True)
    size = as_float_tensor(size, device=device, dtype=dtype, name="room size")
    size = ensure_dim(size)
    if not torch.all(torch.isfinite(size)) or torch.any(size <= 0):
        raise ValueError("room size must contain finite positive values")
    dim = size.numel()
    measure, boundary, surfaces = _sabine_geometry(size)
    coefficient = _sabine_coefficient(dim, c)
    numerator = _finite_product(
        coefficient,
        measure,
        name="c and room size Sabine numerator",
    )
    denominator = _finite_product(
        t60,
        boundary,
        name="t60 and room size Sabine denominator",
    )
    alpha = _finite_ratio(
        numerator,
        denominator,
        name="t60, c, and room size Sabine absorption",
    )
    if alpha > 1.0 and not math.isclose(alpha, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-12):
        minimum_t60 = _finite_ratio(
            numerator,
            boundary,
            name="c and room size minimum t60",
        )
        raise ValueError(
            "requested t60 is too short for Sabine absorption alpha <= 1; "
            f"minimum is {minimum_t60:.6g} s"
        )
    alpha = min(alpha, 1.0)
    beta = math.sqrt(1.0 - alpha)
    result = torch.full((len(surfaces),), beta, device=size.device, dtype=size.dtype)
    stored_beta = float(result[0].detach().cpu().item())
    if (0.0 < beta < 1.0) and stored_beta in (0.0, 1.0):
        raise ValueError(
            "requested t60 cannot be represented by beta at the room dtype"
        )
    return result


def estimate_t60_from_beta(
    size: Tensor,
    beta: Tensor,
    *,
    c: float = _DEF_SPEED_OF_SOUND,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> float:
    """Estimate T60 from reflection coefficients using Sabine's formula.

    Reflection coefficients are pressure-amplitude ratios in ``[0, 1]``. The
    corresponding energy absorption is ``1 - beta**2``. Perfectly reflecting
    walls therefore produce an infinite T60.

    Examples:
        ```python
        t60 = estimate_t60_from_beta(torch.tensor([6.0, 4.0, 3.0]), beta=torch.full((6,), 0.9))
        ```
    """
    c = normalize_finite_real(c, name="c", positive=True)
    size = as_float_tensor(size, device=device, dtype=dtype, name="room size")
    size = ensure_dim(size)
    if not torch.all(torch.isfinite(size)) or torch.any(size <= 0):
        raise ValueError("room size must contain finite positive values")
    beta = as_float_tensor(
        beta, device=size.device, dtype=size.dtype, name="beta"
    ).reshape(-1)
    if not torch.all(torch.isfinite(beta)):
        raise ValueError("beta must contain finite values")
    if torch.any(beta < 0) or torch.any(beta > 1):
        raise ValueError("beta values must be in [0, 1]")
    dim = size.numel()
    if dim == 2:
        if beta.numel() != 4:
            raise ValueError("beta must have 4 elements for 2D t60 estimation")
    else:
        if beta.numel() != 6:
            raise ValueError("beta must have 6 elements for 3D t60 estimation")

    measure, _boundary, surfaces = _sabine_geometry(size)
    coefficient = _sabine_coefficient(dim, c)
    beta_values = tuple(float(value) for value in beta.detach().cpu().tolist())
    alpha_values = tuple((1.0 - value) * (1.0 + value) for value in beta_values)
    absorption_terms = tuple(
        _finite_product(
            surface,
            alpha,
            name="room size and beta Sabine absorption term",
        )
        for surface, alpha in zip(surfaces, alpha_values, strict=True)
    )
    absorption = _finite_sum(
        absorption_terms,
        name="room size and beta Sabine absorption",
    )
    if absorption == 0.0:
        return float("inf")
    numerator = _finite_product(
        coefficient,
        measure,
        name="c and room size Sabine numerator",
    )
    return _finite_ratio(
        numerator,
        absorption,
        name="room size, beta, and c estimated t60",
    )


def attenuation_db_to_time_sabine(att_db: float, t60: float) -> float:
    """Convert attenuation (dB) to time based on T60.

    Note:
        This function corresponds to gpuRIR's ``att2t_SabineEstimation``. TorchRIR
        uses snake_case naming for consistency.

    Examples:
        ```python
        t = attenuation_db_to_time_sabine(att_db=60.0, t60=0.4)
        ```
    """
    t60 = normalize_finite_real(t60, name="t60", positive=True)
    att_db = normalize_finite_real(att_db, name="att_db", positive=True)
    attenuation_time = (att_db / 60.0) * t60
    if not math.isfinite(attenuation_time) or attenuation_time <= 0.0:
        raise ValueError(
            "att_db and t60 must produce a positive finite attenuation time"
        )
    return attenuation_time


def estimate_image_counts_from_tmax(
    tmax: float, room_size: Tensor, c: float = _DEF_SPEED_OF_SOUND
) -> Tensor:
    """Estimate image counts per dimension needed to cover tmax.

    Note:
        This function uses TorchRIR's per-axis image-index half-width. gpuRIR's
        ``t2n`` returns a total count whose rough equivalent is ``2 * n + 1``;
        the two helpers intentionally do not return the same values.

    Examples:
        ```python
        nb_img = estimate_image_counts_from_tmax(0.3, torch.tensor([6.0, 4.0, 3.0]))
        ```
    """
    tmax = normalize_finite_real(tmax, name="tmax", positive=True)
    c = normalize_finite_real(c, name="c", positive=True)
    size = as_float_tensor(room_size, name="room size")
    size = ensure_dim(size)
    if not torch.all(torch.isfinite(size)) or torch.any(size <= 0):
        raise ValueError("room size must contain finite positive values")
    travel_distance = _finite_product(tmax, c, name="tmax and c travel distance")
    counts: list[int] = []
    for dimension in size.detach().cpu().tolist():
        ratio = travel_distance / float(dimension)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(
                "tmax, c, and room size must produce positive finite image counts"
            )
        count = math.ceil(ratio)
        if count > _INT64_MAX:
            raise ValueError(
                "tmax, c, and room size must produce image counts in the "
                "positive int64 range"
            )
        counts.append(count)
    return torch.tensor(counts, dtype=torch.int64, device=size.device)
