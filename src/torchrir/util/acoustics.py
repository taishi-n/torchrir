from __future__ import annotations

"""Acoustic utility formulas."""

import math
from typing import Optional

import torch
from torch import Tensor

from .tensor import as_tensor, ensure_dim, extend_size

_DEF_SPEED_OF_SOUND = 343.0


def estimate_beta_from_t60(
    size: Tensor,
    t60: float,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Estimate reflection coefficients from T60 using Sabine's formula.

    Note:
        This function corresponds to gpuRIR's ``beta_SabineEstimation``. TorchRIR
        uses snake_case naming for consistency.

    Example:
        >>> beta = estimate_beta_from_t60(torch.tensor([6.0, 4.0, 3.0]), t60=0.4)
    """
    if t60 <= 0:
        raise ValueError("t60 must be positive")
    size = as_tensor(size, device=device, dtype=dtype)
    size = ensure_dim(size)
    dim = size.numel()
    if dim == 2:
        lx, ly = size.tolist()
        lz = 1.0
        volume = lx * ly * lz
        surface = 2.0 * (lx + ly) * lz
        alpha = 0.161 * volume / (t60 * surface)
        alpha = max(0.0, min(alpha, 0.999))
        beta = math.sqrt(1.0 - alpha)
        return torch.full((4,), beta, device=size.device, dtype=size.dtype)
    size = extend_size(size, 3)
    lx, ly, lz = size.tolist()
    volume = lx * ly * lz
    surface = 2.0 * (lx * ly + ly * lz + lx * lz)
    alpha = 0.161 * volume / (t60 * surface)
    alpha = max(0.0, min(alpha, 0.999))
    beta = math.sqrt(1.0 - alpha)
    return torch.full((6,), beta, device=size.device, dtype=size.dtype)


def estimate_t60_from_beta(
    size: Tensor,
    beta: Tensor,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> float:
    """Estimate T60 from reflection coefficients using Sabine's formula.

    Example:
        >>> t60 = estimate_t60_from_beta(torch.tensor([6.0, 4.0, 3.0]), beta=torch.full((6,), 0.9))
    """
    size = as_tensor(size, device=device, dtype=dtype)
    size = ensure_dim(size)
    beta = as_tensor(beta, device=size.device, dtype=size.dtype)
    dim = size.numel()
    if dim == 2:
        if beta.numel() != 4:
            raise ValueError("beta must have 4 elements for 2D t60 estimation")
        lx, ly = size.tolist()
        lz = 1.0
        volume = lx * ly * lz
        surfaces = torch.tensor(
            [ly * lz, ly * lz, lx * lz, lx * lz],
            device=size.device,
            dtype=size.dtype,
        )
        alpha = 1.0 - beta**2
        absorption = torch.sum(surfaces * alpha).item()
        if absorption <= 0.0:
            return float("inf")
        return 0.161 * volume / absorption
    size = extend_size(size, 3)
    if beta.numel() != 6:
        raise ValueError("beta must have 6 elements for t60 estimation")
    lx, ly, lz = size.tolist()
    volume = lx * ly * lz
    surfaces = torch.tensor(
        [ly * lz, ly * lz, lx * lz, lx * lz, lx * ly, lx * ly],
        device=size.device,
        dtype=size.dtype,
    )
    alpha = 1.0 - beta**2
    absorption = torch.sum(surfaces * alpha).item()
    if absorption <= 0.0:
        return float("inf")
    return 0.161 * volume / absorption


def att2t_sabine_estimation(att_db: float, t60: float) -> float:
    """Convert attenuation (dB) to time based on T60.

    Note:
        This function corresponds to gpuRIR's ``att2t_SabineEstimation``. TorchRIR
        uses snake_case naming for consistency.

    Example:
        >>> t = att2t_sabine_estimation(att_db=60.0, t60=0.4)
    """
    if t60 <= 0:
        raise ValueError("t60 must be positive")
    if att_db <= 0:
        raise ValueError("att_db must be positive")
    return (att_db / 60.0) * t60


def estimate_image_counts(
    tmax: float, room_size: Tensor, c: float = _DEF_SPEED_OF_SOUND
) -> Tensor:
    """Estimate image counts per dimension needed to cover tmax.

    Note:
        This function corresponds to gpuRIR's ``t2n`` helper, renamed for clarity.

    Example:
        >>> nb_img = estimate_image_counts(0.3, torch.tensor([6.0, 4.0, 3.0]))
    """
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    size = as_tensor(room_size)
    size = ensure_dim(size)
    n = torch.ceil((tmax * c) / size).to(torch.int64)
    return n
