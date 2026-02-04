from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor


_DEF_SPEED_OF_SOUND = 343.0


def as_tensor(
    value,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    if torch.is_tensor(value):
        out = value
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch.as_tensor(value, device=device, dtype=dtype)


def infer_device_dtype(
    *values,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.device, torch.dtype]:
    if device is None or dtype is None:
        for value in values:
            if torch.is_tensor(value):
                if device is None:
                    device = value.device
                if dtype is None:
                    dtype = value.dtype
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32
    return device, dtype


def ensure_dim(size: Tensor) -> Tensor:
    if size.ndim != 1 or size.numel() not in (2, 3):
        raise ValueError("room size must be a 1D tensor of length 2 or 3")
    return size


def extend_size(size: Tensor, dim: int) -> Tensor:
    if size.numel() == dim:
        return size
    if size.numel() == 2 and dim == 3:
        pad = torch.tensor([1.0], device=size.device, dtype=size.dtype)
        return torch.cat([size, pad])
    raise ValueError("unsupported room dimension")


def estimate_beta_from_t60(
    size: Tensor,
    t60: float,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
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


def orientation_to_unit(orientation: Tensor, dim: int) -> Tensor:
    if dim == 2:
        if orientation.ndim == 0:
            angle = orientation
            vec = torch.stack([torch.cos(angle), torch.sin(angle)])
            return normalize_orientation(vec)
        if orientation.shape[-1] == 1:
            angle = orientation.squeeze(-1)
            vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            return normalize_orientation(vec)
        if orientation.ndim == 1 and orientation.numel() != 2:
            angle = orientation
            vec = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
            return normalize_orientation(vec)
        if orientation.shape[-1] == 2:
            return normalize_orientation(orientation)
        raise ValueError("2D orientation must be angle or 2D vector")
    if dim == 3:
        if orientation.shape[-1] == 3:
            return normalize_orientation(orientation)
        if orientation.shape[-1] == 2:
            azimuth = orientation[..., 0]
            elevation = orientation[..., 1]
            x = torch.cos(elevation) * torch.cos(azimuth)
            y = torch.cos(elevation) * torch.sin(azimuth)
            z = torch.sin(elevation)
            vec = torch.stack([x, y, z], dim=-1)
            return normalize_orientation(vec)
        raise ValueError("3D orientation must be vector or (azimuth, elevation)")
    raise ValueError("unsupported dimension for orientation")


def att2t_sabine_estimation(att_db: float, t60: float) -> float:
    if t60 <= 0:
        raise ValueError("t60 must be positive")
    if att_db <= 0:
        raise ValueError("att_db must be positive")
    return (att_db / 60.0) * t60


def att2t_SabineEstimation(att_db: float, t60: float) -> float:
    return att2t_sabine_estimation(att_db, t60)


def beta_SabineEstimation(room_size: Tensor, t60: float) -> Tensor:
    return estimate_beta_from_t60(room_size, t60)


def t2n(tmax: float, room_size: Tensor, c: float = _DEF_SPEED_OF_SOUND) -> Tensor:
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    size = as_tensor(room_size)
    size = ensure_dim(size)
    # number of images in each dimension needed to cover the maximum distance
    # uses the same heuristic as gpuRIR: n = ceil(tmax * c / room_size)
    n = torch.ceil((tmax * c) / size).to(torch.int64)
    return n


def normalize_orientation(orientation: Tensor, *, eps: float = 1e-8) -> Tensor:
    norm = torch.linalg.norm(orientation, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return orientation / norm
