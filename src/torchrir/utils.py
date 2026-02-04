from __future__ import annotations

"""Utility functions for geometry, acoustics, and tensor handling."""

import math
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor


_DEF_SPEED_OF_SOUND = 343.0


def as_tensor(
    value: Tensor | Iterable[float] | float | int,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Convert a value to a tensor while preserving device/dtype when possible."""
    if isinstance(device, str):
        device = resolve_device(device)
    if torch.is_tensor(value):
        out = value
        if device is not None:
            out = out.to(device)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch.as_tensor(value, device=device, dtype=dtype)


def resolve_device(
    device: Optional[torch.device | str],
    *,
    prefer: Tuple[str, ...] = ("cuda", "mps", "cpu"),
) -> torch.device:
    """Resolve a device string (including 'auto') into a torch.device.

    Falls back to CPU when the requested backend is unavailable.

    Example:
        >>> device = resolve_device("auto")
    """
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device

    dev = str(device).lower()
    if dev == "auto":
        for backend in prefer:
            if backend == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if backend == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            if backend == "cpu":
                return torch.device("cpu")
        return torch.device("cpu")

    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device)
        warnings.warn("CUDA not available; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    if dev == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        warnings.warn("MPS not available; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    if dev == "cpu":
        return torch.device("cpu")

    return torch.device(device)


@dataclass(frozen=True)
class DeviceSpec:
    """Resolve device + dtype defaults consistently.

    Example:
        >>> spec = DeviceSpec(device="auto", dtype=torch.float32)
        >>> device, dtype = spec.resolve(tensor)
    """

    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None
    prefer: Tuple[str, ...] = ("cuda", "mps", "cpu")

    def resolve(self, *values) -> Tuple[torch.device, torch.dtype]:
        """Resolve device/dtype from inputs with overrides."""
        tensor_device: Optional[torch.device] = None
        tensor_dtype: Optional[torch.dtype] = None
        for value in values:
            if torch.is_tensor(value):
                if tensor_device is None:
                    tensor_device = value.device
                if tensor_dtype is None:
                    tensor_dtype = value.dtype

        if isinstance(self.device, str) and self.device.lower() == "auto":
            device = tensor_device or resolve_device("auto", prefer=self.prefer)
        elif self.device is None:
            device = tensor_device or torch.device("cpu")
        else:
            device = resolve_device(self.device, prefer=self.prefer)

        if self.dtype is None:
            dtype = tensor_dtype or torch.float32
        else:
            dtype = self.dtype
        return device, dtype


def infer_device_dtype(
    *values,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.device, torch.dtype]:
    """Infer device/dtype from inputs with optional overrides."""
    return DeviceSpec(device=device, dtype=dtype).resolve(*values)


def ensure_dim(size: Tensor) -> Tensor:
    """Validate room size dimensionality (2D or 3D)."""
    if size.ndim != 1 or size.numel() not in (2, 3):
        raise ValueError("room size must be a 1D tensor of length 2 or 3")
    return size


def extend_size(size: Tensor, dim: int) -> Tensor:
    """Extend 2D room size to 3D by adding a dummy z dimension."""
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
    """Estimate reflection coefficients from T60 using Sabine's formula.

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


def orientation_to_unit(orientation: Tensor, dim: int) -> Tensor:
    """Convert orientation representation to unit vectors in 2D/3D."""
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
    """Convert attenuation (dB) to time based on T60.

    Example:
        >>> t = att2t_sabine_estimation(att_db=60.0, t60=0.4)
    """
    if t60 <= 0:
        raise ValueError("t60 must be positive")
    if att_db <= 0:
        raise ValueError("att_db must be positive")
    return (att_db / 60.0) * t60


def att2t_SabineEstimation(att_db: float, t60: float) -> float:
    """Legacy alias for att2t_sabine_estimation.

    Example:
        >>> t = att2t_SabineEstimation(att_db=60.0, t60=0.4)
    """
    return att2t_sabine_estimation(att_db, t60)


def beta_SabineEstimation(room_size: Tensor, t60: float) -> Tensor:
    """Legacy alias for estimate_beta_from_t60.

    Example:
        >>> beta = beta_SabineEstimation(torch.tensor([6.0, 4.0, 3.0]), t60=0.4)
    """
    return estimate_beta_from_t60(room_size, t60)


def t2n(tmax: float, room_size: Tensor, c: float = _DEF_SPEED_OF_SOUND) -> Tensor:
    """Estimate image counts per dimension needed to cover tmax.

    Example:
        >>> nb_img = t2n(0.3, torch.tensor([6.0, 4.0, 3.0]))
    """
    if tmax <= 0:
        raise ValueError("tmax must be positive")
    size = as_tensor(room_size)
    size = ensure_dim(size)
    # number of images in each dimension needed to cover the maximum distance
    # uses the same heuristic as gpuRIR: n = ceil(tmax * c / room_size)
    n = torch.ceil((tmax * c) / size).to(torch.int64)
    return n


def normalize_orientation(orientation: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Normalize orientation vectors with numerical stability."""
    norm = torch.linalg.norm(orientation, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    return orientation / norm
