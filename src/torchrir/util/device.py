"""Device and dtype helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


def resolve_device(
    device: Optional[torch.device | str],
    *,
    prefer: Tuple[str, ...] = ("cuda", "mps", "cpu"),
) -> torch.device:
    """Resolve a device string (including 'auto') into a torch.device.

    Falls back to CPU when the requested backend is unavailable.

    Examples:
        ```python
        device = resolve_device("auto")
        ```
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

    Examples:
        ```python
        spec = DeviceSpec(device="auto", dtype=torch.float32)
        device, dtype = spec.resolve(tensor)
        ```
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
