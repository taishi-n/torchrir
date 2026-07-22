"""Device and dtype helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from ._dtypes import validate_supported_float_dtype


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
    _validate_preference_order(prefer)
    if device is None:
        return torch.device("cpu")
    if not isinstance(device, (str, torch.device)):
        raise TypeError("device must be a string, torch.device, or None")
    dev = str(device).strip().lower()
    if not dev:
        raise ValueError("device must be non-empty")
    if dev == "auto":
        for backend in prefer:
            if backend == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if backend == "mps" and torch.backends.mps.is_available():
                return torch.device("mps")
            if backend == "cpu":
                return torch.device("cpu")
        return torch.device("cpu")

    try:
        requested = torch.device(dev)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device: {device}") from exc
    _validate_device_kind(requested)
    if requested.type == "cuda":
        if torch.cuda.is_available():
            if (
                requested.index is not None
                and requested.index >= torch.cuda.device_count()
            ):
                raise ValueError(f"CUDA device index is unavailable: {requested.index}")
            return requested
        warnings.warn("CUDA not available; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    if requested.type == "mps":
        if torch.backends.mps.is_available():
            return requested
        warnings.warn("MPS not available; falling back to CPU.", RuntimeWarning)
        return torch.device("cpu")
    return torch.device("cpu")


@dataclass(frozen=True, slots=True, kw_only=True)
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

    def __post_init__(self) -> None:
        _validate_preference_order(self.prefer)
        if self.dtype is not None:
            _validate_dtype(self.dtype)
        if self.device is not None:
            if not isinstance(self.device, (str, torch.device)):
                raise TypeError("device must be a string, torch.device, or None")
            normalized = str(self.device).strip().lower()
            if not normalized:
                raise ValueError("device must be non-empty")
            if normalized != "auto":
                try:
                    parsed = torch.device(normalized)
                except (RuntimeError, ValueError) as exc:
                    raise ValueError(f"invalid device: {self.device}") from exc
                _validate_device_kind(parsed)
            if isinstance(self.device, str):
                object.__setattr__(self, "device", normalized)

    def resolve(self, *values: object) -> Tuple[torch.device, torch.dtype]:
        """Resolve device/dtype from inputs with overrides."""
        tensors = [value for value in values if torch.is_tensor(value)]
        tensor_devices = {value.device for value in tensors}
        tensor_dtypes = {value.dtype for value in tensors}

        if self.device is None and len(tensor_devices) > 1:
            devices = ", ".join(sorted(str(device) for device in tensor_devices))
            raise ValueError(
                f"input tensor devices must match when device=None; found {devices}"
            )
        if self.dtype is None and len(tensor_dtypes) > 1:
            dtypes = ", ".join(sorted(str(dtype) for dtype in tensor_dtypes))
            raise ValueError(
                f"input tensor dtypes must match when dtype=None; found {dtypes}"
            )

        if self.dtype is None:
            dtype = next(iter(tensor_dtypes), torch.float32)
        else:
            dtype = self.dtype
        _validate_dtype(dtype)

        if isinstance(self.device, str) and self.device.lower() == "auto":
            prefer = tuple(
                backend
                for backend in self.prefer
                if not (backend == "mps" and dtype == torch.float64)
            )
            device = resolve_device("auto", prefer=prefer or ("cpu",))
        elif self.device is None:
            device = next(iter(tensor_devices), torch.device("cpu"))
        else:
            device = resolve_device(self.device, prefer=self.prefer)

        _validate_device_kind(device)
        if device.type == "mps" and dtype == torch.float64:
            raise ValueError(
                "MPS does not support float64; use dtype=torch.float32 or select "
                "CPU/CUDA"
            )
        return device, dtype


def _validate_preference_order(prefer: Tuple[str, ...]) -> None:
    if not isinstance(prefer, tuple) or not prefer:
        raise ValueError("prefer must be a non-empty tuple")
    if any(item not in ("cuda", "mps", "cpu") for item in prefer):
        raise ValueError("prefer entries must be cuda, mps, or cpu")
    if len(set(prefer)) != len(prefer):
        raise ValueError("prefer must not contain duplicates")


def _validate_dtype(dtype: object) -> None:
    validate_supported_float_dtype(dtype)


def _validate_device_kind(device: torch.device) -> None:
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device type must be cpu, cuda, or mps")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU device must not include an index")
    if device.type == "mps" and device.index not in (None, 0):
        raise ValueError("MPS device index must be 0 when specified")
    if device.type == "cuda" and device.index is not None and device.index < 0:
        raise ValueError("CUDA device index must be non-negative")
