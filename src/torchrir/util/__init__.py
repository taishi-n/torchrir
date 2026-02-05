"""General-purpose math, device, and tensor utilities for torchrir."""

from .acoustics import (
    att2t_sabine_estimation,
    estimate_beta_from_t60,
    estimate_image_counts,
    estimate_t60_from_beta,
)
from .device import DeviceSpec, infer_device_dtype, resolve_device
from .orientation import normalize_orientation, orientation_to_unit
from .tensor import as_tensor, ensure_dim, extend_size

__all__ = [
    "DeviceSpec",
    "as_tensor",
    "att2t_sabine_estimation",
    "ensure_dim",
    "estimate_beta_from_t60",
    "estimate_image_counts",
    "estimate_t60_from_beta",
    "extend_size",
    "infer_device_dtype",
    "normalize_orientation",
    "orientation_to_unit",
    "resolve_device",
]
