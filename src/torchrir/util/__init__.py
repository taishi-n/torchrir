"""General-purpose math, device, and tensor utilities for torchrir."""

from .acoustics import (
    attenuation_db_to_time_sabine,
    estimate_beta_from_t60,
    estimate_image_counts_from_tmax,
    estimate_t60_from_beta,
)
from .cli import add_output_args
from .device import DeviceSpec, infer_device_dtype, resolve_device
from .orientation import normalize_orientation, orientation_to_unit
from .tensor import as_tensor, ensure_dim, extend_size

__all__ = [
    "DeviceSpec",
    "add_output_args",
    "as_tensor",
    "attenuation_db_to_time_sabine",
    "ensure_dim",
    "estimate_beta_from_t60",
    "estimate_image_counts_from_tmax",
    "estimate_t60_from_beta",
    "extend_size",
    "infer_device_dtype",
    "normalize_orientation",
    "orientation_to_unit",
    "resolve_device",
]
