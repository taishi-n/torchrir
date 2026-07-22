"""Internal real floating-point dtype contracts."""

from __future__ import annotations

import torch
from torch import Tensor


SUPPORTED_FLOAT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
)


def validate_supported_float_dtype(dtype: object, *, name: str = "dtype") -> None:
    """Reject floating formats unsupported by the public tensor kernels."""

    if dtype not in SUPPORTED_FLOAT_DTYPES:
        raise TypeError(
            f"{name} must be a supported real floating-point dtype "
            "(torch.float16, torch.bfloat16, torch.float32, or torch.float64)"
        )


def validate_supported_float_tensor(value: Tensor, *, name: str) -> None:
    """Validate one Tensor's real floating-point dtype without running a kernel."""

    validate_materialized_tensor(value, name=name)
    validate_supported_float_dtype(value.dtype, name=f"{name} dtype")


def validate_materialized_tensor(value: Tensor, *, name: str) -> None:
    """Require a dense, materialized Tensor on an advertised backend."""

    if value.is_quantized:
        raise TypeError(f"{name} must not be quantized")
    if value.layout != torch.strided or value.is_nested:
        raise TypeError(f"{name} must use a dense strided layout")
    if value.device.type not in ("cpu", "cuda", "mps"):
        raise ValueError(f"{name} must be on CPU, CUDA, or MPS")


__all__: list[str] = []
