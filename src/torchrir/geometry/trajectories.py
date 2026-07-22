"""Trajectory helpers for dynamic scenes."""

from __future__ import annotations

import torch
from torch import Tensor

from ..util._dtypes import validate_supported_float_tensor
from ..util.tensor import as_float_tensor


def linear_trajectory(
    start: Tensor,
    end: Tensor,
    *,
    progress: Tensor,
) -> Tensor:
    """Interpolate a line at explicit normalized progress values.

    ``progress`` makes the trajectory's time grid explicit. For a dynamic RIR
    scene, obtain it from
    [`FrameSchedule.normalized_progress`][torchrir.signal.FrameSchedule.normalized_progress]
    so every geometry frame is evaluated at its exact sample start.
    """

    if not torch.is_tensor(start) or not torch.is_tensor(end):
        raise TypeError("start and end must be Tensors")
    if start.shape != end.shape:
        raise ValueError("start and end must have matching shapes")
    if start.numel() == 0:
        raise ValueError("start and end must be non-empty")
    start = as_float_tensor(start, name="start")
    end = as_float_tensor(end, name="end")
    if start.device != end.device:
        raise ValueError("start and end must be on the same device")
    if start.dtype != end.dtype:
        raise ValueError("start and end must use the same dtype")
    if not torch.all(torch.isfinite(start)) or not torch.all(torch.isfinite(end)):
        raise ValueError("start and end must contain finite values")
    if not torch.is_tensor(progress):
        raise TypeError("progress must be a Tensor")
    if progress.ndim != 1 or progress.numel() == 0:
        raise ValueError("progress must be a non-empty 1D tensor")
    validate_supported_float_tensor(progress, name="progress")
    if progress.device != start.device:
        raise ValueError("progress and endpoints must be on the same device")
    if progress.dtype != start.dtype:
        raise ValueError("progress and endpoints must use the same dtype")
    if not torch.all(torch.isfinite(progress)):
        raise ValueError("progress must contain finite values")
    if torch.any(progress < 0) or torch.any(progress > 1):
        raise ValueError("progress values must be between 0 and 1")
    if progress.numel() > 1 and torch.any(progress[1:] < progress[:-1]):
        raise ValueError("progress must be non-decreasing")
    weight = progress.reshape((-1,) + (1,) * start.ndim)
    return (1 - weight) * start.unsqueeze(0) + weight * end.unsqueeze(0)
