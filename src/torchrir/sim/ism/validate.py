"""Validation helpers for ISM inputs."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from ...config import SimulationConfig, default_config
from ...models import Room


def _resolve_config(
    *,
    config: Optional[SimulationConfig],
    device: Optional[torch.device | str],
    max_order: int | None,
    tmax: Optional[float],
    directivity: str | tuple[str, str] | None,
) -> tuple[
    SimulationConfig,
    Optional[torch.device | str],
    int,
    Optional[float],
    str | tuple[str, str],
]:
    cfg = config or default_config()
    cfg.validate()

    if device is None and cfg.device is not None:
        device = cfg.device

    if max_order is None:
        if cfg.max_order is None:
            raise ValueError("max_order must be provided if not set in config")
        max_order = cfg.max_order

    if tmax is None and cfg.tmax is not None:
        tmax = cfg.tmax

    if directivity is None:
        directivity = cfg.directivity or "omni"

    return cfg, device, max_order, tmax, directivity


def _validate_static_args(
    *,
    room: Room,
    nsample: Optional[int],
    tmax: Optional[float],
    max_order: int,
) -> int:
    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None:
        if tmax is None:
            raise ValueError("nsample or tmax must be provided")
        nsample = int(math.ceil(tmax * room.fs))
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    return nsample


def _validate_dynamic_args(
    *,
    room: Room,
    nsample: Optional[int],
    tmax: Optional[float],
    max_order: int,
) -> int:
    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None:
        if tmax is None:
            raise ValueError("nsample or tmax must be provided")
        nsample = int(math.ceil(tmax * room.fs))
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    return nsample


def _validate_pos_shapes(src_pos: Tensor, mic_pos: Tensor, dim: int) -> None:
    if src_pos.ndim != 2 or src_pos.shape[1] != dim:
        raise ValueError("sources must be of shape (n_src, dim)")
    if mic_pos.ndim != 2 or mic_pos.shape[1] != dim:
        raise ValueError("mics must be of shape (n_mic, dim)")


def _validate_traj_shapes(src_traj: Tensor, mic_traj: Tensor, dim: int) -> None:
    if src_traj.ndim != 3:
        raise ValueError("src_traj must be of shape (T, n_src, dim)")
    if mic_traj.ndim != 3:
        raise ValueError("mic_traj must be of shape (T, n_mic, dim)")
    if src_traj.shape[0] != mic_traj.shape[0]:
        raise ValueError("src_traj and mic_traj must have the same time length")
    if src_traj.shape[2] != dim:
        raise ValueError("src_traj must match room dimension")
    if mic_traj.shape[2] != dim:
        raise ValueError("mic_traj must match room dimension")
