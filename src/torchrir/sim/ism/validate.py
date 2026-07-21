"""Validation helpers for ISM inputs."""

from __future__ import annotations

import math
from typing import Optional
import warnings

import torch
from torch import Tensor

from ...config import SimulationConfig, default_config
from ...models import Room


def _resolve_config(
    *,
    config: Optional[SimulationConfig],
    device: Optional[torch.device | str],
    max_order: int | None,
    nsample: Optional[int],
    tmax: Optional[float],
    tdiff: Optional[float],
    directivity: str | tuple[str, str] | None,
    dtype: Optional[torch.dtype],
    nb_img: Optional[Tensor | tuple[int, ...]],
) -> tuple[
    SimulationConfig,
    Optional[torch.device | str],
    int,
    Optional[int],
    Optional[float],
    Optional[float],
    str | tuple[str, str],
    Optional[torch.dtype],
    Optional[Tensor | tuple[int, ...]],
]:
    cfg = config or default_config()
    cfg.validate()

    device = _merge_setting("device", device, cfg.device)
    dtype = _merge_setting("dtype", dtype, cfg.dtype)
    max_order = _merge_setting("max_order", max_order, cfg.max_order)
    nsample = _merge_setting("nsample", nsample, cfg.nsample)
    tmax = _merge_setting("tmax", tmax, cfg.tmax)
    tdiff = _merge_setting("tdiff", tdiff, cfg.tdiff)
    directivity = _merge_setting("directivity", directivity, cfg.directivity)
    nb_img = _merge_setting("nb_img", nb_img, cfg.nb_img)

    if max_order is None:
        raise ValueError("max_order must be provided if not set in config")
    if nsample is not None and tmax is not None:
        raise ValueError("nsample and tmax are mutually exclusive")
    directivity = directivity or "omni"

    return cfg, device, max_order, nsample, tmax, tdiff, directivity, dtype, nb_img


def _merge_setting(name: str, explicit, configured):
    if explicit is None:
        return configured
    if configured is None:
        return explicit
    explicit_cmp = (
        torch.device(explicit) if name == "device" and explicit != "auto" else explicit
    )
    configured_cmp = (
        torch.device(configured)
        if name == "device" and configured != "auto"
        else configured
    )
    if torch.is_tensor(explicit_cmp) or torch.is_tensor(configured_cmp):
        equal = torch.equal(
            torch.as_tensor(explicit_cmp), torch.as_tensor(configured_cmp)
        )
    else:
        equal = explicit_cmp == configured_cmp
    if not equal:
        raise ValueError(
            f"conflicting '{name}' values: argument has {explicit}, "
            f"config has {configured}"
        )
    return explicit


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
    elif tmax is not None:
        raise ValueError("nsample and tmax are mutually exclusive")
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    return nsample


def _validate_config_for_room(config: SimulationConfig, room: Room) -> None:
    if config.fs is not None:
        warnings.warn(
            "SimulationConfig.fs is deprecated; Room.fs is the source of truth.",
            DeprecationWarning,
            stacklevel=3,
        )
        if config.fs != room.fs:
            raise ValueError(
                f"SimulationConfig.fs ({config.fs}) conflicts with Room.fs ({room.fs})"
            )
    if config.mixed_precision:
        warnings.warn(
            "SimulationConfig.mixed_precision is not implemented; use dtype instead.",
            DeprecationWarning,
            stacklevel=3,
        )


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
    elif tmax is not None:
        raise ValueError("nsample and tmax are mutually exclusive")
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


def _validate_positions_in_room(
    positions: Tensor, room_size: Tensor, *, name: str
) -> None:
    if not torch.all(torch.isfinite(positions)):
        raise ValueError(f"{name} must contain finite values")
    if torch.any(positions < 0) or torch.any(positions > room_size):
        raise ValueError(f"{name} must lie within room bounds [0, room.size]")


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
