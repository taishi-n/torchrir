"""Preprocessing helpers for ISM inputs."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...models import MicrophoneArray, Room, Source
from ...util.device import infer_device_dtype, resolve_device
from ...util.tensor import as_tensor, ensure_dim
from .helpers import _prepare_entities


def _prepare_static_tensors(
    *,
    room: Room,
    sources: Source | Tensor,
    mics: MicrophoneArray | Tensor,
    orientation: Optional[Tensor | tuple[Tensor, Tensor]],
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> tuple[
    Tensor,
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
    Tensor,
    int,
    torch.device,
    torch.dtype,
]:
    if isinstance(device, str):
        device = resolve_device(device)

    src_pos, src_ori = _prepare_entities(
        sources, orientation, which="source", device=device, dtype=dtype
    )
    mic_pos, mic_ori = _prepare_entities(
        mics, orientation, which="mic", device=device, dtype=dtype
    )

    device, dtype = infer_device_dtype(
        src_pos, mic_pos, room.size, device=device, dtype=dtype
    )
    src_pos = as_tensor(src_pos, device=device, dtype=dtype)
    mic_pos = as_tensor(mic_pos, device=device, dtype=dtype)
    if src_ori is not None:
        src_ori = as_tensor(src_ori, device=device, dtype=dtype)
    if mic_ori is not None:
        mic_ori = as_tensor(mic_ori, device=device, dtype=dtype)

    room_size = as_tensor(room.size, device=device, dtype=dtype)
    room_size = ensure_dim(room_size)
    dim = int(room_size.numel())

    if src_pos.ndim == 1:
        src_pos = src_pos.unsqueeze(0)
    if mic_pos.ndim == 1:
        mic_pos = mic_pos.unsqueeze(0)

    return src_pos, mic_pos, src_ori, mic_ori, room_size, dim, device, dtype


def _prepare_dynamic_tensors(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    orientation: Optional[Tensor | tuple[Tensor, Tensor]],
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> tuple[
    Tensor,
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
    Tensor,
    int,
    torch.device,
    torch.dtype,
]:
    if isinstance(device, str):
        device = resolve_device(device)

    src_traj = as_tensor(src_traj, device=device, dtype=dtype)
    mic_traj = as_tensor(mic_traj, device=device, dtype=dtype)
    device, dtype = infer_device_dtype(
        src_traj, mic_traj, room.size, device=device, dtype=dtype
    )
    src_traj = as_tensor(src_traj, device=device, dtype=dtype)
    mic_traj = as_tensor(mic_traj, device=device, dtype=dtype)

    if src_traj.ndim == 2:
        src_traj = src_traj.unsqueeze(1)
    if mic_traj.ndim == 2:
        mic_traj = mic_traj.unsqueeze(1)

    src_ori, mic_ori = _split_orientation(orientation)
    if src_ori is not None:
        src_ori = as_tensor(src_ori, device=device, dtype=dtype)
    if mic_ori is not None:
        mic_ori = as_tensor(mic_ori, device=device, dtype=dtype)

    room_size = as_tensor(room.size, device=device, dtype=dtype)
    room_size = ensure_dim(room_size)
    dim = int(room_size.numel())

    return src_traj, mic_traj, src_ori, mic_ori, room_size, dim, device, dtype


def _split_orientation(
    orientation: Optional[Tensor | tuple[Tensor, Tensor]],
) -> tuple[Optional[Tensor], Optional[Tensor]]:
    if orientation is None:
        return None, None
    if isinstance(orientation, (list, tuple)):
        if len(orientation) != 2:
            raise ValueError("orientation tuple must have length 2")
        return orientation[0], orientation[1]
    return orientation, orientation
