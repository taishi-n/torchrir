"""Move validated scene tensors to the resolved execution device and dtype."""

from __future__ import annotations

import torch
from torch import Tensor

from ...models import MicrophoneArray, Room, Source
from ...util.tensor import as_float_tensor, ensure_dim


def _prepare_static_tensors(
    *,
    room: Room,
    sources: Source,
    microphones: MicrophoneArray,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    int,
]:
    source_positions = as_float_tensor(
        sources.positions,
        device=device,
        dtype=dtype,
        name="source positions",
    )
    microphone_positions = as_float_tensor(
        microphones.positions,
        device=device,
        dtype=dtype,
        name="microphone positions",
    )
    source_orientation = _move_optional_tensor(
        sources.orientation,
        device=device,
        dtype=dtype,
        name="source orientation",
    )
    microphone_orientation = _move_optional_tensor(
        microphones.orientation,
        device=device,
        dtype=dtype,
        name="microphone orientation",
    )
    room_size = ensure_dim(
        as_float_tensor(room.size, device=device, dtype=dtype, name="room size")
    )
    return (
        source_positions,
        microphone_positions,
        source_orientation,
        microphone_orientation,
        room_size,
        int(room_size.numel()),
    )


def _prepare_dynamic_tensors(
    *,
    room: Room,
    source_trajectory: Tensor,
    microphone_trajectory: Tensor,
    source_orientation: Tensor | None,
    microphone_orientation: Tensor | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    int,
]:
    source_trajectory = as_float_tensor(
        source_trajectory,
        device=device,
        dtype=dtype,
        name="source trajectory",
    )
    microphone_trajectory = as_float_tensor(
        microphone_trajectory,
        device=device,
        dtype=dtype,
        name="microphone trajectory",
    )
    if source_trajectory.ndim == 2:
        source_trajectory = source_trajectory.unsqueeze(1)
    if microphone_trajectory.ndim == 2:
        microphone_trajectory = microphone_trajectory.unsqueeze(1)

    source_orientation = _move_optional_tensor(
        source_orientation,
        device=device,
        dtype=dtype,
        name="source orientation",
    )
    microphone_orientation = _move_optional_tensor(
        microphone_orientation,
        device=device,
        dtype=dtype,
        name="microphone orientation",
    )
    room_size = ensure_dim(
        as_float_tensor(room.size, device=device, dtype=dtype, name="room size")
    )
    return (
        source_trajectory,
        microphone_trajectory,
        source_orientation,
        microphone_orientation,
        room_size,
        int(room_size.numel()),
    )


def _move_optional_tensor(
    value: Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor | None:
    if value is None:
        return None
    return as_float_tensor(value, device=device, dtype=dtype, name=name)
