"""Validation helpers for ISM inputs."""

from __future__ import annotations

import torch
from torch import Tensor

from ...util.tensor import stable_vector_norm


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
    if torch.any(positions <= 0) or torch.any(positions >= room_size):
        raise ValueError(f"{name} must lie strictly inside room bounds (0, room.size)")


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


def _validate_source_mic_separation(
    source_positions: Tensor,
    microphone_positions: Tensor,
    *,
    min_distance: float,
) -> None:
    """Reject singular source--microphone pairs with actionable indices."""
    if source_positions.ndim == 2:
        delta = source_positions[:, None, :] - microphone_positions[None, :, :]
    elif source_positions.ndim == 3:
        if microphone_positions.ndim != 3:
            raise ValueError("source and microphone trajectories must both be 3D")
        delta = source_positions[:, :, None, :] - microphone_positions[:, None, :, :]
    else:
        raise ValueError("source positions must be static positions or a trajectory")

    distance = stable_vector_norm(delta, dim=-1)
    invalid = torch.nonzero(distance < min_distance, as_tuple=False)
    if invalid.numel() == 0:
        return

    first = invalid[0].tolist()
    if distance.ndim == 2:
        source_index, microphone_index = first
        where = f"source {source_index}, microphone {microphone_index}"
    else:
        frame_index, source_index, microphone_index = first
        where = (
            f"frame {frame_index}, source {source_index}, microphone {microphone_index}"
        )
    value = float(distance[tuple(first)].item())
    raise ValueError(
        f"source--microphone distance at {where} is {value:.6g} m; "
        f"it must be at least {min_distance:.6g} m"
    )
