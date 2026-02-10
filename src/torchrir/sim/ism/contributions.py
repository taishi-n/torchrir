"""Compute image-source contributions for ISM."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ..directivity import directivity_gain
from ...models import Room
from .helpers import _cos_between
from .images import _image_positions, _image_positions_batch


def _compute_image_contributions(
    src: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    fdl2: int,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dir: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute sample positions and attenuation for a source and all mics."""
    img = _image_positions(src, room_size, n_vec)
    vec = mic_pos[:, None, :] - img[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    dist = torch.clamp(dist, min=1e-6)
    time = dist / room.c
    time = time + (fdl2 / room.fs)
    sample = time * room.fs

    gain = refl[None, :]
    if src_pattern != "omni":
        if src_dir is None:
            raise ValueError("source orientation required for non-omni directivity")
        cos_theta = _cos_between(vec, src_dir)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        cos_theta = _cos_between(-vec, mic_dir)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = gain / dist
    return sample, attenuation


def _compute_image_contributions_batch(
    src_pos: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    fdl2: int,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dirs: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute samples/attenuation for all sources/mics/images in batch."""
    img = _image_positions_batch(src_pos, room_size, n_vec)
    vec = mic_pos[None, :, None, :] - img[:, None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    dist = torch.clamp(dist, min=1e-6)
    time = dist / room.c
    time = time + (fdl2 / room.fs)
    sample = time * room.fs

    gain = refl.view(1, 1, -1)
    if src_pattern != "omni":
        if src_dirs is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = src_dirs[:, None, None, :]
        cos_theta = _cos_between(vec, src_dirs)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = (
            mic_dir[None, :, None, :]
            if mic_dir.ndim == 2
            else mic_dir.view(1, 1, 1, -1)
        )
        cos_theta = _cos_between(-vec, mic_dir)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = gain / dist
    return sample, attenuation


def _compute_image_contributions_time_batch(
    src_traj: Tensor,
    mic_traj: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    fdl2: int,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dirs: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute samples/attenuation for all time steps in batch."""
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(dtype=src_traj.dtype)
    n = torch.floor_divide(n_vec + 1, 2).to(dtype=src_traj.dtype)
    base = 2.0 * room_size * n
    img = base[None, None, :, :] + sign[None, None, :, :] * src_traj[:, :, None, :]
    vec = mic_traj[:, None, :, None, :] - img[:, :, None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    dist = torch.clamp(dist, min=1e-6)
    time = dist / room.c
    time = time + (fdl2 / room.fs)
    sample = time * room.fs

    gain = refl.view(1, 1, 1, -1)
    if src_pattern != "omni":
        if src_dirs is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs_b = src_dirs[None, :, None, None, :]
        cos_theta = _cos_between(vec, src_dirs_b)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir_b = (
            mic_dir[None, None, :, None, :]
            if mic_dir.ndim == 2
            else mic_dir.view(1, 1, 1, 1, -1)
        )
        cos_theta = _cos_between(-vec, mic_dir_b)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = gain / dist
    return sample, attenuation
