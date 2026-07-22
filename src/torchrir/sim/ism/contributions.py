"""Compute image-source contributions for ISM."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..directivity import directivity_gain
from ...models import Room
from ...util.tensor import stable_vector_norm
from .helpers import _cos_between
from .images import _image_positions, _image_positions_batch


def _reflected_source_directions(src_dirs: Tensor, n_vec: Tensor) -> Tensor:
    """Mirror source orientations into each image-source coordinate system.

    Mirroring a source across a wall flips the orientation component normal to
    that wall.  The image index parity therefore applies the same sign change
    to the source direction as it does to the source position.
    """
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(
        device=src_dirs.device, dtype=src_dirs.dtype
    )
    return src_dirs[..., None, :] * sign


def _compute_image_contributions(
    src: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dir: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute sample positions and attenuation for a source and all mics."""
    img = _image_positions(src, room_size, n_vec)
    vec = mic_pos[:, None, :] - img[None, :, :]
    dist, sample = _path_distances_and_samples(vec, room)

    gain = refl[None, :]
    if src_pattern != "omni":
        if src_dir is None:
            raise ValueError("source orientation required for non-omni directivity")
        reflected_src_dir = _reflected_source_directions(src_dir, n_vec)
        cos_theta = _cos_between(vec, reflected_src_dir)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir_b = mic_dir[:, None, :] if mic_dir.ndim == 2 else mic_dir
        cos_theta = _cos_between(-vec, mic_dir_b)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = _validated_attenuation(gain / dist)
    return sample, attenuation


def _compute_image_contributions_batch(
    src_pos: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dirs: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute samples/attenuation for all sources/mics/images in batch."""
    img = _image_positions_batch(src_pos, room_size, n_vec)
    vec = mic_pos[None, :, None, :] - img[:, None, :, :]
    dist, sample = _path_distances_and_samples(vec, room)

    gain = refl.view(1, 1, -1)
    if src_pattern != "omni":
        if src_dirs is None:
            raise ValueError("source orientation required for non-omni directivity")
        reflected_src_dirs = _reflected_source_directions(src_dirs, n_vec)
        cos_theta = _cos_between(vec, reflected_src_dirs[:, None, :, :])
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

    attenuation = _validated_attenuation(gain / dist)
    return sample, attenuation


def _compute_image_contributions_time_batch(
    src_traj: Tensor,
    mic_traj: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dirs: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute samples/attenuation for all time steps in batch."""
    img = _image_positions_batch(src_traj, room_size, n_vec)
    vec = mic_traj[:, None, :, None, :] - img[:, :, None, :, :]
    dist, sample = _path_distances_and_samples(vec, room)

    gain = refl.view(1, 1, 1, -1)
    if src_pattern != "omni":
        if src_dirs is None:
            raise ValueError("source orientation required for non-omni directivity")
        reflected_src_dirs = _reflected_source_directions(src_dirs, n_vec)
        src_dirs_b = reflected_src_dirs[None, :, None, :, :]
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

    attenuation = _validated_attenuation(gain / dist)
    return sample, attenuation


def _path_distances_and_samples(vec: Tensor, room: Room) -> tuple[Tensor, Tensor]:
    if not torch.all(torch.isfinite(vec)):
        raise ValueError(
            "image-source displacement must be representable in the simulation dtype"
        )
    distance = stable_vector_norm(vec, dim=-1)
    if torch.any(torch.isnan(distance)) or torch.any(distance <= 0):
        raise ValueError("image-source distances must be positive and representable")
    return distance, _distances_to_samples(distance, fs=room.fs, c=room.c)


def _distances_to_samples(distance: Tensor, *, fs: float, c: float) -> Tensor:
    """Scale distances by ``fs / c`` without overflowing a temporary ratio."""

    fs_mantissa, fs_exponent = math.frexp(fs)
    c_mantissa, c_exponent = math.frexp(c)
    ratio_mantissa, ratio_adjustment = math.frexp(fs_mantissa / c_mantissa)
    ratio_exponent = fs_exponent - c_exponent + ratio_adjustment

    if ratio_exponent > 0:
        ratio_mantissa *= 2.0
        ratio_exponent -= 1
    sample = distance * ratio_mantissa
    step_limit = 500 if distance.dtype == torch.float64 else 60
    while ratio_exponent != 0:
        step = max(-step_limit, min(step_limit, ratio_exponent))
        sample = sample * math.ldexp(1.0, step)
        ratio_exponent -= step

    if torch.any(torch.isnan(sample)) or torch.any(sample < 0):
        raise ValueError("image-source sample delays must be non-negative real values")
    return torch.clamp(sample, max=torch.finfo(distance.dtype).max)


def _validated_attenuation(attenuation: Tensor) -> Tensor:
    if not torch.all(torch.isfinite(attenuation)):
        raise ValueError(
            "image-source attenuation must be representable in the simulation dtype"
        )
    return attenuation
