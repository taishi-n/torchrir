from __future__ import annotations

"""ISM API for static and dynamic RIR simulation."""

from typing import Optional, Tuple

import torch
from torch import Tensor

from ...config import SimulationConfig
from ..directivity import split_directivity
from ...models import MicrophoneArray, Room, Source
from ...util.orientation import orientation_to_unit
from .accumulate import _accumulate_rir_batch
from .contributions import (
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
)
from .diffuse import _apply_diffuse_tail
from .helpers import _resolve_beta, _validate_beta
from .images import _image_source_indices, _reflection_coefficients
from .prepare import _prepare_dynamic_tensors, _prepare_static_tensors
from .validate import (
    _resolve_config,
    _validate_dynamic_args,
    _validate_pos_shapes,
    _validate_static_args,
    _validate_traj_shapes,
)


def simulate_rir(
    *,
    room: Room,
    sources: Source | Tensor,
    mics: MicrophoneArray | Tensor,
    max_order: int | None,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate a static RIR using the image source method."""
    cfg, device, max_order, tmax, directivity = _resolve_config(
        config=config,
        device=device,
        max_order=max_order,
        tmax=tmax,
        directivity=directivity,
    )
    nsample = _validate_static_args(
        room=room, nsample=nsample, tmax=tmax, max_order=max_order
    )
    (
        src_pos,
        mic_pos,
        src_ori,
        mic_ori,
        room_size,
        dim,
        device,
        dtype,
    ) = _prepare_static_tensors(
        room=room,
        sources=sources,
        mics=mics,
        orientation=orientation,
        device=device,
        dtype=dtype,
    )
    _validate_pos_shapes(src_pos, mic_pos, dim)

    beta = _resolve_beta(room, room_size, device=device, dtype=dtype)
    beta = _validate_beta(beta, dim)

    n_vec = _image_source_indices(max_order, dim, device=device, nb_img=nb_img)
    refl = _reflection_coefficients(n_vec, beta)

    src_pattern, mic_pattern = split_directivity(directivity)
    mic_dir = None
    if mic_pattern != "omni":
        if mic_ori is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = orientation_to_unit(mic_ori, dim)

    n_src = src_pos.shape[0]
    n_mic = mic_pos.shape[0]
    rir = torch.zeros((n_src, n_mic, nsample), device=device, dtype=dtype)
    fdl = cfg.frac_delay_length
    fdl2 = (fdl - 1) // 2
    img_chunk = cfg.image_chunk_size
    if img_chunk <= 0:
        img_chunk = n_vec.shape[0]

    src_dirs = None
    if src_pattern != "omni":
        if src_ori is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = orientation_to_unit(src_ori, dim)
        if src_dirs.ndim == 1:
            src_dirs = src_dirs.unsqueeze(0).repeat(n_src, 1)
        if src_dirs.ndim != 2 or src_dirs.shape[0] != n_src:
            raise ValueError("source orientation must match number of sources")

    for start in range(0, n_vec.shape[0], img_chunk):
        end = min(start + img_chunk, n_vec.shape[0])
        n_vec_chunk = n_vec[start:end]
        refl_chunk = refl[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_batch(
            src_pos,
            mic_pos,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=src_pattern,
            mic_pattern=mic_pattern,
            src_dirs=src_dirs,
            mic_dir=mic_dir,
        )
        _accumulate_rir_batch(rir, sample_chunk, attenuation_chunk, cfg)

    if tdiff is not None and tmax is not None and tdiff < tmax:
        rir = _apply_diffuse_tail(rir, room, beta, tdiff, tmax, seed=cfg.seed)
    return rir


def simulate_dynamic_rir(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    max_order: int | None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate time-varying RIRs for source/mic trajectories."""
    cfg, device, max_order, tmax, directivity = _resolve_config(
        config=config,
        device=device,
        max_order=max_order,
        tmax=tmax,
        directivity=directivity,
    )
    nsample = _validate_dynamic_args(
        room=room, nsample=nsample, tmax=tmax, max_order=max_order
    )
    (
        src_traj,
        mic_traj,
        src_ori,
        mic_ori,
        room_size,
        dim,
        device,
        dtype,
    ) = _prepare_dynamic_tensors(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        orientation=orientation,
        device=device,
        dtype=dtype,
    )
    _validate_traj_shapes(src_traj, mic_traj, dim)

    beta = _resolve_beta(room, room_size, device=device, dtype=dtype)
    beta = _validate_beta(beta, dim)
    n_vec = _image_source_indices(max_order, dim, device=device, nb_img=None)
    refl = _reflection_coefficients(n_vec, beta)

    src_pattern, mic_pattern = split_directivity(directivity)
    mic_dir = None
    if mic_pattern != "omni":
        if mic_ori is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = orientation_to_unit(mic_ori, dim)

    n_src = src_traj.shape[1]
    n_mic = mic_traj.shape[1]
    rirs = torch.zeros(
        (src_traj.shape[0], n_src, n_mic, nsample), device=device, dtype=dtype
    )
    fdl = cfg.frac_delay_length
    fdl2 = (fdl - 1) // 2
    img_chunk = cfg.image_chunk_size
    if img_chunk <= 0:
        img_chunk = n_vec.shape[0]

    src_dirs = None
    if src_pattern != "omni":
        if src_ori is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = orientation_to_unit(src_ori, dim)
        if src_dirs.ndim == 1:
            src_dirs = src_dirs.unsqueeze(0).repeat(n_src, 1)
        if src_dirs.ndim != 2 or src_dirs.shape[0] != n_src:
            raise ValueError("source orientation must match number of sources")

    for start in range(0, n_vec.shape[0], img_chunk):
        end = min(start + img_chunk, n_vec.shape[0])
        n_vec_chunk = n_vec[start:end]
        refl_chunk = refl[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_time_batch(
            src_traj,
            mic_traj,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=src_pattern,
            mic_pattern=mic_pattern,
            src_dirs=src_dirs,
            mic_dir=mic_dir,
        )
        t_steps = src_traj.shape[0]
        sample_flat = sample_chunk.reshape(t_steps * n_src, n_mic, -1)
        attenuation_flat = attenuation_chunk.reshape(t_steps * n_src, n_mic, -1)
        rir_flat = rirs.view(t_steps * n_src, n_mic, nsample)
        _accumulate_rir_batch(rir_flat, sample_flat, attenuation_flat, cfg)

    return rirs
