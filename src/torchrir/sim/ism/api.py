"""ISM API for static and dynamic RIR simulation."""

from __future__ import annotations

from typing import Optional, Tuple
import warnings

import torch
from torch import Tensor

from ...config import SimulationConfig
from ...models import MicrophoneArray, Room, Source
from .accumulate import _accumulate_rir_batch
from .context import prepare_ism_context, prepare_source_directions
from .contributions import (
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
)
from .diffuse import _apply_diffuse_tail
from .hpf import apply_rir_hpf
from .prepare import _prepare_dynamic_tensors, _prepare_static_tensors
from .validate import (
    _resolve_config,
    _validate_dynamic_args,
    _validate_config_for_room,
    _validate_pos_shapes,
    _validate_positions_in_room,
    _validate_static_args,
    _validate_traj_shapes,
)


def simulate_rir(
    *,
    room: Room,
    sources: Source | Tensor,
    mics: MicrophoneArray | Tensor,
    max_order: int | None = None,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] | None = None,
    orientation: Optional[Tensor | tuple[Optional[Tensor], Optional[Tensor]]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate a static RIR using the image source method."""
    _warn_legacy_settings(
        max_order, nb_img, nsample, tmax, tdiff, directivity, device, dtype
    )
    (
        cfg,
        device,
        max_order,
        nsample,
        tmax,
        tdiff,
        directivity,
        dtype,
        nb_img,
    ) = _resolve_config(
        config=config,
        device=device,
        max_order=max_order,
        nsample=nsample,
        tmax=tmax,
        tdiff=tdiff,
        directivity=directivity,
        dtype=dtype,
        nb_img=nb_img,
    )
    _validate_config_for_room(cfg, room)
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
    _validate_positions_in_room(src_pos, room_size, name="source positions")
    _validate_positions_in_room(mic_pos, room_size, name="microphone positions")

    context = prepare_ism_context(
        room=room,
        room_size=room_size,
        dim=dim,
        device=device,
        dtype=dtype,
        max_order=max_order,
        nb_img=nb_img,
        directivity=directivity,
        microphone_orientation=mic_ori,
        config=cfg,
    )

    n_src = src_pos.shape[0]
    n_mic = mic_pos.shape[0]
    rir = torch.zeros((n_src, n_mic, nsample), device=device, dtype=dtype)
    fdl2 = context.fractional_delay_half_length
    img_chunk = context.image_chunk_size
    if img_chunk <= 0:
        img_chunk = context.image_indices.shape[0]

    src_dirs = prepare_source_directions(
        src_ori, pattern=context.source_pattern, dim=dim, count=n_src
    )

    for start in range(0, context.image_indices.shape[0], img_chunk):
        end = min(start + img_chunk, context.image_indices.shape[0])
        n_vec_chunk = context.image_indices[start:end]
        refl_chunk = context.reflection_coefficients[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_batch(
            src_pos,
            mic_pos,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=context.source_pattern,
            mic_pattern=context.microphone_pattern,
            src_dirs=src_dirs,
            mic_dir=context.microphone_directions,
        )
        _accumulate_rir_batch(rir, sample_chunk, attenuation_chunk, cfg)

    duration = nsample / room.fs
    if tdiff is not None:
        if tdiff >= duration:
            raise ValueError("tdiff must be smaller than the RIR duration")
        rir = _apply_diffuse_tail(
            rir, room, context.beta, tdiff, duration, seed=cfg.seed
        )
    rir = apply_rir_hpf(rir, room.fs, cfg)
    return rir


def simulate_dynamic_rir(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    max_order: int | None = None,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] | None = None,
    orientation: Optional[Tensor | tuple[Optional[Tensor], Optional[Tensor]]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate time-varying RIRs for source/mic trajectories."""
    _warn_legacy_settings(
        max_order, nb_img, nsample, tmax, tdiff, directivity, device, dtype
    )
    (
        cfg,
        device,
        max_order,
        nsample,
        tmax,
        tdiff,
        directivity,
        dtype,
        nb_img,
    ) = _resolve_config(
        config=config,
        device=device,
        max_order=max_order,
        nsample=nsample,
        tmax=tmax,
        tdiff=tdiff,
        directivity=directivity,
        dtype=dtype,
        nb_img=nb_img,
    )
    _validate_config_for_room(cfg, room)
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
    _validate_positions_in_room(src_traj, room_size, name="src_traj")
    _validate_positions_in_room(mic_traj, room_size, name="mic_traj")

    context = prepare_ism_context(
        room=room,
        room_size=room_size,
        dim=dim,
        device=device,
        dtype=dtype,
        max_order=max_order,
        nb_img=nb_img,
        directivity=directivity,
        microphone_orientation=mic_ori,
        config=cfg,
    )

    n_src = src_traj.shape[1]
    n_mic = mic_traj.shape[1]
    rirs = torch.zeros(
        (src_traj.shape[0], n_src, n_mic, nsample), device=device, dtype=dtype
    )
    fdl2 = context.fractional_delay_half_length
    img_chunk = context.image_chunk_size
    if img_chunk <= 0:
        img_chunk = context.image_indices.shape[0]

    src_dirs = prepare_source_directions(
        src_ori, pattern=context.source_pattern, dim=dim, count=n_src
    )

    for start in range(0, context.image_indices.shape[0], img_chunk):
        end = min(start + img_chunk, context.image_indices.shape[0])
        n_vec_chunk = context.image_indices[start:end]
        refl_chunk = context.reflection_coefficients[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_time_batch(
            src_traj,
            mic_traj,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=context.source_pattern,
            mic_pattern=context.microphone_pattern,
            src_dirs=src_dirs,
            mic_dir=context.microphone_directions,
        )
        t_steps = src_traj.shape[0]
        sample_flat = sample_chunk.reshape(t_steps * n_src, n_mic, -1)
        attenuation_flat = attenuation_chunk.reshape(t_steps * n_src, n_mic, -1)
        rir_flat = rirs.view(t_steps * n_src, n_mic, nsample)
        _accumulate_rir_batch(rir_flat, sample_flat, attenuation_flat, cfg)

    duration = nsample / room.fs
    if tdiff is not None:
        if tdiff >= duration:
            raise ValueError("tdiff must be smaller than the RIR duration")
        rirs = _apply_diffuse_tail(
            rirs, room, context.beta, tdiff, duration, seed=cfg.seed
        )
    rirs = apply_rir_hpf(rirs, room.fs, cfg)
    return rirs


def _warn_legacy_settings(*values: object) -> None:
    if any(value is not None for value in values):
        warnings.warn(
            "Passing simulation settings as individual arguments is deprecated "
            "and will be removed in TorchRIR 1.0. Use SimulationConfig.",
            DeprecationWarning,
            stacklevel=3,
        )
