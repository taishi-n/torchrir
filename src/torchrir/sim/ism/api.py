"""Private static and dynamic image-source kernels."""

from __future__ import annotations

import torch
from torch import Tensor

from ...config import ResolvedSimulationConfig
from ...models import DynamicScene, StaticScene
from .accumulate import _accumulate_rir_batch
from .context import prepare_ism_context, prepare_source_directions
from .contributions import (
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
)
from .diffuse import _apply_diffuse_tail
from .hpf import apply_rir_hpf
from .images import _iter_image_source_index_chunks, _reflection_coefficients
from .prepare import _prepare_dynamic_tensors, _prepare_static_tensors
from .validate import (
    _validate_pos_shapes,
    _validate_positions_in_room,
    _validate_source_mic_separation,
    _validate_traj_shapes,
)


def _simulate_static_rir(
    scene: StaticScene,
    config: ResolvedSimulationConfig,
) -> Tensor:
    """Execute the static ISM kernel for a validated scene."""

    room = scene.room
    (
        source_positions,
        microphone_positions,
        source_orientation,
        microphone_orientation,
        room_size,
        dimension,
    ) = _prepare_static_tensors(
        room=room,
        sources=scene.sources,
        microphones=scene.mics,
        device=config.device,
        dtype=config.dtype,
    )
    _validate_pos_shapes(source_positions, microphone_positions, dimension)
    _validate_positions_in_room(source_positions, room_size, name="source positions")
    _validate_positions_in_room(
        microphone_positions, room_size, name="microphone positions"
    )
    _validate_source_mic_separation(
        source_positions,
        microphone_positions,
        min_distance=config.min_source_mic_distance,
    )

    context = prepare_ism_context(
        room=room,
        room_size=room_size,
        dim=dimension,
        dtype=config.dtype,
        max_order=config.max_order,
        nb_img=config.nb_img,
        source_directivity=scene.sources.directivity,
        microphone_directivity=scene.mics.directivity,
        microphone_orientation=microphone_orientation,
        config=config,
    )

    n_sources = source_positions.shape[0]
    n_microphones = microphone_positions.shape[0]
    rir = torch.zeros(
        (n_sources, n_microphones, config.nsample),
        device=config.device,
        dtype=config.dtype,
    )
    source_directions = prepare_source_directions(
        source_orientation,
        pattern=context.source_pattern,
        dim=dimension,
        count=n_sources,
    )
    max_arrival_sample = (
        torch.full(
            (n_sources, n_microphones),
            -torch.inf,
            device=config.device,
            dtype=config.dtype,
        )
        if _needs_pyroom_zero_phase_lengths(config)
        else None
    )

    for image_indices in _iter_image_source_index_chunks(
        context.max_order,
        context.dimension,
        device=config.device,
        nb_img=context.nb_img,
        chunk_size=context.image_chunk_size,
    ):
        reflection_coefficients = _reflection_coefficients(
            image_indices,
            context.beta,
        )
        sample, attenuation = _compute_image_contributions_batch(
            source_positions,
            microphone_positions,
            room_size,
            image_indices,
            reflection_coefficients,
            room,
            src_pattern=context.source_pattern,
            mic_pattern=context.microphone_pattern,
            src_dirs=source_directions,
            mic_dir=context.microphone_directions,
        )
        if max_arrival_sample is not None:
            max_arrival_sample = torch.maximum(
                max_arrival_sample,
                torch.amax(sample, dim=-1),
            )
        _accumulate_rir_batch(rir, sample, attenuation, config)

    zero_phase_lengths = (
        _pyroom_zero_phase_lengths(max_arrival_sample, config)
        if max_arrival_sample is not None
        else None
    )
    if config.tdiff is not None:
        rir = _apply_diffuse_tail(
            rir,
            room_size,
            context.beta,
            config.tdiff,
            config.tmax,
            fs=room.fs,
            c=room.c,
            seed=config.seed,
        )
    return apply_rir_hpf(
        rir,
        room.fs,
        config.high_pass,
        zero_phase_lengths=zero_phase_lengths,
    )


def _simulate_dynamic_rir(
    scene: DynamicScene,
    config: ResolvedSimulationConfig,
) -> Tensor:
    """Execute the dynamic ISM kernel for a validated scene."""

    room = scene.room
    (
        source_trajectory,
        microphone_trajectory,
        source_orientation,
        microphone_orientation,
        room_size,
        dimension,
    ) = _prepare_dynamic_tensors(
        room=room,
        source_trajectory=scene.src_traj,
        microphone_trajectory=scene.mic_traj,
        source_orientation=scene.sources.orientation,
        microphone_orientation=scene.mics.orientation,
        device=config.device,
        dtype=config.dtype,
    )
    _validate_traj_shapes(source_trajectory, microphone_trajectory, dimension)
    _validate_positions_in_room(source_trajectory, room_size, name="src_traj")
    _validate_positions_in_room(microphone_trajectory, room_size, name="mic_traj")
    _validate_source_mic_separation(
        source_trajectory,
        microphone_trajectory,
        min_distance=config.min_source_mic_distance,
    )

    context = prepare_ism_context(
        room=room,
        room_size=room_size,
        dim=dimension,
        dtype=config.dtype,
        max_order=config.max_order,
        nb_img=config.nb_img,
        source_directivity=scene.sources.directivity,
        microphone_directivity=scene.mics.directivity,
        microphone_orientation=microphone_orientation,
        config=config,
    )

    time_steps, n_sources = source_trajectory.shape[:2]
    n_microphones = microphone_trajectory.shape[1]
    rirs = torch.zeros(
        (time_steps, n_sources, n_microphones, config.nsample),
        device=config.device,
        dtype=config.dtype,
    )
    source_directions = prepare_source_directions(
        source_orientation,
        pattern=context.source_pattern,
        dim=dimension,
        count=n_sources,
    )
    max_arrival_sample = (
        torch.full(
            (time_steps, n_sources, n_microphones),
            -torch.inf,
            device=config.device,
            dtype=config.dtype,
        )
        if _needs_pyroom_zero_phase_lengths(config)
        else None
    )

    for image_indices in _iter_image_source_index_chunks(
        context.max_order,
        context.dimension,
        device=config.device,
        nb_img=context.nb_img,
        chunk_size=context.image_chunk_size,
    ):
        reflection_coefficients = _reflection_coefficients(
            image_indices,
            context.beta,
        )
        sample, attenuation = _compute_image_contributions_time_batch(
            source_trajectory,
            microphone_trajectory,
            room_size,
            image_indices,
            reflection_coefficients,
            room,
            src_pattern=context.source_pattern,
            mic_pattern=context.microphone_pattern,
            src_dirs=source_directions,
            mic_dir=context.microphone_directions,
        )
        if max_arrival_sample is not None:
            max_arrival_sample = torch.maximum(
                max_arrival_sample,
                torch.amax(sample, dim=-1),
            )
        sample_flat = sample.reshape(time_steps * n_sources, n_microphones, -1)
        attenuation_flat = attenuation.reshape(
            time_steps * n_sources, n_microphones, -1
        )
        rir_flat = rirs.view(
            time_steps * n_sources,
            n_microphones,
            config.nsample,
        )
        _accumulate_rir_batch(rir_flat, sample_flat, attenuation_flat, config)

    zero_phase_lengths = (
        _pyroom_zero_phase_lengths(max_arrival_sample, config)
        if max_arrival_sample is not None
        else None
    )
    if config.tdiff is not None:
        rirs = _apply_diffuse_tail(
            rirs,
            room_size,
            context.beta,
            config.tdiff,
            config.tmax,
            fs=room.fs,
            c=room.c,
            seed=config.seed,
        )
    return apply_rir_hpf(
        rirs,
        room.fs,
        config.high_pass,
        zero_phase_lengths=zero_phase_lengths,
    )


def _pyroom_zero_phase_lengths(
    max_arrival_sample: Tensor,
    config: ResolvedSimulationConfig,
) -> Tensor:
    """Return pyroomacoustics-compatible physical-axis RIR lengths."""

    fractional_delay_half = (config.frac_delay_length - 1) // 2
    lengths = torch.ceil(max_arrival_sample) + float(fractional_delay_half) + 2.0
    return torch.clamp(lengths, min=1.0, max=float(config.nsample)).to(torch.int64)


def _needs_pyroom_zero_phase_lengths(config: ResolvedSimulationConfig) -> bool:
    """Return whether simulation must track finite pyroomacoustics RIR horizons."""

    return (
        config.high_pass is not None
        and config.high_pass.phase == "zero_phase"
        and config.tdiff is None
    )


__all__: list[str] = []
