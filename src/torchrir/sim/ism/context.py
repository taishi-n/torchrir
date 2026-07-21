"""Shared preparation for static and dynamic image-source simulations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...config import SimulationConfig
from ...models import Room
from ...util.orientation import orientation_to_unit
from ..directivity import split_directivity
from .helpers import _resolve_beta, _validate_beta
from .images import _image_source_indices, _reflection_coefficients


@dataclass(frozen=True)
class ISMContext:
    beta: Tensor
    image_indices: Tensor
    reflection_coefficients: Tensor
    source_pattern: str
    microphone_pattern: str
    microphone_directions: Tensor | None
    fractional_delay_half_length: int
    image_chunk_size: int


def prepare_ism_context(
    *,
    room: Room,
    room_size: Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    max_order: int,
    nb_img: Tensor | tuple[int, ...] | None,
    directivity: str | tuple[str, str],
    microphone_orientation: Tensor | None,
    config: SimulationConfig,
) -> ISMContext:
    beta = _validate_beta(
        _resolve_beta(room, room_size, device=device, dtype=dtype), dim
    )
    image_indices = _image_source_indices(max_order, dim, device=device, nb_img=nb_img)
    reflection_coefficients = _reflection_coefficients(image_indices, beta)
    source_pattern, microphone_pattern = split_directivity(directivity)

    microphone_directions = None
    if microphone_pattern != "omni":
        if microphone_orientation is None:
            raise ValueError("mic orientation required for non-omni directivity")
        microphone_directions = orientation_to_unit(microphone_orientation, dim)

    return ISMContext(
        beta=beta,
        image_indices=image_indices,
        reflection_coefficients=reflection_coefficients,
        source_pattern=source_pattern,
        microphone_pattern=microphone_pattern,
        microphone_directions=microphone_directions,
        fractional_delay_half_length=(config.frac_delay_length - 1) // 2,
        image_chunk_size=config.image_chunk_size,
    )


def prepare_source_directions(
    orientation: Tensor | None,
    *,
    pattern: str,
    dim: int,
    count: int,
) -> Tensor | None:
    if pattern == "omni":
        return None
    if orientation is None:
        raise ValueError("source orientation required for non-omni directivity")
    directions = orientation_to_unit(orientation, dim)
    if directions.ndim == 1:
        directions = directions.unsqueeze(0).repeat(count, 1)
    if directions.ndim != 2 or directions.shape[0] != count:
        raise ValueError("source orientation must match number of sources")
    return directions
