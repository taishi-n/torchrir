"""Shared preparation for static and dynamic image-source simulations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ...config import ResolvedSimulationConfig
from ...models import Room
from .helpers import _resolve_beta, _validate_beta
from .images import _image_source_count


@dataclass(frozen=True, eq=False)
class ISMContext:
    beta: Tensor
    dimension: int
    max_order: int | None
    nb_img: tuple[int, ...] | None
    image_count: int
    source_pattern: str
    microphone_pattern: str
    microphone_directions: Tensor | None
    image_chunk_size: int


def prepare_ism_context(
    *,
    room: Room,
    room_size: Tensor,
    dim: int,
    dtype: torch.dtype,
    max_order: int | None,
    nb_img: tuple[int, ...] | None,
    source_directivity: str,
    microphone_directivity: str,
    microphone_orientation: Tensor | None,
    config: ResolvedSimulationConfig,
) -> ISMContext:
    beta = _validate_beta(
        _resolve_beta(room, room_size, device=room_size.device, dtype=dtype), dim
    )
    image_count = _image_source_count(max_order, dim, nb_img=nb_img)
    source_pattern = source_directivity
    microphone_pattern = microphone_directivity

    microphone_directions = None
    if microphone_pattern != "omni":
        if microphone_orientation is None:
            raise ValueError("mic orientation required for non-omni directivity")
        if microphone_orientation.ndim != 2 or microphone_orientation.shape[1] != dim:
            raise ValueError(
                "microphone orientation must use canonical (microphones, dimensions) shape"
            )
        microphone_directions = microphone_orientation

    return ISMContext(
        beta=beta,
        dimension=dim,
        max_order=max_order,
        nb_img=nb_img,
        image_count=image_count,
        source_pattern=source_pattern,
        microphone_pattern=microphone_pattern,
        microphone_directions=microphone_directions,
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
    if orientation.shape != (count, dim):
        raise ValueError(
            "source orientation must use canonical (sources, dimensions) shape"
        )
    return orientation
