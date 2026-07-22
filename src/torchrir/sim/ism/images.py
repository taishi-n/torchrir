"""Image source indexing and reflection coefficient helpers."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product
from math import comb
from numbers import Integral
from typing import Optional

import torch
from torch import Tensor


_INT64_MAX = torch.iinfo(torch.int64).max


def _image_source_indices(
    max_order: Optional[int],
    dim: int,
    *,
    device: torch.device,
    nb_img: tuple[int, ...] | None = None,
) -> Tensor:
    """Materialize image indices for low-level consumers and tests."""
    count = _image_source_count(max_order, dim, nb_img=nb_img)
    chunks = tuple(
        _iter_image_source_index_chunks(
            max_order,
            dim,
            device=device,
            nb_img=nb_img,
            chunk_size=min(count, 65_536),
        )
    )
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


def _image_source_count(
    max_order: int | None,
    dim: int,
    *,
    nb_img: tuple[int, ...] | None = None,
) -> int:
    """Return the checked number of image indices for one request."""

    if dim not in (2, 3):
        raise ValueError("room dimension must be 2 or 3")
    if nb_img is not None:
        nb = _normalize_nb_img(nb_img, dim=dim)
        image_count = 1
        for value in nb:
            if value > (_INT64_MAX - 1) // 2:
                raise ValueError("nb_img axis width must fit in int64")
            width = 2 * value + 1
            if image_count > _INT64_MAX // width:
                raise ValueError("nb_img image count must fit in int64")
            image_count *= width
        return image_count
    if max_order is None:
        raise ValueError("max_order is required when nb_img is not provided")
    if isinstance(max_order, bool) or not isinstance(max_order, Integral):
        raise TypeError("max_order must be an integer")
    normalized_order = int(max_order)
    if normalized_order < 0:
        raise ValueError("max_order must be non-negative")
    count = sum(
        (2**axis_count) * comb(dim, axis_count) * comb(normalized_order, axis_count)
        for axis_count in range(min(dim, normalized_order) + 1)
    )
    if count > _INT64_MAX:
        raise ValueError("max_order image count must fit in int64")
    return count


def _iter_image_source_index_chunks(
    max_order: int | None,
    dim: int,
    *,
    device: torch.device,
    nb_img: tuple[int, ...] | None,
    chunk_size: int,
) -> Iterator[Tensor]:
    """Yield bounded integer image-index chunks without a full grid."""

    _image_source_count(max_order, dim, nb_img=nb_img)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if nb_img is not None:
        normalized_nb_img = _normalize_nb_img(nb_img, dim=dim)
        rows: Iterable[tuple[int, ...]] = product(
            *(range(-value, value + 1) for value in normalized_nb_img)
        )
    else:
        assert max_order is not None
        rows = _iter_l1_rows(int(max_order), dim=dim)
    return _batch_index_rows(rows, chunk_size=chunk_size, device=device)


def _normalize_nb_img(nb_img: tuple[int, ...], *, dim: int) -> tuple[int, ...]:
    if not isinstance(nb_img, tuple) or len(nb_img) != dim:
        raise ValueError("nb_img must match room dimension")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in nb_img
    ):
        raise TypeError("nb_img must contain integers")
    normalized = tuple(int(value) for value in nb_img)
    if any(value < 0 for value in normalized):
        raise ValueError("nb_img must contain non-negative integers")
    return normalized


def _iter_l1_rows(
    remaining_order: int,
    *,
    dim: int,
    prefix: tuple[int, ...] = (),
) -> Iterator[tuple[int, ...]]:
    if len(prefix) == dim:
        yield prefix
        return
    for value in range(-remaining_order, remaining_order + 1):
        yield from _iter_l1_rows(
            remaining_order - abs(value),
            dim=dim,
            prefix=(*prefix, value),
        )


def _batch_index_rows(
    rows: Iterable[tuple[int, ...]],
    *,
    chunk_size: int,
    device: torch.device,
) -> Iterator[Tensor]:
    chunk: list[tuple[int, ...]] = []
    for row in rows:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield torch.tensor(chunk, device=device, dtype=torch.int64)
            chunk = []
    if chunk:
        yield torch.tensor(chunk, device=device, dtype=torch.int64)


def _image_positions(src: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image source positions for a given source."""

    return _image_positions_for_sources(src, room_size, n_vec)


def _image_positions_batch(src_pos: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image positions for sources with arbitrary leading dimensions."""

    return _image_positions_for_sources(src_pos, room_size, n_vec)


def _image_positions_for_sources(
    sources: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
) -> Tensor:
    """Evaluate image-source affine geometry without overflowing intermediates."""

    source = sources.unsqueeze(-2)
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(
        device=sources.device,
        dtype=sources.dtype,
    )
    tile = torch.floor_divide(n_vec + 1, 2).to(
        device=sources.device,
        dtype=sources.dtype,
    )
    scale = torch.maximum(torch.abs(source), torch.abs(room_size))
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled_positions = 2.0 * (room_size / safe_scale) * tile + sign * (
        source / safe_scale
    )
    positions = scaled_positions * scale
    if not torch.all(torch.isfinite(positions)):
        raise ValueError(
            "image-source positions must be representable in the simulation dtype"
        )
    return positions


def _reflection_coefficients(n_vec: Tensor, beta: Tensor) -> Tensor:
    """Compute reflection coefficients for each image source."""
    dim = n_vec.shape[1]
    beta = beta.view(dim, 2)
    beta_lo = beta[:, 0]
    beta_hi = beta[:, 1]

    n = n_vec
    k = torch.abs(n)
    n_hi = torch.where(n >= 0, (n + 1) // 2, k // 2)
    n_lo = torch.where(n >= 0, n // 2, (k + 1) // 2)

    n_hi = n_hi.to(dtype=beta.dtype)
    n_lo = n_lo.to(dtype=beta.dtype)

    coeff = (beta_hi**n_hi) * (beta_lo**n_lo)
    return torch.prod(coeff, dim=1)
