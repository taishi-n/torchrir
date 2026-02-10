"""Image source indexing and reflection coefficient helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from ...util.tensor import as_tensor


def _image_source_indices(
    max_order: int,
    dim: int,
    *,
    device: torch.device,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
) -> Tensor:
    """Generate image source index vectors up to the given order."""
    if nb_img is not None:
        nb = as_tensor(nb_img, device=device, dtype=torch.int64)
        if nb.numel() != dim:
            raise ValueError("nb_img must match room dimension")
        ranges = [torch.arange(-n, n + 1, device=device, dtype=torch.int64) for n in nb]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=-1)
    rng = torch.arange(-max_order, max_order + 1, device=device, dtype=torch.int64)
    grids = torch.meshgrid(*([rng] * dim), indexing="ij")
    n_vec = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    order = torch.sum(torch.abs(n_vec), dim=-1)
    return n_vec[order <= max_order]


def _image_positions(src: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image source positions for a given source."""
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(dtype=src.dtype)
    n = torch.floor_divide(n_vec + 1, 2).to(dtype=src.dtype)
    return 2.0 * room_size * n + sign * src


def _image_positions_batch(src_pos: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image source positions for multiple sources."""
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(dtype=src_pos.dtype)
    n = torch.floor_divide(n_vec + 1, 2).to(dtype=src_pos.dtype)
    base = 2.0 * room_size * n
    return base[None, :, :] + sign[None, :, :] * src_pos[:, None, :]


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
