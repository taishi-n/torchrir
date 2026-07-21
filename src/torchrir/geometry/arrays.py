"""Microphone array geometry helpers."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ._eigenmike import EM32_PHI_DEG, EM32_THETA_DEG, EM64_PHI_DEG, EM64_THETA_DEG
from ..util.tensor import as_float_tensor


def binaural_array(
    center: Sequence[float] | Tensor,
    *,
    offset: float = 0.08,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a two-mic binaural layout around a center point."""
    if not math.isfinite(offset) or offset <= 0:
        raise ValueError("offset must be positive and finite")
    center_t = _prepare_center(center, device=device, dtype=dtype)
    dim = center_t.numel()
    offset_vec = torch.zeros((dim,), device=center_t.device, dtype=center_t.dtype)
    offset_vec[0] = offset
    left = center_t - offset_vec
    right = center_t + offset_vec
    return torch.stack([left, right], dim=0)


def linear_array(
    center: Sequence[float] | Tensor,
    *,
    num: int,
    spacing: float,
    axis: int = 0,
    direction: Sequence[float] | Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create an equally spaced linear microphone array."""
    if num <= 0:
        raise ValueError("num must be positive")
    if not math.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be positive and finite")
    center_t = _prepare_center(center, device=device, dtype=dtype)
    dim = center_t.numel()
    if direction is None:
        if axis < 0 or axis >= dim:
            raise ValueError("axis out of range for center dimensionality")
        direction_vec = torch.zeros(
            (dim,), device=center_t.device, dtype=center_t.dtype
        )
        direction_vec[axis] = 1.0
    else:
        direction_vec = as_float_tensor(
            direction,
            device=center_t.device,
            dtype=center_t.dtype,
            name="direction",
        )
        if direction_vec.numel() != dim:
            raise ValueError("direction must match center dimensionality")
        direction_norm = torch.linalg.vector_norm(direction_vec)
        if not torch.isfinite(direction_norm) or direction_norm <= 1.0e-8:
            raise ValueError("direction must be a finite non-zero vector")
        direction_vec = direction_vec / direction_norm

    offsets = (
        torch.arange(num, device=center_t.device, dtype=center_t.dtype)
        - (num - 1) / 2.0
    ) * spacing
    return center_t + offsets[:, None] * direction_vec[None, :]


def circular_array(
    center: Sequence[float] | Tensor,
    *,
    num: int,
    radius: float,
    plane: str = "xy",
    normal: Sequence[float] | Tensor | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create an equally spaced circular microphone array."""
    if num <= 0:
        raise ValueError("num must be positive")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    center_t = _prepare_center(center, device=device, dtype=dtype)
    dim = center_t.numel()

    angles = torch.linspace(
        0.0, 2.0 * math.pi, num + 1, device=center_t.device, dtype=center_t.dtype
    )[:-1]
    xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    if dim == 2:
        return center_t + radius * xy
    if dim != 3:
        raise ValueError("center must be 2D or 3D")

    if normal is not None:
        normal_t = as_float_tensor(
            normal,
            device=center_t.device,
            dtype=center_t.dtype,
            name="normal",
        )
        basis_x, basis_y = _basis_from_normal(normal_t)
    else:
        plane_l = plane.lower()
        if plane_l == "xy":
            basis_x = torch.tensor(
                [1.0, 0.0, 0.0], device=center_t.device, dtype=center_t.dtype
            )
            basis_y = torch.tensor(
                [0.0, 1.0, 0.0], device=center_t.device, dtype=center_t.dtype
            )
        elif plane_l == "xz":
            basis_x = torch.tensor(
                [1.0, 0.0, 0.0], device=center_t.device, dtype=center_t.dtype
            )
            basis_y = torch.tensor(
                [0.0, 0.0, 1.0], device=center_t.device, dtype=center_t.dtype
            )
        elif plane_l == "yz":
            basis_x = torch.tensor(
                [0.0, 1.0, 0.0], device=center_t.device, dtype=center_t.dtype
            )
            basis_y = torch.tensor(
                [0.0, 0.0, 1.0], device=center_t.device, dtype=center_t.dtype
            )
        else:
            raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

    circle = xy[:, 0:1] * basis_x[None, :] + xy[:, 1:2] * basis_y[None, :]
    return center_t + radius * circle


def polyhedron_array(
    center: Sequence[float] | Tensor,
    *,
    kind: str = "tetrahedron",
    radius: float = 0.1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a regular polyhedron microphone array (3D only)."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    vertices = _polyhedron_vertices(kind, device=center_t.device, dtype=center_t.dtype)
    norms = torch.linalg.norm(vertices, dim=-1, keepdim=True)
    vertices = vertices / norms
    return center_t + radius * vertices


def eigenmike_em32(
    center: Sequence[float] | Tensor,
    *,
    radius: float = 0.042,
    azimuth_offset_deg: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create the mh acoustics Eigenmike em32 geometry (3D only)."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(azimuth_offset_deg):
        raise ValueError("azimuth_offset_deg must be finite")
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    theta_deg = torch.tensor(
        EM32_THETA_DEG, device=center_t.device, dtype=center_t.dtype
    )
    phi_deg = torch.tensor(EM32_PHI_DEG, device=center_t.device, dtype=center_t.dtype)
    return _spherical_array_from_angles(
        center=center_t,
        radius=radius,
        theta_deg=theta_deg,
        phi_deg=phi_deg + azimuth_offset_deg,
    )


def eigenmike_em64(
    center: Sequence[float] | Tensor,
    *,
    radius: float = 0.042,
    azimuth_offset_deg: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create the mh acoustics Eigenmike em64 geometry (3D only)."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(azimuth_offset_deg):
        raise ValueError("azimuth_offset_deg must be finite")
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    theta_deg = torch.tensor(
        EM64_THETA_DEG, device=center_t.device, dtype=center_t.dtype
    )
    phi_deg = torch.tensor(EM64_PHI_DEG, device=center_t.device, dtype=center_t.dtype)
    return _spherical_array_from_angles(
        center=center_t,
        radius=radius,
        theta_deg=theta_deg,
        phi_deg=phi_deg + azimuth_offset_deg,
    )


def _basis_from_normal(normal: Tensor) -> tuple[Tensor, Tensor]:
    if normal.numel() != 3:
        raise ValueError("normal must be a 3D vector")
    norm = torch.linalg.vector_norm(normal)
    if not torch.isfinite(norm) or norm <= 1.0e-8:
        raise ValueError("normal must be a finite non-zero vector")
    n = normal / norm
    # Select the coordinate axis least aligned with the normal. This remains
    # stable for both positive and negative axis-aligned normals.
    ref = torch.zeros(3, device=normal.device, dtype=normal.dtype)
    ref[int(torch.argmin(torch.abs(n)).item())] = 1.0
    basis_x = torch.linalg.cross(n, ref)
    basis_x = basis_x / torch.linalg.vector_norm(basis_x)
    basis_y = torch.linalg.cross(n, basis_x)
    basis_y = basis_y / torch.linalg.vector_norm(basis_y)
    return basis_x, basis_y


def _prepare_center(
    center: Sequence[float] | Tensor,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
    required_dim: int | None = None,
) -> Tensor:
    center_t = as_float_tensor(
        center, device=device, dtype=dtype, name="array center"
    ).reshape(-1)
    allowed = (required_dim,) if required_dim is not None else (2, 3)
    if center_t.numel() not in allowed:
        expected = str(required_dim) if required_dim is not None else "2D or 3D"
        raise ValueError(f"array center must be {expected}")
    if not torch.all(torch.isfinite(center_t)):
        raise ValueError("array center must contain finite values")
    return center_t


def _polyhedron_vertices(
    kind: str, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    kind_l = kind.lower()
    if kind_l == "tetrahedron":
        return torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
    if kind_l == "cube":
        coords = [-1.0, 1.0]
        return torch.tensor(
            [[x, y, z] for x in coords for y in coords for z in coords],
            device=device,
            dtype=dtype,
        )
    if kind_l == "octahedron":
        return torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            device=device,
            dtype=dtype,
        )
    if kind_l == "dodecahedron":
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi
        verts = []
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                verts.append([a, b, b])
                verts.append([a, b, -b])
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                verts.append([0.0, a * inv_phi, b * phi])
                verts.append([a * inv_phi, b * phi, 0.0])
                verts.append([a * phi, 0.0, b * inv_phi])
        return torch.tensor(verts, device=device, dtype=dtype)
    if kind_l == "icosahedron":
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        verts = []
        for a in (-1.0, 1.0):
            for b in (-1.0, 1.0):
                verts.append([0.0, a, b * phi])
                verts.append([a, b * phi, 0.0])
                verts.append([a * phi, 0.0, b])
        return torch.tensor(verts, device=device, dtype=dtype)
    raise ValueError(
        "kind must be one of 'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'"
    )


def _spherical_array_from_angles(
    *, center: Tensor, radius: float, theta_deg: Tensor, phi_deg: Tensor
) -> Tensor:
    theta = torch.deg2rad(theta_deg)
    phi = torch.deg2rad(phi_deg)
    x = radius * torch.sin(theta) * torch.cos(phi)
    y = radius * torch.sin(theta) * torch.sin(phi)
    z = radius * torch.cos(theta)
    return center + torch.stack([x, y, z], dim=-1)
