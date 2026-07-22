"""Microphone array geometry helpers."""

from __future__ import annotations

import math
import sys
from typing import Literal, Sequence

import torch
from torch import Tensor

from ._eigenmike import EM32_PHI_DEG, EM32_THETA_DEG, EM64_PHI_DEG, EM64_THETA_DEG
from ..util._scalars import normalize_finite_real, normalize_integer
from ..util.orientation import normalize_orientation
from ..util.tensor import as_float_tensor


_MAX_TENSOR_COUNT = min(sys.maxsize, torch.iinfo(torch.int64).max)


def binaural_array(
    center: Sequence[float] | Tensor,
    *,
    offset: float = 0.08,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a two-mic binaural layout around a center point."""
    offset = normalize_finite_real(offset, name="offset", positive=True)
    center_t = _prepare_center(center, device=device, dtype=dtype)
    _require_scalar_representable(offset, reference=center_t, name="offset")
    dim = center_t.numel()
    offset_vec = torch.zeros((dim,), device=center_t.device, dtype=center_t.dtype)
    offset_vec[0] = offset
    left = center_t - offset_vec
    right = center_t + offset_vec
    return _validate_array_output(
        torch.stack([left, right], dim=0),
        expected_unique_rows=2,
    )


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
    num = normalize_integer(
        num,
        name="num",
        minimum=1,
        maximum=_MAX_TENSOR_COUNT,
    )
    spacing = normalize_finite_real(spacing, name="spacing", positive=True)
    center_t = _prepare_center(center, device=device, dtype=dtype)
    _require_scalar_representable(spacing, reference=center_t, name="spacing")
    dim = center_t.numel()
    if direction is None:
        axis = normalize_integer(axis, name="axis")
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
        ).reshape(-1)
        if direction_vec.numel() != dim:
            raise ValueError("direction must match center dimensionality")
        direction_vec = _normalize_array_vector(direction_vec, name="direction")

    maximum_offset = spacing * ((num - 1) / 2.0)
    _require_scalar_representable(
        maximum_offset,
        reference=center_t,
        name="maximum linear-array offset",
    )
    offsets = torch.linspace(
        -maximum_offset,
        maximum_offset,
        num,
        device=center_t.device,
        dtype=center_t.dtype,
    )
    return _validate_array_output(
        center_t + offsets[:, None] * direction_vec[None, :],
        expected_unique_rows=num,
        distinctness="adjacent",
    )


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
    num = normalize_integer(
        num,
        name="num",
        minimum=1,
        maximum=_MAX_TENSOR_COUNT - 1,
    )
    radius = normalize_finite_real(radius, name="radius", positive=True)
    center_t = _prepare_center(center, device=device, dtype=dtype)
    _require_scalar_representable(radius, reference=center_t, name="radius")
    dim = center_t.numel()

    angles = torch.linspace(
        0.0, 2.0 * math.pi, num + 1, device=center_t.device, dtype=center_t.dtype
    )[:-1]
    xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    if dim == 2:
        return _validate_array_output(
            center_t + radius * xy,
            expected_unique_rows=num,
            distinctness="cyclic",
        )
    if dim != 3:
        raise ValueError("center must be 2D or 3D")

    if normal is not None:
        normal_t = as_float_tensor(
            normal,
            device=center_t.device,
            dtype=center_t.dtype,
            name="normal",
        ).reshape(-1)
        basis_x, basis_y = _basis_from_normal(normal_t)
    else:
        if not isinstance(plane, str):
            raise TypeError("plane must be a string")
        plane_l = plane.strip().lower()
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
    return _validate_array_output(
        center_t + radius * circle,
        expected_unique_rows=num,
        distinctness="cyclic",
    )


def polyhedron_array(
    center: Sequence[float] | Tensor,
    *,
    kind: str = "tetrahedron",
    radius: float = 0.1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a regular polyhedron microphone array (3D only)."""
    radius = normalize_finite_real(radius, name="radius", positive=True)
    if not isinstance(kind, str):
        raise TypeError("kind must be a string")
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    _require_scalar_representable(radius, reference=center_t, name="radius")
    vertices = _polyhedron_vertices(
        kind.strip().lower(),
        device=center_t.device,
        dtype=center_t.dtype,
    )
    norms = torch.linalg.norm(vertices, dim=-1, keepdim=True)
    vertices = vertices / norms
    return _validate_array_output(
        center_t + radius * vertices,
        expected_unique_rows=int(vertices.shape[0]),
    )


def eigenmike_em32(
    center: Sequence[float] | Tensor,
    *,
    radius: float = 0.042,
    azimuth_offset_deg: float = 0.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create the mh acoustics Eigenmike em32 geometry (3D only)."""
    radius = normalize_finite_real(radius, name="radius", positive=True)
    azimuth_offset_deg = normalize_finite_real(
        azimuth_offset_deg,
        name="azimuth_offset_deg",
    )
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    _require_scalar_representable(radius, reference=center_t, name="radius")
    _require_scalar_representable(
        azimuth_offset_deg,
        reference=center_t,
        name="azimuth_offset_deg",
    )
    theta_deg = torch.tensor(
        EM32_THETA_DEG, device=center_t.device, dtype=center_t.dtype
    )
    phi_deg = torch.tensor(EM32_PHI_DEG, device=center_t.device, dtype=center_t.dtype)
    return _validate_array_output(
        _spherical_array_from_angles(
            center=center_t,
            radius=radius,
            theta_deg=theta_deg,
            phi_deg=phi_deg + azimuth_offset_deg,
        ),
        expected_unique_rows=len(EM32_THETA_DEG),
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
    radius = normalize_finite_real(radius, name="radius", positive=True)
    azimuth_offset_deg = normalize_finite_real(
        azimuth_offset_deg,
        name="azimuth_offset_deg",
    )
    center_t = _prepare_center(center, device=device, dtype=dtype, required_dim=3)
    _require_scalar_representable(radius, reference=center_t, name="radius")
    _require_scalar_representable(
        azimuth_offset_deg,
        reference=center_t,
        name="azimuth_offset_deg",
    )
    theta_deg = torch.tensor(
        EM64_THETA_DEG, device=center_t.device, dtype=center_t.dtype
    )
    phi_deg = torch.tensor(EM64_PHI_DEG, device=center_t.device, dtype=center_t.dtype)
    return _validate_array_output(
        _spherical_array_from_angles(
            center=center_t,
            radius=radius,
            theta_deg=theta_deg,
            phi_deg=phi_deg + azimuth_offset_deg,
        ),
        expected_unique_rows=len(EM64_THETA_DEG),
    )


def _basis_from_normal(normal: Tensor) -> tuple[Tensor, Tensor]:
    if normal.numel() != 3:
        raise ValueError("normal must be a 3D vector")
    n = _normalize_array_vector(normal, name="normal")
    # Select the coordinate axis least aligned with the normal. This remains
    # stable for both positive and negative axis-aligned normals.
    ref = torch.zeros(3, device=normal.device, dtype=normal.dtype)
    ref[int(torch.argmin(torch.abs(n)).item())] = 1.0
    basis_x = _normalize_array_vector(torch.linalg.cross(n, ref), name="basis")
    basis_y = _normalize_array_vector(
        torch.linalg.cross(n, basis_x),
        name="basis",
    )
    return basis_x, basis_y


def _normalize_array_vector(vector: Tensor, *, name: str) -> Tensor:
    try:
        return normalize_orientation(vector)
    except (TypeError, ValueError) as error:
        message = str(error).replace("orientation", name)
        raise type(error)(message) from error


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


def _require_scalar_representable(
    value: float,
    *,
    reference: Tensor,
    name: str,
) -> None:
    limit = torch.finfo(reference.dtype).max
    if not math.isfinite(value) or abs(value) > limit:
        raise ValueError(f"{name} must be representable as {reference.dtype}")


def _validate_array_output(
    positions: Tensor,
    *,
    expected_unique_rows: int,
    distinctness: Literal["all", "adjacent", "cyclic"] = "all",
) -> Tensor:
    if not torch.all(torch.isfinite(positions)):
        raise ValueError("array positions must be finite in the requested dtype")
    if distinctness == "all":
        rows_are_distinct = (
            torch.unique(positions.detach().cpu(), dim=0).shape[0]
            == expected_unique_rows
        )
    else:
        rows_are_distinct = positions.shape[0] == expected_unique_rows
        if rows_are_distinct and expected_unique_rows > 1:
            adjacent_rows_differ = torch.any(
                positions[1:] != positions[:-1],
                dim=1,
            )
            rows_are_distinct = bool(torch.all(adjacent_rows_differ).item())
            if rows_are_distinct and distinctness == "cyclic":
                rows_are_distinct = bool(
                    torch.any(positions[0] != positions[-1]).item()
                )
    if not rows_are_distinct:
        raise ValueError(
            "array positions must remain distinct in the requested dtype; "
            "increase the scale or use a higher-precision dtype"
        )
    return positions


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
