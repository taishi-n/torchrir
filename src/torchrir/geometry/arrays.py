"""Microphone array geometry helpers."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ..util.tensor import as_tensor


def binaural_array(
    center: Sequence[float] | Tensor,
    *,
    offset: float = 0.08,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a two-mic binaural layout around a center point."""
    center_t = as_tensor(center, device=device, dtype=dtype)
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
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    center_t = as_tensor(center, device=device, dtype=dtype)
    dim = center_t.numel()
    if direction is None:
        if axis < 0 or axis >= dim:
            raise ValueError("axis out of range for center dimensionality")
        direction_vec = torch.zeros(
            (dim,), device=center_t.device, dtype=center_t.dtype
        )
        direction_vec[axis] = 1.0
    else:
        direction_vec = as_tensor(
            direction, device=center_t.device, dtype=center_t.dtype
        )
        if direction_vec.numel() != dim:
            raise ValueError("direction must match center dimensionality")
        direction_vec = direction_vec / torch.linalg.norm(direction_vec)

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
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_t = as_tensor(center, device=device, dtype=dtype)
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
        normal_t = as_tensor(normal, device=center_t.device, dtype=center_t.dtype)
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
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_t = as_tensor(center, device=device, dtype=dtype)
    if center_t.numel() != 3:
        raise ValueError("polyhedron arrays require 3D centers")
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
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_t = as_tensor(center, device=device, dtype=dtype)
    if center_t.numel() != 3:
        raise ValueError("Eigenmike em32 requires a 3D center")
    theta_deg = torch.tensor(
        [
            69.0,
            90.0,
            111.0,
            90.0,
            32.0,
            55.0,
            90.0,
            125.0,
            148.0,
            125.0,
            90.0,
            55.0,
            21.0,
            58.0,
            121.0,
            159.0,
            69.0,
            90.0,
            111.0,
            90.0,
            32.0,
            55.0,
            90.0,
            125.0,
            148.0,
            125.0,
            90.0,
            55.0,
            21.0,
            58.0,
            122.0,
            159.0,
        ],
        device=center_t.device,
        dtype=center_t.dtype,
    )
    phi_deg = torch.tensor(
        [
            0.0,
            32.0,
            0.0,
            328.0,
            0.0,
            45.0,
            69.0,
            45.0,
            0.0,
            315.0,
            291.0,
            315.0,
            91.0,
            90.0,
            90.0,
            89.0,
            180.0,
            212.0,
            180.0,
            148.0,
            180.0,
            225.0,
            249.0,
            225.0,
            180.0,
            135.0,
            111.0,
            135.0,
            269.0,
            270.0,
            270.0,
            271.0,
        ],
        device=center_t.device,
        dtype=center_t.dtype,
    )
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
    if radius <= 0:
        raise ValueError("radius must be positive")
    center_t = as_tensor(center, device=device, dtype=dtype)
    if center_t.numel() != 3:
        raise ValueError("Eigenmike em64 requires a 3D center")
    theta_deg = torch.tensor(
        [
            16.7656,
            21.9677,
            42.3941,
            13.2817,
            22.6728,
            52.6925,
            37.806,
            43.3944,
            43.9386,
            70.3132,
            33.2231,
            60.0257,
            56.4763,
            67.4936,
            93.2735,
            48.423,
            78.0793,
            62.0685,
            38.7171,
            63.8004,
            70.1946,
            96.246,
            81.0992,
            106.094,
            67.7533,
            91.7061,
            39.9985,
            68.7726,
            60.8869,
            82.2833,
            63.0247,
            89.794,
            137.5166,
            139.7604,
            135.2133,
            160.3628,
            162.577,
            142.0685,
            161.1987,
            162.577,
            115.536,
            86.2594,
            116.0164,
            95.3313,
            90.0637,
            111.4549,
            85.8671,
            130.8398,
            102.5775,
            142.6375,
            117.032,
            117.5631,
            115.8884,
            89.69,
            118.4478,
            93.9338,
            106.3875,
            81.0511,
            135.9764,
            142.6771,
            120.6556,
            133.8834,
            116.3591,
            107.464,
        ],
        device=center_t.device,
        dtype=center_t.dtype,
    )
    phi_deg = torch.tensor(
        [
            197.4561,
            115.734,
            81.911,
            313.3592,
            43.1785,
            46.7324,
            335.9958,
            14.5398,
            204.4547,
            206.542,
            247.3219,
            233.817,
            264.5437,
            99.6669,
            104.6842,
            120.9227,
            126.513,
            148.2368,
            162.6381,
            178.5498,
            21.2715,
            25.7834,
            47.8607,
            55.9075,
            71.4285,
            78.4921,
            293.221,
            290.5683,
            318.1354,
            334.0042,
            352.0227,
            0.0,
            174.0335,
            212.7205,
            251.9179,
            150.6471,
            240.8266,
            293.0625,
            331.0098,
            60.8266,
            226.9135,
            233.9255,
            193.6382,
            209.6696,
            183.169,
            163.7105,
            156.9524,
            139.4318,
            135.9729,
            102.3273,
            112.5511,
            83.1464,
            307.7078,
            309.1392,
            278.2519,
            282.9735,
            253.147,
            260.0688,
            59.7394,
            14.2241,
            32.4901,
            334.0753,
            2.0842,
            335.0677,
        ],
        device=center_t.device,
        dtype=center_t.dtype,
    )
    return _spherical_array_from_angles(
        center=center_t,
        radius=radius,
        theta_deg=theta_deg,
        phi_deg=phi_deg + azimuth_offset_deg,
    )


def _basis_from_normal(normal: Tensor) -> tuple[Tensor, Tensor]:
    if normal.numel() != 3:
        raise ValueError("normal must be a 3D vector")
    n = normal / torch.linalg.norm(normal)
    ref = torch.tensor([1.0, 0.0, 0.0], device=normal.device, dtype=normal.dtype)
    if torch.allclose(n, ref):
        ref = torch.tensor([0.0, 1.0, 0.0], device=normal.device, dtype=normal.dtype)
    basis_x = torch.cross(n, ref)
    basis_x = basis_x / torch.linalg.norm(basis_x)
    basis_y = torch.cross(n, basis_x)
    basis_y = basis_y / torch.linalg.norm(basis_y)
    return basis_x, basis_y


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
