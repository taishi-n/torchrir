from __future__ import annotations

"""Room, source, and microphone geometry models."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import math

import torch
from torch import Tensor

from .utils import as_tensor, ensure_dim


@dataclass(frozen=True)
class Room:
    """Room geometry and acoustic parameters.

    Example:
        >>> room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    """

    size: Tensor
    fs: float
    c: float = 343.0
    beta: Optional[Tensor] = None
    t60: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate room size and reflection parameters."""
        size = ensure_dim(self.size)
        object.__setattr__(self, "size", size)
        if self.beta is not None and self.t60 is not None:
            raise ValueError("beta and t60 are mutually exclusive")

    def replace(self, **kwargs) -> "Room":
        """Return a new Room with updated fields."""
        return replace(self, **kwargs)

    @staticmethod
    def shoebox(
        size: Sequence[float] | Tensor,
        *,
        fs: float,
        c: float = 343.0,
        beta: Optional[Sequence[float] | Tensor] = None,
        t60: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Room":
        """Create a rectangular (shoebox) room.

        Example:
            >>> room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
        """
        size_t = as_tensor(size, device=device, dtype=dtype)
        size_t = ensure_dim(size_t)
        beta_t = None
        if beta is not None:
            beta_t = as_tensor(beta, device=device, dtype=dtype)
        return Room(size=size_t, fs=fs, c=c, beta=beta_t, t60=t60)


@dataclass(frozen=True)
class Source:
    """Source container with positions and optional orientation.

    Example:
        >>> sources = Source.from_positions([[1.0, 2.0, 1.5]])
    """

    positions: Tensor
    orientation: Optional[Tensor] = None

    def __post_init__(self) -> None:
        pos = as_tensor(self.positions)
        object.__setattr__(self, "positions", pos)
        if self.orientation is not None:
            ori = as_tensor(self.orientation)
            object.__setattr__(self, "orientation", ori)

    def replace(self, **kwargs) -> "Source":
        """Return a new Source with updated fields."""
        return replace(self, **kwargs)

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Source":
        """Convert positions/orientation to tensors and build a Source."""
        pos = as_tensor(positions, device=device, dtype=dtype)
        ori = None
        if orientation is not None:
            ori = as_tensor(orientation, device=device, dtype=dtype)
        return cls(pos, ori)


@dataclass(frozen=True)
class MicrophoneArray:
    """Microphone array container.

    Example:
        >>> mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])
    """

    positions: Tensor
    orientation: Optional[Tensor] = None

    def __post_init__(self) -> None:
        pos = as_tensor(self.positions)
        object.__setattr__(self, "positions", pos)
        if self.orientation is not None:
            ori = as_tensor(self.orientation)
            object.__setattr__(self, "orientation", ori)

    def replace(self, **kwargs) -> "MicrophoneArray":
        """Return a new MicrophoneArray with updated fields."""
        return replace(self, **kwargs)

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Convert positions/orientation to tensors and build a MicrophoneArray."""
        pos = as_tensor(positions, device=device, dtype=dtype)
        ori = None
        if orientation is not None:
            ori = as_tensor(orientation, device=device, dtype=dtype)
        return cls(pos, ori)

    @classmethod
    def binaural(
        cls,
        center: Sequence[float] | Tensor,
        *,
        offset: float = 0.08,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create a two-mic binaural layout around a center point."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _binaural_positions(center_t, offset=offset)
        return cls.from_positions(positions, orientation=orientation)

    @classmethod
    def linear(
        cls,
        center: Sequence[float] | Tensor,
        *,
        num: int,
        spacing: float,
        axis: int = 0,
        direction: Tensor | Sequence[float] | None = None,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create an equally spaced linear microphone array."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _linear_array_positions(
            center_t, num=num, spacing=spacing, axis=axis, direction=direction
        )
        return cls.from_positions(positions, orientation=orientation)

    @classmethod
    def circular(
        cls,
        center: Sequence[float] | Tensor,
        *,
        num: int,
        radius: float,
        plane: str = "xy",
        normal: Tensor | Sequence[float] | None = None,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create an equally spaced circular microphone array."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _circular_array_positions(
            center_t, num=num, radius=radius, plane=plane, normal=normal
        )
        return cls.from_positions(positions, orientation=orientation)

    @classmethod
    def polyhedron(
        cls,
        center: Sequence[float] | Tensor,
        *,
        kind: str = "tetrahedron",
        radius: float = 0.1,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create a regular polyhedron microphone array (3D only)."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _polyhedron_array_positions(center_t, kind=kind, radius=radius)
        return cls.from_positions(positions, orientation=orientation)

    @classmethod
    def eigenmike_em32(
        cls,
        center: Sequence[float] | Tensor,
        *,
        radius: float = 0.042,
        azimuth_offset_deg: float = 0.0,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create the mh acoustics Eigenmike em32 geometry (3D only)."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _eigenmike_em32_positions(
            center_t, radius=radius, azimuth_offset_deg=azimuth_offset_deg
        )
        return cls.from_positions(positions, orientation=orientation)

    @classmethod
    def eigenmike_em64(
        cls,
        center: Sequence[float] | Tensor,
        *,
        radius: float = 0.042,
        azimuth_offset_deg: float = 0.0,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Create the mh acoustics Eigenmike em64 geometry (3D only)."""
        center_t = as_tensor(center, device=device, dtype=dtype)
        positions = _eigenmike_em64_positions(
            center_t, radius=radius, azimuth_offset_deg=azimuth_offset_deg
        )
        return cls.from_positions(positions, orientation=orientation)


def _binaural_positions(center: Tensor, *, offset: float) -> Tensor:
    dim = center.numel()
    offset_vec = torch.zeros((dim,), device=center.device, dtype=center.dtype)
    offset_vec[0] = offset
    left = center - offset_vec
    right = center + offset_vec
    return torch.stack([left, right], dim=0)


def _linear_array_positions(
    center: Tensor,
    *,
    num: int,
    spacing: float,
    axis: int,
    direction: Tensor | Sequence[float] | None,
) -> Tensor:
    if num <= 0:
        raise ValueError("num must be positive")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    dim = center.numel()
    if direction is None:
        if axis < 0 or axis >= dim:
            raise ValueError("axis out of range for center dimensionality")
        direction_vec = torch.zeros((dim,), device=center.device, dtype=center.dtype)
        direction_vec[axis] = 1.0
    else:
        direction_vec = as_tensor(direction, device=center.device, dtype=center.dtype)
        if direction_vec.numel() != dim:
            raise ValueError("direction must match center dimensionality")
        direction_vec = direction_vec / torch.linalg.norm(direction_vec)

    offsets = (
        torch.arange(num, device=center.device, dtype=center.dtype)
        - (num - 1) / 2.0
    ) * spacing
    return center + offsets[:, None] * direction_vec[None, :]


def _circular_array_positions(
    center: Tensor,
    *,
    num: int,
    radius: float,
    plane: str,
    normal: Tensor | Sequence[float] | None,
) -> Tensor:
    if num <= 0:
        raise ValueError("num must be positive")
    if radius <= 0:
        raise ValueError("radius must be positive")
    dim = center.numel()

    angles = torch.linspace(
        0.0, 2.0 * math.pi, num + 1, device=center.device, dtype=center.dtype
    )[:-1]
    xy = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

    if dim == 2:
        return center + radius * xy
    if dim != 3:
        raise ValueError("center must be 2D or 3D")

    if normal is not None:
        normal_t = as_tensor(normal, device=center.device, dtype=center.dtype)
        basis_x, basis_y = _basis_from_normal(normal_t)
    else:
        plane_l = plane.lower()
        if plane_l == "xy":
            basis_x = torch.tensor(
                [1.0, 0.0, 0.0], device=center.device, dtype=center.dtype
            )
            basis_y = torch.tensor(
                [0.0, 1.0, 0.0], device=center.device, dtype=center.dtype
            )
        elif plane_l == "xz":
            basis_x = torch.tensor(
                [1.0, 0.0, 0.0], device=center.device, dtype=center.dtype
            )
            basis_y = torch.tensor(
                [0.0, 0.0, 1.0], device=center.device, dtype=center.dtype
            )
        elif plane_l == "yz":
            basis_x = torch.tensor(
                [0.0, 1.0, 0.0], device=center.device, dtype=center.dtype
            )
            basis_y = torch.tensor(
                [0.0, 0.0, 1.0], device=center.device, dtype=center.dtype
            )
        else:
            raise ValueError("plane must be one of 'xy', 'xz', 'yz'")

    circle = xy[:, 0:1] * basis_x[None, :] + xy[:, 1:2] * basis_y[None, :]
    return center + radius * circle


def _polyhedron_array_positions(center: Tensor, *, kind: str, radius: float) -> Tensor:
    dim = center.numel()
    if dim != 3:
        raise ValueError("polyhedron arrays require 3D centers")
    if radius <= 0:
        raise ValueError("radius must be positive")
    vertices = _polyhedron_vertices(kind, device=center.device, dtype=center.dtype)
    norms = torch.linalg.norm(vertices, dim=-1, keepdim=True)
    vertices = vertices / norms
    return center + radius * vertices


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


def _polyhedron_vertices(kind: str, *, device: torch.device, dtype: torch.dtype) -> Tensor:
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


def _eigenmike_em32_positions(
    center: Tensor, *, radius: float, azimuth_offset_deg: float
) -> Tensor:
    if center.numel() != 3:
        raise ValueError("Eigenmike em32 requires a 3D center")
    if radius <= 0:
        raise ValueError("radius must be positive")
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
        device=center.device,
        dtype=center.dtype,
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
        device=center.device,
        dtype=center.dtype,
    )
    return _spherical_array_from_angles(
        center=center,
        radius=radius,
        theta_deg=theta_deg,
        phi_deg=phi_deg + azimuth_offset_deg,
    )


def _eigenmike_em64_positions(
    center: Tensor, *, radius: float, azimuth_offset_deg: float
) -> Tensor:
    if center.numel() != 3:
        raise ValueError("Eigenmike em64 requires a 3D center")
    if radius <= 0:
        raise ValueError("radius must be positive")
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
        device=center.device,
        dtype=center.dtype,
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
        device=center.device,
        dtype=center.dtype,
    )
    return _spherical_array_from_angles(
        center=center,
        radius=radius,
        theta_deg=theta_deg,
        phi_deg=phi_deg + azimuth_offset_deg,
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
