"""Room, source, and microphone geometry models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

from .._directivity import canonicalize_directivity
from ..util.acoustics import estimate_beta_from_t60
from ..util._dtypes import validate_supported_float_tensor
from ..util.orientation import orientation_to_unit
from ..util._scalars import normalize_finite_real
from ..util.tensor import as_float_tensor, ensure_dim


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Room:
    """Room geometry and acoustic parameters.

    Reflection coefficients use wall order ``[x-low, x-high, y-low, y-high]``
    in 2D and append ``[z-low, z-high]`` in 3D.

    Examples:
        ```python
        room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
        ```
    """

    size: Tensor
    fs: float
    c: float = 343.0
    beta: Optional[Tensor] = None
    t60: Optional[float] = None

    def __post_init__(self) -> None:
        """Validate room size and reflection parameters."""
        size = ensure_dim(as_float_tensor(self.size, name="room size"))
        if not torch.all(torch.isfinite(size)):
            raise ValueError("room size must contain finite values")
        if torch.any(size <= 0):
            raise ValueError("room size must be strictly positive")
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "fs", _positive_real(self.fs, name="fs"))
        object.__setattr__(self, "c", _positive_real(self.c, name="c"))
        if self.beta is not None and self.t60 is not None:
            raise ValueError("beta and t60 are mutually exclusive")
        if self.t60 is not None:
            object.__setattr__(
                self,
                "t60",
                _positive_real(self.t60, name="t60"),
            )
        if self.beta is not None:
            beta = as_float_tensor(self.beta, name="beta").reshape(-1)
            if beta.device != size.device or beta.dtype != size.dtype:
                raise ValueError("beta device and dtype must match room size")
            expected = 4 if size.numel() == 2 else 6
            if beta.numel() != expected:
                raise ValueError(
                    f"beta must have {expected} elements for {size.numel()}D rooms"
                )
            if not torch.all(torch.isfinite(beta)):
                raise ValueError("beta must contain finite values")
            if torch.any(beta < 0) or torch.any(beta > 1):
                raise ValueError("beta values must be in [0, 1]")
            object.__setattr__(self, "beta", beta)
        self._validate_internal()

    def replace(self, **kwargs) -> "Room":
        """Return a new Room with updated fields."""
        return replace(self, **kwargs)

    def _validate_internal(self) -> None:
        if not torch.is_tensor(self.size) or not self.size.is_floating_point():
            raise TypeError("room size must be a real floating-point Tensor")
        validate_supported_float_tensor(self.size, name="room size")
        ensure_dim(self.size)
        if not torch.all(torch.isfinite(self.size)):
            raise ValueError("room size must contain finite values")
        if torch.any(self.size <= 0):
            raise ValueError("room size must be strictly positive")
        _positive_real(self.fs, name="fs")
        _positive_real(self.c, name="c")
        if self.beta is not None and self.t60 is not None:
            raise ValueError("beta and t60 are mutually exclusive")
        if self.t60 is not None:
            _positive_real(self.t60, name="t60")
            estimate_beta_from_t60(self.size, self.t60, c=self.c)
        if self.beta is not None:
            if not torch.is_tensor(self.beta) or not self.beta.is_floating_point():
                raise TypeError("beta must be a real floating-point Tensor")
            validate_supported_float_tensor(self.beta, name="beta")
            if (
                self.beta.device != self.size.device
                or self.beta.dtype != self.size.dtype
            ):
                raise ValueError("beta device and dtype must match room size")
            expected = 4 if self.size.numel() == 2 else 6
            if self.beta.ndim != 1 or self.beta.numel() != expected:
                raise ValueError(
                    f"beta must have {expected} elements for {self.size.numel()}D rooms"
                )
            if not torch.all(torch.isfinite(self.beta)):
                raise ValueError("beta must contain finite values")
            if torch.any(self.beta < 0) or torch.any(self.beta > 1):
                raise ValueError("beta values must be in [0, 1]")

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

        Examples:
            ```python
            room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
            ```
        """
        size_t = as_float_tensor(size, device=device, dtype=dtype, name="room size")
        size_t = ensure_dim(size_t)
        beta_t = None
        if beta is not None:
            if torch.is_tensor(beta):
                beta_t = as_float_tensor(
                    beta,
                    device=device,
                    dtype=dtype,
                    name="beta",
                )
            else:
                beta_t = as_float_tensor(
                    beta,
                    device=size_t.device,
                    dtype=size_t.dtype,
                    name="beta",
                )
        return Room(size=size_t, fs=fs, c=c, beta=beta_t, t60=t60)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class Source:
    """Source geometry, orientation, and directivity.

    Orientations are normalized to unit vectors with the same ``(n, dim)``
    shape as positions. 2D angles use a scalar or ``(n, 1)`` radians; vectors
    use ``(2,)`` or ``(n, 2)``. 3D angles use azimuth/elevation pairs.

    Examples:
        ```python
        sources = Source.from_positions([[1.0, 2.0, 1.5]])
        ```
    """

    positions: Tensor
    orientation: Optional[Tensor] = None
    directivity: str = "omni"

    def __post_init__(self) -> None:
        pos = _normalize_entity_positions(self.positions, name="source")
        object.__setattr__(self, "positions", pos)
        ori = _normalize_entity_orientation(
            self.orientation,
            n_entities=pos.shape[0],
            dim=pos.shape[1],
            name="source",
            device=pos.device,
            dtype=pos.dtype,
        )
        if ori is not None:
            object.__setattr__(self, "orientation", ori)
        directivity = canonicalize_directivity(self.directivity, endpoint="source")
        if directivity != "omni" and ori is None:
            raise ValueError("source orientation is required for non-omni directivity")
        object.__setattr__(self, "directivity", directivity)
        self._validate_internal()

    def replace(self, **kwargs) -> "Source":
        """Return a new Source with updated fields."""
        return replace(self, **kwargs)

    def _validate_internal(self) -> None:
        _validate_entity_state(
            positions=self.positions,
            orientation=self.orientation,
            directivity=self.directivity,
            name="source",
        )

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[
            float | Sequence[float] | Sequence[Sequence[float]] | Tensor
        ] = None,
        directivity: str = "omni",
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Source":
        """Convert positions/orientation to tensors and build a Source."""
        pos = as_float_tensor(
            positions, device=device, dtype=dtype, name="source positions"
        )
        ori = None
        if orientation is not None:
            ori = _prepare_orientation_input(
                orientation,
                positions=pos,
                device=device,
                dtype=dtype,
                name="source orientation",
            )
        return cls(positions=pos, orientation=ori, directivity=directivity)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class MicrophoneArray:
    """Microphone-array geometry, orientation, and directivity.

    Orientations follow the same canonical unit-vector contract as
    `Source`.

    Examples:
        ```python
        mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])
        ```
    """

    positions: Tensor
    orientation: Optional[Tensor] = None
    directivity: str = "omni"

    def __post_init__(self) -> None:
        pos = _normalize_entity_positions(self.positions, name="mic")
        object.__setattr__(self, "positions", pos)
        ori = _normalize_entity_orientation(
            self.orientation,
            n_entities=pos.shape[0],
            dim=pos.shape[1],
            name="mic",
            device=pos.device,
            dtype=pos.dtype,
        )
        if ori is not None:
            object.__setattr__(self, "orientation", ori)
        directivity = canonicalize_directivity(
            self.directivity,
            endpoint="microphone",
        )
        if directivity != "omni" and ori is None:
            raise ValueError(
                "microphone orientation is required for non-omni directivity"
            )
        object.__setattr__(self, "directivity", directivity)
        self._validate_internal()

    def replace(self, **kwargs) -> "MicrophoneArray":
        """Return a new MicrophoneArray with updated fields."""
        return replace(self, **kwargs)

    def _validate_internal(self) -> None:
        _validate_entity_state(
            positions=self.positions,
            orientation=self.orientation,
            directivity=self.directivity,
            name="microphone",
        )

    @classmethod
    def from_positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[
            float | Sequence[float] | Sequence[Sequence[float]] | Tensor
        ] = None,
        directivity: str = "omni",
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Convert positions/orientation to tensors and build a MicrophoneArray."""
        pos = as_float_tensor(
            positions, device=device, dtype=dtype, name="microphone positions"
        )
        ori = None
        if orientation is not None:
            ori = _prepare_orientation_input(
                orientation,
                positions=pos,
                device=device,
                dtype=dtype,
                name="microphone orientation",
            )
        return cls(positions=pos, orientation=ori, directivity=directivity)


def _normalize_entity_positions(positions: Tensor, *, name: str) -> Tensor:
    pos = as_float_tensor(positions, name=f"{name} positions")
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError(f"{name} positions must have shape (n, 2) or (n, 3)")
    if pos.shape[0] == 0:
        raise ValueError(f"{name} positions must contain at least one entity")
    if not torch.all(torch.isfinite(pos)):
        raise ValueError(f"{name} positions must contain finite values")
    return pos


def _validate_entity_state(
    *,
    positions: Tensor,
    orientation: Tensor | None,
    directivity: str,
    name: str,
) -> None:
    if not torch.is_tensor(positions) or not positions.is_floating_point():
        raise TypeError(f"{name} positions must be a real floating-point Tensor")
    validate_supported_float_tensor(positions, name=f"{name} positions")
    if (
        positions.ndim != 2
        or positions.shape[0] == 0
        or positions.shape[1] not in (2, 3)
    ):
        raise ValueError(f"{name} positions must have shape (n, 2) or (n, 3)")
    if not torch.all(torch.isfinite(positions)):
        raise ValueError(f"{name} positions must contain finite values")

    canonical = canonicalize_directivity(directivity, endpoint=name)
    if canonical != directivity:
        raise ValueError(f"{name} directivity must use its canonical name")
    if orientation is None:
        if directivity != "omni":
            raise ValueError(f"{name} orientation is required for non-omni directivity")
        return
    if not torch.is_tensor(orientation) or not orientation.is_floating_point():
        raise TypeError(f"{name} orientation must be a real floating-point Tensor")
    validate_supported_float_tensor(orientation, name=f"{name} orientation")
    if orientation.device != positions.device or orientation.dtype != positions.dtype:
        raise ValueError(
            f"{name} orientation device and dtype must match its positions"
        )
    if orientation.shape != positions.shape:
        raise ValueError(
            f"{name} orientation must have canonical shape {tuple(positions.shape)}"
        )
    if not torch.all(torch.isfinite(orientation)):
        raise ValueError(f"{name} orientation must contain finite values")
    norms = torch.linalg.vector_norm(orientation, dim=-1)
    tolerance = max(1.0e-12, 4.0 * torch.finfo(orientation.dtype).eps)
    if not torch.all(torch.abs(norms - 1.0) <= tolerance):
        raise ValueError(f"{name} orientation must contain unit vectors")


def _normalize_entity_orientation(
    orientation: Optional[Tensor],
    *,
    n_entities: int,
    dim: int,
    name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    if orientation is None:
        return None
    ori = as_float_tensor(
        orientation,
        name=f"{name} orientation",
    )
    if ori.device != device or ori.dtype != dtype:
        raise ValueError(
            f"{name} orientation device and dtype must match its positions"
        )
    if not torch.all(torch.isfinite(ori)):
        raise ValueError(f"{name} orientation must contain finite values")

    if ori.ndim == 0:
        if dim != 2:
            raise ValueError(
                f"{name} orientation for 3D must be an azimuth/elevation pair "
                "or a 3D vector"
            )
        canonical = orientation_to_unit(ori, dim).unsqueeze(0)
    elif ori.ndim == 1:
        expected = (1, 2) if dim == 2 else (2, 3)
        if ori.numel() not in expected:
            if dim == 2:
                message = (
                    f"{name} shared orientation for 2D must contain one angle "
                    "or a 2D vector"
                )
            else:
                message = (
                    f"{name} shared orientation for 3D must contain "
                    "azimuth/elevation or a 3D vector"
                )
            raise ValueError(message)
        canonical = orientation_to_unit(ori, dim).reshape(1, dim)
    elif ori.ndim == 2:
        if ori.shape[0] not in (1, n_entities):
            raise ValueError(
                f"{name} orientation must have one row or {n_entities} rows"
            )
        valid_widths = (1, 2) if dim == 2 else (2, 3)
        if ori.shape[1] not in valid_widths:
            raise ValueError(f"{name} orientation for {dim}D has an unsupported shape")
        canonical = orientation_to_unit(ori, dim)
    else:
        raise ValueError(f"{name} orientation for {dim}D has an unsupported shape")

    if canonical.shape[0] == 1 and n_entities > 1:
        canonical = canonical.expand(n_entities, -1).clone()
    if canonical.shape != (n_entities, dim):
        raise ValueError(
            f"{name} orientation must resolve to shape ({n_entities}, {dim})"
        )
    return canonical


def _prepare_orientation_input(
    orientation: float | Sequence[float] | Sequence[Sequence[float]] | Tensor,
    *,
    positions: Tensor,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
    name: str,
) -> Tensor:
    if torch.is_tensor(orientation):
        return as_float_tensor(
            orientation,
            device=device,
            dtype=dtype,
            name=name,
        )
    return as_float_tensor(
        orientation,
        device=positions.device,
        dtype=positions.dtype,
        name=name,
    )


def _positive_real(value: object, *, name: str) -> float:
    return normalize_finite_real(value, name=name, positive=True)
