"""Room, source, and microphone geometry models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

from ..util.tensor import as_tensor, ensure_dim


@dataclass(frozen=True)
class Room:
    """Room geometry and acoustic parameters.

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
        size = ensure_dim(self.size)
        if not torch.all(torch.isfinite(size)):
            raise ValueError("room size must contain finite values")
        if torch.any(size <= 0):
            raise ValueError("room size must be strictly positive")
        object.__setattr__(self, "size", size)
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if self.c <= 0:
            raise ValueError("c must be positive")
        if self.beta is not None and self.t60 is not None:
            raise ValueError("beta and t60 are mutually exclusive")
        if self.t60 is not None and self.t60 <= 0:
            raise ValueError("t60 must be positive")
        if self.beta is not None:
            beta = as_tensor(self.beta, dtype=size.dtype).view(-1)
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

        Examples:
            ```python
            room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
            ```
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

    Examples:
        ```python
        sources = Source.from_positions([[1.0, 2.0, 1.5]])
        ```
    """

    positions: Tensor
    orientation: Optional[Tensor] = None

    def __post_init__(self) -> None:
        pos = _normalize_entity_positions(self.positions, name="source")
        object.__setattr__(self, "positions", pos)
        ori = _normalize_entity_orientation(
            self.orientation, n_entities=pos.shape[0], dim=pos.shape[1], name="source"
        )
        if ori is not None:
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

    Examples:
        ```python
        mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])
        ```
    """

    positions: Tensor
    orientation: Optional[Tensor] = None

    def __post_init__(self) -> None:
        pos = _normalize_entity_positions(self.positions, name="mic")
        object.__setattr__(self, "positions", pos)
        ori = _normalize_entity_orientation(
            self.orientation, n_entities=pos.shape[0], dim=pos.shape[1], name="mic"
        )
        if ori is not None:
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


def _validate_orientation(
    orientation: Tensor, *, n_entities: int, dim: int, name: str
) -> None:
    if not torch.all(torch.isfinite(orientation)):
        raise ValueError(f"{name} orientation must contain finite values")

    if dim == 2:
        if orientation.ndim == 0:
            return
        if orientation.ndim == 1:
            if orientation.numel() in (1, 2, n_entities):
                return
            raise ValueError(
                f"{name} orientation for 2D must be angle, 2D vector, or per-entity angles"
            )
        if orientation.ndim == 2:
            if orientation.shape[0] != n_entities or orientation.shape[1] not in (1, 2):
                raise ValueError(
                    f"{name} orientation for 2D must have shape (n, 1) or (n, 2)"
                )
            return
        raise ValueError(f"{name} orientation for 2D has unsupported shape")

    if dim == 3:
        if orientation.ndim == 1:
            if orientation.numel() in (2, 3):
                return
            raise ValueError(
                f"{name} orientation for 3D must be a 3D vector or (azimuth, elevation)"
            )
        if orientation.ndim == 2:
            if orientation.shape[0] != n_entities or orientation.shape[1] not in (2, 3):
                raise ValueError(
                    f"{name} orientation for 3D must have shape (n, 2) or (n, 3)"
                )
            return
        raise ValueError(f"{name} orientation for 3D has unsupported shape")


def _normalize_entity_positions(positions: Tensor, *, name: str) -> Tensor:
    pos = as_tensor(positions)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError(f"{name} positions must have shape (n, 2) or (n, 3)")
    if not torch.all(torch.isfinite(pos)):
        raise ValueError(f"{name} positions must contain finite values")
    return pos


def _normalize_entity_orientation(
    orientation: Optional[Tensor],
    *,
    n_entities: int,
    dim: int,
    name: str,
) -> Optional[Tensor]:
    if orientation is None:
        return None
    ori = as_tensor(orientation)
    _validate_orientation(ori, n_entities=n_entities, dim=dim, name=name)
    return ori
