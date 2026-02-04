from __future__ import annotations

"""Room, source, and microphone geometry models."""

from dataclasses import dataclass, replace
from typing import Optional, Sequence

import torch
from torch import Tensor

from .utils import as_tensor, ensure_dim


@dataclass(frozen=True)
class Room:
    """Room geometry and acoustic parameters."""

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
        """Create a rectangular (shoebox) room."""
        size_t = as_tensor(size, device=device, dtype=dtype)
        size_t = ensure_dim(size_t)
        beta_t = None
        if beta is not None:
            beta_t = as_tensor(beta, device=device, dtype=dtype)
        return Room(size=size_t, fs=fs, c=c, beta=beta_t, t60=t60)


@dataclass(frozen=True)
class Source:
    """Source container with positions and optional orientation."""

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
    def positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Source":
        """Construct a Source from positions."""
        return cls.from_positions(
            positions, orientation=orientation, device=device, dtype=dtype
        )

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
    """Microphone array container."""

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
    def positions(
        cls,
        positions: Sequence[Sequence[float]] | Tensor,
        *,
        orientation: Optional[Sequence[float] | Tensor] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        """Construct a MicrophoneArray from positions."""
        return cls.from_positions(
            positions, orientation=orientation, device=device, dtype=dtype
        )

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
