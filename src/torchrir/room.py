from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .utils import as_tensor, ensure_dim


@dataclass(frozen=True)
class Room:
    size: Tensor
    fs: float
    c: float = 343.0
    beta: Optional[Tensor] = None
    t60: Optional[float] = None

    def __post_init__(self) -> None:
        size = ensure_dim(self.size)
        object.__setattr__(self, "size", size)
        if self.beta is not None and self.t60 is not None:
            raise ValueError("beta and t60 are mutually exclusive")

    @staticmethod
    def shoebox(
        size,
        *,
        fs: float,
        c: float = 343.0,
        beta=None,
        t60: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Room":
        size_t = as_tensor(size, device=device, dtype=dtype)
        size_t = ensure_dim(size_t)
        beta_t = None
        if beta is not None:
            beta_t = as_tensor(beta, device=device, dtype=dtype)
        return Room(size=size_t, fs=fs, c=c, beta=beta_t, t60=t60)


class Source:
    def __init__(self, positions: Tensor, orientation: Optional[Tensor] = None) -> None:
        self.positions = positions
        self.orientation = orientation

    @classmethod
    def positions(
        cls,
        positions,
        *,
        orientation=None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Source":
        return cls.from_positions(
            positions, orientation=orientation, device=device, dtype=dtype
        )

    @classmethod
    def from_positions(
        cls,
        positions,
        *,
        orientation=None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Source":
        pos = as_tensor(positions, device=device, dtype=dtype)
        ori = None
        if orientation is not None:
            ori = as_tensor(orientation, device=device, dtype=dtype)
        return cls(pos, ori)


class MicrophoneArray:
    def __init__(self, positions: Tensor, orientation: Optional[Tensor] = None) -> None:
        self.positions = positions
        self.orientation = orientation

    @classmethod
    def positions(
        cls,
        positions,
        *,
        orientation=None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        return cls.from_positions(
            positions, orientation=orientation, device=device, dtype=dtype
        )

    @classmethod
    def from_positions(
        cls,
        positions,
        *,
        orientation=None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MicrophoneArray":
        pos = as_tensor(positions, device=device, dtype=dtype)
        ori = None
        if orientation is not None:
            ori = as_tensor(orientation, device=device, dtype=dtype)
        return cls(pos, ori)
