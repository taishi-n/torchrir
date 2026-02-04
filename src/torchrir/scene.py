from __future__ import annotations

"""Scene container for simulation inputs."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .room import MicrophoneArray, Room, Source


@dataclass(frozen=True)
class Scene:
    """Container for room, sources, microphones, and optional trajectories.

    Example:
        >>> scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
        >>> scene.validate()
    """

    room: Room
    sources: Source
    mics: MicrophoneArray
    src_traj: Optional[Tensor] = None
    mic_traj: Optional[Tensor] = None

    def is_dynamic(self) -> bool:
        """Return True if any trajectory is provided."""
        return self.src_traj is not None or self.mic_traj is not None

    def validate(self) -> None:
        """Validate scene consistency and trajectory shapes."""
        if not isinstance(self.room, Room):
            raise TypeError("room must be a Room instance")
        if not isinstance(self.sources, Source):
            raise TypeError("sources must be a Source instance")
        if not isinstance(self.mics, MicrophoneArray):
            raise TypeError("mics must be a MicrophoneArray instance")

        dim = self.room.size.numel()
        n_src = self.sources.positions.shape[0]
        n_mic = self.mics.positions.shape[0]

        t_src = _validate_traj(self.src_traj, n_src, dim, "src_traj")
        t_mic = _validate_traj(self.mic_traj, n_mic, dim, "mic_traj")
        if t_src is not None and t_mic is not None and t_src != t_mic:
            raise ValueError("src_traj and mic_traj must have matching time steps")


def _validate_traj(
    traj: Optional[Tensor],
    count: int,
    dim: int,
    name: str,
) -> Optional[int]:
    if traj is None:
        return None
    if not torch.is_tensor(traj):
        raise TypeError(f"{name} must be a Tensor")
    if traj.ndim == 2:
        if count != 1:
            raise ValueError(f"{name} must have shape (T, {count}, {dim})")
        if traj.shape[1] != dim:
            raise ValueError(f"{name} must have shape (T, {dim}) for single entity")
        return traj.shape[0]
    if traj.ndim == 3:
        if traj.shape[1] != count or traj.shape[2] != dim:
            raise ValueError(f"{name} must have shape (T, {count}, {dim})")
        return traj.shape[0]
    raise ValueError(f"{name} must have shape (T, {count}, {dim})")
