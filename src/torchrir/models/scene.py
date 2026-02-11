"""Scene containers for simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import torch
from torch import Tensor

from .room import MicrophoneArray, Room, Source
from ..util.tensor import as_tensor


@dataclass(frozen=True)
class StaticScene:
    """Container for static scene simulation inputs.

    Examples:
        ```python
        scene = StaticScene(room=room, sources=sources, mics=mics)
        ```
    """

    room: Room
    sources: Source
    mics: MicrophoneArray

    def __post_init__(self) -> None:
        _validate_scene_entities(self.room, self.sources, self.mics)

    def is_dynamic(self) -> bool:
        return False

    def validate(self) -> None:
        _validate_scene_entities(self.room, self.sources, self.mics)


@dataclass(frozen=True)
class DynamicScene:
    """Container for dynamic scene simulation inputs.

    Examples:
        ```python
        scene = DynamicScene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
        ```
    """

    room: Room
    sources: Source
    mics: MicrophoneArray
    src_traj: Tensor
    mic_traj: Tensor

    def __post_init__(self) -> None:
        src_traj = as_tensor(self.src_traj)
        mic_traj = as_tensor(self.mic_traj)
        object.__setattr__(self, "src_traj", src_traj)
        object.__setattr__(self, "mic_traj", mic_traj)
        self._validate_internal()

    def is_dynamic(self) -> bool:
        return True

    def validate(self) -> None:
        self._validate_internal()

    def _validate_internal(self) -> None:
        _validate_scene_entities(self.room, self.sources, self.mics)
        dim = int(self.room.size.numel())
        n_src = int(self.sources.positions.shape[0])
        n_mic = int(self.mics.positions.shape[0])
        t_src = _validate_traj(self.src_traj, n_src, dim, "src_traj")
        t_mic = _validate_traj(self.mic_traj, n_mic, dim, "mic_traj")
        if t_src != t_mic:
            raise ValueError("src_traj and mic_traj must have matching time steps")


@dataclass(frozen=True)
class Scene:
    """Deprecated scene wrapper.

    `Scene` is kept for backward compatibility. Prefer `StaticScene` and
    `DynamicScene` to avoid ambiguous states.
    """

    room: Room
    sources: Source
    mics: MicrophoneArray
    src_traj: Optional[Tensor] = None
    mic_traj: Optional[Tensor] = None

    def __post_init__(self) -> None:
        warnings.warn(
            "Scene is deprecated and will be removed in a future release. "
            "Use StaticScene or DynamicScene.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._validate_internal()

    def _validate_internal(self) -> None:
        _validate_scene_entities(self.room, self.sources, self.mics)
        has_src = self.src_traj is not None
        has_mic = self.mic_traj is not None
        if has_src != has_mic:
            raise ValueError(
                "Scene requires both src_traj and mic_traj for dynamic scenes. "
                "Use StaticScene for static inputs."
            )
        if has_src and has_mic:
            assert self.src_traj is not None
            assert self.mic_traj is not None
            dim = int(self.room.size.numel())
            n_src = int(self.sources.positions.shape[0])
            n_mic = int(self.mics.positions.shape[0])
            t_src = _validate_traj(self.src_traj, n_src, dim, "src_traj")
            t_mic = _validate_traj(self.mic_traj, n_mic, dim, "mic_traj")
            if t_src != t_mic:
                raise ValueError("src_traj and mic_traj must have matching time steps")

    def is_dynamic(self) -> bool:
        return self.src_traj is not None and self.mic_traj is not None

    def validate(self) -> None:
        self._validate_internal()

    def to_static_scene(self) -> StaticScene:
        if self.is_dynamic():
            raise ValueError("dynamic Scene cannot be converted to StaticScene")
        return StaticScene(room=self.room, sources=self.sources, mics=self.mics)

    def to_dynamic_scene(self) -> DynamicScene:
        if not self.is_dynamic() or self.src_traj is None or self.mic_traj is None:
            raise ValueError("static Scene cannot be converted to DynamicScene")
        return DynamicScene(
            room=self.room,
            sources=self.sources,
            mics=self.mics,
            src_traj=self.src_traj,
            mic_traj=self.mic_traj,
        )


SceneLike = StaticScene | DynamicScene | Scene


def _validate_scene_entities(room: Room, sources: Source, mics: MicrophoneArray) -> None:
    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if not isinstance(sources, Source):
        raise TypeError("sources must be a Source instance")
    if not isinstance(mics, MicrophoneArray):
        raise TypeError("mics must be a MicrophoneArray instance")

    dim = int(room.size.numel())
    if sources.positions.shape[1] != dim:
        raise ValueError("source position dimension must match room dimension")
    if mics.positions.shape[1] != dim:
        raise ValueError("mic position dimension must match room dimension")


def _validate_traj(
    traj: Tensor,
    count: int,
    dim: int,
    name: str,
) -> int:
    if not torch.is_tensor(traj):
        raise TypeError(f"{name} must be a Tensor")
    if not torch.all(torch.isfinite(traj)):
        raise ValueError(f"{name} must contain finite values")
    if traj.ndim == 2:
        if count != 1:
            raise ValueError(f"{name} must have shape (T, {count}, {dim})")
        if traj.shape[1] != dim:
            raise ValueError(f"{name} must have shape (T, {dim}) for single entity")
        return int(traj.shape[0])
    if traj.ndim == 3:
        if traj.shape[1] != count or traj.shape[2] != dim:
            raise ValueError(f"{name} must have shape (T, {count}, {dim})")
        return int(traj.shape[0])
    raise ValueError(f"{name} must have shape (T, {count}, {dim})")
