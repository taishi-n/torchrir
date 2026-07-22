"""Scene containers for simulation inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor

from .room import MicrophoneArray, Room, Source
from .schedule import FrameSchedule
from ..util.tensor import as_float_tensor


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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

    def validate(self) -> None:
        """Revalidate geometry tensors after possible external mutation."""

        _validate_scene_entities(self.room, self.sources, self.mics)


@dataclass(frozen=True, slots=True, init=False, eq=False)
class DynamicScene:
    """Container for dynamic scene simulation inputs.

    An optional `FrameSchedule` stores exact sample-domain frame starts.
    It can be consumed directly by dynamic convolution without a lossy
    seconds-to-samples round trip.

    Examples:
        ```python
        scene = DynamicScene(
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            schedule=schedule,
        )
        ```
    """

    room: Room
    sources: Source
    mics: MicrophoneArray
    src_traj: Tensor
    mic_traj: Tensor
    schedule: Optional[FrameSchedule] = None

    def __init__(
        self,
        *,
        room: Room,
        sources: Source,
        mics: MicrophoneArray,
        src_traj: (
            Tensor | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
        ),
        mic_traj: (
            Tensor | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
        ),
        schedule: FrameSchedule | None = None,
    ) -> None:
        object.__setattr__(self, "room", room)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "mics", mics)
        object.__setattr__(self, "src_traj", src_traj)
        object.__setattr__(self, "mic_traj", mic_traj)
        object.__setattr__(self, "schedule", schedule)
        self.__post_init__()

    def __post_init__(self) -> None:
        _validate_scene_entity_types(self.room, self.sources, self.mics)
        src_traj = _normalize_trajectory(
            self.src_traj,
            reference=self.sources.positions,
            name="src_traj",
        )
        mic_traj = _normalize_trajectory(
            self.mic_traj,
            reference=self.mics.positions,
            name="mic_traj",
        )
        object.__setattr__(self, "src_traj", src_traj)
        object.__setattr__(self, "mic_traj", mic_traj)
        if self.schedule is not None and not isinstance(self.schedule, FrameSchedule):
            raise TypeError("schedule must be a FrameSchedule or None")
        self._validate_internal()

    def validate(self) -> None:
        """Revalidate geometry and trajectory tensors after external mutation."""

        self._validate_internal()

    def _validate_internal(self) -> None:
        _validate_scene_entities(self.room, self.sources, self.mics)
        dim = int(self.room.size.numel())
        n_src = int(self.sources.positions.shape[0])
        n_mic = int(self.mics.positions.shape[0])
        t_src = _validate_traj(self.src_traj, n_src, dim, "src_traj")
        t_mic = _validate_traj(self.mic_traj, n_mic, dim, "mic_traj")
        _validate_tensor_layout(
            (
                ("src_traj", self.src_traj),
                ("mic_traj", self.mic_traj),
            ),
            reference_name="room size",
            reference=self.room.size,
        )
        if t_src != t_mic:
            raise ValueError("src_traj and mic_traj must have matching time steps")
        src_first = self.src_traj[0] if self.src_traj.ndim == 3 else self.src_traj[0:1]
        mic_first = self.mic_traj[0] if self.mic_traj.ndim == 3 else self.mic_traj[0:1]
        source_positions = self.sources.positions.to(src_first.device)
        microphone_positions = self.mics.positions.to(mic_first.device)
        if not torch.equal(src_first, source_positions):
            raise ValueError("sources.positions must match the first src_traj frame")
        if not torch.equal(mic_first, microphone_positions):
            raise ValueError("mics.positions must match the first mic_traj frame")
        _validate_positions_in_room(self.src_traj, self.room, "src_traj")
        _validate_positions_in_room(self.mic_traj, self.room, "mic_traj")
        if self.schedule is not None and len(self.schedule) != t_src:
            raise ValueError(
                f"schedule has {len(self.schedule)} frames, but trajectories have {t_src}"
            )
        if (
            self.schedule is not None
            and self.schedule.conversion_sample_rate is not None
            and self.schedule.conversion_sample_rate != float(self.room.fs)
        ):
            raise ValueError(
                "schedule seconds-conversion sample rate "
                f"{self.schedule.conversion_sample_rate:g} conflicts "
                f"with room sampling rate {float(self.room.fs):g}"
            )


def _validate_scene_entities(
    room: Room, sources: Source, mics: MicrophoneArray
) -> None:
    _validate_scene_entity_types(room, sources, mics)

    room._validate_internal()
    sources._validate_internal()
    mics._validate_internal()

    dim = int(room.size.numel())
    if sources.positions.shape[1] != dim:
        raise ValueError("source position dimension must match room dimension")
    if mics.positions.shape[1] != dim:
        raise ValueError("mic position dimension must match room dimension")
    _validate_tensor_layout(
        (
            ("source positions", sources.positions),
            ("mic positions", mics.positions),
        ),
        reference_name="room size",
        reference=room.size,
    )
    _validate_positions_in_room(sources.positions, room, "source positions")
    _validate_positions_in_room(mics.positions, room, "mic positions")


def _validate_scene_entity_types(
    room: object,
    sources: object,
    mics: object,
) -> None:
    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if not isinstance(sources, Source):
        raise TypeError("sources must be a Source instance")
    if not isinstance(mics, MicrophoneArray):
        raise TypeError("mics must be a MicrophoneArray instance")


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
        if traj.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one time step")
        return int(traj.shape[0])
    if traj.ndim == 3:
        if traj.shape[1] != count or traj.shape[2] != dim:
            raise ValueError(f"{name} must have shape (T, {count}, {dim})")
        if traj.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one time step")
        return int(traj.shape[0])
    raise ValueError(f"{name} must have shape (T, {count}, {dim})")


def _validate_positions_in_room(positions: Tensor, room: Room, name: str) -> None:
    room_size = room.size.to(device=positions.device, dtype=positions.dtype)
    if torch.any(positions <= 0) or torch.any(positions >= room_size):
        raise ValueError(f"{name} must lie strictly within room bounds (0, room.size)")


def _normalize_trajectory(
    trajectory: Tensor
    | Sequence[Sequence[float]]
    | Sequence[Sequence[Sequence[float]]],
    *,
    reference: Tensor,
    name: str,
) -> Tensor:
    if torch.is_tensor(trajectory):
        return as_float_tensor(trajectory, name=name)
    return as_float_tensor(
        trajectory,
        device=reference.device,
        dtype=reference.dtype,
        name=name,
    )


def _validate_tensor_layout(
    tensors: tuple[tuple[str, Tensor], ...],
    *,
    reference_name: str,
    reference: Tensor,
) -> None:
    for name, value in tensors:
        if value.device != reference.device:
            raise ValueError(
                f"{name} device {value.device} must match {reference_name} "
                f"device {reference.device}"
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"{name} dtype {value.dtype} must match {reference_name} "
                f"dtype {reference.dtype}"
            )


def _validate_dynamic_time_reference(
    scene: DynamicScene,
    *,
    time_reference: str,
) -> None:
    """Reject motion that does not match a one-time dynamic convolution model."""

    source_moves = _trajectory_is_moving(scene.src_traj)
    microphone_moves = _trajectory_is_moving(scene.mic_traj)
    if source_moves and microphone_moves:
        raise ValueError(
            "simultaneous source and microphone motion requires a retarded-time "
            "propagation model, which is not implemented"
        )
    if time_reference == "emission" and microphone_moves:
        raise ValueError(
            "emission-time convolution does not support moving microphones; "
            "use time_reference='observation' with fixed sources"
        )
    if time_reference == "observation" and source_moves:
        raise ValueError(
            "observation-time convolution requires fixed sources; use "
            "time_reference='emission' with fixed microphones"
        )


def _trajectory_is_moving(trajectory: Tensor) -> bool:
    first_frame = trajectory[0:1]
    return not torch.equal(trajectory, first_frame.expand_as(trajectory))
