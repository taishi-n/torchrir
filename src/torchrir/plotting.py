from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .room import MicrophoneArray, Room, Source
from .utils import as_tensor, ensure_dim


def plot_scene_static(
    *,
    room: Room | Sequence[float] | Tensor,
    sources: Source | Tensor | Sequence,
    mics: MicrophoneArray | Tensor | Sequence,
    ax=None,
    title: Optional[str] = None,
    show: bool = False,
):
    plt, ax = _setup_axes(ax, room)

    size = _room_size(room, ax)
    _draw_room(ax, size)

    src = _extract_positions(sources, ax)
    mic = _extract_positions(mics, ax)

    _scatter_positions(ax, src, label="sources", marker="^")
    _scatter_positions(ax, mic, label="mics", marker="o")

    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def plot_scene_dynamic(
    *,
    room: Room | Sequence[float] | Tensor,
    src_traj: Tensor | Sequence,
    mic_traj: Tensor | Sequence,
    step: int = 1,
    ax=None,
    title: Optional[str] = None,
    show: bool = False,
):
    plt, ax = _setup_axes(ax, room)

    size = _room_size(room, ax)
    _draw_room(ax, size)

    src_traj = _as_trajectory(src_traj, ax)
    mic_traj = _as_trajectory(mic_traj, ax)

    _plot_trajectories(ax, src_traj, step=step, label="source path")
    _plot_trajectories(ax, mic_traj, step=step, label="mic path")

    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def _setup_axes(ax, room):
    import matplotlib.pyplot as plt

    size = _room_size(room, ax)
    dim = size.numel()
    if ax is None:
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            _, ax = plt.subplots()
    return plt, ax


def _room_size(room, ax) -> Tensor:
    if isinstance(room, Room):
        size = room.size
    else:
        size = room
    size = as_tensor(size)
    size = ensure_dim(size)
    return size


def _draw_room(ax, size: Tensor) -> None:
    dim = size.numel()
    if dim == 2:
        _draw_room_2d(ax, size)
    else:
        _draw_room_3d(ax, size)


def _draw_room_2d(ax, size: Tensor) -> None:
    import matplotlib.patches as patches

    rect = patches.Rectangle((0.0, 0.0), size[0].item(), size[1].item(),
                             fill=False, edgecolor="black")
    ax.add_patch(rect)
    ax.set_xlim(0, size[0].item())
    ax.set_ylim(0, size[1].item())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_room_3d(ax, size: Tensor) -> None:
    x, y, z = size.tolist()
    corners = torch.tensor(
        [
            [0, 0, 0],
            [x, 0, 0],
            [x, y, 0],
            [0, y, 0],
            [0, 0, z],
            [x, 0, z],
            [x, y, z],
            [0, y, z],
        ],
        dtype=torch.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in edges:
        ax.plot(
            [corners[a, 0], corners[b, 0]],
            [corners[a, 1], corners[b, 1]],
            [corners[a, 2], corners[b, 2]],
            color="black",
        )
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.set_zlim(0, z)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def _extract_positions(entity, ax) -> Tensor:
    if isinstance(entity, (Source, MicrophoneArray)):
        pos = entity.positions
    else:
        pos = entity
    pos = as_tensor(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _scatter_positions(ax, positions: Tensor, *, label: str, marker: str) -> None:
    if positions.numel() == 0:
        return
    dim = positions.shape[1]
    if dim == 2:
        ax.scatter(positions[:, 0], positions[:, 1], label=label, marker=marker)
    else:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], label=label, marker=marker)


def _as_trajectory(traj, ax) -> Tensor:
    traj = as_tensor(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _plot_trajectories(ax, traj: Tensor, *, step: int, label: str) -> None:
    if traj.numel() == 0:
        return
    dim = traj.shape[2]
    if dim == 2:
        for idx in range(traj.shape[1]):
            xy = traj[::step, idx]
            ax.plot(xy[:, 0], xy[:, 1], label=f"{label} {idx}")
    else:
        for idx in range(traj.shape[1]):
            xyz = traj[::step, idx]
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=f"{label} {idx}")
