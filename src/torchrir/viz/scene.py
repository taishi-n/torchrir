from __future__ import annotations

"""Matplotlib-based plotting helpers for room scenes."""

from typing import Any, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor

from ..models import MicrophoneArray, Room, Source
from ..util.tensor import as_tensor, ensure_dim


def plot_scene_static(
    *,
    room: Room | Sequence[float] | Tensor,
    sources: Source | Tensor | Sequence,
    mics: MicrophoneArray | Tensor | Sequence,
    ax: Any | None = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot a static room with source and mic positions.

    Example:
        >>> ax = plot_scene_static(
        ...     room=[6.0, 4.0, 3.0],
        ...     sources=[[1.0, 2.0, 1.5]],
        ...     mics=[[2.0, 2.0, 1.5]],
        ... )
    """
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
    src_pos: Optional[Tensor | Sequence] = None,
    mic_pos: Optional[Tensor | Sequence] = None,
    ax: Any | None = None,
    title: Optional[str] = None,
    show: bool = False,
):
    """Plot source and mic trajectories within a room.

    If trajectories are static, only positions are plotted.

    Example:
        >>> ax = plot_scene_dynamic(
        ...     room=[6.0, 4.0, 3.0],
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ... )
    """
    plt, ax = _setup_axes(ax, room)

    size = _room_size(room, ax)
    _draw_room(ax, size)

    src_traj = _as_trajectory(src_traj)
    mic_traj = _as_trajectory(mic_traj)
    src_pos_t = _extract_positions(src_pos, ax) if src_pos is not None else src_traj[0]
    mic_pos_t = _extract_positions(mic_pos, ax) if mic_pos is not None else mic_traj[0]

    _plot_entity(ax, src_traj, src_pos_t, step=step, label="sources", marker="^")
    _plot_entity(ax, mic_traj, mic_pos_t, step=step, label="mics", marker="o")

    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def _setup_axes(
    ax: Any | None, room: Room | Sequence[float] | Tensor
) -> tuple[Any, Any]:
    """Create 2D/3D axes based on room dimension."""
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


def _room_size(room: Room | Sequence[float] | Tensor, ax: Any | None) -> Tensor:
    """Normalize room size input to a 1D tensor."""
    if isinstance(room, Room):
        size = room.size
    else:
        size = room
    size = as_tensor(size)
    size = ensure_dim(size)
    return size


def _draw_room(ax: Any, size: Tensor) -> None:
    """Draw a 2D or 3D room outline."""
    dim = size.numel()
    if dim == 2:
        _draw_room_2d(ax, size)
    else:
        _draw_room_3d(ax, size)


def _draw_room_2d(ax: Any, size: Tensor) -> None:
    """Draw a 2D rectangular room."""
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (0.0, 0.0), size[0].item(), size[1].item(), fill=False, edgecolor="black"
    )
    ax.add_patch(rect)
    ax.set_xlim(0, size[0].item())
    ax.set_ylim(0, size[1].item())
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def _draw_room_3d(ax: Any, size: Tensor) -> None:
    """Draw a 3D box representing the room."""
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


def _extract_positions(
    entity: Source | MicrophoneArray | Tensor | Sequence, ax: Any | None
) -> Tensor:
    """Extract positions from Source/MicrophoneArray or raw tensor."""
    if isinstance(entity, (Source, MicrophoneArray)):
        pos = entity.positions
    else:
        pos = entity
    pos = as_tensor(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _scatter_positions(
    ax: Any,
    positions: Tensor,
    *,
    label: str,
    marker: str,
    color: Optional[str] = None,
) -> None:
    """Scatter-plot positions in 2D or 3D."""
    if positions.numel() == 0:
        return
    dim = positions.shape[1]
    if dim == 2:
        ax.scatter(
            positions[:, 0], positions[:, 1], label=label, marker=marker, color=color
        )
    else:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            label=label,
            marker=marker,
            color=color,
        )


def _as_trajectory(traj: Tensor | Sequence) -> Tensor:
    """Validate and normalize a trajectory tensor."""
    traj = as_tensor(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _plot_entity(
    ax: Any,
    traj: Tensor,
    positions: Tensor,
    *,
    step: int,
    label: str,
    marker: str,
) -> None:
    """Plot trajectories and/or static positions with a unified legend entry."""
    if traj.numel() == 0:
        return
    import matplotlib.pyplot as plt

    if positions.shape != traj.shape[1:]:
        positions = traj[0]
    moving = _is_moving(traj, positions)
    colors = plt.rcParams.get("axes.prop_cycle", None)
    if colors is not None:
        palette = colors.by_key().get("color", [])
    else:
        palette = []
    if not palette:
        palette = ["C0", "C1", "C2", "C3", "C4", "C5"]

    dim = traj.shape[2]
    for idx in range(traj.shape[1]):
        color = palette[idx % len(palette)]
        lbl = label if idx == 0 else "_nolegend_"
        if moving:
            if dim == 2:
                xy = traj[::step, idx]
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    label=lbl,
                    color=color,
                    marker=marker,
                    markevery=[0],
                )
            else:
                xyz = traj[::step, idx]
                ax.plot(
                    xyz[:, 0],
                    xyz[:, 1],
                    xyz[:, 2],
                    label=lbl,
                    color=color,
                    marker=marker,
                    markevery=[0],
                )
        pos = positions[idx : idx + 1]
        _scatter_positions(ax, pos, label="_nolegend_", marker=marker, color=color)
    if not moving:
        _scatter_positions(
            ax,
            positions[:1],
            label=label,
            marker=marker,
            color=palette[0],
        )


def _is_moving(traj: Tensor, positions: Tensor, *, tol: float = 1e-6) -> bool:
    """Return True if any trajectory deviates from the provided positions."""
    if traj.numel() == 0:
        return False
    pos0 = positions.unsqueeze(0).expand_as(traj)
    return bool(torch.any(torch.linalg.norm(traj - pos0, dim=-1) > tol).item())
