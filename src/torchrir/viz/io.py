from __future__ import annotations

"""I/O helpers for visualization."""

from pathlib import Path
from typing import Optional, Sequence

import torch

from .scene import plot_scene_dynamic, plot_scene_static
from .utils import (
    _positions_to_cpu,
    _save_axes,
    _to_cpu,
    _traj_steps,
    _trajectory_to_cpu,
)


def plot_scene_and_save(
    *,
    out_dir: Path,
    room: Sequence[float] | torch.Tensor,
    sources: object | torch.Tensor | Sequence,
    mics: object | torch.Tensor | Sequence,
    src_traj: Optional[torch.Tensor | Sequence] = None,
    mic_traj: Optional[torch.Tensor | Sequence] = None,
    prefix: str = "scene",
    step: int = 1,
    show: bool = False,
    plot_2d: bool = True,
    plot_3d: bool = True,
) -> tuple[list[Path], list[Path]]:
    """Plot static and dynamic scenes and save images to disk."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    room_size = _to_cpu(room)
    src_pos = _positions_to_cpu(sources)
    mic_pos = _positions_to_cpu(mics)
    dim = int(room_size.numel())

    static_paths: list[Path] = []
    dynamic_paths: list[Path] = []

    for view_dim, enabled in ((2, plot_2d), (3, plot_3d)):
        if not enabled:
            continue
        if view_dim == 2 and dim < 2:
            continue
        if view_dim == 3 and dim < 3:
            continue
        view_room = room_size[:view_dim]
        view_src = src_pos[:, :view_dim]
        view_mic = mic_pos[:, :view_dim]

        ax = plot_scene_static(
            room=view_room,
            sources=view_src,
            mics=view_mic,
            title=f"Room scene ({view_dim}D static)",
            show=False,
        )
        static_path = out_dir / f"{prefix}_static_{view_dim}d.png"
        _save_axes(ax, static_path, show=show)
        static_paths.append(static_path)

        if src_traj is not None or mic_traj is not None:
            steps = _traj_steps(src_traj, mic_traj)
            src_traj = _trajectory_to_cpu(src_traj, src_pos, steps)
            mic_traj = _trajectory_to_cpu(mic_traj, mic_pos, steps)
            view_src_traj = src_traj[:, :, :view_dim]
            view_mic_traj = mic_traj[:, :, :view_dim]
            ax = plot_scene_dynamic(
                room=view_room,
                src_traj=view_src_traj,
                mic_traj=view_mic_traj,
                src_pos=view_src,
                mic_pos=view_mic,
                step=step,
                title=f"Room scene ({view_dim}D trajectories)",
                show=False,
            )
            dynamic_path = out_dir / f"{prefix}_dynamic_{view_dim}d.png"
            _save_axes(ax, dynamic_path, show=show)
            dynamic_paths.append(dynamic_path)

    return static_paths, dynamic_paths
