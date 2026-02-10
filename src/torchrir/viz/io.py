"""I/O helpers for visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import logging

import torch

from .animation import animate_scene_gif
from .scene import plot_scene_dynamic, plot_scene_static
from .utils import (
    _positions_to_cpu,
    _save_axes,
    _to_cpu,
    _traj_steps,
    _trajectory_to_cpu,
)


def render_scene_plots(
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


def save_scene_plots(
    *,
    out_dir: Path,
    room: torch.Tensor | Sequence[float],
    sources: object,
    mics: object,
    src_traj: Optional[torch.Tensor | Sequence] = None,
    mic_traj: Optional[torch.Tensor | Sequence] = None,
    prefix: str,
    show: bool,
    logger: logging.Logger,
    plot_2d: bool = True,
    plot_3d: bool = True,
) -> None:
    """Plot and save scene images."""
    try:
        static_paths, dynamic_paths = render_scene_plots(
            out_dir=out_dir,
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix=prefix,
            show=show,
            plot_2d=plot_2d,
            plot_3d=plot_3d,
        )
        for path in static_paths + dynamic_paths:
            logger.info("saved: %s", path)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Plot skipped: %s", exc)


def save_scene_gifs(
    *,
    out_dir: Path,
    room: torch.Tensor | Sequence[float],
    sources: object,
    mics: object,
    src_traj: torch.Tensor,
    mic_traj: torch.Tensor,
    prefix: str,
    signal_len: int,
    fs: int,
    gif_fps: int,
    logger: logging.Logger,
) -> None:
    """Render trajectory GIFs."""
    try:
        gif_path = out_dir / f"{prefix}.gif"
        animate_scene_gif(
            out_path=gif_path,
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            fps=gif_fps if gif_fps > 0 else None,
            signal_len=signal_len,
            fs=fs,
        )
        logger.info("saved: %s", gif_path)
        if torch.as_tensor(room).numel() == 3:
            gif_path_3d = out_dir / f"{prefix}_3d.gif"
            animate_scene_gif(
                out_path=gif_path_3d,
                room=room,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                fps=gif_fps if gif_fps > 0 else None,
                signal_len=signal_len,
                fs=fs,
                plot_2d=False,
                plot_3d=True,
            )
            logger.info("saved: %s", gif_path_3d)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("GIF skipped: %s", exc)
