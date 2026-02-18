"""I/O helpers for visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import logging

import torch

from .animation import animate_scene_gif, animate_scene_mp4
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
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
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
            annotate_sources=annotate_sources,
            annotation_lines=annotation_lines,
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
                annotate_sources=annotate_sources,
                annotation_lines=annotation_lines,
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
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
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
            annotate_sources=annotate_sources,
            annotation_lines=annotation_lines,
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
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
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
            annotate_sources=annotate_sources,
            annotation_lines=annotation_lines,
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
                annotate_sources=annotate_sources,
                annotation_lines=annotation_lines,
            )
            logger.info("saved: %s", gif_path_3d)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("GIF skipped: %s", exc)


def save_scene_videos(
    *,
    out_dir: Path,
    room: torch.Tensor | Sequence[float],
    sources: object,
    mics: object,
    src_traj: torch.Tensor,
    mic_traj: torch.Tensor,
    signal_len: int,
    fs: int,
    logger: logging.Logger,
    mp4_fps: float | None = None,
    save_3d: bool = True,
    mixture_path: Path | None = None,
    mux_audio: bool = True,
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
) -> None:
    """Render trajectory MP4 videos.

    Output names follow oobss-compatible conventions:
    - ``room_layout_2d.mp4``
    - ``room_layout_3d.mp4`` (3D rooms when ``save_3d`` is enabled)
    """
    try:
        path_2d = out_dir / "room_layout_2d.mp4"
        animate_scene_mp4(
            out_path=path_2d,
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            fps=mp4_fps,
            signal_len=signal_len,
            fs=fs,
            plot_2d=True,
            plot_3d=False,
            annotate_sources=annotate_sources,
            annotation_lines=annotation_lines,
            mixture_path=mixture_path,
            mux_audio=mux_audio,
        )
        logger.info("saved: %s", path_2d)

        if torch.as_tensor(room).numel() == 3 and save_3d:
            path_3d = out_dir / "room_layout_3d.mp4"
            animate_scene_mp4(
                out_path=path_3d,
                room=room,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                fps=mp4_fps,
                signal_len=signal_len,
                fs=fs,
                plot_2d=False,
                plot_3d=True,
                annotate_sources=annotate_sources,
                annotation_lines=annotation_lines,
                mixture_path=mixture_path,
                mux_audio=mux_audio,
            )
            logger.info("saved: %s", path_3d)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("MP4 skipped: %s", exc)


def save_scene_layout_images(
    *,
    out_dir: Path,
    room: torch.Tensor | Sequence[float],
    sources: object,
    mics: object,
    logger: logging.Logger,
    src_traj: torch.Tensor | Sequence | None = None,
    mic_traj: torch.Tensor | Sequence | None = None,
    save_2d: bool = True,
    save_3d: bool = True,
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
    show: bool = False,
) -> None:
    """Save static layout images with explicit 2D/3D filenames."""
    try:
        room_size = _to_cpu(room)
        src_pos = _positions_to_cpu(sources)
        mic_pos = _positions_to_cpu(mics)
        dim = int(room_size.numel())

        out_dir.mkdir(parents=True, exist_ok=True)

        has_traj = src_traj is not None or mic_traj is not None
        src_traj_t = None
        mic_traj_t = None
        if has_traj:
            steps = _traj_steps(src_traj, mic_traj)
            src_traj_t = _trajectory_to_cpu(src_traj, src_pos, steps)
            mic_traj_t = _trajectory_to_cpu(mic_traj, mic_pos, steps)

        if save_2d and dim >= 2:
            if has_traj and src_traj_t is not None and mic_traj_t is not None:
                ax2d = plot_scene_dynamic(
                    room=room_size[:2],
                    src_traj=src_traj_t[:, :, :2],
                    mic_traj=mic_traj_t[:, :, :2],
                    src_pos=src_pos[:, :2],
                    mic_pos=mic_pos[:, :2],
                    title="Room layout and source trajectories (top view)",
                    show=False,
                    annotate_sources=annotate_sources,
                    annotation_lines=annotation_lines,
                )
            else:
                ax2d = plot_scene_static(
                    room=room_size[:2],
                    sources=src_pos[:, :2],
                    mics=mic_pos[:, :2],
                    title="Room layout (top view)",
                    show=False,
                    annotate_sources=annotate_sources,
                    annotation_lines=annotation_lines,
                )
            path_2d = out_dir / "room_layout_2d.png"
            _save_axes(ax2d, path_2d, show=show)
            logger.info("saved: %s", path_2d)

        if save_3d and dim >= 3:
            if has_traj and src_traj_t is not None and mic_traj_t is not None:
                ax3d = plot_scene_dynamic(
                    room=room_size[:3],
                    src_traj=src_traj_t[:, :, :3],
                    mic_traj=mic_traj_t[:, :, :3],
                    src_pos=src_pos[:, :3],
                    mic_pos=mic_pos[:, :3],
                    title="Room layout and source trajectories",
                    show=False,
                    annotate_sources=annotate_sources,
                    annotation_lines=annotation_lines,
                )
            else:
                ax3d = plot_scene_static(
                    room=room_size[:3],
                    sources=src_pos[:, :3],
                    mics=mic_pos[:, :3],
                    title="Room layout",
                    show=False,
                    annotate_sources=annotate_sources,
                    annotation_lines=annotation_lines,
                )
            path_3d = out_dir / "room_layout_3d.png"
            _save_axes(ax3d, path_3d, show=show)
            logger.info("saved: %s", path_3d)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Layout image skipped: %s", exc)
