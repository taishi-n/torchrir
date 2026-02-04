from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .plotting import plot_scene_dynamic, plot_scene_static


def plot_scene_and_save(
    *,
    out_dir: Path,
    room,
    sources,
    mics,
    src_traj=None,
    mic_traj=None,
    prefix: str = "scene",
    step: int = 1,
    show: bool = False,
    plot_2d: bool = True,
    plot_3d: bool = True,
) -> tuple[list[Path], list[Path]]:
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
                step=step,
                title=f"Room scene ({view_dim}D trajectories)",
                show=False,
            )
            _overlay_positions(ax, view_src, view_mic)
            dynamic_path = out_dir / f"{prefix}_dynamic_{view_dim}d.png"
            _save_axes(ax, dynamic_path, show=show)
            dynamic_paths.append(dynamic_path)

    return static_paths, dynamic_paths


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _positions_to_cpu(entity: torch.Tensor | object) -> torch.Tensor:
    pos = getattr(entity, "positions", entity)
    pos = _to_cpu(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _traj_steps(src_traj, mic_traj) -> int:
    if src_traj is not None:
        return int(_to_cpu(src_traj).shape[0])
    return int(_to_cpu(mic_traj).shape[0])


def _trajectory_to_cpu(traj, fallback_pos: torch.Tensor, steps: int) -> torch.Tensor:
    if traj is None:
        return fallback_pos.unsqueeze(0).repeat(steps, 1, 1)
    traj = _to_cpu(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _save_axes(ax, path: Path, *, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def _overlay_positions(ax, sources: torch.Tensor, mics: torch.Tensor) -> None:
    if sources.numel() > 0:
        if sources.shape[1] == 2:
            ax.scatter(sources[:, 0], sources[:, 1], marker="^", label="sources", color="tab:green")
        else:
            ax.scatter(
                sources[:, 0],
                sources[:, 1],
                sources[:, 2],
                marker="^",
                label="sources",
                color="tab:green",
            )
    if mics.numel() > 0:
        if mics.shape[1] == 2:
            ax.scatter(mics[:, 0], mics[:, 1], marker="o", label="mics", color="tab:orange")
        else:
            ax.scatter(
                mics[:, 0],
                mics[:, 1],
                mics[:, 2],
                marker="o",
                label="mics",
                color="tab:orange",
            )
    ax.legend(loc="best")
