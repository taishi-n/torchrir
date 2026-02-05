from __future__ import annotations

"""Higher-level plotting utilities used by examples."""

from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from .plotting import plot_scene_dynamic, plot_scene_static


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
    """Plot static and dynamic scenes and save images to disk.

    Dynamic plots show trajectories for moving entities and points for fixed ones.

    Args:
        out_dir: Output directory for PNGs.
        room: Room size tensor or sequence.
        sources: Source positions or Source-like object.
        mics: Microphone positions or MicrophoneArray-like object.
        src_traj: Optional source trajectory (T, n_src, dim).
        mic_traj: Optional mic trajectory (T, n_mic, dim).
        prefix: Filename prefix for saved images.
        step: Subsampling step for trajectories.
        show: Whether to show figures interactively.
        plot_2d: Save 2D projections.
        plot_3d: Save 3D projections (only if dim == 3).

    Returns:
        Tuple of (static_paths, dynamic_paths).

    Example:
        >>> plot_scene_and_save(
        ...     out_dir=Path("outputs"),
        ...     room=[6.0, 4.0, 3.0],
        ...     sources=[[1.0, 2.0, 1.5]],
        ...     mics=[[2.0, 2.0, 1.5]],
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ...     prefix="scene",
        ... )
    """
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


def _to_cpu(value: Any) -> torch.Tensor:
    """Move a value to CPU as a tensor."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _positions_to_cpu(entity: torch.Tensor | object) -> torch.Tensor:
    """Extract positions from an entity and move to CPU."""
    pos = getattr(entity, "positions", entity)
    pos = _to_cpu(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _traj_steps(
    src_traj: Optional[torch.Tensor | Sequence],
    mic_traj: Optional[torch.Tensor | Sequence],
) -> int:
    """Infer the number of trajectory steps."""
    if src_traj is not None:
        return int(_to_cpu(src_traj).shape[0])
    return int(_to_cpu(mic_traj).shape[0])


def _trajectory_to_cpu(
    traj: Optional[torch.Tensor | Sequence], fallback_pos: torch.Tensor, steps: int
) -> torch.Tensor:
    """Normalize trajectory to CPU tensor with shape (T, N, dim)."""
    if traj is None:
        return fallback_pos.unsqueeze(0).repeat(steps, 1, 1)
    traj = _to_cpu(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _save_axes(ax: Any, path: Path, *, show: bool) -> None:
    """Save a matplotlib axis to disk."""
    import matplotlib.pyplot as plt

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
