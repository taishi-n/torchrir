from __future__ import annotations

"""Animation helpers for dynamic scenes."""

from pathlib import Path
from typing import Optional, Sequence

import torch

from .plotting_utils import _positions_to_cpu, _to_cpu, _traj_steps, _trajectory_to_cpu


def animate_scene_gif(
    *,
    out_path: Path,
    room: Sequence[float] | torch.Tensor,
    sources: object | torch.Tensor | Sequence,
    mics: object | torch.Tensor | Sequence,
    src_traj: Optional[torch.Tensor | Sequence] = None,
    mic_traj: Optional[torch.Tensor | Sequence] = None,
    step: int = 1,
    fps: Optional[float] = None,
    signal_len: Optional[int] = None,
    fs: Optional[float] = None,
    duration_s: Optional[float] = None,
    plot_2d: bool = True,
    plot_3d: bool = False,
) -> Path:
    """Render a GIF showing source/mic trajectories.

    Args:
        out_path: Destination GIF path.
        room: Room size tensor or sequence.
        sources: Source positions or Source-like object.
        mics: Microphone positions or MicrophoneArray-like object.
        src_traj: Optional source trajectory (T, n_src, dim).
        mic_traj: Optional mic trajectory (T, n_mic, dim).
        step: Subsampling step for trajectories.
        fps: Frames per second for the GIF (auto if None).
        signal_len: Optional signal length (samples) to infer elapsed time.
        fs: Sample rate used with signal_len.
        duration_s: Optional total duration in seconds (overrides signal_len/fs).
        plot_2d: Use 2D projection if True.
        plot_3d: Use 3D projection if True and dim == 3.

    Returns:
        The output path.

    Example:
        >>> animate_scene_gif(
        ...     out_path=Path("outputs/scene.gif"),
        ...     room=[6.0, 4.0, 3.0],
        ...     sources=[[1.0, 2.0, 1.5]],
        ...     mics=[[2.0, 2.0, 1.5]],
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ...     signal_len=16000,
        ...     fs=16000,
        ... )
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    room_size = _to_cpu(room)
    src_pos = _positions_to_cpu(sources)
    mic_pos = _positions_to_cpu(mics)
    dim = int(room_size.numel())
    view_dim = 3 if (plot_3d and dim == 3) else 2
    view_room = room_size[:view_dim]
    view_src = src_pos[:, :view_dim]
    view_mic = mic_pos[:, :view_dim]

    if src_traj is None and mic_traj is None:
        raise ValueError("at least one trajectory is required for animation")
    steps = _traj_steps(src_traj, mic_traj)
    src_traj = _trajectory_to_cpu(src_traj, src_pos, steps)
    mic_traj = _trajectory_to_cpu(mic_traj, mic_pos, steps)
    view_src_traj = src_traj[:, :, :view_dim]
    view_mic_traj = mic_traj[:, :, :view_dim]

    if view_dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, view_room[0].item())
        ax.set_ylim(0, view_room[1].item())
        ax.set_zlim(0, view_room[2].item())
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        fig, ax = plt.subplots()
        ax.set_xlim(0, view_room[0].item())
        ax.set_ylim(0, view_room[1].item())
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    src_scatter = ax.scatter([], [], marker="^", color="tab:green", label="sources")
    mic_scatter = ax.scatter([], [], marker="o", color="tab:orange", label="mics")
    src_lines = []
    mic_lines = []
    for _ in range(view_src_traj.shape[1]):
        if view_dim == 2:
            (line,) = ax.plot([], [], color="tab:green", alpha=0.6)
        else:
            (line,) = ax.plot([], [], [], color="tab:green", alpha=0.6)
        src_lines.append(line)
    for _ in range(view_mic_traj.shape[1]):
        if view_dim == 2:
            (line,) = ax.plot([], [], color="tab:orange", alpha=0.6)
        else:
            (line,) = ax.plot([], [], [], color="tab:orange", alpha=0.6)
        mic_lines.append(line)

    ax.legend(loc="best")

    if duration_s is None and signal_len is not None and fs is not None:
        duration_s = float(signal_len) / float(fs)

    def _frame(i: int):
        idx = min(i * step, view_src_traj.shape[0] - 1)
        src_frame = view_src_traj[: idx + 1]
        mic_frame = view_mic_traj[: idx + 1]
        src_pos_frame = view_src_traj[idx]
        mic_pos_frame = view_mic_traj[idx]

        if view_dim == 2:
            src_scatter.set_offsets(src_pos_frame)
            mic_scatter.set_offsets(mic_pos_frame)
            for s_idx, line in enumerate(src_lines):
                xy = src_frame[:, s_idx, :]
                line.set_data(xy[:, 0], xy[:, 1])
            for m_idx, line in enumerate(mic_lines):
                xy = mic_frame[:, m_idx, :]
                line.set_data(xy[:, 0], xy[:, 1])
        else:
            setattr(
                src_scatter,
                "_offsets3d",
                (src_pos_frame[:, 0], src_pos_frame[:, 1], src_pos_frame[:, 2]),
            )
            setattr(
                mic_scatter,
                "_offsets3d",
                (mic_pos_frame[:, 0], mic_pos_frame[:, 1], mic_pos_frame[:, 2]),
            )
            for s_idx, line in enumerate(src_lines):
                xyz = src_frame[:, s_idx, :]
                line.set_data(xyz[:, 0], xyz[:, 1])
                line.set_3d_properties(xyz[:, 2])
            for m_idx, line in enumerate(mic_lines):
                xyz = mic_frame[:, m_idx, :]
                line.set_data(xyz[:, 0], xyz[:, 1])
                line.set_3d_properties(xyz[:, 2])
        if duration_s is not None and steps > 1:
            t = (idx / (steps - 1)) * duration_s
            ax.set_title(f"t = {t:.2f} s")
        return [src_scatter, mic_scatter, *src_lines, *mic_lines]

    frames = max(1, (view_src_traj.shape[0] + step - 1) // step)
    if fps is None or fps <= 0:
        if duration_s is not None and duration_s > 0:
            fps = frames / duration_s
        else:
            fps = 6.0
    anim = animation.FuncAnimation(
        fig, _frame, frames=frames, interval=1000 / fps, blit=False
    )
    fps_int = None if fps is None else max(1, int(round(fps)))
    anim.save(out_path, writer="pillow", fps=fps_int)
    plt.close(fig)
    return out_path
