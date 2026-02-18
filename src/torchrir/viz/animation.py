"""Animation helpers for dynamic scenes."""

from __future__ import annotations

from pathlib import Path
import logging
import shutil
import subprocess
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch

from .utils import (
    _add_axes_annotation,
    _ensure_default_mpl_style,
    _positions_to_cpu,
    _to_cpu,
    _traj_steps,
    _trajectory_to_cpu,
)

LOGGER = logging.getLogger(__name__)
_MP4_WIDTH_PX = 1280
_MP4_HEIGHT_PX = 720
_MP4_DPI = 100
_MP4_FIGSIZE_INCHES = (_MP4_WIDTH_PX / _MP4_DPI, _MP4_HEIGHT_PX / _MP4_DPI)
_VIDEO_FONT_SIZE_PT = 24.0


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
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
) -> Path:
    """Render a GIF showing source/mic trajectories."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, anim, fps_out = _build_scene_animation(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=src_traj,
        mic_traj=mic_traj,
        step=step,
        fps=fps,
        signal_len=signal_len,
        fs=fs,
        duration_s=duration_s,
        plot_2d=plot_2d,
        plot_3d=plot_3d,
        annotate_sources=annotate_sources,
        annotation_lines=annotation_lines,
    )
    anim.save(out_path, writer="pillow", fps=fps_out)
    plt.close(fig)
    return out_path


def animate_scene_mp4(
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
    annotate_sources: bool = True,
    annotation_lines: Optional[Sequence[str]] = None,
    mixture_path: Path | None = None,
    mux_audio: bool = True,
    audio_channels: tuple[int, int] = (0, 1),
) -> Path:
    """Render an MP4 showing source/mic trajectories.

    When ``mux_audio`` is enabled and ``mixture_path`` is given, a stereo track
    is added with ffmpeg using the requested channel indices.
    The video canvas defaults to HD (1280x720).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, anim, fps_out = _build_scene_animation(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=src_traj,
        mic_traj=mic_traj,
        step=step,
        fps=fps,
        signal_len=signal_len,
        fs=fs,
        duration_s=duration_s,
        plot_2d=plot_2d,
        plot_3d=plot_3d,
        annotate_sources=annotate_sources,
        annotation_lines=annotation_lines,
        figsize=_MP4_FIGSIZE_INCHES,
    )
    writer = FFMpegWriter(fps=fps_out)
    anim.save(out_path, writer=writer, dpi=_MP4_DPI)
    plt.close(fig)

    if mux_audio and mixture_path is not None:
        _add_stereo_audio_to_mp4(
            video_path=out_path,
            mixture_path=Path(mixture_path),
            audio_channels=audio_channels,
        )
    return out_path


def _build_scene_animation(
    *,
    room: Sequence[float] | torch.Tensor,
    sources: object | torch.Tensor | Sequence,
    mics: object | torch.Tensor | Sequence,
    src_traj: Optional[torch.Tensor | Sequence],
    mic_traj: Optional[torch.Tensor | Sequence],
    step: int,
    fps: Optional[float],
    signal_len: Optional[int],
    fs: Optional[float],
    duration_s: Optional[float],
    plot_2d: bool,
    plot_3d: bool,
    annotate_sources: bool,
    annotation_lines: Optional[Sequence[str]] = None,
    figsize: tuple[float, float] | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    _ensure_default_mpl_style()

    room_size = _to_cpu(room)
    src_pos = _positions_to_cpu(sources)
    mic_pos = _positions_to_cpu(mics)
    dim = int(room_size.numel())
    view_dim = 3 if (plot_3d and dim == 3) else 2
    view_room = room_size[:view_dim]

    if not plot_2d and not plot_3d:
        raise ValueError("Either plot_2d or plot_3d must be True")
    if src_traj is None and mic_traj is None:
        raise ValueError("at least one trajectory is required for animation")

    steps = _traj_steps(src_traj, mic_traj)
    src_traj_t = _trajectory_to_cpu(src_traj, src_pos, steps)
    mic_traj_t = _trajectory_to_cpu(mic_traj, mic_pos, steps)
    view_src_traj = src_traj_t[:, :, :view_dim]
    view_mic_traj = mic_traj_t[:, :, :view_dim]

    if view_dim == 3:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, view_room[0].item())
        ax.set_ylim(0, view_room[1].item())
        ax.set_zlim(0, view_room[2].item())
        ax.set_xlabel("x", fontsize=_VIDEO_FONT_SIZE_PT)
        ax.set_ylabel("y", fontsize=_VIDEO_FONT_SIZE_PT)
        ax.set_zlabel("z", fontsize=_VIDEO_FONT_SIZE_PT)
        ax.tick_params(axis="both", labelsize=_VIDEO_FONT_SIZE_PT)
        ax.tick_params(axis="z", labelsize=_VIDEO_FONT_SIZE_PT)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, view_room[0].item())
        ax.set_ylim(0, view_room[1].item())
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x", fontsize=_VIDEO_FONT_SIZE_PT)
        ax.set_ylabel("y", fontsize=_VIDEO_FONT_SIZE_PT)
        ax.tick_params(axis="both", labelsize=_VIDEO_FONT_SIZE_PT)

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
    source_texts = []
    if annotate_sources:
        for idx in range(view_src_traj.shape[1]):
            text = ax.text(0.0, 0.0, f"S{idx}", fontsize=_VIDEO_FONT_SIZE_PT)
            source_texts.append(text)

    ax.legend(loc="best", fontsize=_VIDEO_FONT_SIZE_PT)
    annotation_text = _add_axes_annotation(
        ax, annotation_lines, fontsize=_VIDEO_FONT_SIZE_PT
    )

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
            if annotate_sources:
                for s_idx, text in enumerate(source_texts):
                    pos = src_pos_frame[s_idx]
                    text.set_position((float(pos[0]), float(pos[1])))
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
            if annotate_sources:
                for s_idx, text in enumerate(source_texts):
                    pos = src_pos_frame[s_idx]
                    text.set_position((float(pos[0]), float(pos[1])))
                    text.set_3d_properties(float(pos[2]))
        if duration_s is not None and steps > 1:
            t = (idx / (steps - 1)) * duration_s
            ax.set_title(f"t = {t:.2f} s", fontsize=_VIDEO_FONT_SIZE_PT)
        artists = [src_scatter, mic_scatter, *src_lines, *mic_lines, *source_texts]
        if annotation_text is not None:
            artists.append(annotation_text)
        return artists

    frames = max(1, (view_src_traj.shape[0] + step - 1) // step)
    if fps is None or fps <= 0:
        if duration_s is not None and duration_s > 0:
            fps = frames / duration_s
        else:
            fps = 6.0
    fps_out = max(1, int(round(float(fps))))

    anim = animation.FuncAnimation(
        fig,
        _frame,
        frames=frames,
        interval=1000 / float(fps_out),
        blit=False,
    )
    return fig, anim, fps_out


def _add_stereo_audio_to_mp4(
    *,
    video_path: Path,
    mixture_path: Path,
    audio_channels: tuple[int, int] = (0, 1),
) -> None:
    if not video_path.exists():
        return
    if not mixture_path.exists():
        LOGGER.warning("mixture file not found. Skip audio mux for %s", video_path.name)
        return

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        LOGGER.warning("ffmpeg not found. Skip audio mux for %s", video_path.name)
        return

    mixture, sample_rate = sf.read(mixture_path, always_2d=True)
    mixture = np.asarray(mixture, dtype=np.float64)
    n_channels = int(mixture.shape[1])
    if n_channels <= 0:
        LOGGER.warning(
            "mixture has no channels. Skip audio mux for %s", video_path.name
        )
        return

    if n_channels == 1:
        stereo = np.repeat(mixture[:, :1], repeats=2, axis=1)
    else:
        ch_l, ch_r = audio_channels
        if ch_l < 0 or ch_r < 0 or ch_l >= n_channels or ch_r >= n_channels:
            LOGGER.warning(
                "Requested channels %s unavailable for %s. Using first two channels.",
                audio_channels,
                video_path.name,
            )
            stereo = mixture[:, :2]
        else:
            stereo = mixture[:, [ch_l, ch_r]]

    tmp_audio = video_path.with_name(video_path.stem + "_tmp_audio.wav")
    tmp_video = video_path.with_name(video_path.stem + "_tmp_mux.mp4")
    sf.write(tmp_audio, stereo, int(sample_rate))

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(tmp_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(tmp_video),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.warning(
            "Failed to mux audio into %s: %s",
            video_path.name,
            (result.stderr or result.stdout).strip(),
        )
    else:
        tmp_video.replace(video_path)

    if tmp_audio.exists():
        tmp_audio.unlink()
    if tmp_video.exists():
        tmp_video.unlink()
