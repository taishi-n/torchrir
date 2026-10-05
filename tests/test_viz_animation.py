"""Integration checks for rendered scene animations."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from PIL.GifImagePlugin import GifImageFile
import pytest
import torch

from torchrir.viz.animation import animate_scene_gif, animate_scene_mp4
from torchrir.signal import FrameSchedule


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("annotate", [False, True])
@pytest.mark.parametrize("format_name", ["gif", "mp4"])
def test_render_scene_animation(
    tmp_path: Path, dimension: int, annotate: bool, format_name: str
) -> None:
    if format_name == "mp4" and not all(
        shutil.which(command) for command in ("ffmpeg", "ffprobe")
    ):
        pytest.skip("MP4 rendering requires ffmpeg and ffprobe")
    source = torch.tensor([[[1.0, 1.0, 0.5]], [[2.0, 1.5, 2.0]]])[..., :dimension]
    microphones = torch.tensor([[3.0, 2.0, 1.0]])[:, :dimension]
    path = tmp_path / f"scene.{format_name}"
    render = animate_scene_gif if format_name == "gif" else animate_scene_mp4
    initial_figures = plt.get_fignums()
    try:
        render(
            out_path=path,
            room=[4.0, 3.0, 2.5][:dimension],
            sources=source[0],
            mics=microphones,
            src_traj=source,
            fps=2,
            schedule=FrameSchedule.from_samples([0, 4]),
            fs=8,
            stop_sample=8,
            plot_2d=dimension == 2,
            plot_3d=dimension == 3,
            annotate_sources=annotate,
        )
        assert plt.get_fignums() == initial_figures
        if format_name == "gif":
            with Image.open(path) as gif:
                assert isinstance(gif, GifImageFile)
                assert gif.n_frames == 2
                assert min(gif.size) > 0
                for frame in range(gif.n_frames):
                    gif.seek(frame)
                    gif.load()
        else:
            probe = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames",
                    "-of",
                    "json",
                    str(path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            assert (stream["width"], stream["height"]) == (1280, 720)
            assert int(stream["nb_frames"]) == 2
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
                check=True,
                capture_output=True,
            )
    finally:
        for figure in set(plt.get_fignums()) - set(initial_figures):
            plt.close(figure)


@pytest.mark.parametrize(
    "starts,step,expected",
    [
        ([0, 2, 4, 6], 1, [0, 1, 2, 3]),
        ([0, 1, 6, 7], 1, [0, 1, 1, 2]),
        ([0, 1, 6, 7], 2, [0, 0, 0, 2]),
        ([0, 1, 2, 3], 1, [0, 2, 3, 3]),
    ],
)
def test_animation_uses_sample_schedule_and_output_clock(starts, step, expected):
    from torchrir.viz.animation import _build_scene_animation

    trajectory = torch.tensor([[[float(i), 1.0]] for i in range(4)])
    fig, animation, fps = _build_scene_animation(
        room=[5.0, 4.0],
        sources=trajectory[0],
        mics=[[3.0, 2.0]],
        src_traj=trajectory,
        mic_traj=None,
        step=step,
        fps=4.0,
        schedule=FrameSchedule.from_samples(starts),
        fs=8,
        stop_sample=8,
        plot_2d=True,
        plot_3d=False,
        annotate_sources=True,
    )
    try:
        assert fps == 4.0
        for index, source_index in enumerate(expected):
            animation._draw_next_frame(index, blit=False)
            assert fig.axes[0].get_title() == f"t = {index / 4:.2f} s"
            assert fig.axes[0].collections[0].get_offsets().tolist() == [
                [source_index, 1.0]
            ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("format_name", ["gif", "mp4"])
@pytest.mark.parametrize("fps,step,frames", [(None, 1, 4), (None, 2, 2), (1.5, 1, 5)])
def test_animation_preserves_duration(tmp_path, format_name, fps, step, frames):
    if format_name == "mp4" and not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    trajectory = torch.tensor([[[float(i), 1.0]] for i in range(4)])
    path = tmp_path / f"timed.{format_name}"
    render = animate_scene_gif if format_name == "gif" else animate_scene_mp4
    render(
        out_path=path,
        room=[5.0, 4.0],
        sources=trajectory[0],
        mics=[[3.0, 2.0]],
        src_traj=trajectory,
        schedule=FrameSchedule.from_samples([0, 3, 6, 9]),
        fs=4,
        stop_sample=12,
        step=step,
        fps=fps,
    )
    if format_name == "gif":
        with Image.open(path) as gif:
            assert isinstance(gif, GifImageFile)
            assert gif.n_frames == frames
            duration_ms = 0
            for index in range(gif.n_frames):
                gif.seek(index)
                duration_ms += gif.info["duration"]
            assert duration_ms == 3000
    else:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        video = json.loads(probe.stdout)["streams"][0]
        assert int(video["nb_frames"]) == frames
        assert float(video["duration"]) == pytest.approx(3.0, abs=0.001)


def test_gif_rounds_cumulative_boundaries(tmp_path):
    path = tmp_path / "fractional.gif"
    trajectory = torch.tensor([[[1.0, 1.0]], [[2.0, 1.0]], [[3.0, 1.0]]])
    animate_scene_gif(
        out_path=path,
        room=[4.0, 3.0],
        sources=trajectory[0],
        mics=[[2.0, 2.0]],
        src_traj=trajectory,
        schedule=FrameSchedule.from_samples([0, 33, 66]),
        fs=100,
        stop_sample=100,
    )
    with Image.open(path) as gif:
        durations = []
        for i in range(3):
            gif.seek(i)
            durations.append(gif.info["duration"])
    assert durations == [330, 340, 330]


def test_mp4_audio_and_video_cover_full_tail(tmp_path):
    import numpy as np
    import soundfile as sf

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg unavailable")
    audio = tmp_path / "mixture.wav"
    sf.write(audio, np.zeros((24000, 2)), 8000, subtype="FLOAT")
    path = tmp_path / "tail.mp4"
    trajectory = torch.tensor([[[1.0, 1.0]], [[2.0, 1.0]]])
    animate_scene_mp4(
        out_path=path,
        room=[4.0, 3.0],
        sources=trajectory[0],
        mics=[[2.0, 2.0]],
        src_traj=trajectory,
        schedule=FrameSchedule.from_samples([0, 8000]),
        fs=8000,
        stop_sample=24000,
        mixture_path=audio,
    )
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = {
        stream["codec_type"]: stream for stream in json.loads(probe.stdout)["streams"]
    }
    assert set(streams) == {"audio", "video"}
    for stream in streams.values():
        assert float(stream["duration"]) == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize(
    "updates,error",
    [
        ({"schedule": [0, 1]}, TypeError),
        ({"schedule": FrameSchedule.from_samples([0])}, ValueError),
        ({"stop_sample": 4}, ValueError),
        ({"stop_sample": True}, TypeError),
        ({"fs": 0}, ValueError),
        ({"step": 0}, ValueError),
        ({"step": 1.5}, TypeError),
        ({"fps": float("nan")}, ValueError),
        ({"fps": 0}, ValueError),
        ({"mic_traj": torch.ones(1, 1, 2)}, ValueError),
        (
            {"schedule": FrameSchedule.from_seconds([0.0, 0.5], sample_rate=16)},
            ValueError,
        ),
        ({"fps": 101}, ValueError),
    ],
)
def test_animation_rejects_invalid_timeline_before_creating_figure(
    tmp_path, updates, error
):
    options: dict[str, Any] = dict(
        out_path=tmp_path / "invalid.gif",
        room=[4.0, 3.0],
        sources=[[1.0, 1.0]],
        mics=[[2.0, 2.0]],
        src_traj=torch.tensor([[[1.0, 1.0]], [[2.0, 1.0]]]),
        schedule=FrameSchedule.from_samples([0, 4]),
        fs=8,
        stop_sample=8,
    )
    options.update(updates)
    figures = plt.get_fignums()
    with pytest.raises(error):
        animate_scene_gif(**options)
    assert plt.get_fignums() == figures
    assert not options["out_path"].exists()


def test_mp4_rejects_mismatched_audio_duration(tmp_path):
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "short.wav"
    sf.write(audio, np.zeros((8000, 2)), 8000, subtype="FLOAT")
    with pytest.raises(ValueError, match="audio duration"):
        animate_scene_mp4(
            out_path=tmp_path / "mismatch.mp4",
            room=[4.0, 3.0],
            sources=[[1.0, 1.0]],
            mics=[[2.0, 2.0]],
            src_traj=torch.ones(2, 1, 2),
            schedule=FrameSchedule.from_samples([0, 8000]),
            fs=8000,
            stop_sample=16000,
            mixture_path=audio,
        )
    assert not (tmp_path / "mismatch.mp4").exists()
