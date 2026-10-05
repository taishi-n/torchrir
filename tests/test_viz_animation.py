"""Integration checks for rendered scene animations."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from PIL.GifImagePlugin import GifImageFile
import pytest
import torch

from torchrir.viz.animation import animate_scene_gif, animate_scene_mp4


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
            duration_s=1.0,
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
