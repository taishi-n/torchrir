"""Requested media failures must be visible and leave no partial output."""

import logging
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
import torch
from matplotlib.animation import Animation

from torchrir.signal import FrameSchedule
from torchrir.viz import animation, io, utils


def options():
    return dict(
        room=[4.0, 3.0],
        sources=[[1.0, 1.0]],
        mics=[[2.0, 2.0]],
        src_traj=torch.tensor([[[1.0, 1.0]], [[2.0, 1.0]]]),
        mic_traj=torch.tensor([[[2.0, 2.0]], [[2.0, 2.0]]]),
        schedule=FrameSchedule.from_samples([0, 4000]),
        fs=8000,
        stop_sample=8000,
    )


@pytest.mark.parametrize("kind", ["gifs", "videos", "plots", "layout_images"])
def test_save_helpers_propagate_requested_output_errors(tmp_path, monkeypatch, kind):
    def fail(**kwargs):
        raise RuntimeError("render failed")

    params = options()
    if kind == "gifs":
        monkeypatch.setattr(io, "animate_scene_gif", fail)
        params.update(prefix="scene", gif_fps=2)
    elif kind == "videos":
        monkeypatch.setattr(io, "animate_scene_mp4", fail)
    else:
        for key in ("schedule", "fs", "stop_sample"):
            del params[key]
        monkeypatch.setattr(io, "plot_scene_static", fail)
        monkeypatch.setattr(io, "plot_scene_dynamic", fail)
        if kind == "plots":
            params.update(prefix="scene", show=False)
    with pytest.raises(RuntimeError, match="render failed"):
        getattr(io, f"save_scene_{kind}")(
            out_dir=tmp_path, logger=logging.getLogger("test"), **params
        )


@pytest.mark.parametrize("kind", ["gif", "mp4"])
def test_encoder_failure_preserves_destination_and_closes_figure(
    tmp_path, monkeypatch, kind
):
    path = tmp_path / f"scene.{kind}"
    path.write_bytes(b"existing output")
    before = plt.get_fignums()

    def fail(self, filename, **kwargs):
        self._draw_was_started = True
        Path(filename).write_bytes(b"partial")
        raise OSError("encoder failed")

    monkeypatch.setattr(Animation, "save", fail)
    with pytest.raises(OSError, match="encoder failed"):
        getattr(animation, f"animate_scene_{kind}")(out_path=path, **options())
    assert path.read_bytes() == b"existing output"
    assert list(tmp_path.iterdir()) == [path]
    assert plt.get_fignums() == before


def test_animation_construction_failure_closes_owned_figure(tmp_path, monkeypatch):
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise RuntimeError("annotation failed")

    monkeypatch.setattr(animation, "_add_axes_annotation", fail)
    try:
        with pytest.raises(RuntimeError, match="annotation failed"):
            animation.animate_scene_gif(out_path=tmp_path / "scene.gif", **options())
        assert plt.get_fignums() == before
    finally:
        for figure in set(plt.get_fignums()) - set(before):
            plt.close(figure)


def test_plot_save_failure_closes_figure_and_preserves_file(tmp_path, monkeypatch):
    fig, ax = plt.subplots()
    path = tmp_path / "scene.png"
    path.write_bytes(b"existing")

    def fail(filename, **kwargs):
        Path(filename).write_bytes(b"partial")
        raise OSError("save failed")

    monkeypatch.setattr(fig, "savefig", fail)
    try:
        with pytest.raises(OSError, match="save failed"):
            utils._save_axes(ax, path, show=False)
        assert path.read_bytes() == b"existing"
        assert not plt.fignum_exists(fig.number)
        assert list(tmp_path.iterdir()) == [path]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "failure", ["exit", "exception", "missing_ffmpeg", "missing_audio", "collision"]
)
def test_mux_failure_and_temporary_file_isolation(tmp_path, monkeypatch, failure):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"existing")
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros((8000, 2)), 8000, subtype="FLOAT")
    sentinels = [tmp_path / "video_tmp_audio.wav", tmp_path / "video_tmp_mux.mp4"]
    for path in sentinels:
        path.write_bytes(b"unrelated")
    monkeypatch.setattr(
        animation.shutil,
        "which",
        lambda name: None if failure == "missing_ffmpeg" else "ffmpeg",
    )

    def mux(command, **kwargs):
        Path(command[-1]).write_bytes(b"encoded")
        if failure == "exception":
            raise OSError("spawn failed")
        return SimpleNamespace(
            returncode=1 if failure == "exit" else 0, stderr="encode failed", stdout=""
        )

    monkeypatch.setattr(animation.subprocess, "run", mux)
    if failure == "missing_audio":
        audio.unlink()
    before = set(tmp_path.iterdir())
    if failure == "collision":
        animation._add_stereo_audio_to_mp4(video_path=video, mixture_path=audio)
        assert video.read_bytes() == b"encoded"
    else:
        with pytest.raises((RuntimeError, OSError)):
            animation._add_stereo_audio_to_mp4(video_path=video, mixture_path=audio)
        assert video.read_bytes() == b"existing"
    assert set(tmp_path.iterdir()) == before
    assert all(path.read_bytes() == b"unrelated" for path in sentinels)


def test_requested_audio_requires_input_before_rendering(tmp_path):
    with pytest.raises(ValueError, match="mixture_path"):
        animation.animate_scene_mp4(
            out_path=tmp_path / "scene.mp4", mux_audio=True, **options()
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("caller_owned", [False, True])
def test_plot_construction_failure_respects_figure_ownership(
    monkeypatch, dynamic, caller_owned
):
    from torchrir.viz import scene

    fig, supplied = plt.subplots() if caller_owned else (None, None)
    before = plt.get_fignums()

    def fail(*args, **kwargs):
        raise RuntimeError("annotation failed")

    monkeypatch.setattr(scene, "_add_axes_annotation", fail)
    try:
        with pytest.raises(RuntimeError, match="annotation failed"):
            if dynamic:
                scene.plot_scene_dynamic(
                    room=[4.0, 3.0],
                    ax=supplied,
                    src_traj=torch.ones(2, 1, 2),
                    mic_traj=torch.ones(2, 1, 2),
                )
            else:
                scene.plot_scene_static(
                    room=[4.0, 3.0],
                    ax=supplied,
                    sources=[[1.0, 1.0]],
                    mics=[[2.0, 2.0]],
                )
        assert plt.get_fignums() == before
    finally:
        for number in set(plt.get_fignums()) - set(before):
            plt.close(number)
        if fig is not None:
            plt.close(fig)
