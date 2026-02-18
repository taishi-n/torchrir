from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

import torchrir.viz.io as viz_io


def _traj(dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if dim == 3:
        src = torch.tensor([[[1.0, 1.0, 1.5]], [[2.0, 1.5, 1.5]]], dtype=torch.float32)
        mic = torch.tensor([[[3.0, 2.0, 1.2]], [[3.0, 2.0, 1.2]]], dtype=torch.float32)
    else:
        src = torch.tensor([[[1.0, 1.0]], [[2.0, 1.5]]], dtype=torch.float32)
        mic = torch.tensor([[[3.0, 2.0]], [[3.0, 2.0]]], dtype=torch.float32)
    return src, mic


def test_save_scene_videos_3d_calls_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_traj, mic_traj = _traj(3)
    calls: list[dict[str, object]] = []

    def _fake_animate_scene_mp4(**kwargs):
        calls.append(dict(kwargs))
        return kwargs["out_path"]

    monkeypatch.setattr(viz_io, "animate_scene_mp4", _fake_animate_scene_mp4)

    viz_io.save_scene_videos(
        out_dir=tmp_path,
        room=[6.0, 4.0, 3.0],
        sources=[[1.0, 1.0, 1.5]],
        mics=[[3.0, 2.0, 1.2]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=1600,
        fs=16000,
        logger=logging.getLogger("test"),
        save_3d=True,
        mixture_path=tmp_path / "mixture.wav",
        mux_audio=True,
    )

    assert len(calls) == 2
    assert calls[0]["out_path"] == tmp_path / "room_layout_2d.mp4"
    assert calls[0]["plot_2d"] is True
    assert calls[0]["plot_3d"] is False
    assert calls[0]["annotate_sources"] is True
    assert calls[1]["out_path"] == tmp_path / "room_layout_3d.mp4"
    assert calls[1]["plot_2d"] is False
    assert calls[1]["plot_3d"] is True
    assert calls[1]["annotate_sources"] is True


def test_save_scene_videos_2d_calls_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_traj, mic_traj = _traj(2)
    calls: list[dict[str, object]] = []

    def _fake_animate_scene_mp4(**kwargs):
        calls.append(dict(kwargs))
        return kwargs["out_path"]

    monkeypatch.setattr(viz_io, "animate_scene_mp4", _fake_animate_scene_mp4)

    viz_io.save_scene_videos(
        out_dir=tmp_path,
        room=[6.0, 4.0],
        sources=[[1.0, 1.0]],
        mics=[[3.0, 2.0]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=1600,
        fs=16000,
        logger=logging.getLogger("test"),
        save_3d=True,
        mixture_path=tmp_path / "mixture.wav",
    )

    assert len(calls) == 1
    assert calls[0]["out_path"] == tmp_path / "room_layout_2d.mp4"


def test_save_scene_videos_warns_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    src_traj, mic_traj = _traj(3)

    def _raise(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(viz_io, "animate_scene_mp4", _raise)

    with caplog.at_level(logging.WARNING):
        viz_io.save_scene_videos(
            out_dir=tmp_path,
            room=[6.0, 4.0, 3.0],
            sources=[[1.0, 1.0, 1.5]],
            mics=[[3.0, 2.0, 1.2]],
            src_traj=src_traj,
            mic_traj=mic_traj,
            signal_len=1600,
            fs=16000,
            logger=logging.getLogger("test"),
            save_3d=True,
            mixture_path=tmp_path / "mixture.wav",
        )
    assert "MP4 skipped" in caplog.text


def test_save_scene_videos_forwards_annotation_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_traj, mic_traj = _traj(3)
    calls: list[dict[str, object]] = []
    annotation_lines = [
        "scene:scene_0000",
        "move:7.00-13.00 s",
        "speed:S1=0.52m/s",
    ]

    def _fake_animate_scene_mp4(**kwargs):
        calls.append(dict(kwargs))
        return kwargs["out_path"]

    monkeypatch.setattr(viz_io, "animate_scene_mp4", _fake_animate_scene_mp4)

    viz_io.save_scene_videos(
        out_dir=tmp_path,
        room=[6.0, 4.0, 3.0],
        sources=[[1.0, 1.0, 1.5]],
        mics=[[3.0, 2.0, 1.2]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=1600,
        fs=16000,
        logger=logging.getLogger("test"),
        save_3d=True,
        annotation_lines=annotation_lines,
    )

    assert len(calls) == 2
    assert calls[0]["annotation_lines"] == annotation_lines
    assert calls[1]["annotation_lines"] == annotation_lines


def test_save_scene_layout_images_uses_explicit_dim_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_paths: list[Path] = []
    calls = {"static": 0, "dynamic": 0}

    class _DummyAx:
        figure = object()

    def _fake_plot_scene_static(**kwargs):
        del kwargs
        calls["static"] += 1
        return _DummyAx()

    def _fake_plot_scene_dynamic(**kwargs):
        del kwargs
        calls["dynamic"] += 1
        return _DummyAx()

    def _fake_save_axes(ax, path, *, show):
        del ax, show
        saved_paths.append(path)

    monkeypatch.setattr(viz_io, "plot_scene_static", _fake_plot_scene_static)
    monkeypatch.setattr(viz_io, "plot_scene_dynamic", _fake_plot_scene_dynamic)
    monkeypatch.setattr(viz_io, "_save_axes", _fake_save_axes)

    viz_io.save_scene_layout_images(
        out_dir=tmp_path,
        room=[6.0, 4.0, 3.0],
        sources=[[1.0, 1.0, 1.5]],
        mics=[[3.0, 2.0, 1.2]],
        logger=logging.getLogger("test"),
        save_2d=True,
        save_3d=True,
        annotate_sources=True,
    )

    assert tmp_path / "room_layout_2d.png" in saved_paths
    assert tmp_path / "room_layout_3d.png" in saved_paths
    assert calls["static"] >= 1


def test_save_scene_layout_images_uses_dynamic_when_trajectory_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved_paths: list[Path] = []
    calls = {"static": 0, "dynamic": 0}

    class _DummyAx:
        figure = object()

    def _fake_plot_scene_static(**kwargs):
        del kwargs
        calls["static"] += 1
        return _DummyAx()

    def _fake_plot_scene_dynamic(**kwargs):
        del kwargs
        calls["dynamic"] += 1
        return _DummyAx()

    def _fake_save_axes(ax, path, *, show):
        del ax, show
        saved_paths.append(path)

    monkeypatch.setattr(viz_io, "plot_scene_static", _fake_plot_scene_static)
    monkeypatch.setattr(viz_io, "plot_scene_dynamic", _fake_plot_scene_dynamic)
    monkeypatch.setattr(viz_io, "_save_axes", _fake_save_axes)

    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.5]],
            [[1.5, 1.2, 1.5]],
        ],
        dtype=torch.float32,
    )
    mic_traj = torch.tensor(
        [
            [[3.0, 2.0, 1.2]],
            [[3.0, 2.0, 1.2]],
        ],
        dtype=torch.float32,
    )

    viz_io.save_scene_layout_images(
        out_dir=tmp_path,
        room=[6.0, 4.0, 3.0],
        sources=[[1.0, 1.0, 1.5]],
        mics=[[3.0, 2.0, 1.2]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        logger=logging.getLogger("test"),
        save_2d=True,
        save_3d=True,
        annotate_sources=True,
    )

    assert tmp_path / "room_layout_2d.png" in saved_paths
    assert tmp_path / "room_layout_3d.png" in saved_paths
    assert calls["dynamic"] == 2
    assert calls["static"] == 0


def test_save_scene_layout_images_forwards_annotation_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dynamic_kwargs: list[dict[str, object]] = []

    class _DummyAx:
        figure = object()

    def _fake_plot_scene_dynamic(**kwargs):
        dynamic_kwargs.append(dict(kwargs))
        return _DummyAx()

    monkeypatch.setattr(viz_io, "plot_scene_dynamic", _fake_plot_scene_dynamic)
    monkeypatch.setattr(viz_io, "_save_axes", lambda ax, path, *, show: None)

    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.5]],
            [[1.5, 1.2, 1.5]],
        ],
        dtype=torch.float32,
    )
    mic_traj = torch.tensor(
        [
            [[3.0, 2.0, 1.2]],
            [[3.0, 2.0, 1.2]],
        ],
        dtype=torch.float32,
    )
    annotation_lines = [
        "scene:scene_0000",
        "move:7.00-13.00 s",
        "speed:S1=0.52m/s",
    ]

    viz_io.save_scene_layout_images(
        out_dir=tmp_path,
        room=[6.0, 4.0, 3.0],
        sources=[[1.0, 1.0, 1.5]],
        mics=[[3.0, 2.0, 1.2]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        logger=logging.getLogger("test"),
        save_2d=True,
        save_3d=True,
        annotation_lines=annotation_lines,
    )

    assert len(dynamic_kwargs) == 2
    assert dynamic_kwargs[0]["annotation_lines"] == annotation_lines
    assert dynamic_kwargs[1]["annotation_lines"] == annotation_lines


def test_save_axes_uses_fixed_static_size_and_dpi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt
    from torchrir.viz import utils as viz_utils

    calls: dict[str, object] = {}

    class _DummyFig:
        def set_size_inches(self, width: float, height: float) -> None:
            calls["figsize"] = (width, height)

        def tight_layout(self) -> None:
            calls["tight_layout"] = True

        def savefig(self, path: Path, *, dpi: int) -> None:
            calls["savefig"] = (path, dpi)

    class _DummyAx:
        figure = _DummyFig()

    monkeypatch.setattr(plt, "close", lambda fig: calls.setdefault("closed", fig))
    monkeypatch.setattr(plt, "show", lambda: calls.setdefault("shown", True))

    out_path = tmp_path / "room_layout_2d.png"
    viz_utils._save_axes(_DummyAx(), out_path, show=False)

    assert calls["figsize"] == pytest.approx(viz_utils._STATIC_FIGSIZE_INCHES)
    assert calls["tight_layout"] is True
    assert calls["savefig"] == (out_path, viz_utils._STATIC_SAVE_DPI)


def test_animate_scene_mp4_uses_hd_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt
    from torchrir.viz import animation as viz_animation

    calls: dict[str, object] = {}

    class _DummyAnim:
        def save(self, path: Path, *, writer: object, dpi: int) -> None:
            calls["anim_save"] = (path, writer, dpi)

    class _DummyWriter:
        def __init__(self, *, fps: int) -> None:
            calls["writer_fps"] = fps

    def _fake_build_scene_animation(**kwargs):
        calls["figsize"] = kwargs["figsize"]
        return object(), _DummyAnim(), 12

    monkeypatch.setattr(
        viz_animation, "_build_scene_animation", _fake_build_scene_animation
    )
    monkeypatch.setattr(mpl_animation, "FFMpegWriter", _DummyWriter)
    monkeypatch.setattr(plt, "close", lambda fig: calls.setdefault("closed", fig))

    src_traj, mic_traj = _traj(2)
    out_path = tmp_path / "room_layout_2d.mp4"
    path = viz_animation.animate_scene_mp4(
        out_path=out_path,
        room=[6.0, 4.0],
        sources=[[1.0, 1.0]],
        mics=[[3.0, 2.0]],
        src_traj=src_traj,
        mic_traj=mic_traj,
        mux_audio=False,
    )

    assert path == out_path
    assert viz_animation._MP4_WIDTH_PX == 1280
    assert viz_animation._MP4_HEIGHT_PX == 720
    assert viz_animation._VIDEO_FONT_SIZE_PT == 24.0
    assert calls["figsize"] == viz_animation._MP4_FIGSIZE_INCHES
    assert calls["writer_fps"] == 12
    saved_path, writer, dpi = calls["anim_save"]
    assert saved_path == out_path
    assert writer.__class__.__name__ == "_DummyWriter"
    assert dpi == viz_animation._MP4_DPI
