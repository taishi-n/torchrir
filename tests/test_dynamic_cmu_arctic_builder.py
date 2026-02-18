from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import torchrir.datasets.dynamic_cmu_arctic as dynamic_builder
from torchrir.datasets import build_dynamic_cmu_arctic_dataset


def _write_fake_cmu_speaker(root: Path, speaker: str, sample_rate: int = 16000) -> None:
    speaker_root = root / "ARCTIC" / f"cmu_us_{speaker}_arctic"
    wav_dir = speaker_root / "wav"
    etc_dir = speaker_root / "etc"
    wav_dir.mkdir(parents=True, exist_ok=True)
    etc_dir.mkdir(parents=True, exist_ok=True)

    utterances = ["arctic_a0001", "arctic_a0002"]
    text_lines = []
    t = np.arange(2000, dtype=np.float64) / float(sample_rate)
    for idx, utt in enumerate(utterances):
        freq = 220.0 + 40.0 * idx
        wav = 0.1 * np.sin(2.0 * np.pi * freq * t)
        sf.write(wav_dir / f"{utt}.wav", wav, sample_rate)
        text_lines.append(f'( {utt} "dummy text {idx}" )')
    (etc_dir / "txt.done.data").write_text(
        "\n".join(text_lines) + "\n", encoding="utf-8"
    )


@pytest.fixture()
def built_dataset(tmp_path: Path) -> Path:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)

    dataset_root = tmp_path / "out_ds"
    build_dynamic_cmu_arctic_dataset(
        cmu_root=cmu_root,
        dataset_root=dataset_root,
        speakers=["bdl", "slt", "clb"],
        n_scenes=1,
        duration_sec=0.1,
        trajectory_steps=8,
        rir_samples=64,
        max_order=1,
        overwrite=False,
        save_layout_mp4=False,
        save_layout_images=False,
    )
    return dataset_root


def test_dynamic_cmu_arctic_builder_smoke(built_dataset: Path) -> None:
    scene_dir = built_dataset / "scene_0000"
    assert scene_dir.exists()
    assert scene_dir.is_dir()


def test_dynamic_cmu_arctic_builder_expected_files(built_dataset: Path) -> None:
    scene_dir = built_dataset / "scene_0000"
    expected = [
        scene_dir / "mixture.wav",
        scene_dir / "source_00.wav",
        scene_dir / "source_01.wav",
        scene_dir / "source_02.wav",
        scene_dir / "metadata.json",
        scene_dir / "source_info.json",
    ]
    for path in expected:
        assert path.exists(), f"missing file: {path}"


def test_dynamic_cmu_arctic_builder_metadata_source_info_keys(
    built_dataset: Path,
) -> None:
    scene_dir = built_dataset / "scene_0000"
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    source_info = json.loads(
        (scene_dir / "source_info.json").read_text(encoding="utf-8")
    )

    assert "source_info" in metadata
    assert "extra" in metadata
    assert "dynamic" in metadata
    assert metadata["extra"]["motion_profile"]["pre_static_ratio"] == pytest.approx(
        0.35
    )
    assert metadata["extra"]["motion_profile"]["move_ratio"] == pytest.approx(0.30)
    assert metadata["extra"]["motion_profile"]["post_static_ratio"] == pytest.approx(
        0.35
    )

    assert isinstance(source_info, list)
    assert len(source_info) == 3
    required = {
        "speaker",
        "utterance_ids",
        "source_index",
        "is_moving",
        "velocity_mps",
        "motion_type",
        "angular_velocity_rad_s",
        "turn_direction",
        "move_start_sec",
        "move_end_sec",
    }
    for item in source_info:
        assert required.issubset(item.keys())


def test_dynamic_cmu_arctic_builder_calls_video_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)

    calls: list[dict[str, object]] = []

    def _fake_save_scene_videos(**kwargs) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(dynamic_builder, "save_scene_videos", _fake_save_scene_videos)

    dataset_root = tmp_path / "out_ds"
    build_dynamic_cmu_arctic_dataset(
        cmu_root=cmu_root,
        dataset_root=dataset_root,
        speakers=["bdl", "slt", "clb"],
        n_scenes=1,
        duration_sec=0.1,
        trajectory_steps=8,
        rir_samples=64,
        max_order=1,
        overwrite=False,
        save_layout_mp4=True,
        save_layout_mp4_3d=False,
        layout_video_fps=12.0,
        layout_video_mux_audio=False,
        save_layout_images=False,
    )

    assert len(calls) == 1
    call = calls[0]
    assert (dataset_root / "scene_0000") == call["out_dir"]
    assert (dataset_root / "scene_0000" / "mixture.wav") == call["mixture_path"]
    assert call["save_3d"] is False
    assert call["mp4_fps"] == pytest.approx(12.0)
    assert call["mux_audio"] is False
    annotation_lines = call["annotation_lines"]
    assert isinstance(annotation_lines, list)
    assert len(annotation_lines) == 3
    assert annotation_lines[0] == "scene:scene_0000"
    assert annotation_lines[1].startswith("move:")
    assert annotation_lines[2].startswith("speed:")


def test_dynamic_cmu_arctic_builder_defaults_enable_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)

    video_calls: list[dict[str, object]] = []
    image_calls: list[dict[str, object]] = []

    def _fake_save_scene_videos(**kwargs) -> None:
        video_calls.append(dict(kwargs))

    def _fake_save_scene_layout_images(**kwargs) -> None:
        image_calls.append(dict(kwargs))

    monkeypatch.setattr(dynamic_builder, "save_scene_videos", _fake_save_scene_videos)
    monkeypatch.setattr(
        dynamic_builder, "save_scene_layout_images", _fake_save_scene_layout_images
    )

    dataset_root = tmp_path / "out_ds"
    build_dynamic_cmu_arctic_dataset(
        cmu_root=cmu_root,
        dataset_root=dataset_root,
        speakers=["bdl", "slt", "clb"],
        n_scenes=1,
        duration_sec=0.1,
        trajectory_steps=8,
        rir_samples=64,
        max_order=1,
        overwrite=False,
    )

    assert len(image_calls) == 1
    assert image_calls[0]["annotate_sources"] is True
    assert image_calls[0]["src_traj"] is not None
    assert image_calls[0]["mic_traj"] is not None
    assert image_calls[0]["save_3d"] is True
    image_annotation_lines = image_calls[0]["annotation_lines"]
    assert isinstance(image_annotation_lines, list)
    assert len(image_annotation_lines) == 3
    assert image_annotation_lines[0] == "scene:scene_0000"
    assert image_annotation_lines[1].startswith("move:")
    assert image_annotation_lines[2].startswith("speed:")
    assert len(video_calls) == 1
    assert video_calls[0]["annotate_sources"] is True
    video_annotation_lines = video_calls[0]["annotation_lines"]
    assert isinstance(video_annotation_lines, list)
    assert len(video_annotation_lines) == 3
    assert video_annotation_lines[0] == "scene:scene_0000"
    assert video_annotation_lines[1].startswith("move:")
    assert video_annotation_lines[2].startswith("speed:")


def test_build_layout_annotation_lines_format() -> None:
    lines = dynamic_builder._build_layout_annotation_lines(
        scene_id="scene_0123",
        move_start_sec=7.0,
        move_end_sec=13.0,
        source_velocity_mps=np.array([0.0, 0.52, 0.8], dtype=np.float64),
    )
    assert lines == [
        "scene:scene_0123",
        "move:7.00-13.00 s",
        "speed:S1=0.52m/s, S2=0.80m/s",
    ]
