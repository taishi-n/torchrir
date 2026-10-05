from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
from threading import Event, Lock
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
import torch

import torchrir.datasets._archive as archive_utils
import torchrir.datasets.dynamic_cmu_arctic as dynamic_builder
from torchrir.config import SimulationConfig
from torchrir.datasets._archive import _TRANSACTION_MARKER
from torchrir.datasets import (
    DynamicCmuArcticBuildConfig,
    DynamicDatasetBuildResult,
    build_dynamic_cmu_arctic,
)
from torchrir.signal import FrameSchedule


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


def _small_build_config(
    *,
    cmu_root: Path,
    dataset_root: Path,
    **updates: object,
) -> DynamicCmuArcticBuildConfig:
    config = DynamicCmuArcticBuildConfig(
        cmu_root=cmu_root,
        dataset_root=dataset_root,
        speakers=("bdl", "slt", "clb"),
        n_scenes=1,
        duration_sec=0.1,
        trajectory_steps=8,
        simulation=SimulationConfig(max_order=1, nsample=64),
        save_layout_mp4=False,
        save_layout_images=False,
    )
    return replace(config, **updates)


@pytest.fixture()
def built_dataset(tmp_path: Path) -> Path:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)

    dataset_root = tmp_path / "out_ds"
    result = build_dynamic_cmu_arctic(
        DynamicCmuArcticBuildConfig(
            cmu_root=cmu_root,
            dataset_root=dataset_root,
            speakers=("bdl", "slt", "clb"),
            n_scenes=1,
            duration_sec=0.1,
            trajectory_steps=8,
            simulation=SimulationConfig(max_order=1, nsample=64),
            overwrite=False,
            save_layout_mp4=False,
            save_layout_images=False,
        )
    )
    assert result == DynamicDatasetBuildResult(
        dataset_root=dataset_root,
        sample_rate=16000,
        n_mics=6,
        n_scenes=1,
        scene_dirs=(dataset_root / "scene_0000",),
    )
    return dataset_root


def test_dynamic_builder_rejects_invalid_logger_before_side_effects(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "output" / "dataset"
    config = _small_build_config(
        cmu_root=tmp_path / "cmu",
        dataset_root=dataset_root,
    )

    with pytest.raises(TypeError, match="callable info"):
        build_dynamic_cmu_arctic(config, logger=cast(Any, object()))

    assert not dataset_root.parent.exists()


def test_dynamic_builder_module_help_has_no_eager_import_warning() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchrir.datasets.dynamic_cmu_arctic",
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage:" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr


def test_dynamic_build_config_normalizes_paths_and_copies_sequences() -> None:
    speakers = ["bdl", "slt", "clb"]
    room_size = [8, 6, 3]
    mic_center = [4, 3, 1.5]
    source_margin = [0.5, 0.5, 0.3]
    config = DynamicCmuArcticBuildConfig(
        cmu_root="datasets/cmu_arctic",
        dataset_root="outputs/dynamic",
        speakers=speakers,
        room_size=room_size,
        mic_center=mic_center,
        source_margin=source_margin,
    )

    assert isinstance(config.cmu_root, Path)
    assert isinstance(config.dataset_root, Path)
    assert config.speakers == ("bdl", "slt", "clb")
    assert config.room_size == (8.0, 6.0, 3.0)
    assert config.mic_center == (4.0, 3.0, 1.5)
    assert config.source_margin == (0.5, 0.5, 0.3)
    for value in (
        config.speakers,
        config.room_size,
        config.mic_center,
        config.source_margin,
    ):
        assert isinstance(value, tuple)

    speakers.append("rms")
    room_size[0] = 100
    mic_center[0] = 100
    source_margin[0] = 100
    assert config.speakers == ("bdl", "slt", "clb")
    assert config.room_size == (8.0, 6.0, 3.0)
    assert config.mic_center == (4.0, 3.0, 1.5)
    assert config.source_margin == (0.5, 0.5, 0.3)


def test_dynamic_build_records_normalize_numpy_integers() -> None:
    config = DynamicCmuArcticBuildConfig(
        cmu_root="datasets/cmu_arctic",
        dataset_root="outputs/dynamic",
        n_scenes=cast(Any, np.int64(2)),
        n_sources=cast(Any, np.int64(3)),
        n_moving_sources=cast(Any, np.int64(1)),
        trajectory_steps=cast(Any, np.int64(8)),
        seed=cast(Any, np.int64(4)),
    )
    result = DynamicDatasetBuildResult(
        dataset_root=Path("output"),
        sample_rate=cast(Any, np.int64(16_000)),
        n_mics=cast(Any, np.int64(2)),
        n_scenes=cast(Any, np.int64(1)),
        scene_dirs=(Path("output/scene_0000"),),
    )

    for value in (
        config.n_scenes,
        config.n_sources,
        config.n_moving_sources,
        config.trajectory_steps,
        config.seed,
        result.sample_rate,
        result.n_mics,
        result.n_scenes,
    ):
        assert type(value) is int


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"speakers": None}, TypeError, "speakers must be a sequence"),
        ({"speakers": ()}, ValueError, "speakers must not be empty"),
        ({"speakers": ("bdl", 1, "clb")}, TypeError, "contain strings"),
        (
            {"speakers": ("bdl", "bdl", "clb")},
            ValueError,
            "duplicates",
        ),
        ({"n_scenes": 0}, ValueError, "n_scenes"),
        ({"n_scenes": 10**400}, ValueError, "n_scenes must be at most"),
        ({"n_sources": 4}, ValueError, "number of speakers"),
        ({"n_moving_sources": 4}, ValueError, "n_moving_sources"),
        ({"trajectory_steps": 1}, ValueError, "at least 2"),
        ({"duration_sec": 0}, ValueError, "duration_sec"),
        ({"rt60": float("nan")}, ValueError, "rt60 must be finite"),
        ({"rt60": 0.001}, ValueError, "requested t60 is too short"),
        (
            {"simulation": SimulationConfig(nb_img=(1, 1), nsample=64)},
            ValueError,
            "nb_img must contain three",
        ),
        ({"room_size": (8, -1, 3)}, ValueError, "room_size"),
        ({"source_margin": (4, 0.5, 0.3)}, ValueError, "source_margin"),
        ({"move_start_ratio": 0.8}, ValueError, "motion ratios"),
        ({"moving_speed_max": 0.1}, ValueError, "moving speeds"),
        (
            {
                "randomize_mic_center": False,
                "mic_center": (0.1, 3.0, 1.5),
            },
            ValueError,
            "microphone array",
        ),
        (
            {
                "randomize_mic_center": False,
                "mic_center": (1.0, 1.0, 1.5),
            },
            ValueError,
            "source radius",
        ),
        (
            {"room_size": (4.0, 3.0, 3.0)},
            ValueError,
            "randomized microphone-center",
        ),
        ({"layout_video_fps": 0}, ValueError, "layout_video_fps"),
        ({"overwrite": 1}, TypeError, "overwrite must be a bool"),
    ],
)
def test_invalid_dynamic_build_config_fails_before_build_starts(
    tmp_path: Path,
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    dataset_root = tmp_path / "must-not-be-created"
    valid = DynamicCmuArcticBuildConfig(
        cmu_root=tmp_path / "cmu",
        dataset_root=dataset_root,
        speakers=("bdl", "slt", "clb"),
    )

    with pytest.raises(error, match=message):
        replace(valid, **updates)
    assert not dataset_root.exists()


@pytest.mark.parametrize(
    ("cmu_relative", "dataset_relative"),
    [
        ("shared", "shared"),
        ("cmu", "cmu/generated"),
        ("output/cmu", "output"),
    ],
    ids=("same-root", "output-inside-corpus", "corpus-inside-output"),
)
def test_dynamic_build_config_rejects_overlapping_roots(
    tmp_path: Path,
    cmu_relative: str,
    dataset_relative: str,
) -> None:
    with pytest.raises(ValueError, match="cmu_root and dataset_root must not overlap"):
        DynamicCmuArcticBuildConfig(
            cmu_root=tmp_path / cmu_relative,
            dataset_root=tmp_path / dataset_relative,
        )


def test_dynamic_build_config_accepts_more_than_six_sources() -> None:
    speakers = tuple(f"speaker_{index}" for index in range(7))

    config = DynamicCmuArcticBuildConfig(
        cmu_root=Path("cmu"),
        dataset_root=Path("generated"),
        speakers=speakers,
        n_sources=7,
        n_moving_sources=2,
    )

    assert config.n_sources == 7
    assert config.speakers == speakers


def test_normalize_sources_uses_one_scale_that_also_protects_mixture() -> None:
    stems = [
        np.array([[0.75, 0.25], [-0.50, 0.10]], dtype=np.float64),
        np.array([[0.75, -0.25], [-0.50, 0.20]], dtype=np.float64),
    ]
    expected_scale = 0.99 / 1.50

    normalized = dynamic_builder._normalize_sources(stems)
    mixture = np.sum(np.stack(normalized, axis=0), axis=0)

    for actual, original in zip(normalized, stems, strict=True):
        np.testing.assert_allclose(actual, original * expected_scale)
    np.testing.assert_allclose(mixture, np.sum(normalized, axis=0))
    assert np.max(np.abs(mixture)) <= 0.99
    assert np.max(np.abs(np.stack(normalized, axis=0))) <= 0.99


def test_source_trajectory_uses_uniform_sample_starts_without_endpoint() -> None:
    schedule = FrameSchedule.uniform(
        frame_count=4,
        stop_sample=100,
    )
    timeline = schedule.starts.numpy().astype(np.float64) / 100.0
    starts = np.array([[3.0, 2.0, 1.0]], dtype=np.float64)

    trajectory = dynamic_builder._build_source_trajectory(
        starts=starts,
        mic_center=np.array([2.0, 2.0, 1.0], dtype=np.float64),
        start_azimuth=np.array([0.0], dtype=np.float64),
        end_azimuth=np.array([np.pi / 2.0], dtype=np.float64),
        moving_indices=[0],
        timeline=timeline,
        move_start_ratio=0.0,
        move_end_ratio=1.0,
    )

    np.testing.assert_array_equal(schedule.starts, torch.tensor([0, 25, 50, 75]))
    np.testing.assert_allclose(timeline, [0.0, 0.25, 0.50, 0.75])
    expected_theta = timeline * (np.pi / 2.0)
    np.testing.assert_allclose(trajectory[:, 0, 0], 2.0 + np.cos(expected_theta))
    np.testing.assert_allclose(trajectory[:, 0, 1], 2.0 + np.sin(expected_theta))
    np.testing.assert_allclose(trajectory[:, 0, 2], 1.0)
    assert not np.allclose(trajectory[-1, 0, :2], [2.0, 3.0])


def test_fixed_length_loader_checks_every_utterance_sample_rate() -> None:
    class _Sentence:
        def __init__(self, utterance_id: str) -> None:
            self.utterance_id = utterance_id

    class _MixedRateDataset:
        def available_sentences(self) -> list[_Sentence]:
            return [_Sentence("valid"), _Sentence("invalid")]

        def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
            return torch.ones(2), 8000 if utterance_id == "invalid" else 16000

    with pytest.raises(ValueError, match="sample rate mismatch while loading"):
        dynamic_builder._load_fixed_length_signal(
            cast(Any, _MixedRateDataset()),
            target_samples=5,
            expected_sample_rate=16000,
            rng_py=random.Random(0),
        )


def test_random_geometry_sampling_uses_strict_room_interior() -> None:
    class _LowerBoundRng:
        def uniform(self, low, high):
            del high
            return np.asarray(low)

    rng = cast(Any, _LowerBoundRng())
    room_size = np.array([8.0, 6.0, 3.0])
    center = dynamic_builder._sample_random_mic_center(
        rng=rng,
        room_size=room_size,
        source_margin=np.zeros(3),
        min_source_distance_m=1.0,
        array_radius_m=0.5,
    )
    assert np.all(center > np.array([1.0, 1.0, 0.5]))

    source = dynamic_builder._sample_position_on_azimuth(
        rng=rng,
        azimuth_rad=0.0,
        mic_center=np.array([4.0, 3.0, 1.5]),
        room_size=room_size,
        margin=np.zeros(3),
        min_radius_m=1.0,
    )
    assert np.all(source > 0.0)
    assert np.all(source < room_size)


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


@pytest.mark.numerical
def test_dynamic_cmu_arctic_audio_is_consistent(built_dataset: Path) -> None:
    scene_dir = built_dataset / "scene_0000"
    mixture, mixture_fs = sf.read(scene_dir / "mixture.wav", always_2d=True)
    stems = []
    for source_idx in range(3):
        stem, stem_fs = sf.read(
            scene_dir / f"source_{source_idx:02d}.wav", always_2d=True
        )
        assert stem_fs == mixture_fs
        assert stem.shape == mixture.shape
        stems.append(stem)

    assert mixture_fs == 16000
    assert mixture.shape == (1600 + 64 - 1, 6)
    assert sf.info(scene_dir / "mixture.wav").subtype == "FLOAT"
    assert all(
        sf.info(scene_dir / f"source_{index:02d}.wav").subtype == "FLOAT"
        for index in range(3)
    )
    np.testing.assert_allclose(mixture, np.sum(stems, axis=0), atol=5e-7, rtol=0)
    assert np.max(np.abs(mixture)) <= 0.99


@pytest.mark.numerical
def test_dynamic_cmu_arctic_geometry_stays_within_room(
    built_dataset: Path,
) -> None:
    metadata = json.loads(
        (built_dataset / "scene_0000" / "metadata.json").read_text(encoding="utf-8")
    )
    room_size = np.asarray(metadata["room"]["size"])
    src_traj = np.asarray(metadata["trajectories"]["sources"])
    mic_traj = np.asarray(metadata["trajectories"]["mics"])
    assert np.all(src_traj > 0.0) and np.all(src_traj < room_size)
    assert np.all(mic_traj > 0.0) and np.all(mic_traj < room_size)
    center = np.asarray(metadata["extra"]["mic_center_xyz_m"])
    distances = np.linalg.norm(src_traj - center, axis=-1)
    assert np.all(distances >= 1.8 - 1e-9)


@pytest.mark.numerical
def test_dynamic_cmu_arctic_build_is_seed_reproducible(tmp_path: Path) -> None:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)

    roots = [tmp_path / "first", tmp_path / "second"]
    for dataset_root in roots:
        build_dynamic_cmu_arctic(
            _small_build_config(
                cmu_root=cmu_root,
                dataset_root=dataset_root,
                duration_sec=0.05,
                trajectory_steps=4,
                simulation=SimulationConfig(max_order=0, nsample=32),
                seed=123,
            )
        )

    relative_paths = [
        "mixture.wav",
        "source_00.wav",
        "source_01.wav",
        "source_02.wav",
        "metadata.json",
        "source_info.json",
    ]
    for relative_path in relative_paths:
        first = roots[0] / "scene_0000" / relative_path
        second = roots[1] / "scene_0000" / relative_path
        assert first.read_bytes() == second.read_bytes()


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


def test_dynamic_cmu_arctic_builder_metadata_records_exact_schedule(
    built_dataset: Path,
) -> None:
    metadata = json.loads(
        (built_dataset / "scene_0000" / "metadata.json").read_text(encoding="utf-8")
    )

    expected_starts = list(range(0, 1600, 200))
    assert metadata["frame_schedule"] == {
        "starts_samples": expected_starts,
        "sample_rate": 16000.0,
    }
    assert metadata["schema"] == {"name": "torchrir.scene", "version": 1}
    assert metadata["convolution"] == {
        "time_reference": "emission",
        "output_sample_count": 1663,
    }
    assert metadata["signal"] == {
        "sample_rate": 16000.0,
        "sample_count": 1600,
    }


def test_dynamic_builder_ceil_aligns_signal_motion_and_metadata(tmp_path: Path) -> None:
    cmu_root = tmp_path / "cmu"
    for speaker in ("bdl", "slt", "clb"):
        _write_fake_cmu_speaker(cmu_root, speaker)
    dataset_root = tmp_path / "dataset"
    requested_duration = 0.10001

    build_dynamic_cmu_arctic(
        _small_build_config(
            cmu_root=cmu_root,
            dataset_root=dataset_root,
            duration_sec=requested_duration,
        )
    )
    scene_dir = dataset_root / "scene_0000"
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    mixture, sample_rate = sf.read(scene_dir / "mixture.wav")

    assert sample_rate == 16000
    assert metadata["signal"]["sample_count"] == 1601
    assert mixture.shape[0] == 1601 + 64 - 1
    assert metadata["frame_schedule"]["starts_samples"] == [
        0,
        200,
        400,
        600,
        800,
        1000,
        1200,
        1400,
    ]
    motion_time = metadata["extra"]["motion_time_sec"]
    assert motion_time["requested_total"] == pytest.approx(requested_duration)
    assert motion_time["effective_total"] == pytest.approx(1601 / 16000)
    assert motion_time["move_start"] == pytest.approx((1601 / 16000) * 0.35)
    assert motion_time["move_end"] == pytest.approx((1601 / 16000) * 0.65)


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
    build_dynamic_cmu_arctic(
        _small_build_config(
            cmu_root=cmu_root,
            dataset_root=dataset_root,
            save_layout_mp4=True,
            save_layout_mp4_3d=False,
            layout_video_fps=12.0,
            layout_video_mux_audio=False,
        )
    )

    assert len(calls) == 1
    call = calls[0]
    assert cast(Path, call["out_dir"]).name == "scene_0000"
    assert cast(Path, call["mixture_path"]).name == "mixture.wav"
    assert call["save_3d"] is False
    assert call["mp4_fps"] == pytest.approx(12.0)
    assert call["mux_audio"] is False
    assert (
        call["stop_sample"] == sf.info(dataset_root / "scene_0000/mixture.wav").frames
    )
    assert isinstance(call["schedule"], FrameSchedule)
    assert len(call["schedule"]) == cast(torch.Tensor, call["src_traj"]).shape[0]
    annotation_lines = cast(list[str], call["annotation_lines"])
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
    build_dynamic_cmu_arctic(
        _small_build_config(
            cmu_root=cmu_root,
            dataset_root=dataset_root,
            save_layout_mp4=True,
            save_layout_images=True,
        )
    )

    assert len(image_calls) == 1
    assert image_calls[0]["annotate_sources"] is True
    assert image_calls[0]["src_traj"] is not None
    assert image_calls[0]["mic_traj"] is not None
    assert image_calls[0]["save_3d"] is True
    image_annotation_lines = cast(list[str], image_calls[0]["annotation_lines"])
    assert len(image_annotation_lines) == 3
    assert image_annotation_lines[0] == "scene:scene_0000"
    assert image_annotation_lines[1].startswith("move:")
    assert image_annotation_lines[2].startswith("speed:")
    assert len(video_calls) == 1
    assert video_calls[0]["annotate_sources"] is True
    video_annotation_lines = cast(list[str], video_calls[0]["annotation_lines"])
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


def test_builder_restores_existing_dataset_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_root = tmp_path / "out_ds"
    dataset_root.mkdir()
    marker = dataset_root / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    stale_workspace = tmp_path / ".out_ds.torchrir-build.stale"
    stale_workspace.mkdir()
    (stale_workspace / "owner").write_text(
        archive_utils._TEMPORARY_MARKER,
        encoding="utf-8",
    )
    (stale_workspace / "partial.txt").write_text("stale", encoding="utf-8")

    def _fail_build(
        config: DynamicCmuArcticBuildConfig,
        *,
        logger: object = None,
    ) -> tuple[int, int]:
        del logger
        staging_root = Path(config.dataset_root)
        staging_root.mkdir(parents=True)
        (staging_root / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected failure")

    monkeypatch.setattr(
        dynamic_builder, "_build_dynamic_cmu_arctic_in_place", _fail_build
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        build_dynamic_cmu_arctic(
            DynamicCmuArcticBuildConfig(
                cmu_root=tmp_path / "cmu",
                dataset_root=dataset_root,
                overwrite=True,
            )
        )

    assert marker.read_text(encoding="utf-8") == "original"
    assert not [
        path for path in tmp_path.glob(".out_ds.torchrir-build.*") if path.is_dir()
    ]


def test_commit_staged_dataset_preserves_overwrite_false_under_concurrency(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "out_ds"
    staged_directories = [tmp_path / "staged-a", tmp_path / "staged-b"]
    for index, staged in enumerate(staged_directories):
        staged.mkdir()
        (staged / "value.txt").write_text(str(index), encoding="utf-8")

    def commit(staged: Path) -> str:
        try:
            dynamic_builder._commit_staged_dataset(
                staged,
                dataset_root,
                overwrite=False,
            )
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(commit, staged_directories))

    assert sorted(outcomes) == ["exists", "published"]
    assert (dataset_root / "value.txt").read_text(encoding="utf-8") in {"0", "1"}


def test_builder_serializes_generation_before_target_exists_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "out_ds"
    config = _small_build_config(
        cmu_root=tmp_path / "cmu",
        dataset_root=dataset_root,
        overwrite=False,
    )
    real_lock = dynamic_builder.dataset_write_lock
    state_lock = Lock()
    first_build_entered = Event()
    second_lock_attempted = Event()
    release_first_build = Event()
    lock_attempts = 0
    build_calls = 0

    @contextmanager
    def observed_build_lock(path: Path):
        nonlocal lock_attempts
        with state_lock:
            lock_attempts += 1
            attempt = lock_attempts
        if attempt == 2:
            second_lock_attempted.set()
        with real_lock(path):
            yield

    def fake_build(
        request: DynamicCmuArcticBuildConfig,
        *,
        logger: object = None,
    ) -> tuple[int, int]:
        nonlocal build_calls
        del logger
        with state_lock:
            build_calls += 1
            call = build_calls
        staging_root = Path(request.dataset_root)
        staging_root.mkdir(parents=True)
        if call == 1:
            first_build_entered.set()
            assert second_lock_attempted.wait(timeout=2.0)
            assert release_first_build.wait(timeout=2.0)
        return 8000, 1

    monkeypatch.setattr(dynamic_builder, "dataset_write_lock", observed_build_lock)
    monkeypatch.setattr(
        dynamic_builder, "_build_dynamic_cmu_arctic_in_place", fake_build
    )

    def build() -> str:
        try:
            build_dynamic_cmu_arctic(config)
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(build)
        assert first_build_entered.wait(timeout=2.0)
        second = executor.submit(build)
        assert second_lock_attempted.wait(timeout=2.0)
        release_first_build.set()
        outcomes = [first.result(timeout=2.0), second.result(timeout=2.0)]

    assert sorted(outcomes) == ["exists", "published"]
    assert build_calls == 1


def test_builder_recovers_completed_transaction_before_exists_check(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "out_ds"
    dataset_root.mkdir()
    (dataset_root / "current.txt").write_text("current", encoding="utf-8")
    transaction = tmp_path / ".out_ds.torchrir-transaction"
    backup = transaction / "previous"
    backup.mkdir(parents=True)
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    (backup / "old.txt").write_text("old", encoding="utf-8")
    target_status = dataset_root.stat()
    backup_status = backup.stat()
    archive_utils._write_transaction_manifest(
        transaction,
        archive_utils.TransactionManifest(
            staged=archive_utils.FileIdentity(
                int(target_status.st_dev),
                int(target_status.st_ino),
            ),
            original=archive_utils.FileIdentity(
                int(backup_status.st_dev),
                int(backup_status.st_ino),
            ),
            phase="published",
        ),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        build_dynamic_cmu_arctic(
            _small_build_config(
                cmu_root=tmp_path / "cmu",
                dataset_root=dataset_root,
                overwrite=False,
            )
        )

    assert not transaction.exists()
    assert (dataset_root / "current.txt").read_text(encoding="utf-8") == "current"
