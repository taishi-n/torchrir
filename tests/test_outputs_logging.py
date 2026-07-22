"""Output persistence and logging behavior."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import stat
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.datasets.attribution import DatasetAttribution, attribution_for
from torchrir.io.outputs import (
    save_attribution_file,
    save_result_metadata,
    save_scene_audio,
    save_scene_metadata,
)
from torchrir.io import build_metadata, build_result_metadata
from torchrir.io import save_metadata_json
from torchrir.logging import LoggingConfig, get_logger, setup_logging
from torchrir.sim import simulate
from torchrir.signal import FrameSchedule


def _scene() -> StaticScene:
    return StaticScene(
        room=Room.shoebox([4.0, 3.0], fs=8000, beta=[0.8] * 4),
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.5]]),
    )


def test_logging_config_resolves_levels_and_rejects_invalid_values() -> None:
    assert LoggingConfig(level="debug").resolve_level() == logging.DEBUG
    assert LoggingConfig(level=logging.WARNING).resolve_level() == logging.WARNING
    assert LoggingConfig().replace(level="ERROR").level == "ERROR"
    with pytest.raises(ValueError, match="unknown log level"):
        LoggingConfig(level="verbose")
    with pytest.raises(TypeError, match="str or int"):
        LoggingConfig(level=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="str or int"):
        LoggingConfig(level=True)
    with pytest.raises(ValueError, match="non-empty"):
        LoggingConfig(format="")
    with pytest.raises(TypeError, match="format"):
        LoggingConfig(format=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="datefmt"):
        LoggingConfig(datefmt=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid logging format"):
        LoggingConfig(format="%(")


def test_setup_logging_is_idempotent_and_namespaces_children() -> None:
    logger = logging.getLogger("torchrir")
    logger.handlers.clear()
    configured = setup_logging(
        LoggingConfig(level="DEBUG", format="%(message)s"),
    )
    repeated = setup_logging(LoggingConfig(level="INFO"))
    assert configured is repeated
    assert len(configured.handlers) == 1
    assert configured.level == logging.INFO
    assert configured.handlers[0].level == logging.INFO
    assert configured.propagate is False
    setup_logging(
        LoggingConfig(level="DEBUG", format="%(levelname)s:%(message)s"),
    )
    assert configured.handlers[0].level == logging.DEBUG
    assert configured.handlers[0].formatter is not None
    assert configured.handlers[0].formatter._fmt == "%(levelname)s:%(message)s"
    assert get_logger() is logging.getLogger("torchrir")
    assert get_logger("examples") is logging.getLogger("torchrir.examples")
    assert get_logger("torchrir.custom") is logging.getLogger("torchrir.custom")
    assert get_logger("torchrir_custom") is logging.getLogger(
        "torchrir.torchrir_custom"
    )
    with pytest.raises(ValueError, match="non-empty"):
        get_logger("")
    with pytest.raises(TypeError, match="name"):
        get_logger(cast(Any, 1))
    with pytest.raises(TypeError, match="LoggingConfig"):
        setup_logging(cast(Any, object()))
    logger.handlers.clear()


def test_setup_logging_is_atomic_on_formatter_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = logging.getLogger("torchrir")
    logger.handlers.clear()
    logger.setLevel(logging.WARNING)
    logger.propagate = True
    config = LoggingConfig(level="DEBUG")

    def fail_formatter(*args: object, **kwargs: object) -> logging.Formatter:
        del args, kwargs
        raise ValueError("injected formatter failure")

    monkeypatch.setattr(logging, "Formatter", fail_formatter)
    with pytest.raises(ValueError, match="formatter failure"):
        setup_logging(config)

    assert logger.level == logging.WARNING
    assert logger.propagate is True
    assert logger.handlers == []


def test_setup_logging_serializes_concurrent_handler_creation() -> None:
    logger = logging.getLogger("torchrir")
    logger.handlers.clear()
    config = LoggingConfig(level="INFO")

    with ThreadPoolExecutor(max_workers=8) as executor:
        configured = list(executor.map(lambda _: setup_logging(config), range(32)))

    assert all(candidate is logger for candidate in configured)
    assert (
        len(
            [
                handler
                for handler in logger.handlers
                if getattr(handler, "_torchrir_managed", False)
            ]
        )
        == 1
    )
    logger.handlers.clear()


def test_save_scene_audio_writes_file_and_logs(tmp_path: Path) -> None:
    logger = Mock(spec=logging.Logger)
    output = save_scene_audio(
        out_dir=tmp_path / "scene",
        audio=torch.linspace(-0.2, 0.2, 32),
        fs=8000,
        audio_name="mixture.wav",
        logger=logger,
    )
    assert output.exists()
    logger.info.assert_called_once()


@pytest.mark.parametrize(
    "attribution",
    [
        attribution_for("librispeech", subset="dev-clean"),
        attribution_for("librispeech", subset="dev-clean").to_dict(),
    ],
)
def test_save_attribution_file_accepts_objects_and_mappings(
    tmp_path: Path, attribution: object
) -> None:
    output = save_attribution_file(
        out_dir=tmp_path,
        dataset_attribution=attribution,
        modifications=["trimmed", "convolved"],
    )
    text = output.read_text(encoding="utf-8")
    assert "Dataset: LibriSpeech" in text
    assert "Subset: dev-clean" in text
    assert "- trimmed" in text
    assert "THIRD_PARTY_DATASETS.md" in text


def test_save_attribution_file_validates_input_contract(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="mapping, dataclass"):
        save_attribution_file(
            out_dir=tmp_path, dataset_attribution=object(), modifications=[]
        )
    with pytest.raises(ValueError, match="missing required keys"):
        save_attribution_file(
            out_dir=tmp_path,
            dataset_attribution={"dataset": "incomplete"},
            modifications=[],
        )


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"dataset": None}, TypeError, "must be a string"),
        ({"source": ""}, ValueError, "non-empty"),
        ({"license_name": "license\ninjected"}, ValueError, "single-line"),
        ({"attribution_required": 1}, TypeError, "must be a bool"),
        ({"subset": "bad\rsubset"}, ValueError, "single-line"),
    ],
)
def test_save_attribution_file_rejects_invalid_mapping_before_writing(
    tmp_path: Path,
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    out_dir = tmp_path / "output"
    attribution = attribution_for("librispeech", subset="dev-clean").to_dict()
    attribution.update(updates)

    with pytest.raises(error, match=message):
        save_attribution_file(
            out_dir=out_dir,
            dataset_attribution=attribution,
            modifications=[],
        )

    assert not out_dir.exists()


@pytest.mark.parametrize("modifications", [None, [None], [""], ["ok\ninjected"]])
def test_save_attribution_file_rejects_invalid_modifications_before_writing(
    tmp_path: Path,
    modifications: object,
) -> None:
    out_dir = tmp_path / "output"
    with pytest.raises((TypeError, ValueError), match="modifications"):
        save_attribution_file(
            out_dir=out_dir,
            dataset_attribution=attribution_for("cmu_arctic"),
            modifications=cast(Any, modifications),
        )
    assert not out_dir.exists()


def test_save_attribution_file_respects_optional_attribution_flag(
    tmp_path: Path,
) -> None:
    attribution = DatasetAttribution(
        dataset_key="example",
        dataset="Example",
        source="https://example.invalid",
        license_name="Public domain",
        license_url="https://example.invalid/license",
        required_attribution="Example provenance",
        attribution_required=False,
    )

    output = save_attribution_file(
        out_dir=tmp_path,
        dataset_attribution=attribution,
        modifications=["simulated"],
    )

    text = output.read_text(encoding="utf-8")
    assert "Attribution required: no" in text
    assert "not required" in text
    assert "keep this attribution file" not in text


def test_output_helpers_reject_unsafe_names_and_logger_before_side_effects(
    tmp_path: Path,
) -> None:
    scene = _scene()
    invalid_dirs = [tmp_path / "audio", tmp_path / "notice", tmp_path / "metadata"]

    with pytest.raises(ValueError, match="single file name"):
        save_scene_audio(
            out_dir=invalid_dirs[0],
            audio=torch.ones(2),
            fs=8000,
            audio_name="../outside.wav",
        )
    with pytest.raises(ValueError, match="single file name"):
        save_attribution_file(
            out_dir=invalid_dirs[1],
            dataset_attribution=attribution_for("cmu_arctic"),
            modifications=[],
            attribution_name="../outside.txt",
        )
    with pytest.raises(ValueError, match="single file name"):
        save_scene_metadata(
            out_dir=invalid_dirs[2],
            metadata_name="../outside.json",
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 4),
        )
    for index, helper in enumerate((save_scene_audio, save_attribution_file)):
        out_dir = tmp_path / f"logger-{index}"
        kwargs: dict[str, object]
        if helper is save_scene_audio:
            kwargs = {
                "out_dir": out_dir,
                "audio": torch.ones(2),
                "fs": 8000,
                "audio_name": "audio.wav",
                "logger": object(),
            }
        else:
            kwargs = {
                "out_dir": out_dir,
                "dataset_attribution": attribution_for("cmu_arctic"),
                "modifications": [],
                "logger": object(),
            }
        with pytest.raises(TypeError, match="callable info"):
            helper(**cast(Any, kwargs))
        assert not out_dir.exists()

    assert all(not path.exists() for path in invalid_dirs)
    assert not (tmp_path / "outside.wav").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "outside.json").exists()


@pytest.mark.parametrize(
    "audio_name",
    ["nested/audio.wav", r"nested\audio.wav", r"C:\audio.wav", "NUL", "audio.wav."],
)
def test_output_names_are_safe_on_posix_and_windows(
    tmp_path: Path,
    audio_name: str,
) -> None:
    out_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="single file name"):
        save_scene_audio(
            out_dir=out_dir,
            audio=torch.ones(2),
            fs=8000,
            audio_name=audio_name,
        )

    assert not out_dir.exists()


def test_scene_and_result_metadata_round_trip_to_json(tmp_path: Path) -> None:
    scene = _scene()
    rirs = torch.zeros((1, 1, 32))
    logger = Mock(spec=logging.Logger)
    metadata = save_scene_metadata(
        out_dir=tmp_path,
        metadata_name="scene.json",
        room=scene.room,
        sources=scene.sources,
        mics=scene.mics,
        rirs=rirs,
        signal_len=100,
        source_info=[{"speaker": "test"}],
        extra={"purpose": "unit-test"},
        logger=logger,
    )
    persisted = json.loads((tmp_path / "scene.json").read_text(encoding="utf-8"))
    assert persisted == metadata
    assert persisted["extra"]["purpose"] == "unit-test"

    result = simulate(
        scene,
        SimulationConfig(max_order=0, nsample=32, seed=5),
    )
    result_metadata = save_result_metadata(
        out_dir=tmp_path,
        result=result,
        metadata_name="result.json",
        extra={"kind": "result"},
        logger=logger,
    )
    assert result_metadata["simulation"]["method"] == "ism"
    assert result_metadata["simulation"]["config"]["seed"] == 5
    assert json.loads((tmp_path / "result.json").read_text()) == result_metadata
    assert logger.info.call_count == 2


def test_one_frame_dynamic_result_remains_dynamic_in_metadata() -> None:
    static = _scene()
    dynamic = DynamicScene(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        src_traj=static.sources.positions.unsqueeze(0),
        mic_traj=static.mics.positions.unsqueeze(0),
        schedule=FrameSchedule.from_samples([0]),
    )
    result = simulate(dynamic, SimulationConfig(max_order=0, nsample=32))
    metadata = build_result_metadata(result)
    assert metadata["dynamic"] is True
    assert metadata["trajectories"]["sources"] is not None
    assert metadata["trajectories"]["mics"] is not None
    assert metadata["frame_schedule"]["starts_samples"] == [0]


def test_dynamic_metadata_preserves_exact_sample_schedule_and_time_reference() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(4, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(4, -1, -1).clone()
    schedule = FrameSchedule.from_samples([0, 25, 50, 75])

    metadata = build_metadata(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        rirs=torch.zeros(4, 1, 1, 16),
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=schedule,
        time_reference="emission",
        signal_len=100,
    )

    assert metadata["schema"] == {"name": "torchrir.scene", "version": 1}
    assert metadata["generator"]["name"] == "torchrir"
    assert metadata["generator"]["version"]
    assert metadata["generator"]["torch_version"] == str(torch.__version__)
    assert metadata["frame_schedule"]["starts_samples"] == [0, 25, 50, 75]
    assert metadata["frame_schedule"]["sample_rate"] == 8000.0
    assert metadata["signal"] == {"sample_rate": 8000.0, "sample_count": 100}
    assert metadata["convolution"] == {
        "time_reference": "emission",
        "output_sample_count": 115,
    }
    assert metadata["rir"] == {
        "shape": [4, 1, 1, 16],
        "sample_rate": 8000.0,
        "sample_count": 16,
        "origin_sample": 0,
    }
    assert set(metadata) == {
        "schema",
        "generator",
        "room",
        "sources",
        "mics",
        "trajectories",
        "rir",
        "doa",
        "frame_schedule",
        "signal",
        "convolution",
        "dynamic",
    }
    serialized = json.dumps(metadata)
    for removed_key in (
        "array",
        "rirs_shape",
        "signal_samples",
        "starts_seconds",
        "time_axis",
        "timestamps",
    ):
        assert f'"{removed_key}"' not in serialized


def test_observation_metadata_accepts_moving_mic_and_schedule_in_tail() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(3, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(3, -1, -1).clone()
    mic_traj[1, 0, 0] += 0.1
    mic_traj[2, 0, 0] += 0.2
    metadata = build_metadata(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        rirs=torch.zeros(3, 1, 1, 16),
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=FrameSchedule.from_samples([0, 50, 110]),
        time_reference="observation",
        signal_len=100,
    )

    assert metadata["frame_schedule"]["starts_samples"] == [0, 50, 110]
    assert metadata["convolution"] == {
        "time_reference": "observation",
        "output_sample_count": 115,
    }
    assert metadata["trajectories"]["mics"][2][0][0] == pytest.approx(
        float(static.mics.positions[0, 0]) + 0.2
    )


@pytest.mark.parametrize(
    ("moving_endpoint", "time_reference", "message"),
    [
        ("source", "observation", "requires fixed sources"),
        ("microphone", "emission", "does not support moving microphones"),
        ("both", "emission", "simultaneous source and microphone motion"),
    ],
)
def test_metadata_rejects_motion_incompatible_with_time_reference(
    moving_endpoint: str,
    time_reference: str,
    message: str,
) -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(2, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(2, -1, -1).clone()
    if moving_endpoint in ("source", "both"):
        src_traj[1, 0, 0] += 0.1
    if moving_endpoint in ("microphone", "both"):
        mic_traj[1, 0, 0] += 0.1

    with pytest.raises(ValueError, match=message):
        build_metadata(
            room=static.room,
            sources=static.sources,
            mics=static.mics,
            rirs=torch.zeros(2, 1, 1, 16),
            src_traj=src_traj,
            mic_traj=mic_traj,
            schedule=FrameSchedule.from_samples([0, 50]),
            time_reference=cast(Any, time_reference),
            signal_len=100,
        )


def test_metadata_rejects_invalid_time_reference_literal() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(2, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(2, -1, -1).clone()

    with pytest.raises(ValueError, match="time_reference"):
        build_metadata(
            room=static.room,
            sources=static.sources,
            mics=static.mics,
            rirs=torch.zeros(2, 1, 1, 16),
            src_traj=src_traj,
            mic_traj=mic_traj,
            schedule=FrameSchedule.from_samples([0, 50]),
            time_reference=cast(Any, "invalid"),
            signal_len=100,
        )


def test_result_metadata_rejects_motion_reference_mismatch() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(2, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(2, -1, -1).clone()
    mic_traj[1, 0, 0] += 0.1
    scene = DynamicScene(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=FrameSchedule.from_samples([0, 50]),
    )
    result = simulate(scene, SimulationConfig(max_order=0, nsample=16))

    with pytest.raises(ValueError, match="does not support moving microphones"):
        build_result_metadata(
            result,
            time_reference="emission",
            signal_len=100,
        )


def test_dynamic_metadata_without_schedule_does_not_infer_one() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(4, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(4, -1, -1).clone()

    metadata = build_metadata(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        rirs=torch.zeros(4, 1, 1, 16),
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=100,
    )

    assert metadata["dynamic"] is True
    assert metadata["frame_schedule"] is None
    assert metadata["convolution"] is None


def test_result_metadata_rejects_invalid_explicit_schedule_type() -> None:
    static = _scene()
    dynamic = DynamicScene(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        src_traj=static.sources.positions.unsqueeze(0),
        mic_traj=static.mics.positions.unsqueeze(0),
    )
    result = simulate(dynamic, SimulationConfig(max_order=0, nsample=16))
    with pytest.raises(TypeError, match="FrameSchedule"):
        build_result_metadata(result, schedule=cast(Any, object()))


def test_dynamic_metadata_rejects_incomplete_convolution_context() -> None:
    static = _scene()
    src_traj = static.sources.positions.unsqueeze(0).expand(4, -1, -1).clone()
    mic_traj = static.mics.positions.unsqueeze(0).expand(4, -1, -1).clone()
    with pytest.raises(
        ValueError, match="requires dynamic trajectories.*FrameSchedule"
    ):
        build_metadata(
            room=static.room,
            sources=static.sources,
            mics=static.mics,
            rirs=torch.zeros(4, 1, 1, 16),
            src_traj=src_traj,
            mic_traj=mic_traj,
            time_reference="emission",
            signal_len=100,
        )
    with pytest.raises(ValueError, match="signal_len is required"):
        build_metadata(
            room=static.room,
            sources=static.sources,
            mics=static.mics,
            rirs=torch.zeros(4, 1, 1, 16),
            src_traj=src_traj,
            mic_traj=mic_traj,
            schedule=FrameSchedule.from_samples([0, 25, 50, 75]),
            time_reference="emission",
        )


@pytest.mark.parametrize("signal_len", [True, 1.5, "100"])
def test_metadata_rejects_non_integer_signal_length(signal_len: Any) -> None:
    scene = _scene()
    with pytest.raises(TypeError, match="signal_len must be an integer"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            signal_len=signal_len,
        )


def test_metadata_rejects_non_positive_signal_length_and_non_mapping_extra() -> None:
    scene = _scene()
    with pytest.raises(ValueError, match="signal_len must be positive"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            signal_len=0,
        )
    with pytest.raises(TypeError, match="extra must be a mapping"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            extra=cast(Any, ["not", "a", "mapping"]),
        )


def test_metadata_rejects_signal_and_output_lengths_outside_int64() -> None:
    scene = _scene()
    int64_max = torch.iinfo(torch.int64).max
    with pytest.raises(ValueError, match="at most"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            signal_len=int64_max + 1,
        )

    with pytest.raises(ValueError, match="int64 output length"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 1, 16),
            src_traj=scene.sources.positions.unsqueeze(0),
            mic_traj=scene.mics.positions.unsqueeze(0),
            schedule=FrameSchedule.from_samples([0]),
            time_reference="emission",
            signal_len=int64_max,
        )


def test_metadata_serializes_extended_numpy_scalars_without_recursion() -> None:
    scene = _scene()
    metadata = build_metadata(
        room=scene.room,
        sources=scene.sources,
        mics=scene.mics,
        rirs=torch.zeros(1, 1, 16),
        extra={
            "extended": np.longdouble("1.5"),
            "array": np.array([1.25, 2.5], dtype=np.longdouble),
            "integer": np.int64(7),
            "boolean": np.bool_(True),
        },
    )

    assert metadata["extra"] == {
        "extended": 1.5,
        "array": [1.25, 2.5],
        "integer": 7,
        "boolean": True,
    }


def test_metadata_rejects_extended_numpy_complex_scalars() -> None:
    scene = _scene()
    with pytest.raises(TypeError, match="real"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            extra={"invalid": np.clongdouble(1 + 2j)},
        )


def test_metadata_rejects_cycles_but_allows_shared_containers() -> None:
    scene = _scene()
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="must not contain cycles"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            extra=cyclic,
        )

    shared = [1, 2]
    metadata = build_metadata(
        room=scene.room,
        sources=scene.sources,
        mics=scene.mics,
        rirs=torch.zeros(1, 1, 16),
        extra={"first": shared, "second": shared},
    )
    assert metadata["extra"] == {"first": [1, 2], "second": [1, 2]}


def test_metadata_preserves_an_explicit_empty_extra_mapping() -> None:
    scene = _scene()
    metadata = build_metadata(
        room=scene.room,
        sources=scene.sources,
        mics=scene.mics,
        rirs=torch.zeros(1, 1, 16),
        extra={},
    )

    assert metadata["extra"] == {}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            torch.sparse_coo_tensor(
                torch.tensor([[0]]),
                torch.tensor([1.0]),
                size=(1,),
            ),
            "dense strided",
        ),
        (
            torch.quantize_per_tensor(
                torch.tensor([1.0]),
                scale=0.1,
                zero_point=0,
                dtype=torch.qint8,
            ),
            "quantized",
        ),
        (torch.empty(1, device="meta"), "CPU, CUDA, or MPS"),
    ],
)
def test_metadata_rejects_non_materializable_tensors(
    value: torch.Tensor,
    message: str,
) -> None:
    scene = _scene()
    with pytest.raises(TypeError, match=message):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 16),
            extra={"invalid": value},
        )


def test_result_metadata_rejects_schedule_competing_with_scene_schedule() -> None:
    static = _scene()
    schedule = FrameSchedule.from_samples([0, 25, 50, 75])
    dynamic = DynamicScene(
        room=static.room,
        sources=static.sources,
        mics=static.mics,
        src_traj=static.sources.positions.unsqueeze(0).expand(4, -1, -1).clone(),
        mic_traj=static.mics.positions.unsqueeze(0).expand(4, -1, -1).clone(),
        schedule=schedule,
    )
    result = simulate(dynamic, SimulationConfig(max_order=0, nsample=16))

    with pytest.raises(ValueError, match="schedule must be omitted"):
        build_result_metadata(result, schedule=schedule)


def test_build_metadata_revalidates_mutated_scene_tensors() -> None:
    scene = _scene()
    assert scene.room.beta is not None
    scene.room.beta.fill_(2.0)
    with pytest.raises(ValueError, match="beta values"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=torch.zeros(1, 1, 32),
        )


def test_build_metadata_rejects_nonfinite_rirs() -> None:
    scene = _scene()
    rirs = torch.zeros(1, 1, 32)
    rirs[..., 0] = float("nan")
    with pytest.raises(ValueError, match="rirs must contain finite"):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=rirs,
        )


@pytest.mark.parametrize(
    ("rirs", "error", "message"),
    [
        (
            torch.zeros(1, 1, 16).to_sparse(),
            TypeError,
            "dense strided",
        ),
        (
            torch.empty(1, 1, 16, device="meta"),
            ValueError,
            "CPU, CUDA, or MPS",
        ),
    ],
)
def test_build_metadata_rejects_nonmaterialized_rirs(
    rirs: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    scene = _scene()
    with pytest.raises(error, match=message):
        build_metadata(
            room=scene.room,
            sources=scene.sources,
            mics=scene.mics,
            rirs=rirs,
        )


def test_metadata_does_not_infer_binaural_layout_from_two_mics() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, beta=[0.8] * 4)
    metadata = build_metadata(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0], [3.0, 2.0]]),
        rirs=torch.zeros(1, 2, 32),
    )
    assert metadata["mics"]["layout"]["kind"] == "custom"
    assert metadata["mics"]["layout"]["minimum_pair_distance"] == pytest.approx(2**0.5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metadata_layout_supports_low_precision_cpu_geometry(
    dtype: torch.dtype,
) -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, beta=[0.8] * 4, dtype=dtype)
    metadata = build_metadata(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]], dtype=dtype),
        mics=MicrophoneArray.from_positions([[2.0, 1.0], [3.0, 2.0]], dtype=dtype),
        rirs=torch.zeros(1, 2, 32, dtype=dtype),
    )
    assert metadata["mics"]["layout"]["minimum_pair_distance"] == pytest.approx(2**0.5)
    assert metadata["doa"]["azimuth"][0][0][0] == pytest.approx(
        torch.pi,
        abs=1e-12,
    )


def test_metadata_geometry_is_stable_for_extreme_float64_positions() -> None:
    room = Room.shoebox(
        [1.7e308, 1.7e308, 1.7e308],
        fs=8000,
        beta=[0.8] * 6,
        dtype=torch.float64,
    )
    source_position = [2.0e307, 2.0e307, 1.2e308]
    microphone_positions = [
        [1.0e308, 1.0e308, 2.0e307],
        [1.4e308, 1.4e308, 2.0e307],
    ]
    metadata = build_metadata(
        room=room,
        sources=Source.from_positions([source_position], dtype=torch.float64),
        mics=MicrophoneArray.from_positions(
            microphone_positions,
            dtype=torch.float64,
        ),
        rirs=torch.zeros(1, 2, 4, dtype=torch.float64),
    )

    center = metadata["mics"]["layout"]["center"]
    minimum_pair_distance = metadata["mics"]["layout"]["minimum_pair_distance"]
    azimuth = metadata["doa"]["azimuth"][0][0]
    elevation = metadata["doa"]["elevation"][0][0]

    assert all(math.isfinite(value) for value in center)
    assert math.isfinite(minimum_pair_distance)
    assert all(math.isfinite(value) for value in azimuth)
    assert all(math.isfinite(value) for value in elevation)
    assert center == pytest.approx([1.2e308, 1.2e308, 2.0e307], rel=1e-15)

    pair_delta = microphone_positions[1][0] - microphone_positions[0][0]
    assert minimum_pair_distance == pytest.approx(
        math.hypot(pair_delta, pair_delta),
        rel=1e-15,
    )
    x = source_position[0] - microphone_positions[0][0]
    y = source_position[1] - microphone_positions[0][1]
    z = source_position[2] - microphone_positions[0][2]
    assert azimuth[0] == pytest.approx(math.atan2(y, x), abs=1e-15)
    assert elevation[0] == pytest.approx(
        math.atan2(z, math.hypot(x, y)),
        abs=1e-15,
    )


def test_save_metadata_json_rejects_nonfinite_values_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    path.write_text('{"preserved": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Out of range float values"):
        save_metadata_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"preserved": True}


def test_save_metadata_json_preserves_existing_file_mode(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o640)
    save_metadata_json(path, {"valid": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o640
