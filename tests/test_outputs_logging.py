"""Output persistence and logging behavior."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from torchrir import MicrophoneArray, RIRResult, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.datasets.attribution import attribution_for
from torchrir.io.outputs import (
    save_attribution_file,
    save_result_metadata,
    save_scene_audio,
    save_scene_metadata,
)
from torchrir.logging import LoggingConfig, get_logger, setup_logging


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
        LoggingConfig(level="verbose").resolve_level()
    with pytest.raises(TypeError, match="str or int"):
        LoggingConfig(level=object()).resolve_level()  # type: ignore[arg-type]


def test_setup_logging_is_idempotent_and_namespaces_children() -> None:
    name = "torchrir.test.outputs"
    logger = logging.getLogger(name)
    logger.handlers.clear()
    configured = setup_logging(
        LoggingConfig(level="DEBUG", format="%(message)s", propagate=True),
        name=name,
    )
    repeated = setup_logging(LoggingConfig(level="INFO"), name=name)
    assert configured is repeated
    assert len(configured.handlers) == 1
    assert configured.level == logging.INFO
    assert get_logger() is logging.getLogger("torchrir")
    assert get_logger("examples") is logging.getLogger("torchrir.examples")
    assert get_logger("torchrir.custom") is logging.getLogger("torchrir.custom")
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

    result = RIRResult(
        rirs=rirs,
        scene=scene,
        config=SimulationConfig(max_order=0, nsample=32),
        seed=5,
        backend="test-backend",
    )
    result_metadata = save_result_metadata(
        out_dir=tmp_path,
        result=result,
        metadata_name="result.json",
        extra={"kind": "result"},
        logger=logger,
    )
    assert result_metadata["simulation"]["backend"] == "test-backend"
    assert result_metadata["simulation"]["seed"] == 5
    assert json.loads((tmp_path / "result.json").read_text()) == result_metadata
    assert logger.info.call_count == 2
