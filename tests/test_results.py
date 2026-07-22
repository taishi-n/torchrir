"""RIRResult shape and metadata invariants."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, RIRResult, Room, Source, StaticScene
from torchrir.config import (
    ResolvedSimulationConfig,
    SimulationConfig,
    _resolve_simulation_config,
)
from torchrir.signal import FrameSchedule
from torchrir.sim import simulate


def _static_scene() -> StaticScene:
    return StaticScene(
        room=Room.shoebox([4.0, 3.0], fs=8000),
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0], [3.0, 2.0]]),
    )


def _dynamic_scene() -> DynamicScene:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    sources = Source.from_positions([[1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]])
    return DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=[[[1.0, 1.0]], [[1.5, 1.0]]],
        mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        schedule=FrameSchedule.from_samples([0, 800]),
    )


def _config(scene: StaticScene | DynamicScene) -> ResolvedSimulationConfig:
    values = (
        (scene.src_traj, scene.mic_traj, scene.room.size)
        if isinstance(scene, DynamicScene)
        else (scene.sources.positions, scene.mics.positions, scene.room.size)
    )
    return _resolve_simulation_config(
        SimulationConfig(max_order=0, nsample=16),
        fs=scene.room.fs,
        room_dimension=int(scene.room.size.numel()),
        tensor_values=values,
    )


@pytest.mark.parametrize(
    ("rirs", "error", "message"),
    [
        (torch.zeros(1, 2, 16, dtype=torch.int64), TypeError, "floating-point"),
        (torch.zeros(1, 16), ValueError, "must be 3D"),
        (torch.zeros(1, 1, 16), ValueError, "microphones"),
        (torch.zeros(2, 2, 16), ValueError, "sources"),
        (torch.zeros(1, 2, 0), ValueError, "at least one sample"),
    ],
)
def test_static_rir_result_rejects_invalid_tensor_contracts(
    rirs: torch.Tensor, error: type[Exception], message: str
) -> None:
    scene = _static_scene()
    with pytest.raises(error, match=message):
        RIRResult(rirs=rirs, scene=scene, config=_config(scene))


def test_rir_result_rejects_unresolved_request_config() -> None:
    scene = _static_scene()
    with pytest.raises(TypeError, match="ResolvedSimulationConfig"):
        RIRResult(
            rirs=torch.zeros(1, 2, 16),
            scene=scene,
            config=cast(Any, SimulationConfig(max_order=0, nsample=16)),
        )


@pytest.mark.parametrize(
    ("rirs", "config_update", "message"),
    [
        (torch.zeros(1, 2, 15), {}, "15 samples"),
        (
            torch.zeros(1, 2, 16),
            {"fs": 16000.0, "tmax": 16 / 16000},
            "sampling rate",
        ),
        (
            torch.zeros(1, 2, 16),
            {"device": torch.device("mps:0")},
            "device",
        ),
        (torch.zeros(1, 2, 16), {"dtype": torch.float64}, "dtype"),
    ],
)
def test_rir_result_rejects_resolved_metadata_mismatch(
    rirs: torch.Tensor, config_update: dict[str, object], message: str
) -> None:
    scene = _static_scene()
    config = replace(_config(scene), **config_update)
    with pytest.raises(ValueError, match=message):
        RIRResult(rirs=rirs, scene=scene, config=config)


def test_dynamic_result_keeps_exact_frame_schedule_on_scene() -> None:
    scene = _dynamic_scene()
    result = RIRResult(
        rirs=torch.zeros(2, 1, 1, 16), scene=scene, config=_config(scene)
    )
    assert result.scene is scene
    assert scene.schedule is not None
    torch.testing.assert_close(scene.schedule.starts, torch.tensor([0, 800]))


def test_dynamic_result_rejects_frame_count_mismatch() -> None:
    scene = _dynamic_scene()
    with pytest.raises(ValueError, match="frames"):
        RIRResult(
            rirs=torch.zeros(3, 1, 1, 16),
            scene=scene,
            config=_config(scene),
        )


def test_result_rejects_nb_img_dimension_mismatch() -> None:
    scene = _static_scene()
    wrong = replace(_config(scene), max_order=None, nb_img=(1, 1, 1))
    with pytest.raises(ValueError, match="nb_img dimension"):
        RIRResult(rirs=torch.zeros(1, 2, 16), scene=scene, config=wrong)


def test_result_detects_valid_in_place_scene_mutation() -> None:
    scene = _dynamic_scene()
    result = RIRResult(
        rirs=torch.zeros(2, 1, 1, 16), scene=scene, config=_config(scene)
    )
    scene.src_traj[1, 0, 0] = 1.4
    with pytest.raises(ValueError, match="modified after"):
        result.validate()


def test_result_supports_inference_mode_scene_tensors() -> None:
    with torch.inference_mode():
        scene = _static_scene()
        result = simulate(scene, SimulationConfig(max_order=0, nsample=64))
    result.validate()


def test_result_detects_inference_mode_scene_mutation() -> None:
    with torch.inference_mode():
        scene = _static_scene()
        result = simulate(scene, SimulationConfig(max_order=0, nsample=64))
        scene.sources.positions[0, 0] = 1.25

    with pytest.raises(ValueError, match="modified after"):
        result.validate()


def test_result_detects_numpy_alias_scene_mutation() -> None:
    source_array = np.array([[1.0, 1.0]], dtype=np.float32)
    scene = StaticScene(
        room=Room.shoebox([4.0, 3.0], fs=8000),
        sources=Source.from_positions(torch.from_numpy(source_array)),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    result = simulate(scene, SimulationConfig(max_order=0, nsample=64))

    source_array[0, 0] = 1.25
    with pytest.raises(ValueError, match="modified after"):
        result.validate()


def test_result_rechecks_numpy_alias_rir_finiteness() -> None:
    scene = _static_scene()
    rir_array = np.zeros((1, 2, 16), dtype=np.float32)
    result = RIRResult(
        rirs=torch.from_numpy(rir_array),
        scene=scene,
        config=_config(scene),
    )

    rir_array[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        result.validate()


def test_tensor_model_equality_uses_identity_semantics() -> None:
    scene = _static_scene()
    other = _static_scene()
    result = simulate(scene, SimulationConfig(max_order=0, nsample=16))
    other_result = simulate(other, SimulationConfig(max_order=0, nsample=16))

    assert scene == scene
    assert scene != other
    assert scene.room != other.room
    assert scene.sources != other.sources
    assert scene.mics != other.mics
    assert result == result
    assert result != other_result
