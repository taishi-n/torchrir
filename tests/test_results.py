"""RIRResult shape and metadata invariants."""

from __future__ import annotations

import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, RIRResult, Room, Source, StaticScene
from torchrir.config import SimulationConfig


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
        timestamps=[0.0, 0.1],
    )


def _config() -> SimulationConfig:
    return SimulationConfig(max_order=0, nsample=16)


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
    with pytest.raises(error, match=message):
        RIRResult(rirs=rirs, scene=_static_scene(), config=_config())


def test_rir_result_requires_non_empty_backend() -> None:
    with pytest.raises(ValueError, match="backend"):
        RIRResult(
            rirs=torch.zeros(1, 2, 16),
            scene=_static_scene(),
            config=_config(),
            backend="",
        )


def test_dynamic_result_inherits_scene_timestamps() -> None:
    scene = _dynamic_scene()
    result = RIRResult(rirs=torch.zeros(2, 1, 1, 16), scene=scene, config=_config())
    assert result.timestamps is scene.timestamps


def test_dynamic_result_rejects_frame_count_mismatch() -> None:
    with pytest.raises(ValueError, match="frames"):
        RIRResult(
            rirs=torch.zeros(3, 1, 1, 16),
            scene=_dynamic_scene(),
            config=_config(),
        )


@pytest.mark.parametrize(
    ("timestamps", "error", "message"),
    [
        ([0.0, 0.1], TypeError, "must be a Tensor"),
        (torch.tensor([[0.0, 0.1]]), ValueError, "frame count"),
        (torch.tensor([0.0 + 0.0j, 0.1 + 0.0j]), ValueError, "finite real"),
        (torch.tensor([0.0, float("nan")]), ValueError, "finite real"),
        (torch.tensor([0.1, 0.2]), ValueError, "first timestamp"),
        (torch.tensor([0.0, 0.0]), ValueError, "strictly increasing"),
    ],
)
def test_dynamic_result_validates_timestamps(
    timestamps, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        RIRResult(
            rirs=torch.zeros(2, 1, 1, 16),
            scene=_dynamic_scene(),
            config=_config(),
            timestamps=timestamps,
        )


def test_static_result_rejects_timestamps() -> None:
    with pytest.raises(ValueError, match="only valid for dynamic"):
        RIRResult(
            rirs=torch.zeros(1, 2, 16),
            scene=_static_scene(),
            config=_config(),
            timestamps=torch.tensor([0.0]),
        )
