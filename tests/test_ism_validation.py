"""Focused validation tests for image-source kernels."""

from __future__ import annotations

import pytest
import torch

from torchrir import MicrophoneArray, Room, Source
from torchrir.sim.ism.context import prepare_source_directions
from torchrir.sim.ism.helpers import _resolve_beta, _validate_beta
from torchrir.sim.ism.prepare import _prepare_static_tensors
from torchrir.sim.ism.validate import (
    _validate_pos_shapes,
    _validate_positions_in_room,
    _validate_source_mic_separation,
    _validate_traj_shapes,
)


def test_position_and_trajectory_shape_validators() -> None:
    valid = torch.ones(2, 3)
    _validate_pos_shapes(valid, valid, 3)
    with pytest.raises(ValueError, match="sources"):
        _validate_pos_shapes(torch.ones(3), valid, 3)
    with pytest.raises(ValueError, match="mics"):
        _validate_pos_shapes(valid, torch.ones(2, 2), 3)

    room_size = torch.tensor([4.0, 3.0, 2.0])
    _validate_positions_in_room(valid, room_size, name="positions")
    with pytest.raises(ValueError, match="finite"):
        _validate_positions_in_room(
            torch.tensor([[float("nan"), 1.0, 1.0]]), room_size, name="positions"
        )
    with pytest.raises(ValueError, match="room bounds"):
        _validate_positions_in_room(
            torch.tensor([[5.0, 1.0, 1.0]]), room_size, name="positions"
        )
    for boundary in (
        torch.tensor([[0.0, 1.0, 1.0]]),
        torch.tensor([[4.0, 1.0, 1.0]]),
    ):
        with pytest.raises(ValueError, match="strictly inside"):
            _validate_positions_in_room(boundary, room_size, name="positions")

    src = torch.ones(2, 1, 3)
    mic = torch.ones(2, 2, 3)
    _validate_traj_shapes(src, mic, 3)
    invalid_cases = [
        (torch.ones(2, 3), mic, "src_traj must be of shape"),
        (src, torch.ones(2, 3), "mic_traj must be of shape"),
        (src, torch.ones(3, 2, 3), "same time length"),
        (torch.ones(2, 1, 2), mic, "src_traj must match"),
        (src, torch.ones(2, 2, 2), "mic_traj must match"),
    ]
    for invalid_src, invalid_mic, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            _validate_traj_shapes(invalid_src, invalid_mic, 3)


def test_source_microphone_separation_validator_reports_pair_and_frame() -> None:
    with pytest.raises(ValueError, match="source 1, microphone 0"):
        _validate_source_mic_separation(
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
            torch.tensor([[2.0, 2.0]]),
            min_distance=1e-6,
        )

    with pytest.raises(ValueError, match="frame 1, source 0, microphone 0"):
        _validate_source_mic_separation(
            torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]),
            torch.tensor([[[3.0, 3.0]], [[2.0, 2.0]]]),
            min_distance=1e-6,
        )


def test_source_microphone_separation_uses_a_stable_tiny_distance() -> None:
    _validate_source_mic_separation(
        torch.tensor([[1.0e-308, 1.0e-308]], dtype=torch.float64),
        torch.tensor([[2.0e-308, 1.0e-308]], dtype=torch.float64),
        min_distance=torch.finfo(torch.float64).smallest_normal / 10.0,
    )


def test_beta_preparation_paths() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, t60=0.5)
    beta = _resolve_beta(
        room,
        room.size,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert beta.shape == (4,)
    assert beta.dtype == torch.float64
    default_beta = _resolve_beta(
        Room.shoebox([4.0, 3.0, 2.0], fs=8000),
        torch.tensor([4.0, 3.0, 2.0]),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(default_beta, torch.ones(6))
    with pytest.raises(ValueError, match="4 elements"):
        _validate_beta(torch.ones(6), 2)


def test_static_preparation_uses_entity_orientation_without_override() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    source = Source.from_positions(
        [[1.0, 1.0]],
        orientation=[1.0, 0.0],
        directivity="cardioid",
    )
    microphones = MicrophoneArray.from_positions([[2.0, 1.0]])
    source_positions, _, source_orientation, _, _, dimension = _prepare_static_tensors(
        room=room,
        sources=source,
        microphones=microphones,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert source_positions.dtype == torch.float64
    assert source_orientation is not None
    assert source_orientation.dtype == torch.float64
    torch.testing.assert_close(
        source_orientation, torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    )
    assert dimension == 2


def test_prepare_source_directions_validates_count_and_orientation() -> None:
    assert prepare_source_directions(None, pattern="omni", dim=2, count=2) is None
    with pytest.raises(ValueError, match="required"):
        prepare_source_directions(None, pattern="cardioid", dim=2, count=1)
    canonical = prepare_source_directions(
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        pattern="cardioid",
        dim=2,
        count=2,
    )
    torch.testing.assert_close(
        canonical,
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    with pytest.raises(ValueError, match="canonical"):
        prepare_source_directions(
            torch.tensor([1.0, 0.0]), pattern="cardioid", dim=2, count=2
        )
