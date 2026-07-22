"""Focused validation tests for image-source kernels."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from torchrir import Room, Source
from torchrir.config import SimulationConfig
from torchrir.sim.ism.context import prepare_source_directions
from torchrir.sim.ism.helpers import _prepare_entities, _resolve_beta, _validate_beta
from torchrir.sim.ism.validate import (
    _merge_setting,
    _resolve_config,
    _validate_config_for_room,
    _validate_dynamic_args,
    _validate_pos_shapes,
    _validate_positions_in_room,
    _validate_static_args,
    _validate_traj_shapes,
)


def test_merge_setting_accepts_equal_values_and_rejects_conflicts() -> None:
    assert _merge_setting("max_order", None, 2) == 2
    assert _merge_setting("max_order", 2, None) == 2
    assert _merge_setting("device", "cpu", torch.device("cpu")) == "cpu"
    explicit = torch.tensor([1, 2])
    assert _merge_setting("nb_img", explicit, (1, 2)) is explicit
    with pytest.raises(ValueError, match="conflicting 'max_order'"):
        _merge_setting("max_order", 1, 2)
    with pytest.raises(ValueError, match="conflicting 'nb_img'"):
        _merge_setting("nb_img", torch.tensor([1, 2]), (2, 1))


def test_resolve_config_requires_order_and_compatible_timing() -> None:
    kwargs: dict[str, Any] = dict(
        device=None,
        max_order=None,
        nsample=None,
        tmax=None,
        tdiff=None,
        directivity=None,
        dtype=None,
        nb_img=None,
    )
    with pytest.raises(ValueError, match="max_order"):
        _resolve_config(config=SimulationConfig(), **kwargs)
    with pytest.raises(ValueError, match="conflicting 'max_order'"):
        _resolve_config(
            config=SimulationConfig(max_order=2),
            device=None,
            max_order=1,
            nsample=None,
            tmax=None,
            tdiff=None,
            directivity=None,
            dtype=None,
            nb_img=None,
        )
    resolved = _resolve_config(
        config=SimulationConfig(max_order=2, nsample=64), **kwargs
    )
    assert resolved[2] == 2
    assert resolved[3] == 64
    assert resolved[6] == "omni"


@pytest.mark.parametrize("validator", [_validate_static_args, _validate_dynamic_args])
def test_simulation_argument_validators_cover_timing_and_order(validator) -> None:
    room = Room.shoebox([4.0, 3.0], fs=10)
    assert validator(room=room, nsample=None, tmax=0.25, max_order=0) == 3
    with pytest.raises(TypeError, match="Room"):
        validator(room=cast(Any, object()), nsample=2, tmax=None, max_order=0)
    with pytest.raises(ValueError, match="nsample or tmax"):
        validator(room=room, nsample=None, tmax=None, max_order=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validator(room=room, nsample=2, tmax=0.2, max_order=0)
    with pytest.raises(ValueError, match="nsample must be positive"):
        validator(room=room, nsample=0, tmax=None, max_order=0)
    with pytest.raises(ValueError, match="max_order"):
        validator(room=room, nsample=2, tmax=None, max_order=-1)


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


def test_deprecated_room_config_fields_warn_explicitly() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    with pytest.deprecated_call(match="SimulationConfig.fs"):
        _validate_config_for_room(SimulationConfig(fs=8000), room)
    with pytest.deprecated_call(match="SimulationConfig.fs"):
        with pytest.raises(ValueError, match="conflicts"):
            _validate_config_for_room(SimulationConfig(fs=16000), room)
    with pytest.deprecated_call(match="mixed_precision"):
        _validate_config_for_room(SimulationConfig(mixed_precision=True), room)


def test_entity_and_beta_preparation_paths() -> None:
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

    source = Source.from_positions([[1.0, 1.0]], orientation=[1.0, 0.0])
    position, orientation = _prepare_entities(
        source, None, which="source", device="cpu", dtype=torch.float64
    )
    assert position.dtype == torch.float64
    assert orientation is not None and orientation.dtype == torch.float64
    _, overridden = _prepare_entities(
        source,
        (torch.tensor([0.0, 1.0]), None),
        which="source",
        device="cpu",
        dtype=torch.float32,
    )
    torch.testing.assert_close(overridden, torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="length 2"):
        _prepare_entities(
            source,
            cast(Any, (torch.ones(2),)),
            which="source",
            device="cpu",
            dtype=torch.float32,
        )


def test_prepare_source_directions_validates_count_and_orientation() -> None:
    assert prepare_source_directions(None, pattern="omni", dim=2, count=2) is None
    with pytest.raises(ValueError, match="required"):
        prepare_source_directions(None, pattern="cardioid", dim=2, count=1)
    repeated = prepare_source_directions(
        torch.tensor([1.0, 0.0]), pattern="cardioid", dim=2, count=2
    )
    torch.testing.assert_close(repeated, torch.tensor([[1.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError, match="match number"):
        prepare_source_directions(
            torch.tensor([[1.0, 0.0]]), pattern="cardioid", dim=2, count=2
        )
