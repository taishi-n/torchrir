import pytest
import torch

from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate_rir


def test_room_beta_t60_exclusive():
    with pytest.raises(ValueError):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=[0.9] * 6, t60=0.5)


def test_room_dimension_validation():
    with pytest.raises(ValueError):
        Room.shoebox(size=[4.0], fs=16000)

    room = Room.shoebox(size=[4.0, 3.0], fs=16000)
    assert room.size.shape == (2,)
    room3 = Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000)
    assert room3.size.shape == (3,)


def test_room_positive_parameters_validation():
    with pytest.raises(ValueError, match="fs must be positive"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=0)
    with pytest.raises(ValueError, match="c must be positive"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, c=0.0)
    with pytest.raises(ValueError, match="room size must be strictly positive"):
        Room.shoebox(size=[4.0, -3.0, 2.0], fs=16000)


def test_room_beta_validation():
    with pytest.raises(ValueError, match="beta must have 6 elements"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=[0.9] * 4)
    with pytest.raises(ValueError, match="beta values must be in \\[0, 1\\]"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=[1.1] * 6)


def test_integer_geometry_is_promoted_to_floating_point() -> None:
    room = Room.shoebox(size=[4, 3, 2], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1, 1, 1]])
    mics = MicrophoneArray.from_positions([[2, 1, 1]])
    rir = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        config=SimulationConfig(max_order=0, nsample=128),
    )
    assert room.size.dtype == torch.float32
    assert sources.positions.dtype == torch.float32
    assert rir.dtype == torch.float32


def test_scene_rejects_positions_outside_room() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000)
    with pytest.raises(ValueError, match="within room bounds"):
        StaticScene(
            room=room,
            sources=Source.from_positions([[5.0, 1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0, 1.0]]),
        )


def test_entity_rejects_zero_orientation_vector() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        Source.from_positions([[1.0, 1.0, 1.0]], orientation=[0.0, 0.0, 0.0])
