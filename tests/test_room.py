import pytest

from torchrir import Room


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
