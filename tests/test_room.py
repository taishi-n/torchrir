import pytest
import torch

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
