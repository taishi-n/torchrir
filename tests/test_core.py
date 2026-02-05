import torch
import pytest

from torchrir import (
    MicrophoneArray,
    Room,
    SimulationConfig,
    Source,
    simulate_dynamic_rir,
    simulate_rir,
)


def test_simulate_rir_shape_and_peak():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0, 1.0]])

    nsample = 2048
    rir = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=0,
        nsample=nsample,
        directivity="omni",
    )

    assert rir.shape == (1, 1, nsample)
    dist = 1.0
    expected = dist / room.c * room.fs
    fdl = SimulationConfig().frac_delay_length
    expected += (fdl - 1) / 2
    peak = torch.argmax(torch.abs(rir[0, 0])).item()
    assert abs(peak - expected) <= 1.0


def test_simulate_rir_directivity_requires_orientation():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0, 1.0]])

    with pytest.raises(ValueError):
        simulate_rir(
            room=room,
            sources=sources,
            mics=mics,
            max_order=0,
            nsample=256,
            directivity="cardioid",
        )


def test_simulate_rir_angle_orientation_2d():
    room = Room.shoebox(size=[5.0, 4.0], fs=16000, beta=[0.9] * 4)
    sources = Source.from_positions([[1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]])
    orientation = torch.tensor(0.0)

    rir = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=0,
        nsample=256,
        directivity="cardioid",
        orientation=orientation,
    )

    assert rir.shape == (1, 1, 256)


def test_simulate_dynamic_rir_shape():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)

    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[1.5, 1.0, 1.0]],
            [[2.0, 1.0, 1.0]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[2.5, 1.0, 1.0]],
            [[2.5, 1.2, 1.0]],
            [[2.5, 1.4, 1.0]],
        ]
    )

    nsample = 512
    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=0,
        nsample=nsample,
    )

    assert rirs.shape == (3, 1, 1, nsample)


def test_dynamic_accepts_2d_input():
    room = Room.shoebox(size=[5.0, 4.0], fs=8000, beta=[0.9] * 4)

    src_traj = torch.tensor(
        [
            [1.0, 1.0],
            [1.5, 1.0],
            [2.0, 1.0],
        ]
    )
    mic_traj = torch.tensor(
        [
            [2.5, 1.0],
            [2.5, 1.2],
            [2.5, 1.4],
        ]
    )

    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=0,
        nsample=256,
    )

    assert rirs.shape == (3, 1, 1, 256)
