import pytest
import torch

from torchrir import ISMSimulator, MicrophoneArray, RIRResult, Room, Scene, SimulationConfig, Source


def test_scene_validate_static():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    scene = Scene(room=room, sources=sources, mics=mics)
    scene.validate()


def test_scene_validate_dynamic_shapes():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[1.5, 1.0, 1.0]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[2.0, 1.5, 1.0]],
            [[2.2, 1.5, 1.0]],
        ]
    )
    scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    scene.validate()


def test_scene_validate_mismatch_time():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    src_traj = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[1.5, 1.0, 1.0]],
        ]
    )
    mic_traj = torch.tensor(
        [
            [[2.0, 1.5, 1.0]],
            [[2.2, 1.5, 1.0]],
            [[2.4, 1.5, 1.0]],
        ]
    )
    scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    with pytest.raises(ValueError):
        scene.validate()


def test_rir_result_container():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    scene = Scene(room=room, sources=sources, mics=mics)
    config = SimulationConfig(max_order=1, tmax=0.1)
    rirs = torch.zeros((1, 1, 256))
    result = RIRResult(rirs=rirs, scene=scene, config=config, seed=123)
    assert result.rirs.shape == (1, 1, 256)


def test_ism_simulator_static():
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.positions([[2.0, 1.5, 1.0]])
    scene = Scene(room=room, sources=sources, mics=mics)
    config = SimulationConfig(max_order=1, tmax=0.05)
    result = ISMSimulator().simulate(scene, config)
    assert result.rirs.ndim == 3
