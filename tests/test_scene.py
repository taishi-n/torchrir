import pytest
import torch
import warnings
from typing import Any, cast

from torchrir import (
    DynamicScene,
    MicrophoneArray,
    RIRResult,
    Room,
    Scene,
    Source,
    StaticScene,
)
from torchrir.config import ResolvedSimulationConfig, SimulationConfig
from torchrir.sim import ISMSimulator, simulate


def test_scene_validate_static():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    scene.validate()


def test_scene_validate_dynamic_shapes():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
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
    scene = DynamicScene(
        room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj
    )
    scene.validate()


def test_scene_validate_mismatch_time():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
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
    with pytest.raises(ValueError):
        DynamicScene(
            room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj
        )


def test_rir_result_container():
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    config = SimulationConfig(max_order=1, tmax=0.1)
    rirs = torch.zeros((1, 1, 256))
    result = RIRResult(rirs=rirs, scene=scene, config=config, seed=123)
    assert result.rirs.shape == (1, 1, 256)


def test_ism_simulator_static():
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    config = SimulationConfig()
    result = ISMSimulator(max_order=1, tmax=0.05).simulate(scene, config)
    assert result.rirs.ndim == 3


def test_legacy_scene_deprecated_for_static() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    with pytest.deprecated_call(match="Scene is deprecated"):
        scene = Scene(room=room, sources=sources, mics=mics)
    assert scene.is_dynamic() is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scene.validate()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not deprecations


def test_legacy_scene_rejects_half_dynamic() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    src_traj = torch.tensor([[[1.0, 1.0, 1.0]], [[1.2, 1.1, 1.0]]], dtype=torch.float32)
    with pytest.deprecated_call(match="Scene is deprecated"):
        with pytest.raises(ValueError, match="requires both src_traj and mic_traj"):
            Scene(
                room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=None
            )


def test_ism_simulator_rejects_missing_timing() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.5, 1.0]]),
    )
    with pytest.raises(ValueError, match="nsample or tmax must be provided"):
        ISMSimulator().simulate(scene, SimulationConfig(max_order=1))


def test_dynamic_scene_normalizes_non_tensor_traj() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=cast(Any, [[[1.0, 1.0, 1.0]], [[1.1, 1.0, 1.0]]]),
        mic_traj=cast(Any, [[[2.0, 1.5, 1.0]], [[2.0, 1.5, 1.0]]]),
    )
    assert torch.is_tensor(scene.src_traj)
    assert torch.is_tensor(scene.mic_traj)


def test_ism_simulator_rejects_conflicting_config_max_order() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    cfg = SimulationConfig(max_order=2)
    sim = ISMSimulator(max_order=1, tmax=0.05)
    with pytest.raises(ValueError, match="conflicting 'max_order'"):
        sim.simulate(scene, cfg)


def test_ism_simulator_rejects_conflicting_config_tmax() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    cfg = SimulationConfig(tmax=0.10)
    sim = ISMSimulator(max_order=1, tmax=0.05)
    with pytest.raises(ValueError, match="conflicting 'tmax'"):
        sim.simulate(scene, cfg)


def test_scene_oriented_simulation_returns_effective_config() -> None:
    room = Room.shoebox(size=[4, 3, 2], fs=8000, beta=[0.9] * 6)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1, 1, 1]]),
        mics=MicrophoneArray.from_positions([[2, 1, 1]]),
    )
    result = simulate(scene, SimulationConfig(max_order=0, nsample=128))
    assert isinstance(result.config, ResolvedSimulationConfig)
    assert result.config.max_order == 0
    assert result.config.nsample == 128
    assert result.config.fs == 8000
    assert result.config.dtype == torch.float32


def test_rir_result_rejects_shape_that_conflicts_with_scene() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0], [1.5, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    with pytest.raises(ValueError, match="scene has 2"):
        RIRResult(
            rirs=torch.zeros(1, 1, 32),
            scene=scene,
            config=SimulationConfig(max_order=0, nsample=32),
        )


def test_dynamic_scene_passes_entity_orientation_to_simulator() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    sources = Source.from_positions([[1.0, 1.0]], orientation=[1.0, 0.0])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]], orientation=[-1.0, 0.0])
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
        mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        timestamps=[0.0, 0.01],
    )
    result = simulate(
        scene,
        SimulationConfig(
            max_order=0,
            nsample=128,
            directivity=("cardioid", "cardioid"),
        ),
    )
    assert result.rirs.shape == (2, 1, 1, 128)
    assert result.timestamps is not None
    assert scene.timestamps is not None
    assert torch.equal(result.timestamps, scene.timestamps)


def test_dynamic_scene_rejects_non_monotonic_timestamps() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
            timestamps=[0.0, 0.0],
        )
