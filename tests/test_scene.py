"""Scene validation and scene-oriented simulation contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

import torchrir.models as models
from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import ResolvedSimulationConfig, SimulationConfig
from torchrir.signal import FrameSchedule
from torchrir.sim import simulate


def test_scene_public_surface_excludes_removed_type_helpers() -> None:
    assert "SceneLike" not in models.__all__
    assert not hasattr(models, "SceneLike")
    assert not hasattr(StaticScene, "is_dynamic")
    assert not hasattr(DynamicScene, "is_dynamic")


def test_static_scene_validates_geometry() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.5, 1.0]]),
    )
    scene.validate()


def test_dynamic_scene_validates_matching_trajectories() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    microphones = MicrophoneArray.from_positions([[2.0, 1.5, 1.0]])
    source_trajectory = torch.tensor([[[1.0, 1.0, 1.0]], [[1.5, 1.0, 1.0]]])
    microphone_trajectory = torch.tensor([[[2.0, 1.5, 1.0]], [[2.2, 1.5, 1.0]]])
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=microphones,
        src_traj=source_trajectory,
        mic_traj=microphone_trajectory,
    )
    scene.validate()


def test_dynamic_scene_rejects_mismatched_time_steps() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    with pytest.raises(ValueError, match="matching time steps"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.5, 1.0]]),
            src_traj=[[[1.0, 1.0, 1.0]], [[1.5, 1.0, 1.0]]],
            mic_traj=[
                [[2.0, 1.5, 1.0]],
                [[2.2, 1.5, 1.0]],
                [[2.4, 1.5, 1.0]],
            ],
        )


def test_dynamic_scene_normalizes_tensor_like_trajectories() -> None:
    room = Room.shoebox(size=[4.0, 3.0, 2.5], fs=16000, beta=[0.9] * 6)
    scene = DynamicScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.5, 1.0]]),
        src_traj=cast(Any, [[[1.0, 1.0, 1.0]], [[1.1, 1.0, 1.0]]]),
        mic_traj=cast(Any, [[[2.0, 1.5, 1.0]], [[2.0, 1.5, 1.0]]]),
    )
    assert torch.is_tensor(scene.src_traj)
    assert torch.is_tensor(scene.mic_traj)
    assert scene.src_traj.dtype == scene.sources.positions.dtype
    assert scene.mic_traj.device == scene.room.size.device


def test_scene_rejects_implicit_mixed_geometry_dtypes() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, dtype=torch.float64)
    with pytest.raises(ValueError, match="source positions dtype"):
        StaticScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]], dtype=torch.float32),
            mics=MicrophoneArray.from_positions(
                [[2.0, 1.0]],
                dtype=torch.float64,
            ),
        )


def test_dynamic_scene_preserves_tensor_dtype_and_rejects_mismatch() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    sources = Source.from_positions([[1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]])
    source_trajectory = torch.tensor(
        [[[1.0, 1.0]], [[1.1, 1.0]]],
        dtype=torch.float64,
    )
    with pytest.raises(ValueError, match="src_traj dtype"):
        DynamicScene(
            room=room,
            sources=sources,
            mics=mics,
            src_traj=source_trajectory,
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        )
    assert source_trajectory.dtype == torch.float64


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


def test_kernel_and_result_share_one_resolved_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchrir.sim.simulators as simulator_module

    room = Room.shoebox([4.0, 3.0], fs=8000)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    captured: list[ResolvedSimulationConfig] = []

    def fake_kernel(
        received_scene: StaticScene, resolved: ResolvedSimulationConfig
    ) -> torch.Tensor:
        assert received_scene is scene
        captured.append(resolved)
        return torch.zeros(1, 1, resolved.nsample, dtype=resolved.dtype)

    monkeypatch.setattr(simulator_module, "_simulate_static_rir", fake_kernel)
    result = simulate(scene, SimulationConfig(max_order=0, nsample=32))
    assert captured == [result.config]
    assert captured[0] is result.config


def test_dynamic_scene_directivity_is_read_from_entities() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    sources = Source.from_positions(
        [[1.0, 1.0]],
        orientation=[1.0, 0.0],
        directivity="cardioid",
    )
    microphones = MicrophoneArray.from_positions(
        [[2.0, 1.0]],
        orientation=[-1.0, 0.0],
        directivity="cardioid",
    )
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=microphones,
        src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
        mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        schedule=FrameSchedule.from_samples([0, 80]),
    )
    result = simulate(scene, SimulationConfig(max_order=0, nsample=128))
    assert result.rirs.shape == (2, 1, 1, 128)
    assert result.scene is scene
    assert scene.schedule is not None


def test_dynamic_scene_rejects_schedule_frame_count_mismatch() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    with pytest.raises(ValueError, match="schedule has 1 frames"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
            schedule=FrameSchedule.from_samples([0]),
        )


def test_dynamic_scene_rejects_seconds_conversion_rate_mismatch() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    with pytest.raises(
        ValueError, match="seconds-conversion sample rate.*room sampling rate"
    ):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
            schedule=FrameSchedule.from_seconds([0.0, 80 / 16000], sample_rate=16000),
        )


def test_dynamic_scene_requires_frame_schedule_type() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000, beta=[0.9] * 4)
    with pytest.raises(TypeError, match="FrameSchedule"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
            schedule=cast(Any, [0, 1]),
        )


def test_dynamic_scene_requires_exact_first_frame_identity() -> None:
    room = Room.shoebox(size=[4.0, 3.0], fs=8000)
    with pytest.raises(ValueError, match="first src_traj frame"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.000001, 1.0]], [[1.2, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        )


def test_simulate_revalidates_mutated_room_size() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, beta=[0.8] * 4)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    room.size.fill_(float("nan"))
    with pytest.raises(ValueError, match="room size must contain finite"):
        simulate(scene, SimulationConfig(max_order=0, nsample=32))


def test_simulate_revalidates_mutated_beta() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000, beta=[0.8] * 4)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    assert room.beta is not None
    room.beta.fill_(2.0)
    with pytest.raises(ValueError, match="beta values"):
        simulate(scene, SimulationConfig(max_order=0, nsample=32))


def test_simulate_revalidates_mutated_canonical_orientation() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    sources = Source.from_positions(
        [[1.0, 1.0]],
        orientation=[1.0, 0.0],
        directivity="cardioid",
    )
    scene = StaticScene(
        room=room,
        sources=sources,
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    assert sources.orientation is not None
    sources.orientation.zero_()
    with pytest.raises(ValueError, match="unit vectors"):
        simulate(scene, SimulationConfig(max_order=0, nsample=32))


@pytest.mark.parametrize(
    ("source_position", "microphone_position"),
    [
        ([0.0, 1.0], [2.0, 1.0]),
        ([1.0, 1.0], [4.0, 1.0]),
    ],
)
def test_static_scene_rejects_entities_on_room_walls(
    source_position: list[float], microphone_position: list[float]
) -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    with pytest.raises(ValueError, match="strictly within room bounds"):
        StaticScene(
            room=room,
            sources=Source.from_positions([source_position]),
            mics=MicrophoneArray.from_positions([microphone_position]),
        )


def test_dynamic_scene_rejects_trajectory_point_on_room_wall() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    with pytest.raises(ValueError, match="strictly within room bounds"):
        DynamicScene(
            room=room,
            sources=Source.from_positions([[1.0, 1.0]]),
            mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
            src_traj=[[[1.0, 1.0]], [[4.0, 1.0]]],
            mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        )
