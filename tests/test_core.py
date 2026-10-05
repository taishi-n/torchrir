"""Public scene-oriented simulation contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import RIRHighPassConfig, SimulationConfig
from torchrir.sim import simulate


def _simulate_static(
    room: Room,
    sources: Source,
    microphones: MicrophoneArray,
    config: SimulationConfig,
) -> torch.Tensor:
    scene = StaticScene(room=room, sources=sources, mics=microphones)
    return simulate(scene, config).rirs


def _simulate_dynamic(
    room: Room,
    source_trajectory: torch.Tensor,
    microphone_trajectory: torch.Tensor,
    config: SimulationConfig,
    *,
    sources: Source | None = None,
    microphones: MicrophoneArray | None = None,
) -> torch.Tensor:
    source_first = (
        source_trajectory[0:1] if source_trajectory.ndim == 2 else source_trajectory[0]
    )
    microphone_first = (
        microphone_trajectory[0:1]
        if microphone_trajectory.ndim == 2
        else microphone_trajectory[0]
    )
    scene = DynamicScene(
        room=room,
        sources=sources or Source.from_positions(source_first),
        mics=microphones or MicrophoneArray.from_positions(microphone_first),
        src_traj=source_trajectory,
        mic_traj=microphone_trajectory,
    )
    return simulate(scene, config).rirs


def test_simulate_rir_shape_and_physical_peak() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    rir = _simulate_static(
        room,
        Source.from_positions([[1.0, 1.0, 1.0]]),
        MicrophoneArray.from_positions([[2.0, 1.0, 1.0]]),
        SimulationConfig(max_order=0, nsample=2048),
    )
    assert rir.shape == (1, 1, 2048)
    expected = room.fs / room.c
    peak = torch.argmax(torch.abs(rir[0, 0])).item()
    assert abs(peak - expected) <= 1.0


@pytest.mark.parametrize("frac_delay_length", [1, 9, 81, 129])
def test_physical_arrival_time_is_independent_of_fractional_delay_length(
    frac_delay_length: int,
) -> None:
    room = Room.shoebox(size=[20.0, 4.0, 3.0], fs=343.0, c=343.0, beta=[0.0] * 6)
    rir = _simulate_static(
        room,
        Source.from_positions([[1.0, 1.0, 1.0]]),
        MicrophoneArray.from_positions([[6.0, 1.0, 1.0]]),
        SimulationConfig(
            max_order=0,
            nsample=160,
            frac_delay_length=frac_delay_length,
            use_lut=False,
        ),
    )
    assert int(torch.argmax(torch.abs(rir[0, 0])).item()) == 5


def test_directional_entity_requires_orientation_at_construction() -> None:
    with pytest.raises(ValueError, match="orientation is required"):
        Source.from_positions([[1.0, 1.0, 1.0]], directivity="cardioid")
    with pytest.raises(ValueError, match="orientation is required"):
        MicrophoneArray.from_positions([[2.0, 1.0, 1.0]], directivity="hypercardioid")


def test_simulate_rir_angle_orientation_2d() -> None:
    room = Room.shoebox(size=[5.0, 4.0], fs=16000, beta=[0.9] * 4)
    rir = _simulate_static(
        room,
        Source.from_positions(
            [[1.0, 1.0]],
            orientation=torch.tensor(0.0),
            directivity="cardioid",
        ),
        MicrophoneArray.from_positions([[2.0, 1.0]]),
        SimulationConfig(max_order=0, nsample=256),
    )
    assert rir.shape == (1, 1, 256)


def test_simulate_dynamic_rir_shape() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    source = torch.tensor([[[1.0, 1.0, 1.0]], [[1.5, 1.0, 1.0]], [[2.0, 1.0, 1.0]]])
    microphone = torch.tensor([[[2.5, 1.0, 1.0]], [[2.5, 1.2, 1.0]], [[2.5, 1.4, 1.0]]])
    rirs = _simulate_dynamic(
        room,
        source,
        microphone,
        SimulationConfig(max_order=0, nsample=512),
    )
    assert rirs.shape == (3, 1, 1, 512)


def test_dynamic_zero_phase_hpf_uses_each_frames_natural_horizon() -> None:
    pytest.importorskip("scipy.signal")
    room = Room.shoebox(
        size=[5.0, 4.0, 3.0],
        fs=16000,
        beta=[0.9] * 6,
        dtype=torch.float64,
    )
    source_trajectory = torch.tensor(
        [[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]],
        dtype=torch.float64,
    )
    microphone_trajectory = torch.tensor(
        [[[3.0, 1.0, 1.0]], [[4.0, 1.0, 1.0]]],
        dtype=torch.float64,
    )
    config = SimulationConfig(
        max_order=2,
        nsample=1024,
        use_lut=False,
        dtype=torch.float64,
        high_pass=RIRHighPassConfig(),
    )
    dynamic = _simulate_dynamic(
        room,
        source_trajectory,
        microphone_trajectory,
        config,
    )

    for frame in range(2):
        static = _simulate_static(
            room,
            Source.from_positions(source_trajectory[frame], dtype=torch.float64),
            MicrophoneArray.from_positions(
                microphone_trajectory[frame],
                dtype=torch.float64,
            ),
            config,
        )
        torch.testing.assert_close(dynamic[frame], static, rtol=0, atol=0)


def test_dynamic_accepts_2d_single_entity_trajectory() -> None:
    room = Room.shoebox(size=[5.0, 4.0], fs=8000, beta=[0.9] * 4)
    source = torch.tensor([[1.0, 1.0], [1.5, 1.0], [2.0, 1.0]])
    microphone = torch.tensor([[2.5, 1.0], [2.5, 1.2], [2.5, 1.4]])
    rirs = _simulate_dynamic(
        room,
        source,
        microphone,
        SimulationConfig(max_order=0, nsample=256),
    )
    assert rirs.shape == (3, 1, 1, 256)


def test_dynamic_simulation_accepts_nb_img() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    source = torch.tensor([[[1.0, 1.0, 1.0]], [[1.2, 1.0, 1.0]]])
    microphone = torch.tensor([[[2.5, 1.0, 1.0]], [[2.5, 1.0, 1.0]]])
    rirs = _simulate_dynamic(
        room,
        source,
        microphone,
        SimulationConfig(nb_img=(0, 0, 0), nsample=256),
    )
    assert rirs.shape == (2, 1, 1, 256)


def test_simulation_rejects_fractional_nb_img_at_config_boundary() -> None:
    with pytest.raises(TypeError, match="contain integers"):
        SimulationConfig(nb_img=cast(Any, (0.5, 0, 0)), nsample=32)


@pytest.mark.parametrize("directivity", ["omni", "cardioid"])
def test_simulation_rejects_coincident_source_and_microphone(
    directivity: str,
) -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    source = Source.from_positions(
        [[1.0, 1.0, 1.0]],
        orientation=[1.0, 0.0, 0.0] if directivity != "omni" else None,
        directivity=directivity,
    )
    with pytest.raises(ValueError, match="source--microphone distance"):
        _simulate_static(
            room,
            source,
            MicrophoneArray.from_positions([[1.0, 1.0, 1.0]]),
            SimulationConfig(max_order=0, nsample=128),
        )


def test_dynamic_simulation_reports_coincident_frame_indices() -> None:
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    source = torch.tensor([[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]])
    microphone = torch.tensor([[[3.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]])
    with pytest.raises(ValueError, match="frame 1, source 0, microphone 0"):
        _simulate_dynamic(
            room,
            source,
            microphone,
            SimulationConfig(max_order=0, nsample=128),
        )


def test_physical_time_origin_keeps_arrival_at_requested_horizon() -> None:
    room = Room.shoebox(size=[100.0, 10.0, 10.0], fs=343.0, c=343.0, beta=[0.0] * 6)
    rir = _simulate_static(
        room,
        Source.from_positions([[1.0, 1.0, 1.0]]),
        MicrophoneArray.from_positions([[64.0, 1.0, 1.0]]),
        SimulationConfig(
            max_order=0,
            nsample=64,
            frac_delay_length=9,
        ),
    )
    assert int(torch.argmax(torch.abs(rir[0, 0])).item()) == 63
    assert rir[0, 0, 63].item() == pytest.approx(1.0 / 63.0)


def test_simulate_rir_hpf_changes_output_when_enabled() -> None:
    pytest.importorskip("scipy.signal")
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    microphones = MicrophoneArray.from_positions([[2.0, 1.0, 1.0]])
    no_hpf = _simulate_static(
        room,
        sources,
        microphones,
        SimulationConfig(max_order=3, nsample=1024),
    )
    with_hpf = _simulate_static(
        room,
        sources,
        microphones,
        SimulationConfig(
            max_order=3,
            nsample=1024,
            high_pass=RIRHighPassConfig(cutoff_hz=10.0, order=2),
        ),
    )
    assert no_hpf.shape == with_hpf.shape
    assert not torch.allclose(no_hpf, with_hpf)
    assert abs(with_hpf.mean().item()) < 0.25 * abs(no_hpf.mean().item())


def test_simulate_rir_reports_short_zero_phase_hpf_input_clearly() -> None:
    pytest.importorskip("scipy.signal")
    room = Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000)
    with pytest.raises(ValueError, match="too short"):
        _simulate_static(
            room,
            Source.from_positions([[1.0, 1.0, 1.0]]),
            MicrophoneArray.from_positions([[2.0, 1.0, 1.0]]),
            SimulationConfig(
                max_order=0,
                nsample=4,
                high_pass=RIRHighPassConfig(phase="zero_phase"),
            ),
        )


def test_simulate_rejects_non_scene_and_non_config_inputs() -> None:
    room = Room.shoebox([4.0, 3.0], fs=8000)
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]]),
    )
    with pytest.raises(TypeError, match="scene"):
        simulate(cast(Any, object()), SimulationConfig(max_order=0, nsample=16))
    with pytest.raises(TypeError, match="config"):
        simulate(scene, cast(Any, object()))
