"""Physical and metamorphic invariants for RIR simulation."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate


def _config(**updates: object) -> SimulationConfig:
    values: dict[str, object] = {
        "max_order": 2,
        "nsample": 512,
        "use_lut": False,
        "rir_hpf_enable": False,
    }
    values.update(updates)
    return SimulationConfig(**cast(Any, values))


def _static_rirs(
    room: Room,
    source_positions: list[list[float]],
    microphone_positions: list[list[float]],
    config: SimulationConfig,
) -> torch.Tensor:
    scene = StaticScene(
        room=room,
        sources=Source.from_positions(source_positions, dtype=room.size.dtype),
        mics=MicrophoneArray.from_positions(
            microphone_positions, dtype=room.size.dtype
        ),
    )
    return simulate(scene, config).rirs


@pytest.mark.numerical
def test_direct_path_is_independent_of_room_reflection_parameters() -> None:
    config = _config(max_order=0)
    positions = ([[1.0, 1.0, 1.0]], [[3.0, 2.0, 1.5]])
    absorbing = Room.shoebox([5.0, 4.0, 3.0], fs=8000, beta=[0.0] * 6)
    reflective = Room.shoebox([5.0, 4.0, 3.0], fs=8000, beta=[1.0] * 6)
    torch.testing.assert_close(
        _static_rirs(absorbing, *positions, config),
        _static_rirs(reflective, *positions, config),
        rtol=0,
        atol=0,
    )


@pytest.mark.numerical
def test_direct_path_is_translation_invariant() -> None:
    config = _config(max_order=0)
    base = _static_rirs(
        Room.shoebox([5.0, 4.0, 3.0], fs=8000),
        [[1.0, 1.0, 1.0]],
        [[3.0, 2.0, 1.5]],
        config,
    )
    shifted = _static_rirs(
        Room.shoebox([9.0, 8.0, 7.0], fs=8000),
        [[3.0, 2.0, 2.0]],
        [[5.0, 3.0, 2.5]],
        config,
    )
    torch.testing.assert_close(base, shifted, rtol=0, atol=0)


@pytest.mark.numerical
def test_source_and_microphone_permutations_only_permute_output_axes() -> None:
    room = Room.shoebox(
        [6.0, 5.0, 3.0],
        fs=8000,
        beta=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    sources = [[1.0, 1.0, 1.0], [2.0, 3.0, 1.5]]
    microphones = [[4.0, 1.0, 1.2], [3.0, 4.0, 1.8], [5.0, 2.0, 2.0]]
    config = _config()
    original = _static_rirs(room, sources, microphones, config)
    permuted = _static_rirs(
        room,
        [sources[1], sources[0]],
        [microphones[2], microphones[0], microphones[1]],
        config,
    )
    expected = original[[1, 0]][:, [2, 0, 1]]
    torch.testing.assert_close(permuted, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.numerical
def test_omnidirectional_static_rir_is_reciprocal() -> None:
    room = Room.shoebox(
        [6.0, 5.0, 3.0],
        fs=8000,
        beta=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        dtype=torch.float64,
    )
    left = [[1.0, 1.0, 1.0], [2.0, 3.0, 1.5]]
    right = [[4.0, 1.0, 1.2], [3.0, 4.0, 1.8]]
    config = _config(dtype=torch.float64)
    forward = _static_rirs(room, left, right, config)
    reverse = _static_rirs(room, right, left, config)
    torch.testing.assert_close(forward, reverse.transpose(0, 1), rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
def test_repeated_dynamic_frames_equal_static_rir() -> None:
    room = Room.shoebox([5.0, 4.0, 3.0], fs=8000, beta=[0.8] * 6)
    sources = Source.from_positions([[1.0, 1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[3.0, 2.0, 1.5]])
    config = _config()
    static = simulate(StaticScene(room=room, sources=sources, mics=mics), config).rirs
    dynamic_scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=sources.positions.unsqueeze(0).repeat(3, 1, 1),
        mic_traj=mics.positions.unsqueeze(0).repeat(3, 1, 1),
        timestamps=[0.0, 0.1, 0.2],
    )
    dynamic = simulate(dynamic_scene, config).rirs
    torch.testing.assert_close(dynamic, static.unsqueeze(0).expand_as(dynamic))


@pytest.mark.numerical
def test_image_and_accumulation_chunk_sizes_do_not_change_rir() -> None:
    room = Room.shoebox([5.0, 4.0, 3.0], fs=8000, beta=[0.8] * 6)
    sources = [[1.0, 1.0, 1.0], [2.0, 1.5, 1.5]]
    microphones = [[3.0, 2.0, 1.5], [4.0, 3.0, 2.0]]
    large_chunks = _static_rirs(
        room,
        sources,
        microphones,
        _config(max_order=3, image_chunk_size=2048, accumulate_chunk_size=4096),
    )
    unit_chunks = _static_rirs(
        room,
        sources,
        microphones,
        _config(max_order=3, image_chunk_size=1, accumulate_chunk_size=1),
    )
    torch.testing.assert_close(large_chunks, unit_chunks, rtol=0, atol=0)


@pytest.mark.numerical
def test_diffuse_tail_seed_controls_only_the_tail() -> None:
    room = Room.shoebox([5.0, 4.0, 3.0], fs=8000, beta=[0.8] * 6)
    source_positions = [[1.0, 1.0, 1.0]]
    microphone_positions = [[3.0, 2.0, 1.5]]
    common = dict(max_order=2, nsample=512, tdiff=0.03)
    first = _static_rirs(
        room,
        source_positions,
        microphone_positions,
        _config(**common, seed=7),
    )
    repeated = _static_rirs(
        room,
        source_positions,
        microphone_positions,
        _config(**common, seed=7),
    )
    changed = _static_rirs(
        room,
        source_positions,
        microphone_positions,
        _config(**common, seed=8),
    )
    split = int(0.03 * room.fs)
    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first[..., :split], changed[..., :split], rtol=0, atol=0)
    assert not torch.equal(first[..., split:], changed[..., split:])


@pytest.mark.numerical
def test_public_api_flips_source_directivity_for_reflected_paths() -> None:
    # Only the y-low wall reflects. Its image is behind a +y-facing cardioid,
    # so first-order simulation must reduce exactly to the direct path.
    room = Room.shoebox(
        [10.0, 10.0, 10.0],
        fs=343.0,
        c=343.0,
        beta=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    scene = StaticScene(
        room=room,
        sources=Source.from_positions([[5.0, 2.0, 5.0]], orientation=[0.0, 1.0, 0.0]),
        mics=MicrophoneArray.from_positions([[5.0, 8.0, 5.0]]),
    )
    direct = simulate(
        scene,
        SimulationConfig(
            max_order=0,
            nsample=128,
            directivity=("cardioid", "omni"),
            use_lut=False,
            rir_hpf_enable=False,
        ),
    ).rirs
    first_order = simulate(
        scene,
        SimulationConfig(
            max_order=1,
            nsample=128,
            directivity=("cardioid", "omni"),
            use_lut=False,
            rir_hpf_enable=False,
        ),
    ).rirs
    torch.testing.assert_close(first_order, direct, rtol=0, atol=1e-7)

    omni_scene = StaticScene(
        room=room,
        sources=Source.from_positions([[5.0, 2.0, 5.0]]),
        mics=scene.mics,
    )
    omni = simulate(
        omni_scene,
        SimulationConfig(
            max_order=1,
            nsample=128,
            use_lut=False,
            rir_hpf_enable=False,
        ),
    ).rirs
    assert not torch.allclose(omni, direct)
