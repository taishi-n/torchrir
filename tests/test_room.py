import math

import pytest
import torch

from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate


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
    with pytest.raises(ValueError, match="too short"):
        Room.shoebox(size=[5.0, 4.0, 3.0], fs=16000, t60=1.0e-4)
    with pytest.raises(ValueError, match="fs must be positive"):
        Room.shoebox(size=[4.0, 3.0], fs=10**400)
    with pytest.raises(TypeError, match="fs must be a real number"):
        Room.shoebox(size=[4.0, 3.0], fs="16000")  # type: ignore[arg-type]


def test_room_beta_validation():
    with pytest.raises(ValueError, match="beta must have 6 elements"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=[0.9] * 4)
    with pytest.raises(ValueError, match="beta values must be in \\[0, 1\\]"):
        Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=[1.1] * 6)


def test_room_accepts_noncontiguous_beta_tensor() -> None:
    beta = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]).transpose(0, 1)
    assert not beta.is_contiguous()
    room = Room.shoebox(size=[4.0, 3.0, 2.0], fs=16000, beta=beta)
    assert room.beta is not None
    torch.testing.assert_close(room.beta, beta.reshape(-1))


def test_integer_geometry_is_promoted_to_floating_point() -> None:
    room = Room.shoebox(size=[4, 3, 2], fs=16000, beta=[0.9] * 6)
    sources = Source.from_positions([[1, 1, 1]])
    mics = MicrophoneArray.from_positions([[2, 1, 1]])
    scene = StaticScene(room=room, sources=sources, mics=mics)
    rir = simulate(
        scene,
        SimulationConfig(max_order=0, nsample=128),
    ).rirs
    assert room.size.dtype == torch.float32
    assert sources.positions.dtype == torch.float32
    assert rir.dtype == torch.float32


def test_geometry_models_reject_unsupported_float8() -> None:
    with pytest.raises(TypeError, match="supported"):
        Room.shoebox(
            size=torch.tensor([4.0, 3.0], dtype=torch.float8_e4m3fn),
            fs=16000,
        )
    with pytest.raises(TypeError, match="supported"):
        Source.from_positions(torch.tensor([[1.0, 1.0]], dtype=torch.float8_e4m3fn))


@pytest.mark.parametrize("scene_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("execution_dtype", [torch.float32, torch.float64])
def test_public_simulation_rejects_low_precision_scene_before_upcast(
    scene_dtype: torch.dtype,
    execution_dtype: torch.dtype,
) -> None:
    scene = StaticScene(
        room=Room.shoebox([4.0, 3.0], fs=8000, dtype=scene_dtype),
        sources=Source.from_positions([[1.0, 1.0]], dtype=scene_dtype),
        mics=MicrophoneArray.from_positions([[2.0, 1.0]], dtype=scene_dtype),
    )
    with pytest.raises(TypeError, match="simulation scene tensors"):
        simulate(
            scene,
            SimulationConfig(
                max_order=0,
                nsample=64,
                dtype=execution_dtype,
            ),
        )


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


def test_entity_directivity_is_validated_and_canonicalized() -> None:
    source = Source.from_positions(
        [[1.0, 1.0]], orientation=[1.0, 0.0], directivity="card"
    )
    microphone = MicrophoneArray.from_positions(
        [[2.0, 1.0]], orientation=[-1.0, 0.0], directivity="figure-8"
    )
    assert source.directivity == "cardioid"
    assert microphone.directivity == "bidir"
    assert source.orientation is not None
    assert microphone.orientation is not None
    assert source.orientation.shape == (1, 2)
    assert microphone.orientation.shape == (1, 2)
    with pytest.raises(ValueError, match="unsupported source directivity"):
        Source.from_positions(
            [[1.0, 1.0]], orientation=[1.0, 0.0], directivity="unknown"
        )


def test_2d_per_entity_angles_are_unambiguous_and_canonical() -> None:
    sources = Source.from_positions(
        [[1.0, 1.0], [2.0, 1.0]],
        orientation=[[0.0], [math.pi]],
        directivity="cardioid",
        dtype=torch.float64,
    )
    assert sources.orientation is not None
    assert sources.orientation.shape == (2, 2)
    torch.testing.assert_close(
        sources.orientation,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )


def test_shared_orientation_is_expanded_to_each_entity() -> None:
    mics = MicrophoneArray.from_positions(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        orientation=[0.0, 2.0],
        directivity="cardioid",
    )
    assert mics.orientation is not None
    torch.testing.assert_close(
        mics.orientation,
        torch.tensor([[0.0, 1.0]]).expand(3, -1),
    )


def test_entity_rejects_ambiguous_flat_per_entity_angles() -> None:
    with pytest.raises(ValueError, match="shared orientation"):
        Source.from_positions(
            [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
            orientation=[0.0, math.pi / 2, math.pi],
            directivity="cardioid",
        )


def test_entity_rejects_mixed_orientation_dtype_without_override() -> None:
    with pytest.raises(ValueError, match="orientation device and dtype"):
        Source.from_positions(
            torch.tensor([[1.0, 1.0]], dtype=torch.float64),
            orientation=torch.tensor([1.0, 0.0], dtype=torch.float32),
            directivity="cardioid",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_orientation_survives_canonical_validation(
    dtype: torch.dtype,
) -> None:
    source = Source.from_positions(
        [[1.0, 1.0, 1.0]],
        orientation=[1.0, 2.0, 3.0],
        directivity="cardioid",
        dtype=dtype,
    )
    assert source.orientation is not None
    tolerance = 4.0 * torch.finfo(dtype).eps
    torch.testing.assert_close(
        torch.linalg.vector_norm(source.orientation, dim=-1),
        torch.ones(1, dtype=dtype),
        rtol=0,
        atol=tolerance,
    )
