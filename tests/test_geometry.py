from __future__ import annotations

import random

import pytest
import torch

from torchrir.geometry import (
    binaural_array,
    circular_array,
    eigenmike_em32,
    eigenmike_em64,
    linear_array,
    linear_trajectory,
    polyhedron_array,
)
from torchrir.geometry.sampling import (
    clamp_positions,
    sample_positions,
    sample_positions_min_distance,
    sample_positions_with_z_range,
)


@pytest.mark.numerical
def test_linear_trajectory_has_exact_endpoints_and_equal_steps() -> None:
    start = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    end = torch.tensor([5.0, 4.0, 1.0], dtype=torch.float64)
    trajectory = linear_trajectory(start, end, 5)
    torch.testing.assert_close(trajectory[0], start, rtol=0, atol=0)
    torch.testing.assert_close(trajectory[-1], end, rtol=0, atol=0)
    deltas = trajectory[1:] - trajectory[:-1]
    torch.testing.assert_close(deltas, deltas[0].expand_as(deltas), rtol=0, atol=0)


@pytest.mark.numerical
def test_linear_and_binaural_arrays_match_exact_layouts() -> None:
    linear = linear_array([1.0, 2.0, 3.0], num=4, spacing=0.2, axis=1)
    expected_linear = torch.tensor(
        [[1.0, 1.7, 3.0], [1.0, 1.9, 3.0], [1.0, 2.1, 3.0], [1.0, 2.3, 3.0]]
    )
    torch.testing.assert_close(linear, expected_linear)
    torch.testing.assert_close(
        binaural_array([1.0, 2.0], offset=0.1),
        torch.tensor([[0.9, 2.0], [1.1, 2.0]]),
    )


@pytest.mark.numerical
def test_circular_array_has_requested_radius_and_plane() -> None:
    center = torch.tensor([1.0, 2.0, 3.0])
    positions = circular_array(center, num=8, radius=0.25, plane="xz")
    radii = torch.linalg.vector_norm(positions - center, dim=-1)
    torch.testing.assert_close(radii, torch.full((8,), 0.25))
    torch.testing.assert_close(positions[:, 1], torch.full((8,), 2.0))


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("kind", "count"),
    [
        ("tetrahedron", 4),
        ("cube", 8),
        ("octahedron", 6),
        ("dodecahedron", 20),
        ("icosahedron", 12),
    ],
)
def test_polyhedron_arrays_have_expected_counts_and_radius(
    kind: str, count: int
) -> None:
    center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    positions = polyhedron_array(center, kind=kind, radius=0.2)
    assert positions.shape == (count, 3)
    torch.testing.assert_close(
        torch.linalg.vector_norm(positions - center, dim=-1),
        torch.full((count,), 0.2, dtype=torch.float64),
    )


def test_array_factories_validate_scalar_and_layout_arguments() -> None:
    with pytest.raises(ValueError, match="num must be positive"):
        linear_array([0.0, 0.0], num=0, spacing=0.1)
    with pytest.raises(ValueError, match="spacing"):
        linear_array([0.0, 0.0], num=2, spacing=0.0)
    with pytest.raises(ValueError, match="axis out of range"):
        linear_array([0.0, 0.0], num=2, spacing=0.1, axis=2)
    with pytest.raises(ValueError, match="direction must match"):
        linear_array([0.0, 0.0], num=2, spacing=0.1, direction=[1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="plane"):
        circular_array([0.0, 0.0, 0.0], num=4, radius=0.1, plane="invalid")
    with pytest.raises(ValueError, match="one of"):
        polyhedron_array([0.0, 0.0, 0.0], kind="invalid")
    with pytest.raises(ValueError, match="3"):
        polyhedron_array([0.0, 0.0], kind="cube")


def test_linear_trajectory_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        linear_trajectory(torch.zeros(3), torch.ones(3), 1)


def test_linear_trajectory_validates_endpoints() -> None:
    with pytest.raises(ValueError, match="matching shapes"):
        linear_trajectory(torch.zeros(2), torch.ones(3), 2)
    with pytest.raises(ValueError, match="same device"):
        linear_trajectory(torch.zeros(2), torch.ones(2, device="meta"), 2)
    with pytest.raises(ValueError, match="same dtype"):
        linear_trajectory(
            torch.zeros(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            2,
        )
    with pytest.raises(ValueError, match="finite"):
        linear_trajectory(torch.tensor([0.0, float("nan")]), torch.ones(2), 2)


def test_array_factories_reject_zero_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        linear_array(
            [0.0, 0.0, 0.0],
            num=2,
            spacing=0.1,
            direction=[0.0, 0.0, 0.0],
        )
    with pytest.raises(ValueError, match="non-zero"):
        circular_array([0.0, 0.0, 0.0], num=4, radius=0.1, normal=[0.0, 0.0, 0.0])


def test_circular_array_supports_negative_axis_normal() -> None:
    positions = circular_array(
        [0.0, 0.0, 0.0], num=4, radius=0.1, normal=[-1.0, 0.0, 0.0]
    )
    assert torch.all(torch.isfinite(positions))


def test_eigenmike_tables_preserve_capsule_counts_and_radius() -> None:
    center = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    for factory, count in ((eigenmike_em32, 32), (eigenmike_em64, 64)):
        positions = factory(center)
        assert positions.shape == (count, 3)
        radii = torch.linalg.vector_norm(positions - center, dim=-1)
        assert torch.allclose(radii, torch.full_like(radii, 0.042))


def test_sampling_preserves_device_and_dtype() -> None:
    room = torch.tensor([4.0, 3.0, 2.0], dtype=torch.float64)
    positions = sample_positions(num=3, room_size=room, rng=random.Random(1))
    assert positions.shape == (3, 3)
    assert positions.dtype == torch.float64
    assert positions.device == room.device


def test_sampling_rejects_impossible_margin() -> None:
    with pytest.raises(ValueError, match="no feasible"):
        sample_positions(
            num=1,
            room_size=torch.tensor([1.0, 1.0]),
            rng=random.Random(1),
            margin=0.5,
        )


@pytest.mark.numerical
def test_sampling_respects_margin_z_range_and_minimum_distance() -> None:
    room = torch.tensor([6.0, 5.0, 3.0], dtype=torch.float64)
    positions = sample_positions_with_z_range(
        num=20,
        room_size=room,
        rng=random.Random(3),
        margin=0.5,
        z_range=(1.2, 1.4),
    )
    assert torch.all(positions >= 0.5)
    assert torch.all(positions <= room - 0.5)
    assert torch.all((positions[:, 2] >= 1.2) & (positions[:, 2] <= 1.4))

    center = torch.tensor([3.0, 2.5, 1.5], dtype=torch.float64)
    separated = sample_positions_min_distance(
        num=10,
        room_size=room,
        rng=random.Random(4),
        center=center,
        min_distance=1.0,
        z_range=None,
    )
    assert torch.all(torch.linalg.vector_norm(separated - center, dim=-1) >= 1.0)


@pytest.mark.numerical
def test_clamp_positions_enforces_both_room_margins() -> None:
    positions = torch.tensor([[-1.0, 1.0], [3.9, 8.0], [2.0, 2.0]])
    clamped = clamp_positions(positions, torch.tensor([4.0, 3.0]), margin=0.25)
    expected = torch.tensor([[0.25, 1.0], [3.75, 2.75], [2.0, 2.0]])
    torch.testing.assert_close(clamped, expected)


def test_geometry_rejects_non_finite_scalars() -> None:
    with pytest.raises(ValueError, match="finite"):
        circular_array([0.0, 0.0, 0.0], num=4, radius=float("nan"))
    with pytest.raises(ValueError, match="margin"):
        sample_positions(
            num=1,
            room_size=torch.tensor([4.0, 3.0]),
            rng=random.Random(1),
            margin=float("nan"),
        )
