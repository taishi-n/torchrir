from __future__ import annotations

import random

import pytest
import torch

from torchrir.geometry import (
    circular_array,
    eigenmike_em32,
    eigenmike_em64,
    linear_array,
    linear_trajectory,
)
from torchrir.geometry.sampling import sample_positions


def test_linear_trajectory_rejects_too_few_steps() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        linear_trajectory(torch.zeros(3), torch.ones(3), 1)


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
