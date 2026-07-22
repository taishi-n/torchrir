from __future__ import annotations

import math
import random
from typing import Any, cast

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
    trajectory = linear_trajectory(
        start,
        end,
        progress=torch.linspace(0.0, 1.0, 5, dtype=torch.float64),
    )
    torch.testing.assert_close(trajectory[0], start, rtol=0, atol=0)
    torch.testing.assert_close(trajectory[-1], end, rtol=0, atol=0)
    deltas = trajectory[1:] - trajectory[:-1]
    torch.testing.assert_close(deltas, deltas[0].expand_as(deltas), rtol=0, atol=0)


@pytest.mark.numerical
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_linear_trajectory_avoids_finite_endpoint_difference_overflow(
    dtype: torch.dtype,
) -> None:
    limit = torch.finfo(dtype).max
    start = torch.tensor([limit], dtype=dtype)
    end = torch.tensor([-limit], dtype=dtype)
    progress = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=dtype)

    trajectory = linear_trajectory(start, end, progress=progress)

    assert torch.all(torch.isfinite(trajectory))
    torch.testing.assert_close(trajectory[0], start, rtol=0, atol=0)
    torch.testing.assert_close(trajectory[2], torch.zeros_like(start), rtol=0, atol=0)
    torch.testing.assert_close(trajectory[-1], end, rtol=0, atol=0)


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
    with pytest.raises(TypeError, match="num must be an integer"):
        linear_array([0.0, 0.0], num=True, spacing=0.1)
    with pytest.raises(TypeError, match="spacing must be a real number"):
        linear_array([0.0, 0.0], num=2, spacing=cast(Any, "0.1"))
    with pytest.raises(ValueError, match="radius"):
        circular_array([0.0, 0.0], num=4, radius=10**400)
    with pytest.raises(TypeError, match="plane"):
        circular_array(
            [0.0, 0.0, 0.0],
            num=4,
            radius=0.1,
            plane=cast(Any, 1),
        )


def test_array_factories_flatten_single_vector_matrix_inputs() -> None:
    linear = linear_array(
        [0.0, 0.0, 0.0],
        num=2,
        spacing=0.1,
        direction=cast(Any, [[1.0, 0.0, 0.0]]),
    )
    circular = circular_array(
        [0.0, 0.0, 0.0],
        num=4,
        radius=0.1,
        normal=cast(Any, [[0.0, 0.0, 1.0]]),
    )

    assert linear.shape == (2, 3)
    assert circular.shape == (4, 3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_array_factories_reject_scalars_outside_requested_dtype(
    dtype: torch.dtype,
) -> None:
    excessive = float(torch.finfo(dtype).max) * 2.0
    center_2d = torch.zeros(2, dtype=dtype)
    center_3d = torch.zeros(3, dtype=dtype)

    factories = [
        lambda: binaural_array(center_2d, offset=excessive),
        lambda: linear_array(center_2d, num=2, spacing=excessive),
        lambda: circular_array(center_2d, num=4, radius=excessive),
        lambda: polyhedron_array(center_3d, radius=excessive),
        lambda: eigenmike_em32(center_3d, radius=excessive),
        lambda: eigenmike_em64(center_3d, azimuth_offset_deg=excessive),
    ]

    for factory in factories:
        with pytest.raises(ValueError, match="representable"):
            factory()


def test_array_factories_reject_non_finite_requested_dtype_output() -> None:
    limit = torch.finfo(torch.float32).max
    center = torch.tensor([limit, 0.0], dtype=torch.float32)

    with pytest.raises(ValueError, match="array positions must be finite"):
        binaural_array(center, offset=limit)


def test_linear_array_validates_combined_spacing_and_count_range() -> None:
    with pytest.raises(ValueError, match="maximum linear-array offset"):
        linear_array(
            torch.zeros(2, dtype=torch.float32),
            num=4,
            spacing=torch.finfo(torch.float32).max,
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: binaural_array(
            torch.ones(2, dtype=torch.float16),
            offset=1.0e-4,
        ),
        lambda: linear_array(
            torch.ones(2, dtype=torch.float16),
            num=2,
            spacing=1.0e-4,
        ),
        lambda: circular_array(
            torch.ones(2, dtype=torch.float16),
            num=4,
            radius=1.0e-4,
        ),
        lambda: polyhedron_array(
            torch.ones(3, dtype=torch.float16),
            radius=1.0e-4,
        ),
    ],
)
def test_array_factories_reject_dtype_quantized_duplicate_positions(
    factory: Any,
) -> None:
    with pytest.raises(ValueError, match="remain distinct"):
        factory()


def test_variable_size_array_distinctness_checks_do_not_use_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_unique(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("variable-size array validation must not call torch.unique")

    monkeypatch.setattr(torch, "unique", reject_unique)

    assert linear_array([0.0, 0.0], num=4, spacing=0.1).shape == (4, 2)
    assert circular_array([0.0, 0.0], num=8, radius=0.1).shape == (8, 2)


@pytest.mark.numerical
def test_array_basis_normalization_handles_extreme_finite_vectors() -> None:
    center = torch.zeros(3, dtype=torch.float64)
    direction = torch.tensor([1.0e308, 1.0e308, 0.0], dtype=torch.float64)

    linear = linear_array(
        center,
        num=3,
        spacing=0.2,
        direction=direction,
    )

    expected_offset = 0.2 / math.sqrt(2.0)
    expected_linear = torch.tensor(
        [
            [-expected_offset, -expected_offset, 0.0],
            [0.0, 0.0, 0.0],
            [expected_offset, expected_offset, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.all(torch.isfinite(linear))
    torch.testing.assert_close(linear, expected_linear, rtol=1.0e-15, atol=0.0)

    normal = torch.full((3,), 1.0e308, dtype=torch.float64)
    circular = circular_array(center, num=8, radius=0.2, normal=normal)
    offsets = circular - center
    unit_normal = torch.full((3,), 1.0 / math.sqrt(3.0), dtype=torch.float64)

    assert torch.all(torch.isfinite(circular))
    torch.testing.assert_close(
        torch.linalg.vector_norm(offsets, dim=1),
        torch.full((8,), 0.2, dtype=torch.float64),
        rtol=1.0e-15,
        atol=1.0e-15,
    )
    torch.testing.assert_close(
        offsets @ unit_normal,
        torch.zeros(8, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-15,
    )


def test_array_factories_reject_counts_outside_tensor_range() -> None:
    with pytest.raises(ValueError, match="num must be at most"):
        linear_array([0.0, 0.0], num=10**400, spacing=0.1)
    with pytest.raises(ValueError, match="num must be at most"):
        circular_array([0.0, 0.0], num=10**400, radius=0.1)


def test_linear_trajectory_uses_explicit_sample_aligned_progress() -> None:
    start = torch.tensor([0.0, 1.0])
    end = torch.tensor([10.0, 5.0])
    progress = torch.tensor([0.0, 0.2, 0.5, 0.7])

    trajectory = linear_trajectory(start, end, progress=progress)

    expected = start + progress[:, None] * (end - start)
    torch.testing.assert_close(trajectory, expected, rtol=0, atol=0)


def test_linear_trajectory_validates_endpoints() -> None:
    progress = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="matching shapes"):
        linear_trajectory(torch.zeros(2), torch.ones(3), progress=progress)
    with pytest.raises(ValueError, match="CPU, CUDA, or MPS"):
        linear_trajectory(
            torch.zeros(2),
            torch.ones(2, device="meta"),
            progress=progress,
        )
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]),
        torch.tensor([0.0, 1.0]),
        size=(2,),
    )
    with pytest.raises(TypeError, match="dense strided"):
        linear_trajectory(sparse, torch.ones(2), progress=progress)
    with pytest.raises(ValueError, match="same dtype"):
        linear_trajectory(
            torch.zeros(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            progress=progress,
        )
    with pytest.raises(ValueError, match="finite"):
        linear_trajectory(
            torch.tensor([0.0, float("nan")]),
            torch.ones(2),
            progress=progress,
        )
    with pytest.raises(ValueError, match="non-empty"):
        linear_trajectory(torch.empty(0), torch.empty(0), progress=progress)
    with pytest.raises(TypeError, match="real floating-point"):
        linear_trajectory(
            torch.tensor([1 + 2j]),
            torch.tensor([2 + 3j]),
            progress=progress,
        )
    with pytest.raises(TypeError, match="boolean"):
        linear_trajectory(
            torch.tensor([False, True]),
            torch.tensor([True, False]),
            progress=progress,
        )
    with pytest.raises(TypeError, match="Tensors"):
        linear_trajectory(cast(Any, [0.0]), torch.ones(1), progress=progress)


@pytest.mark.parametrize(
    ("progress", "error", "message"),
    [
        (torch.tensor([]), ValueError, "non-empty 1D"),
        (torch.zeros(1, 1), ValueError, "non-empty 1D"),
        (torch.tensor([0, 1]), TypeError, "supported"),
        (torch.tensor([0.0, float("nan")]), ValueError, "finite"),
        (torch.tensor([-0.1, 0.5]), ValueError, "between 0 and 1"),
        (torch.tensor([0.0, 1.1]), ValueError, "between 0 and 1"),
        (torch.tensor([0.0, 0.8, 0.7]), ValueError, "non-decreasing"),
    ],
)
def test_linear_trajectory_validates_progress(
    progress: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        linear_trajectory(torch.zeros(2), torch.ones(2), progress=progress)


def test_linear_trajectory_requires_progress_matching_endpoint_dtype() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        linear_trajectory(
            torch.zeros(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float32),
            progress=torch.tensor([0.0, 1.0], dtype=torch.float64),
        )
    with pytest.raises(TypeError, match="progress must be a Tensor"):
        linear_trajectory(
            torch.zeros(2),
            torch.ones(2),
            progress=cast(Any, [0.0, 1.0]),
        )
    with pytest.raises(ValueError, match="CPU, CUDA, or MPS"):
        linear_trajectory(
            torch.zeros(2),
            torch.ones(2),
            progress=torch.empty(2, device="meta"),
        )


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


def test_sampling_rejects_invalid_scalar_and_shape_contracts() -> None:
    room = torch.tensor([4.0, 3.0, 2.0])
    with pytest.raises(TypeError, match="rng"):
        sample_positions(num=1, room_size=room, rng=cast(Any, object()))
    with pytest.raises(TypeError, match="num must be an integer"):
        sample_positions(num=True, room_size=room, rng=random.Random(1))
    with pytest.raises(ValueError, match="num must be at most"):
        sample_positions(num=10**400, room_size=room, rng=random.Random(1))
    with pytest.raises(ValueError, match="max_attempts must be at most"):
        sample_positions_min_distance(
            num=1,
            room_size=room,
            rng=random.Random(1),
            center=torch.ones(3),
            min_distance=0.1,
            max_attempts=10**400,
        )
    with pytest.raises(ValueError, match="exactly two"):
        sample_positions_with_z_range(
            num=1,
            room_size=room,
            rng=random.Random(1),
            z_range=cast(Any, (1.0, 1.5, 2.0)),
        )
    with pytest.raises(ValueError, match="last dimension"):
        clamp_positions(torch.ones(2, 2), room)
    with pytest.raises(ValueError, match="finite"):
        clamp_positions(torch.tensor([[float("nan"), 1.0]]), room[:2])
