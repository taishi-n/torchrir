"""Deterministic contracts for tensor, orientation, and acoustic helpers."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any, cast

import numpy as np
import pytest
import torch

from torchrir.sim.directivity import directivity_gain
from torchrir.util.acoustics import (
    attenuation_db_to_time_sabine,
    estimate_beta_from_t60,
    estimate_image_counts_from_tmax,
    estimate_t60_from_beta,
)
from torchrir.util.device import DeviceSpec, resolve_device
from torchrir.util.orientation import normalize_orientation, orientation_to_unit
from torchrir.util.tensor import (
    as_float_tensor,
    as_tensor,
    ensure_dim,
    extend_size,
    stable_vector_norm,
)


@pytest.mark.numerical
def test_orientation_representations_map_to_expected_unit_vectors() -> None:
    torch.testing.assert_close(
        orientation_to_unit(torch.tensor(0.0), 2), torch.tensor([1.0, 0.0])
    )
    torch.testing.assert_close(
        orientation_to_unit(torch.tensor([[math.pi / 2]]), 2),
        torch.tensor([[0.0, 1.0]]),
        atol=1e-7,
        rtol=0,
    )
    angles = torch.tensor([[0.0], [math.pi / 2], [math.pi]])
    expected_2d = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    torch.testing.assert_close(
        orientation_to_unit(angles, 2), expected_2d, atol=1e-7, rtol=0
    )
    torch.testing.assert_close(
        orientation_to_unit(torch.tensor([3.0, 4.0]), 2),
        torch.tensor([0.6, 0.8]),
    )
    torch.testing.assert_close(
        orientation_to_unit(torch.tensor([0.0, math.pi / 2]), 3),
        torch.tensor([0.0, 0.0, 1.0]),
        atol=1e-7,
        rtol=0,
    )
    torch.testing.assert_close(
        orientation_to_unit(torch.tensor([0.0, 0.0, -2.0]), 3),
        torch.tensor([0.0, 0.0, -1.0]),
    )


@pytest.mark.parametrize(
    ("orientation", "dim", "message"),
    [
        (torch.tensor([[1.0, 2.0, 3.0]]), 2, "2D orientation"),
        (torch.tensor([0.0, 1.0, 2.0]), 2, "2D orientation"),
        (torch.tensor([1.0]), 3, "3D orientation"),
        (torch.tensor([1.0, 0.0]), 4, "unsupported dimension"),
    ],
)
def test_orientation_rejects_unsupported_representations(
    orientation: torch.Tensor, dim: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        orientation_to_unit(orientation, dim)


def test_normalize_orientation_rejects_zero_and_non_finite_vectors() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        normalize_orientation(torch.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        normalize_orientation(torch.tensor([float("nan"), 1.0]))
    with pytest.raises(TypeError, match="eps"):
        normalize_orientation(torch.ones(2), eps=cast(Any, True))
    with pytest.raises(ValueError, match="eps"):
        normalize_orientation(torch.ones(2), eps=float("nan"))
    with pytest.raises(TypeError, match="orientation must be a Tensor"):
        normalize_orientation(cast(Any, [1.0, 0.0]))


def test_normalize_orientation_stably_handles_extreme_finite_vectors() -> None:
    orientation = torch.tensor([1.0e308, 1.0e308], dtype=torch.float64)

    normalized = normalize_orientation(orientation)

    torch.testing.assert_close(
        normalized,
        torch.full((2,), 2.0**-0.5, dtype=torch.float64),
    )
    assert torch.all(torch.isfinite(normalized))


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("omni", [1.0, 1.0, 1.0]),
        ("half-omni", [0.0, 0.0, 1.0]),
        ("subcard", [0.5, 0.75, 1.0]),
        ("card", [0.0, 0.5, 1.0]),
        ("hypcard", [-0.5, 0.25, 1.0]),
        ("figure-8", [-1.0, 0.0, 1.0]),
    ],
)
def test_directivity_patterns_match_analytic_linear_forms(
    pattern: str, expected: list[float]
) -> None:
    cosine = torch.tensor([-1.0, 0.0, 1.0])
    torch.testing.assert_close(
        directivity_gain(pattern, cosine), torch.tensor(expected)
    )


@pytest.mark.parametrize(
    ("cosine", "error", "message"),
    [
        ([0.0], TypeError, "must be a Tensor"),
        (torch.tensor([0]), TypeError, "floating-point"),
        (torch.tensor([float("nan")]), ValueError, "finite"),
        (torch.tensor([1.01]), ValueError, r"\[-1, 1\]"),
    ],
)
def test_directivity_gain_validates_cosine_contract(
    cosine: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        directivity_gain("cardioid", cast(Any, cosine))


def test_tensor_helpers_preserve_or_explicitly_convert_properties() -> None:
    source = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert as_tensor(source) is source
    converted = as_tensor(source, dtype=torch.float32, device="cpu")
    assert converted.dtype == torch.float32
    promoted = as_float_tensor([1, 2, 3])
    assert promoted.dtype == torch.get_default_dtype()
    with pytest.raises(TypeError, match="floating-point"):
        as_float_tensor([1, 2], dtype=torch.int64)
    with pytest.raises(TypeError, match="real floating-point"):
        as_float_tensor(torch.tensor([1 + 2j]))


def test_stable_vector_norm_handles_large_small_and_zero_float64_vectors() -> None:
    vectors = torch.tensor(
        [[1.0e308, 1.0e308], [1.0e-308, 0.0], [0.0, 0.0]],
        dtype=torch.float64,
    )

    norms = stable_vector_norm(vectors)

    assert torch.all(torch.isfinite(norms))
    torch.testing.assert_close(
        norms,
        torch.tensor(
            [math.hypot(1.0e308, 1.0e308), 1.0e-308, 0.0],
            dtype=torch.float64,
        ),
        rtol=1.0e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "value",
    [
        [True, 1.0],
        [[0.0, 1.0], [False, 2.0]],
        np.array([True, 1], dtype=object),
        (item for item in (0.0, np.bool_(True))),
    ],
)
def test_as_float_tensor_rejects_mixed_boolean_values(value: object) -> None:
    with pytest.raises(TypeError, match="boolean"):
        as_float_tensor(cast(Any, value), name="geometry")


def test_dimension_helpers_validate_and_extend_sizes() -> None:
    size_2d = ensure_dim(torch.tensor([4.0, 3.0]))
    torch.testing.assert_close(extend_size(size_2d, 3), torch.tensor([4.0, 3.0, 1.0]))
    size_3d = torch.tensor([4.0, 3.0, 2.0])
    assert extend_size(size_3d, 3) is size_3d
    with pytest.raises(ValueError, match="length 2 or 3"):
        ensure_dim(torch.ones(4))
    with pytest.raises(ValueError, match="unsupported"):
        extend_size(size_2d, 4)


@pytest.mark.numerical
def test_acoustic_helpers_match_closed_form_values() -> None:
    assert attenuation_db_to_time_sabine(30.0, 0.8) == pytest.approx(0.4)
    torch.testing.assert_close(
        estimate_image_counts_from_tmax(0.1, torch.tensor([10.0, 5.0]), c=100.0),
        torch.tensor([1, 2]),
    )
    size = torch.tensor([100.0, 100.0])
    minimum_t60 = (12.0 * math.log(10.0) / 343.0) * (100.0**2) / 400.0
    fully_absorbing = estimate_beta_from_t60(size, minimum_t60)
    torch.testing.assert_close(
        fully_absorbing,
        torch.zeros(4),
        rtol=0,
        atol=1e-7,
    )
    with pytest.raises(ValueError, match="too short"):
        estimate_beta_from_t60(size, minimum_t60 / 2.0)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: estimate_beta_from_t60(torch.tensor([4.0, 3.0]), 0), "t60"),
        (
            lambda: estimate_beta_from_t60(
                torch.tensor([4.0, 3.0]), 0.5, c=float("nan")
            ),
            "c",
        ),
        (
            lambda: estimate_beta_from_t60(torch.tensor([4.0, 0.0]), 0.5),
            "room size",
        ),
        (lambda: attenuation_db_to_time_sabine(60, 0), "t60"),
        (lambda: attenuation_db_to_time_sabine(60, float("inf")), "t60"),
        (lambda: attenuation_db_to_time_sabine(60, 10**400), "t60"),
        (lambda: attenuation_db_to_time_sabine(0, 0.5), "att_db"),
        (lambda: attenuation_db_to_time_sabine(float("nan"), 0.5), "att_db"),
        (lambda: estimate_image_counts_from_tmax(0, torch.tensor([4.0, 3.0])), "tmax"),
        (
            lambda: estimate_image_counts_from_tmax(
                float("inf"), torch.tensor([4.0, 3.0])
            ),
            "tmax",
        ),
        (
            lambda: estimate_image_counts_from_tmax(0.5, torch.tensor([4.0, 3.0]), c=0),
            "c",
        ),
        (
            lambda: estimate_image_counts_from_tmax(
                0.5, torch.tensor([4.0, float("nan")])
            ),
            "room size",
        ),
        (
            lambda: estimate_t60_from_beta(torch.tensor([4.0, 3.0]), torch.ones(6)),
            "4 elements",
        ),
        (
            lambda: estimate_t60_from_beta(
                torch.tensor([4.0, 3.0, 2.0]), torch.ones(4)
            ),
            "6 elements",
        ),
        (
            lambda: estimate_t60_from_beta(
                torch.tensor([4.0, 3.0]), torch.tensor([1.1] * 4)
            ),
            r"\[0, 1\]",
        ),
    ],
)
def test_acoustic_helpers_reject_invalid_inputs(operation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_acoustic_helpers_reject_non_real_scalar_types() -> None:
    with pytest.raises(TypeError, match="real number"):
        attenuation_db_to_time_sabine(cast(Any, "60"), 0.5)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: estimate_beta_from_t60(
                torch.tensor([1.0e308, 1.0e308, 1.0e308], dtype=torch.float64),
                0.3,
            ),
            "room size",
        ),
        (
            lambda: estimate_t60_from_beta(
                torch.tensor([1.0e308, 1.0e308, 1.0e308], dtype=torch.float64),
                torch.zeros(6, dtype=torch.float64),
            ),
            "room size",
        ),
        (
            lambda: estimate_t60_from_beta(
                torch.ones(2, dtype=torch.float64),
                torch.full((4,), 1.0 - 1.0e-15, dtype=torch.float64),
                c=1.0e-300,
            ),
            "estimated t60",
        ),
        (
            lambda: attenuation_db_to_time_sabine(1.0e300, 1.0e300),
            "att_db.*t60",
        ),
        (
            lambda: estimate_image_counts_from_tmax(
                1.0e300,
                torch.ones(2, dtype=torch.float64),
                c=1.0e300,
            ),
            "tmax.*c",
        ),
        (
            lambda: estimate_image_counts_from_tmax(
                1.0e10,
                torch.ones(2, dtype=torch.float64),
                c=1.0e10,
            ),
            "int64",
        ),
    ],
)
def test_acoustic_helpers_reject_finite_intermediate_overflow(
    operation: Callable[[], object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_t60_estimator_preserves_infinite_perfect_reflection_contract() -> None:
    result = estimate_t60_from_beta(torch.tensor([4.0, 3.0]), torch.ones(4))
    assert math.isinf(result)


def test_beta_estimator_rejects_dtype_endpoint_collapse() -> None:
    with pytest.raises(ValueError, match="cannot be represented"):
        estimate_beta_from_t60(torch.tensor([4.0, 3.0]), 1.0e10)

    beta = estimate_beta_from_t60(
        torch.tensor([4.0, 3.0], dtype=torch.float64),
        1.0e10,
    )
    assert torch.all(beta < 1)


def test_device_spec_uses_tensor_defaults_and_explicit_overrides() -> None:
    tensor = torch.ones(2, dtype=torch.float64)
    assert resolve_device(None).type == "cpu"
    assert resolve_device(torch.device("cpu")).type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    device, dtype = DeviceSpec().resolve(tensor)
    assert device == tensor.device
    assert dtype == torch.float64
    device, dtype = DeviceSpec(device="auto", dtype=torch.float32).resolve(tensor)
    assert device == resolve_device("auto")
    assert dtype == torch.float32


def test_device_spec_rejects_implicit_mixed_tensor_layouts() -> None:
    with pytest.raises(ValueError, match="dtypes must match"):
        DeviceSpec().resolve(torch.ones(1), torch.ones(1, dtype=torch.float64))


def test_device_helpers_reject_unsupported_inferred_and_object_devices() -> None:
    with pytest.raises(ValueError, match="cpu, cuda, or mps"):
        DeviceSpec().resolve(torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="cpu, cuda, or mps"):
        as_tensor([1.0], device=torch.device("meta"))


def test_device_spec_explicit_dtype_resolves_mixed_input_dtypes() -> None:
    device, dtype = DeviceSpec(dtype=torch.float32).resolve(
        torch.ones(1),
        torch.ones(1, dtype=torch.float64),
    )
    assert device == torch.device("cpu")
    assert dtype == torch.float32


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (lambda: DeviceSpec(prefer=()), ValueError, "non-empty tuple"),
        (
            lambda: DeviceSpec(prefer=("cuda", "cuda")),
            ValueError,
            "duplicates",
        ),
        (
            lambda: DeviceSpec(prefer=("invalid",)),
            ValueError,
            "cuda, mps, or cpu",
        ),
        (lambda: DeviceSpec(device="meta"), ValueError, "cpu, cuda, or mps"),
        (
            lambda: DeviceSpec(device=cast(Any, object())),
            TypeError,
            "string, torch.device",
        ),
        (lambda: DeviceSpec(dtype=torch.int64), TypeError, "dtype must be"),
    ],
)
def test_device_spec_rejects_invalid_request_state(
    factory: Callable[[], object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()
