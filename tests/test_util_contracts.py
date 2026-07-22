"""Deterministic contracts for tensor, orientation, and acoustic helpers."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch

from torchrir.sim.directivity import directivity_gain, split_directivity
from torchrir.util.acoustics import (
    attenuation_db_to_time_sabine,
    estimate_beta_from_t60,
    estimate_image_counts_from_tmax,
    estimate_t60_from_beta,
)
from torchrir.util.device import DeviceSpec, resolve_device
from torchrir.util.orientation import normalize_orientation, orientation_to_unit
from torchrir.util.tensor import as_float_tensor, as_tensor, ensure_dim, extend_size


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
    angles = torch.tensor([0.0, math.pi / 2, math.pi])
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


def test_directivity_specification_validation() -> None:
    assert split_directivity("omni") == ("omni", "omni")
    assert split_directivity(("cardioid", "bidir")) == ("cardioid", "bidir")
    with pytest.raises(ValueError, match="length 2"):
        split_directivity(cast(Any, ("omni",)))
    with pytest.raises(TypeError, match="strings"):
        split_directivity(("omni", 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported"):
        split_directivity("unknown")


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
    almost_absorbing = estimate_beta_from_t60(torch.tensor([100.0, 100.0]), 0.001)
    torch.testing.assert_close(
        almost_absorbing,
        torch.full((4,), math.sqrt(0.001)),
        rtol=1e-5,
        atol=1e-7,
    )


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (lambda: estimate_beta_from_t60(torch.tensor([4.0, 3.0]), 0), "t60"),
        (lambda: attenuation_db_to_time_sabine(60, 0), "t60"),
        (lambda: attenuation_db_to_time_sabine(0, 0.5), "att_db"),
        (lambda: estimate_image_counts_from_tmax(0, torch.tensor([4.0, 3.0])), "tmax"),
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
    ],
)
def test_acoustic_helpers_reject_invalid_inputs(operation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_device_spec_uses_tensor_defaults_and_explicit_overrides() -> None:
    tensor = torch.ones(2, dtype=torch.float64)
    assert resolve_device(None).type == "cpu"
    assert resolve_device(torch.device("cpu")).type == "cpu"
    assert resolve_device("cpu").type == "cpu"
    device, dtype = DeviceSpec().resolve(tensor)
    assert device == tensor.device
    assert dtype == torch.float64
    device, dtype = DeviceSpec(device="auto", dtype=torch.float32).resolve(tensor)
    assert device == tensor.device
    assert dtype == torch.float32
