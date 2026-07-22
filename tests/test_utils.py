import math
import warnings

import pytest
import torch

from torchrir.util import (
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    resolve_device,
)


@pytest.mark.numerical
@pytest.mark.parametrize("target_t60", [0.2, 0.5, 1.2])
def test_t60_beta_roundtrip_3d(target_t60: float):
    size = torch.tensor([6.0, 4.0, 3.0])
    beta = estimate_beta_from_t60(size, target_t60)
    assert beta.shape == (6,)
    t60 = estimate_t60_from_beta(size, beta)
    assert t60 == pytest.approx(target_t60, rel=1e-5)


@pytest.mark.numerical
@pytest.mark.parametrize("target_t60", [0.2, 0.5, 1.2])
def test_t60_beta_roundtrip_2d(target_t60: float):
    size = torch.tensor([6.0, 4.0])
    beta = estimate_beta_from_t60(size, target_t60)
    assert beta.shape == (4,)
    t60 = estimate_t60_from_beta(size, beta)
    assert t60 == pytest.approx(target_t60, rel=1e-5)


def test_t60_from_perfect_reflection():
    size = torch.tensor([6.0, 4.0, 3.0])
    beta = torch.ones(6)
    t60 = estimate_t60_from_beta(size, beta)
    assert math.isinf(t60)


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("size", "coefficient_factor"),
    [
        (torch.tensor([6.0, 4.0]), 12.0),
        (torch.tensor([6.0, 4.0, 3.0]), 24.0),
    ],
)
def test_sabine_uses_dimension_and_speed_of_sound(
    size: torch.Tensor, coefficient_factor: float
) -> None:
    c = 300.0
    beta = torch.full((2 * size.numel(),), 0.8)
    if size.numel() == 2:
        measure = 6.0 * 4.0
        surfaces = torch.tensor([4.0, 4.0, 6.0, 6.0])
    else:
        measure = 6.0 * 4.0 * 3.0
        surfaces = torch.tensor([12.0, 12.0, 18.0, 18.0, 24.0, 24.0])
    expected = (
        coefficient_factor
        * math.log(10.0)
        / c
        * measure
        / torch.sum(surfaces * (1.0 - beta**2)).item()
    )
    actual = estimate_t60_from_beta(size, beta, c=c)
    assert actual == pytest.approx(expected, rel=1e-6)
    estimated = estimate_beta_from_t60(size, actual, c=c)
    torch.testing.assert_close(estimated, beta, rtol=1e-6, atol=1e-7)


def test_resolve_device_auto_returns_device():
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda", "mps")


def test_resolve_device_invalid_falls_back_cpu():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        device = resolve_device("cuda")
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"
