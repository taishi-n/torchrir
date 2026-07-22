import math
import warnings

import pytest
import torch

from torchrir.util import (
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    infer_device_dtype,
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


def test_infer_device_dtype_auto_prefers_tensor_device():
    tensor = torch.ones(1)
    device, dtype = infer_device_dtype(tensor, device="auto")
    assert device == tensor.device
    assert dtype == tensor.dtype
