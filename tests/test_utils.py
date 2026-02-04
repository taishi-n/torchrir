import math

import torch

from torchrir.utils import estimate_beta_from_t60, estimate_t60_from_beta


def test_t60_beta_roundtrip_monotonic():
    size = torch.tensor([6.0, 4.0, 3.0])
    beta = estimate_beta_from_t60(size, 0.5)
    assert beta.shape == (6,)
    t60 = estimate_t60_from_beta(size, beta)
    assert 0.1 < t60 < 5.0


def test_t60_beta_roundtrip_2d():
    size = torch.tensor([6.0, 4.0])
    beta = estimate_beta_from_t60(size, 0.5)
    assert beta.shape == (4,)
    t60 = estimate_t60_from_beta(size, beta)
    assert 0.1 < t60 < 5.0


def test_t60_from_perfect_reflection():
    size = torch.tensor([6.0, 4.0, 3.0])
    beta = torch.ones(6)
    t60 = estimate_t60_from_beta(size, beta)
    assert math.isinf(t60)
