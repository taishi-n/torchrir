"""Configuration validation and normalization contracts."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, cast

import pytest
import torch

from torchrir.config import (
    RIRHighPassConfig,
    ResolvedSimulationConfig,
    SimulationConfig,
    default_config,
)


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"fs": 0.0}, ValueError, "fs must be positive"),
        ({"fs": float("nan")}, ValueError, "fs must be positive"),
        ({"max_order": -1}, ValueError, "max_order"),
        ({"tmax": 0.0}, ValueError, "tmax must be positive"),
        ({"tmax": float("inf")}, ValueError, "tmax must be positive"),
        ({"nsample": 0}, ValueError, "nsample must be positive"),
        ({"tmax": 1.0, "nsample": 10}, ValueError, "mutually exclusive"),
        ({"tdiff": -0.1}, ValueError, "tdiff must be non-negative"),
        ({"tdiff": float("nan")}, ValueError, "tdiff must be non-negative"),
        ({"tdiff": 1.0, "tmax": 1.0}, ValueError, "smaller than tmax"),
        ({"seed": -1}, ValueError, "seed must be non-negative"),
        ({"dtype": torch.int64}, TypeError, "floating-point"),
        ({"frac_delay_length": 0}, ValueError, "positive odd"),
        ({"frac_delay_length": 4}, ValueError, "positive odd"),
        ({"sinc_lut_granularity": 0}, ValueError, "granularity"),
        ({"image_chunk_size": 0}, ValueError, "image_chunk_size"),
        ({"accumulate_chunk_size": 0}, ValueError, "accumulate_chunk_size"),
        ({"rir_hpf_fc": 0.0}, ValueError, "rir_hpf_fc"),
        ({"rir_hpf_kwargs": {"n": 0}}, ValueError, r"\['n'\]"),
    ],
)
def test_simulation_config_rejects_invalid_values(
    updates: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        SimulationConfig(**cast(Any, updates))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (RIRHighPassConfig(cutoff_hz=0), "cutoff_hz"),
        (RIRHighPassConfig(order=0), "order"),
        (RIRHighPassConfig(rp=-1), "rp"),
        (RIRHighPassConfig(rs=-1), "rs"),
        (RIRHighPassConfig(filter_type=""), "filter_type"),
    ],
)
def test_high_pass_config_rejects_invalid_values(
    config: RIRHighPassConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_simulation_config_copies_and_freezes_legacy_hpf_mapping() -> None:
    original = {"n": 4, "type": "cheby1", "rp": 1.0, "rs": 40.0}
    config = SimulationConfig(rir_hpf_kwargs=original)
    original["n"] = 8
    assert isinstance(config.rir_hpf_kwargs, MappingProxyType)
    assert config.rir_hpf_kwargs["n"] == 4
    with pytest.raises(TypeError):
        config.rir_hpf_kwargs["n"] = 6  # type: ignore[index]


def test_explicit_high_pass_config_takes_precedence() -> None:
    explicit = RIRHighPassConfig(enabled=False, cutoff_hz=20.0, order=3)
    config = SimulationConfig(
        rir_hpf=explicit,
        rir_hpf_enable=True,
        rir_hpf_fc=10.0,
    )
    assert config.high_pass is explicit
    assert explicit.as_scipy_kwargs() == {
        "n": 3,
        "rp": 5.0,
        "rs": 60.0,
        "type": "butter",
    }


def test_replace_and_default_config_return_valid_independent_configs() -> None:
    base = default_config()
    changed = base.replace(max_order=2, nsample=128)
    assert base.max_order is None
    assert changed.max_order == 2
    assert changed.nsample == 128


def test_resolved_config_records_effective_runtime_values() -> None:
    source = SimulationConfig(
        nb_img=(1, 2, 3), seed=4, use_lut=False, image_chunk_size=11
    )
    resolved = ResolvedSimulationConfig.from_config(
        source,
        fs=8000,
        max_order=2,
        nsample=400,
        directivity=("cardioid", "omni"),
        device=torch.device("cpu"),
        dtype=torch.float64,
        tdiff=0.02,
    )
    assert resolved.tmax == pytest.approx(0.05)
    assert resolved.nb_img == (1, 2, 3)
    assert resolved.seed == 4
    assert resolved.use_lut is False
    assert resolved.image_chunk_size == 11
