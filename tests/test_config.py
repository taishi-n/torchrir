"""Configuration validation and resolution contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import numpy as np
import pytest
import torch

from torchrir.config import (
    RIRHighPassConfig,
    ResolvedSimulationConfig,
    SimulationConfig,
    _resolve_simulation_config,
)


def _config(**updates: object) -> SimulationConfig:
    values: dict[str, object] = {"max_order": 1, "nsample": 64}
    values.update(updates)
    return SimulationConfig(**cast(Any, values))


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"max_order": -1}, ValueError, "max_order"),
        ({"max_order": 1.5}, TypeError, "max_order"),
        ({"tmax": True, "nsample": None}, TypeError, "tmax"),
        ({"tmax": 0.0, "nsample": None}, ValueError, "tmax must be positive"),
        ({"tmax": 10**400, "nsample": None}, ValueError, "tmax must be positive"),
        ({"tmax": "0.1", "nsample": None}, TypeError, "real number"),
        (
            {"tmax": float("inf"), "nsample": None},
            ValueError,
            "tmax must be positive",
        ),
        ({"nsample": 0}, ValueError, "nsample must be positive"),
        ({"tdiff": 0.0}, ValueError, "tdiff must be positive"),
        ({"tdiff": float("nan")}, ValueError, "tdiff must be positive"),
        (
            {"tmax": 1.0, "nsample": None, "tdiff": 1.0},
            ValueError,
            "smaller than tmax",
        ),
        ({"seed": -1}, ValueError, "seed must be non-negative"),
        ({"dtype": torch.int64}, TypeError, "dtype must be"),
        ({"frac_delay_length": 0}, ValueError, "positive odd"),
        ({"frac_delay_length": 4}, ValueError, "positive odd"),
        ({"sinc_lut_granularity": 0}, ValueError, "granularity"),
        ({"image_chunk_size": 0}, ValueError, "image_chunk_size"),
        ({"accumulate_chunk_size": 0}, ValueError, "accumulate_chunk_size"),
        ({"min_source_mic_distance": 0.0}, ValueError, "min_source_mic_distance"),
        (
            {"min_source_mic_distance": float("nan")},
            ValueError,
            "min_source_mic_distance",
        ),
        ({"high_pass": cast(Any, object())}, TypeError, "high_pass"),
        ({"use_lut": 1}, TypeError, "use_lut"),
        ({"use_compile": 0}, TypeError, "use_compile"),
        ({"device": cast(Any, object())}, TypeError, "device"),
    ],
)
def test_simulation_config_rejects_invalid_values(
    updates: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        _config(**updates)


def test_simulation_config_requires_complete_exclusive_limits() -> None:
    with pytest.raises(ValueError, match="max_order or nb_img"):
        SimulationConfig(nsample=64)
    with pytest.raises(ValueError, match="max_order or nb_img"):
        SimulationConfig(max_order=1, nb_img=(1, 1), nsample=64)
    with pytest.raises(ValueError, match="tmax or nsample"):
        SimulationConfig(max_order=1)
    with pytest.raises(ValueError, match="tmax or nsample"):
        SimulationConfig(max_order=1, tmax=0.1, nsample=64)


def test_resolved_config_exposes_no_public_resolution_constructor() -> None:
    assert not hasattr(ResolvedSimulationConfig, "from_config")


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"cutoff_hz": 0}, ValueError, "cutoff_hz"),
        ({"order": 0}, ValueError, "order"),
        ({"order": 1.5}, TypeError, "order"),
        ({"passband_ripple_db": -1}, ValueError, "passband_ripple_db"),
        (
            {"stopband_attenuation_db": -1},
            ValueError,
            "stopband_attenuation_db",
        ),
        ({"filter_family": ""}, ValueError, "filter_family"),
        ({"filter_family": "not-a-filter"}, ValueError, "filter_family"),
        ({"cutoff_hz": True}, TypeError, "cutoff_hz"),
        ({"phase": "forward-backward"}, ValueError, "phase"),
        (
            {"filter_family": "cheby1", "passband_ripple_db": 0},
            ValueError,
            "passband_ripple_db must be positive",
        ),
        (
            {"filter_family": "cheby2", "stopband_attenuation_db": 0},
            ValueError,
            "stopband_attenuation_db must be positive",
        ),
        (
            {
                "filter_family": "ellip",
                "passband_ripple_db": 5,
                "stopband_attenuation_db": 5,
            },
            ValueError,
            "must exceed",
        ),
    ],
)
def test_high_pass_config_validates_at_construction(
    kwargs: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        RIRHighPassConfig(**cast(Any, kwargs))


def test_nb_img_is_normalized_to_an_immutable_tuple() -> None:
    original = torch.tensor([1, 2, 3])
    config = SimulationConfig(nb_img=original, nsample=64)
    original[0] = 9
    assert config.nb_img == (1, 2, 3)
    with pytest.raises(FrozenInstanceError):
        config.nb_img = (0, 0, 0)  # type: ignore[misc]


def test_non_integer_nb_img_tuple_is_rejected_before_tensor_coercion() -> None:
    with pytest.raises(TypeError, match="integers"):
        SimulationConfig(nb_img=cast(Any, (0.5, 0, 0)), nsample=64)


def test_boolean_nb_img_is_rejected_before_tensor_rounding() -> None:
    with pytest.raises(TypeError, match="booleans"):
        SimulationConfig(nb_img=cast(Any, (True, False)), nsample=64)


@pytest.mark.parametrize(
    "nb_img",
    [
        (True, 1),
        (False, 2.0),
        (np.bool_(True), 1),
    ],
)
def test_mixed_nb_img_tuple_rejects_boolean_and_float_coercion(
    nb_img: tuple[object, ...],
) -> None:
    with pytest.raises(TypeError, match="integers|booleans"):
        SimulationConfig(nb_img=cast(Any, nb_img), nsample=64)


@pytest.mark.parametrize(
    "nb_img",
    [
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0 + 0.0j, 2.0 + 0.0j]),
    ],
)
def test_nb_img_tensor_requires_integer_dtype(nb_img: torch.Tensor) -> None:
    with pytest.raises(TypeError, match="integer dtype"):
        SimulationConfig(nb_img=nb_img, nsample=64)


def test_nb_img_tensor_must_be_materialized() -> None:
    with pytest.raises(ValueError, match="materialized"):
        SimulationConfig(
            nb_img=torch.empty(2, dtype=torch.int64, device="meta"),
            nsample=64,
        )


def test_nb_img_tensor_requires_dense_strided_layout() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]),
        torch.tensor([1, 2]),
        size=(2,),
    )
    with pytest.raises(TypeError, match="strided"):
        SimulationConfig(nb_img=sparse, nsample=64)


def test_nb_img_values_must_fit_non_negative_int64() -> None:
    with pytest.raises(ValueError, match="int64"):
        SimulationConfig(nb_img=(torch.iinfo(torch.int64).max + 1, 0), nsample=64)


def test_explicit_sample_count_must_fit_positive_int64() -> None:
    maximum = torch.iinfo(torch.int64).max
    assert _config(nsample=maximum).nsample == maximum
    with pytest.raises(ValueError, match="nsample.*int64"):
        _config(nsample=maximum + 1)


def test_seed_must_fit_non_negative_int64() -> None:
    maximum = torch.iinfo(torch.int64).max
    assert _config(seed=maximum).seed == maximum
    with pytest.raises(ValueError, match="seed must be at most"):
        _config(seed=maximum + 1)


def test_integer_config_fields_normalize_numpy_integers() -> None:
    high_pass = RIRHighPassConfig(order=cast(Any, np.int64(3)))
    config = SimulationConfig(
        max_order=cast(Any, np.int64(2)),
        nsample=cast(Any, np.int64(64)),
        seed=cast(Any, np.int64(7)),
        frac_delay_length=cast(Any, np.int64(81)),
        sinc_lut_granularity=cast(Any, np.int64(20)),
        image_chunk_size=cast(Any, np.int64(32)),
        accumulate_chunk_size=cast(Any, np.int64(48)),
        high_pass=high_pass,
    )
    assert type(high_pass.order) is int
    for value in (
        config.max_order,
        config.nsample,
        config.seed,
        config.frac_delay_length,
        config.sinc_lut_granularity,
        config.image_chunk_size,
        config.accumulate_chunk_size,
    ):
        assert type(value) is int

    resolved = _resolve_simulation_config(
        config,
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    resolved = replace(
        resolved,
        max_order=cast(Any, np.int64(2)),
        nsample=cast(Any, np.int64(64)),
        seed=cast(Any, np.int64(7)),
        frac_delay_length=cast(Any, np.int64(81)),
        sinc_lut_granularity=cast(Any, np.int64(20)),
        image_chunk_size=cast(Any, np.int64(32)),
        accumulate_chunk_size=cast(Any, np.int64(48)),
    )
    for value in (
        resolved.max_order,
        resolved.nsample,
        resolved.seed,
        resolved.frac_delay_length,
        resolved.sinc_lut_granularity,
        resolved.image_chunk_size,
        resolved.accumulate_chunk_size,
    ):
        assert type(value) is int


@pytest.mark.parametrize(
    "updates",
    [
        {"max_order": np.bool_(True)},
        {"nsample": np.bool_(True)},
        {"seed": np.bool_(True)},
        {"frac_delay_length": np.bool_(True)},
        {"sinc_lut_granularity": np.bool_(True)},
        {"image_chunk_size": np.bool_(True)},
        {"accumulate_chunk_size": np.bool_(True)},
    ],
)
def test_integer_config_fields_reject_numpy_booleans(
    updates: dict[str, object],
) -> None:
    with pytest.raises(TypeError, match="integer"):
        _config(**updates)


def test_high_pass_order_rejects_numpy_boolean() -> None:
    with pytest.raises(TypeError, match="integer"):
        RIRHighPassConfig(order=cast(Any, np.bool_(True)))


@pytest.mark.parametrize(
    ("tmax", "fs"),
    [
        (1.0e308, 1.0e308),
        (1.0e10, 1.0e10),
    ],
)
def test_tmax_resolution_rejects_finite_product_overflow_or_int64_excess(
    tmax: float,
    fs: float,
) -> None:
    with pytest.raises(ValueError, match="tmax.*fs.*int64"):
        _resolve_simulation_config(
            SimulationConfig(max_order=1, tmax=tmax),
            fs=fs,
            room_dimension=3,
            tensor_values=(torch.zeros(1),),
        )


def test_high_pass_is_explicit_and_disabled_by_default() -> None:
    base = _config()
    assert base.high_pass is None
    high_pass = RIRHighPassConfig(
        cutoff_hz=20.0,
        order=3,
        passband_ripple_db=1.0,
        stopband_attenuation_db=40.0,
        filter_family="CHEBY1",
        phase="zero_phase",
    )
    enabled = base.replace(high_pass=high_pass)
    assert enabled.high_pass is high_pass
    assert high_pass.filter_family == "cheby1"


def test_replace_returns_a_valid_independent_config() -> None:
    base = _config()
    changed = base.replace(max_order=2, nsample=128)
    assert base.max_order == 1
    assert base.nsample == 64
    assert changed.max_order == 2
    assert changed.nsample == 128


def test_resolved_config_records_effective_runtime_values() -> None:
    source = SimulationConfig(
        nb_img=(1, 2, 3),
        tmax=0.05,
        min_source_mic_distance=2.0e-5,
        seed=4,
        use_lut=False,
        image_chunk_size=11,
    )
    resolved = _resolve_simulation_config(
        source,
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1, dtype=torch.float64),),
    )
    assert resolved.nsample == 400
    assert resolved.tmax == pytest.approx(0.05)
    assert resolved.nb_img == (1, 2, 3)
    assert resolved.max_order is None
    assert resolved.min_source_mic_distance == pytest.approx(2.0e-5)
    assert resolved.seed == 4
    assert resolved.use_lut is False
    assert resolved.image_chunk_size == 11
    assert resolved.device == torch.device("cpu")
    assert resolved.dtype == torch.float64


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"max_order": None}, "max_order or nb_img"),
        ({"tmax": 1.0}, "nsample / fs"),
        ({"nsample": 0, "tmax": 0.0}, "nsample"),
        ({"tdiff": 1.0}, "smaller than tmax"),
        ({"use_lut": 1}, "use_lut"),
    ],
)
def test_resolved_config_cannot_be_replaced_with_invalid_metadata(
    updates: dict[str, object], message: str
) -> None:
    resolved = _resolve_simulation_config(
        _config(),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    with pytest.raises((TypeError, ValueError), match=message):
        replace(resolved, **updates)


@pytest.mark.parametrize(
    "updates",
    [
        {"max_order": np.bool_(True)},
        {"nsample": np.bool_(True)},
        {"seed": np.bool_(True)},
        {"frac_delay_length": np.bool_(True)},
        {"sinc_lut_granularity": np.bool_(True)},
        {"image_chunk_size": np.bool_(True)},
        {"accumulate_chunk_size": np.bool_(True)},
    ],
)
def test_resolved_integer_fields_reject_numpy_booleans(
    updates: dict[str, object],
) -> None:
    resolved = _resolve_simulation_config(
        _config(seed=1),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    with pytest.raises(TypeError, match="integer"):
        replace(resolved, **updates)


def test_resolved_config_validates_room_dependent_values() -> None:
    with pytest.raises(ValueError, match="room dimension"):
        _resolve_simulation_config(
            SimulationConfig(nb_img=(1, 2), nsample=64),
            fs=8000,
            room_dimension=3,
            tensor_values=(torch.zeros(1),),
        )
    with pytest.raises(ValueError, match="RIR duration"):
        _resolve_simulation_config(
            SimulationConfig(max_order=1, nsample=64, tdiff=0.1),
            fs=8000,
            room_dimension=3,
            tensor_values=(torch.zeros(1),),
        )
    with pytest.raises(ValueError, match="fs/2"):
        _resolve_simulation_config(
            SimulationConfig(
                max_order=1,
                nsample=64,
                high_pass=RIRHighPassConfig(cutoff_hz=4000),
            ),
            fs=8000,
            room_dimension=3,
            tensor_values=(torch.zeros(1),),
        )


def test_device_none_infers_scene_tensor_device_and_dtype() -> None:
    tensor = torch.zeros(1, dtype=torch.float64)
    resolved = _resolve_simulation_config(
        _config(),
        fs=8000,
        room_dimension=3,
        tensor_values=(tensor,),
    )
    assert resolved.device == tensor.device
    assert resolved.dtype == tensor.dtype


def test_device_auto_explicitly_prefers_accelerator_over_cpu_scene(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    resolved = _resolve_simulation_config(
        _config(device="auto"),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    assert resolved.device == torch.device("cuda:0")


def test_device_auto_skips_mps_for_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    resolved = _resolve_simulation_config(
        _config(device="auto", dtype=torch.float64),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1, dtype=torch.float64),),
    )
    assert resolved.device == torch.device("cpu")
    assert resolved.dtype == torch.float64


def test_resolved_config_records_effective_backend_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu = _resolve_simulation_config(
        _config(device="cpu", use_compile=True),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    assert cpu.use_compile is False

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    mps = _resolve_simulation_config(
        _config(device="auto", use_lut=True),
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.zeros(1),),
    )
    assert mps.device == torch.device("mps:0")
    assert mps.use_lut is False


@pytest.mark.parametrize("device", ["meta", "xpu", "mps:1", "cpu:1"])
def test_simulation_config_rejects_unsupported_devices(device: str) -> None:
    with pytest.raises(ValueError, match="device"):
        _config(device=device)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float8_e4m3fn],
)
def test_simulation_config_rejects_numerically_unsafe_dtypes(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(TypeError, match="dtype must be"):
        _config(dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("explicit_dtype", [None, torch.float32, torch.float64])
def test_simulation_resolution_rejects_low_precision_scene_tensors(
    dtype: torch.dtype,
    explicit_dtype: torch.dtype | None,
) -> None:
    with pytest.raises(TypeError, match="simulation scene tensors must use"):
        _resolve_simulation_config(
            _config(dtype=explicit_dtype),
            fs=8000,
            room_dimension=3,
            tensor_values=(torch.zeros(1, dtype=dtype),),
        )


def test_explicit_mps_rejects_float64_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    with pytest.raises(ValueError, match="MPS does not support float64"):
        _resolve_simulation_config(
            _config(device="mps", dtype=torch.float64),
            fs=8000,
            room_dimension=3,
            tensor_values=(torch.zeros(1, dtype=torch.float64),),
        )
