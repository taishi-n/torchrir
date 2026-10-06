"""Numerical oracle tests for the image-source implementation.

These tests deliberately exercise low-level kernels.  End-to-end comparison
tests alone cannot identify whether an error comes from image geometry,
reflection gains, directivity, or fractional-delay accumulation.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch

from torchrir.config import (
    ResolvedSimulationConfig,
    SimulationConfig,
    _resolve_simulation_config,
)
from torchrir.models import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.sim import simulate
from torchrir.sim.ism.accumulate import _accumulate_rir_batch
from torchrir.sim.ism.contributions import (
    _compute_image_contributions,
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
    _reflected_source_directions,
)
from torchrir.sim.ism.images import (
    _image_positions,
    _image_source_count,
    _image_source_indices,
    _iter_image_source_index_chunks,
    _reflection_coefficients,
)
from torchrir.sim.ism.helpers import _cos_between


def _resolved_accumulation_config(
    *,
    nsample: int,
    dtype: torch.dtype,
    **updates: object,
) -> ResolvedSimulationConfig:
    values: dict[str, object] = {
        "max_order": 0,
        "nsample": nsample,
        **updates,
    }
    config = SimulationConfig(**cast(Any, values))
    return _resolve_simulation_config(
        config,
        fs=8000,
        room_dimension=3,
        tensor_values=(torch.empty(0, dtype=dtype),),
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("dim", "max_order", "expected_count"),
    [
        (2, 0, 1),
        (2, 1, 5),
        (2, 2, 13),
        (2, 3, 25),
        (3, 0, 1),
        (3, 1, 7),
        (3, 2, 25),
        (3, 3, 63),
    ],
)
def test_image_source_indices_form_integer_l1_ball(
    dim: int, max_order: int, expected_count: int
) -> None:
    indices = _image_source_indices(max_order, dim, device=torch.device("cpu"))
    assert indices.shape == (expected_count, dim)
    assert torch.all(torch.sum(torch.abs(indices), dim=-1) <= max_order)
    assert torch.unique(indices, dim=0).shape[0] == expected_count


@pytest.mark.numerical
def test_nb_img_generates_complete_rectangular_index_grid() -> None:
    indices = _image_source_indices(
        None, 3, device=torch.device("cpu"), nb_img=(1, 2, 0)
    )
    expected = {(x, y, 0) for x in range(-1, 2) for y in range(-2, 3)}
    assert {tuple(row.tolist()) for row in indices} == expected


@pytest.mark.numerical
def test_nb_img_index_generation_is_independent_of_default_float_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        indices = _image_source_indices(
            None,
            2,
            device=torch.device("cpu"),
            nb_img=(2049, 0),
        )
    finally:
        torch.set_default_dtype(previous)

    assert indices.shape == (4099, 2)
    assert indices[:, 0].min().item() == -2049
    assert indices[:, 0].max().item() == 2049


@pytest.mark.parametrize(
    "nb_img",
    [
        ((torch.iinfo(torch.int64).max + 1) // 2, 0),
        (2**31, 2**31),
    ],
)
def test_nb_img_rejects_int64_range_overflow(nb_img: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="fit in int64"):
        _image_source_indices(
            None,
            2,
            device=torch.device("cpu"),
            nb_img=nb_img,
        )


@pytest.mark.parametrize(
    ("max_order", "nb_img"),
    [
        (10**10, None),
        (None, (2**31, 2**31)),
    ],
)
def test_image_source_count_rejects_int64_overflow(
    max_order: int | None,
    nb_img: tuple[int, ...] | None,
) -> None:
    with pytest.raises(ValueError, match="image count must fit in int64"):
        _image_source_count(max_order, 2, nb_img=nb_img)


def test_image_source_indices_stream_in_bounded_chunks() -> None:
    chunks = tuple(
        _iter_image_source_index_chunks(
            2,
            3,
            device=torch.device("cpu"),
            nb_img=None,
            chunk_size=7,
        )
    )

    assert sum(chunk.shape[0] for chunk in chunks) == 25
    assert all(0 < chunk.shape[0] <= 7 for chunk in chunks)
    torch.testing.assert_close(
        torch.cat(chunks),
        _image_source_indices(2, 3, device=torch.device("cpu")),
    )


@pytest.mark.numerical
def test_image_positions_match_mirror_geometry() -> None:
    source = torch.tensor([2.0, 3.0], dtype=torch.float64)
    room_size = torch.tensor([10.0, 8.0], dtype=torch.float64)
    indices = torch.tensor([[0, 0], [1, 0], [-1, 0], [2, 0], [-2, 0], [0, 1], [0, -1]])
    expected = torch.tensor(
        [
            [2.0, 3.0],
            [18.0, 3.0],
            [-2.0, 3.0],
            [22.0, 3.0],
            [-18.0, 3.0],
            [2.0, 13.0],
            [2.0, -3.0],
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        _image_positions(source, room_size, indices), expected, rtol=0, atol=0
    )


@pytest.mark.numerical
def test_reflection_coefficients_use_the_correct_walls() -> None:
    # Wall order is x-low, x-high, y-low, y-high.
    beta = torch.tensor([0.2, 0.3, 0.5, 0.7], dtype=torch.float64)
    indices = torch.tensor([[0, 0], [1, 0], [-1, 0], [2, 0], [-2, 0], [3, 0], [0, -1]])
    expected = torch.tensor(
        [1.0, 0.3, 0.2, 0.3 * 0.2, 0.2 * 0.3, 0.3**2 * 0.2, 0.5],
        dtype=torch.float64,
    )
    torch.testing.assert_close(
        _reflection_coefficients(indices, beta), expected, rtol=1e-14, atol=0
    )


@pytest.mark.numerical
def test_path_delays_and_attenuation_match_analytic_values() -> None:
    room = Room.shoebox(
        [10.0, 8.0],
        fs=8.0,
        c=4.0,
        beta=[0.2, 0.3, 0.5, 0.7],
        dtype=torch.float64,
    )
    source = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    microphone = torch.tensor([[6.0, 3.0]], dtype=torch.float64)
    indices = torch.tensor([[0, 0], [-1, 0], [1, 0]])
    assert room.beta is not None
    reflection = _reflection_coefficients(indices, room.beta)

    sample, attenuation = _compute_image_contributions_batch(
        source,
        microphone,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="omni",
        src_dirs=None,
        mic_dir=None,
    )

    # Direct, x-low reflection, x-high reflection: distances 4, 8, and 12 m.
    torch.testing.assert_close(
        sample[0, 0], torch.tensor([8.0, 16.0, 24.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        attenuation[0, 0],
        torch.tensor([1 / 4, 0.2 / 8, 0.3 / 12], dtype=torch.float64),
        rtol=1e-14,
        atol=0,
    )


@pytest.mark.numerical
def test_extreme_finite_image_geometry_is_stable_in_all_contribution_paths() -> None:
    room = Room.shoebox(
        [1.0e308, 1.0e308],
        fs=1.0,
        c=1.0e308,
        beta=[1.0] * 4,
        dtype=torch.float64,
    )
    source = torch.tensor([[9.0e307, 5.0e307]], dtype=torch.float64)
    microphone = torch.tensor([[8.0e307, 4.0e307]], dtype=torch.float64)
    indices = torch.tensor([[0, 0], [1, 0]], dtype=torch.int64)
    reflection = torch.ones(2, dtype=torch.float64)

    images = _image_positions(source[0], room.size, indices)
    expected_images = torch.tensor(
        [[9.0e307, 5.0e307], [1.1e308, 5.0e307]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(images, expected_images, rtol=1.0e-15, atol=0.0)

    expected_samples = (
        torch.tensor(
            [math.hypot(1.0e307, 1.0e307), math.hypot(3.0e307, 1.0e307)],
            dtype=torch.float64,
        )
        / room.c
    )
    expected_attenuation = torch.tensor(
        [
            1.0 / math.hypot(1.0e307, 1.0e307),
            1.0 / math.hypot(3.0e307, 1.0e307),
        ],
        dtype=torch.float64,
    )

    single = _compute_image_contributions(
        source[0],
        microphone,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="omni",
        src_dir=None,
        mic_dir=None,
    )
    batch = _compute_image_contributions_batch(
        source,
        microphone,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="omni",
        src_dirs=None,
        mic_dir=None,
    )
    time_batch = _compute_image_contributions_time_batch(
        source.unsqueeze(0).expand(2, -1, -1),
        microphone.unsqueeze(0).expand(2, -1, -1),
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="omni",
        src_dirs=None,
        mic_dir=None,
    )

    for samples, attenuation in (single, (batch[0][0], batch[1][0])):
        assert torch.all(torch.isfinite(samples))
        assert torch.all(torch.isfinite(attenuation))
        torch.testing.assert_close(samples[0], expected_samples, rtol=1.0e-15, atol=0.0)
        torch.testing.assert_close(
            attenuation[0], expected_attenuation, rtol=1.0e-15, atol=0.0
        )
    torch.testing.assert_close(
        time_batch[0],
        batch[0].unsqueeze(0).expand(2, -1, -1, -1),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        time_batch[1],
        batch[1].unsqueeze(0).expand(2, -1, -1, -1),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.numerical
def test_extreme_finite_delay_scaling_avoids_division_overflow() -> None:
    room = Room.shoebox(
        [2.0e30, 2.0e30],
        fs=1.0e-308,
        c=1.0e-296,
        beta=[1.0] * 4,
        dtype=torch.float64,
    )
    sample, attenuation = _compute_image_contributions_batch(
        torch.tensor([[5.0e29, 5.0e29]], dtype=torch.float64),
        torch.tensor([[1.5e30, 5.0e29]], dtype=torch.float64),
        room.size,
        torch.zeros((1, 2), dtype=torch.int64),
        torch.ones(1, dtype=torch.float64),
        room,
        src_pattern="omni",
        mic_pattern="omni",
        src_dirs=None,
        mic_dir=None,
    )

    torch.testing.assert_close(
        sample,
        torch.tensor([[[1.0e18]]], dtype=torch.float64),
        rtol=1.0e-15,
        atol=0.0,
    )
    torch.testing.assert_close(
        attenuation,
        torch.tensor([[[1.0e-30]]], dtype=torch.float64),
        rtol=1.0e-15,
        atol=0.0,
    )


def test_cos_between_stably_normalizes_extreme_finite_vectors() -> None:
    cosine = _cos_between(
        torch.tensor([[1.0e308, 1.0e308]], dtype=torch.float64),
        torch.tensor([[1.0, 0.0]], dtype=torch.float64),
    )

    torch.testing.assert_close(
        cosine,
        torch.tensor([2.0**-0.5], dtype=torch.float64),
        rtol=1.0e-15,
        atol=0.0,
    )


@pytest.mark.numerical
def test_reflected_source_directions_flip_odd_axes() -> None:
    source_directions = torch.tensor([[1.0, 2.0, 3.0]])
    indices = torch.tensor([[0, 0, 0], [1, 0, -1], [2, -2, 3]])
    expected = torch.tensor([[[1.0, 2.0, 3.0], [-1.0, 2.0, -3.0], [1.0, 2.0, -3.0]]])
    torch.testing.assert_close(
        _reflected_source_directions(source_directions, indices), expected
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("pattern", "back_gain"),
    [
        ("homni", 0.0),
        ("subcardioid", 0.5),
        ("cardioid", 0.0),
        ("hypercardioid", -0.5),
        ("bidir", -1.0),
    ],
)
def test_reflected_source_directivity_matches_analytic_back_gain(
    pattern: str, back_gain: float
) -> None:
    room = Room.shoebox([10.0, 10.0, 10.0], fs=343.0, c=343.0, beta=[1.0] * 6)
    source = torch.tensor([5.0, 2.0, 5.0])
    microphone = torch.tensor([[5.0, 8.0, 5.0]])
    indices = torch.tensor([[0, -1, 0]])
    reflection = torch.ones(1)
    source_direction = torch.tensor([0.0, 1.0, 0.0])

    _, attenuation = _compute_image_contributions(
        source,
        microphone,
        room.size,
        indices,
        reflection,
        room,
        src_pattern=pattern,
        mic_pattern="omni",
        src_dir=source_direction,
        mic_dir=None,
    )
    assert attenuation.item() == pytest.approx(back_gain / 10.0, abs=1e-7)


@pytest.mark.numerical
def test_reflected_source_directivity_is_consistent_in_all_batch_paths() -> None:
    room = Room.shoebox([10.0, 10.0, 10.0], fs=343.0, c=343.0, beta=[1.0] * 6)
    source = torch.tensor([[5.0, 2.0, 5.0]])
    microphone = torch.tensor([[5.0, 8.0, 5.0]])
    indices = torch.tensor([[0, -1, 0]])
    reflection = torch.ones(1)
    direction = torch.tensor([[0.0, 1.0, 0.0]])

    _, static_attenuation = _compute_image_contributions_batch(
        source,
        microphone,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="cardioid",
        mic_pattern="omni",
        src_dirs=direction,
        mic_dir=None,
    )
    _, dynamic_attenuation = _compute_image_contributions_time_batch(
        source.unsqueeze(0).repeat(2, 1, 1),
        microphone.unsqueeze(0).repeat(2, 1, 1),
        room.size,
        indices,
        reflection,
        room,
        src_pattern="cardioid",
        mic_pattern="omni",
        src_dirs=direction,
        mic_dir=None,
    )

    torch.testing.assert_close(static_attenuation, torch.zeros_like(static_attenuation))
    torch.testing.assert_close(
        dynamic_attenuation, torch.zeros_like(dynamic_attenuation)
    )


@pytest.mark.numerical
def test_single_source_contributions_broadcast_per_microphone_directions() -> None:
    room = Room.shoebox(
        [10.0, 8.0], fs=8.0, c=4.0, beta=[0.2, 0.3, 0.5, 0.7], dtype=torch.float64
    )
    source = torch.tensor([2.0, 3.0], dtype=torch.float64)
    microphones = torch.tensor([[6.0, 3.0], [2.0, 6.0]], dtype=torch.float64)
    indices = torch.tensor([[0, 0], [-1, 0], [1, 0]])
    assert room.beta is not None
    reflection = _reflection_coefficients(indices, room.beta)
    microphone_directions = torch.tensor(
        [[-1.0, 0.0], [0.0, -1.0]], dtype=torch.float64
    )

    samples, attenuation = _compute_image_contributions(
        source,
        microphones,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="cardioid",
        src_dir=None,
        mic_dir=microphone_directions,
    )
    batch_samples, batch_attenuation = _compute_image_contributions_batch(
        source.unsqueeze(0),
        microphones,
        room.size,
        indices,
        reflection,
        room,
        src_pattern="omni",
        mic_pattern="cardioid",
        src_dirs=None,
        mic_dir=microphone_directions,
    )

    torch.testing.assert_close(samples, batch_samples[0], rtol=0, atol=0)
    torch.testing.assert_close(attenuation, batch_attenuation[0], rtol=0, atol=0)


def _fractional_delay_reference(
    samples: torch.Tensor,
    amplitudes: torch.Tensor,
    *,
    nsample: int,
    fdl: int,
) -> torch.Tensor:
    output = torch.zeros(nsample, dtype=samples.dtype)
    half = (fdl - 1) // 2
    window = torch.hann_window(fdl, periodic=False, dtype=samples.dtype)
    for sample, amplitude in zip(samples.flatten(), amplitudes.flatten(), strict=True):
        center = int(torch.floor(sample).item())
        fraction = sample - center
        for tap in range(fdl):
            target = center + tap - half
            if 0 <= target < nsample:
                delay = torch.tensor(tap - half, dtype=samples.dtype) - fraction
                output[target] += amplitude * torch.sinc(delay) * window[tap]
    return output


def _boundary_arrival_reference(
    samples: torch.Tensor,
    *,
    nsample: int,
    fdl: int,
) -> torch.Tensor:
    return torch.stack(
        [
            _fractional_delay_reference(
                sample.reshape(1),
                sample.reciprocal().reshape(1),
                nsample=nsample,
                fdl=fdl,
            )
            for sample in samples
        ]
    )


@pytest.mark.numerical
@pytest.mark.parametrize("use_lut", [False, True])
def test_static_physical_axis_matches_direct_reference_at_fractional_boundaries(
    use_lut: bool,
) -> None:
    nsample = 8
    fdl = 7
    room = Room.shoebox(
        [10.0, 3.0],
        fs=8.0,
        c=8.0,
        beta=[0.0] * 4,
        dtype=torch.float64,
    )
    source = Source.from_positions([[1.0, 1.0]], dtype=torch.float64)
    microphones = MicrophoneArray.from_positions(
        [[1.25, 1.0], [8.75, 1.0]], dtype=torch.float64
    )
    result = simulate(
        StaticScene(room=room, sources=source, mics=microphones),
        SimulationConfig(
            max_order=0,
            nsample=nsample,
            frac_delay_length=fdl,
            use_lut=use_lut,
            dtype=torch.float64,
        ),
    )
    physical_samples = torch.tensor([0.25, 7.75], dtype=torch.float64)
    expected = _boundary_arrival_reference(
        physical_samples,
        nsample=nsample,
        fdl=fdl,
    )

    assert result.rirs.is_contiguous()
    torch.testing.assert_close(result.rirs[0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("use_lut", [False, True])
def test_dynamic_physical_axis_matches_direct_reference_at_fractional_boundaries(
    use_lut: bool,
) -> None:
    nsample = 8
    fdl = 7
    room = Room.shoebox(
        [10.0, 3.0],
        fs=8.0,
        c=8.0,
        beta=[0.0] * 4,
        dtype=torch.float64,
    )
    source_trajectory = torch.tensor(
        [[[1.25, 1.0]], [[8.75, 1.0]]], dtype=torch.float64
    )
    microphone_trajectory = torch.tensor(
        [[[1.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float64
    )
    scene = DynamicScene(
        room=room,
        sources=Source.from_positions(source_trajectory[0], dtype=torch.float64),
        mics=MicrophoneArray.from_positions(
            microphone_trajectory[0], dtype=torch.float64
        ),
        src_traj=source_trajectory,
        mic_traj=microphone_trajectory,
    )
    result = simulate(
        scene,
        SimulationConfig(
            max_order=0,
            nsample=nsample,
            frac_delay_length=fdl,
            use_lut=use_lut,
            dtype=torch.float64,
        ),
    )
    physical_samples = torch.tensor([0.25, 7.75], dtype=torch.float64)
    expected = _boundary_arrival_reference(
        physical_samples,
        nsample=nsample,
        fdl=fdl,
    )

    assert result.rirs.is_contiguous()
    torch.testing.assert_close(result.rirs[:, 0, 0], expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fractional_delay_accumulation_matches_direct_reference(
    dtype: torch.dtype,
) -> None:
    fdl = 7
    nsample = 16
    samples = torch.tensor([[[0.25, 5.5, 15.75, 5.5]]], dtype=dtype)
    amplitudes = torch.tensor([[[1.0, -0.5, 0.25, 0.125]]], dtype=dtype)
    actual = torch.zeros((1, 1, nsample), dtype=dtype)
    config = _resolved_accumulation_config(
        nsample=nsample,
        dtype=dtype,
        frac_delay_length=fdl,
        use_lut=False,
        accumulate_chunk_size=2,
    )
    _accumulate_rir_batch(actual, samples, amplitudes, config)
    expected = _fractional_delay_reference(
        samples, amplitudes, nsample=nsample, fdl=fdl
    )
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual[0, 0], expected, rtol=tolerance, atol=tolerance)


@pytest.mark.numerical
@pytest.mark.parametrize("use_lut", [False, True])
def test_fractional_delay_accumulation_passes_gradcheck(use_lut: bool) -> None:
    samples = torch.tensor(
        [[[0.23, 5.37], [2.13, 11.27]], [[1.17, 6.33], [3.21, 10.39]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    amplitudes = torch.linspace(-0.4, 0.7, samples.numel(), dtype=torch.float64)
    amplitudes = amplitudes.reshape_as(samples).requires_grad_()
    config = _resolved_accumulation_config(
        nsample=12,
        dtype=torch.float64,
        frac_delay_length=7,
        use_lut=use_lut,
        accumulate_chunk_size=1,
    )

    def accumulate(
        sample_values: torch.Tensor, amplitude_values: torch.Tensor
    ) -> torch.Tensor:
        output = sample_values.new_zeros((2, 2, 12))
        _accumulate_rir_batch(output, sample_values, amplitude_values, config)
        return output

    assert torch.autograd.gradcheck(accumulate, (samples, amplitudes))


@pytest.mark.numerical
@pytest.mark.parametrize("use_lut", [False, True])
def test_accumulation_masks_non_castable_samples_before_integer_conversion(
    use_lut: bool,
) -> None:
    samples = torch.tensor(
        [[[1.25, 1.0e20, float("inf"), float("nan")]]],
        dtype=torch.float64,
    )
    amplitudes = torch.tensor(
        [[[1.0, float("nan"), float("nan"), float("nan")]]],
        dtype=torch.float64,
    )
    actual = torch.zeros((1, 1, 8), dtype=torch.float64)
    config = _resolved_accumulation_config(
        nsample=8,
        dtype=torch.float64,
        frac_delay_length=7,
        use_lut=use_lut,
    )

    _accumulate_rir_batch(actual, samples, amplitudes, config)

    expected = _fractional_delay_reference(
        samples[..., :1],
        amplitudes[..., :1],
        nsample=8,
        fdl=7,
    )
    assert torch.all(torch.isfinite(actual))
    tolerance = 2.0e-3 if use_lut else 1.0e-13
    torch.testing.assert_close(actual[0, 0], expected, rtol=tolerance, atol=tolerance)


@pytest.mark.numerical
def test_accumulation_is_independent_of_chunk_size() -> None:
    generator = torch.Generator().manual_seed(1234)
    samples = torch.rand((2, 3, 31), generator=generator) * 60.0
    amplitudes = torch.randn((2, 3, 31), generator=generator)
    outputs = []
    for chunk_size in (1, 7, 4096):
        output = torch.zeros((2, 3, 96))
        config = _resolved_accumulation_config(
            nsample=96,
            dtype=output.dtype,
            frac_delay_length=21,
            use_lut=False,
            accumulate_chunk_size=chunk_size,
        )
        _accumulate_rir_batch(output, samples, amplitudes, config)
        outputs.append(output)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(outputs[0], outputs[2], rtol=0, atol=0)


@pytest.mark.numerical
def test_sinc_lut_stays_close_to_analytic_sinc() -> None:
    generator = torch.Generator().manual_seed(4321)
    samples = 20.0 + torch.rand((1, 2, 100), generator=generator) * 100.0
    amplitudes = torch.randn((1, 2, 100), generator=generator)
    outputs = []
    for use_lut in (False, True):
        output = torch.zeros((1, 2, 160))
        config = _resolved_accumulation_config(
            nsample=160,
            dtype=output.dtype,
            frac_delay_length=81,
            sinc_lut_granularity=20,
            use_lut=use_lut,
        )
        _accumulate_rir_batch(output, samples, amplitudes, config)
        outputs.append(output)
    relative_l2 = torch.linalg.vector_norm(
        outputs[1] - outputs[0]
    ) / torch.linalg.vector_norm(outputs[0])
    assert relative_l2.item() < 2e-3
