"""Numerical oracle tests for the image-source implementation.

These tests deliberately exercise low-level kernels.  End-to-end comparison
tests alone cannot identify whether an error comes from image geometry,
reflection gains, directivity, or fractional-delay accumulation.
"""

from __future__ import annotations

import pytest
import torch

from torchrir.config import SimulationConfig
from torchrir.models import Room
from torchrir.sim.ism.accumulate import _accumulate_rir_batch
from torchrir.sim.ism.contributions import (
    _compute_image_contributions,
    _compute_image_contributions_batch,
    _compute_image_contributions_time_batch,
    _reflected_source_directions,
)
from torchrir.sim.ism.images import (
    _image_positions,
    _image_source_indices,
    _reflection_coefficients,
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
    indices = _image_source_indices(0, 3, device=torch.device("cpu"), nb_img=(1, 2, 0))
    expected = {(x, y, 0) for x in range(-1, 2) for y in range(-2, 3)}
    assert {tuple(row.tolist()) for row in indices} == expected


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
        0,
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
        0,
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
        0,
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
        0,
        src_pattern="cardioid",
        mic_pattern="omni",
        src_dirs=direction,
        mic_dir=None,
    )

    torch.testing.assert_close(static_attenuation, torch.zeros_like(static_attenuation))
    torch.testing.assert_close(
        dynamic_attenuation, torch.zeros_like(dynamic_attenuation)
    )


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
    config = SimulationConfig(
        frac_delay_length=fdl,
        use_lut=False,
        accumulate_chunk_size=2,
        rir_hpf_enable=False,
    )
    _accumulate_rir_batch(actual, samples, amplitudes, config)
    expected = _fractional_delay_reference(
        samples, amplitudes, nsample=nsample, fdl=fdl
    )
    tolerance = 2e-6 if dtype == torch.float32 else 1e-13
    torch.testing.assert_close(actual[0, 0], expected, rtol=tolerance, atol=tolerance)


@pytest.mark.numerical
def test_accumulation_is_independent_of_chunk_size() -> None:
    generator = torch.Generator().manual_seed(1234)
    samples = torch.rand((2, 3, 31), generator=generator) * 60.0
    amplitudes = torch.randn((2, 3, 31), generator=generator)
    outputs = []
    for chunk_size in (1, 7, 4096):
        output = torch.zeros((2, 3, 96))
        config = SimulationConfig(
            frac_delay_length=21,
            use_lut=False,
            accumulate_chunk_size=chunk_size,
            rir_hpf_enable=False,
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
        config = SimulationConfig(
            frac_delay_length=81,
            sinc_lut_granularity=20,
            use_lut=use_lut,
            rir_hpf_enable=False,
        )
        _accumulate_rir_batch(output, samples, amplitudes, config)
        outputs.append(output)
    relative_l2 = torch.linalg.vector_norm(
        outputs[1] - outputs[0]
    ) / torch.linalg.vector_norm(outputs[0])
    assert relative_l2.item() < 2e-3
