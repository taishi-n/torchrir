from typing import Any, cast

import numpy as np
import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, RIRResult, Room, Source
from torchrir.config import SimulationConfig, _resolve_simulation_config
from torchrir.signal import (
    DynamicConvolver,
    FrameSchedule,
    convolve_rir,
    fft_convolve,
)


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("signal_len, rir_len", [(1, 1), (3, 5), (17, 8), (127, 64)])
def test_fft_convolve_matches_numpy_direct_convolution(
    dtype: torch.dtype, signal_len: int, rir_len: int
) -> None:
    generator = torch.Generator().manual_seed(signal_len * 100 + rir_len)
    signal = torch.randn(signal_len, generator=generator, dtype=dtype)
    rir = torch.randn(rir_len, generator=generator, dtype=dtype)

    actual = fft_convolve(signal, rir)
    expected = torch.from_numpy(np.convolve(signal.numpy(), rir.numpy()))

    tolerance = 1e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.numerical
@pytest.mark.parametrize(
    ("signal", "rir", "expected"),
    [
        ([1.0, -2.0, 3.0], [1.0], [1.0, -2.0, 3.0]),
        ([1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        ([0.0, 0.0], [1.0, -1.0], [0.0, 0.0, 0.0]),
    ],
)
def test_fft_convolve_exact_boundary_cases(
    signal: list[float], rir: list[float], expected: list[float]
) -> None:
    actual = fft_convolve(torch.tensor(signal), torch.tensor(rir))
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=2e-7)


def test_frame_schedule_sample_factories() -> None:
    explicit = FrameSchedule.from_samples([0, 3, 8])
    uniform = FrameSchedule.uniform(frame_count=6, stop_sample=10)
    fixed_hop = FrameSchedule.fixed_hop(stop_sample=10, hop_size=4)

    assert explicit.starts.device.type == "cpu"
    assert explicit.starts.dtype == torch.int64
    torch.testing.assert_close(explicit.starts, torch.tensor([0, 3, 8]))
    torch.testing.assert_close(uniform.starts, torch.tensor([0, 1, 3, 5, 6, 8]))
    torch.testing.assert_close(fixed_hop.starts, torch.tensor([0, 4, 8]))


def test_frame_schedule_uniform_avoids_int64_intermediate_overflow() -> None:
    schedule = FrameSchedule.uniform(frame_count=3, stop_sample=2**62)

    assert schedule.starts.tolist() == [
        0,
        1_537_228_672_809_129_301,
        3_074_457_345_618_258_602,
    ]


def test_frame_schedule_fixed_hop_handles_int64_endpoint() -> None:
    maximum = torch.iinfo(torch.int64).max

    one_frame = FrameSchedule.fixed_hop(
        stop_sample=maximum,
        hop_size=maximum,
    )
    two_frames = FrameSchedule.fixed_hop(
        stop_sample=maximum,
        hop_size=2**62,
    )

    assert one_frame.starts.tolist() == [0]
    assert two_frames.starts.tolist() == [0, 2**62]


def test_frame_schedule_starts_are_mutation_safe() -> None:
    schedule = FrameSchedule.from_samples([0, 3, 8])
    snapshot = schedule.starts
    snapshot[0] = 2
    torch.testing.assert_close(schedule.starts, torch.tensor([0, 3, 8]))


def test_frame_schedule_repr_includes_starts_and_conversion_provenance() -> None:
    samples = FrameSchedule.from_samples([0, 3, 8])
    seconds = FrameSchedule.from_seconds([0.0, 0.5, 1.0], sample_rate=8)

    assert repr(samples) == (
        "FrameSchedule(starts=[0, 3, 8], conversion_sample_rate=None)"
    )
    assert repr(seconds) == (
        "FrameSchedule(starts=[0, 4, 8], conversion_sample_rate=8.0)"
    )


def test_frame_schedule_repr_bounds_large_schedule_output() -> None:
    schedule = FrameSchedule.fixed_hop(stop_sample=20, hop_size=1)

    assert repr(schedule) == (
        "FrameSchedule(starts=[0, 1, 2, 3, ..., 17, 18, 19] (20 frames), "
        "conversion_sample_rate=None)"
    )


def test_frame_schedule_seconds_use_cpu_float64_flooring() -> None:
    schedule = FrameSchedule.from_seconds(
        [0.0, 251 / 16000, 500 / 16000],
        sample_rate=16000,
    )
    torch.testing.assert_close(schedule.starts, torch.tensor([0, 251, 500]))


def test_frame_schedule_normalized_progress_uses_exact_sample_starts() -> None:
    schedule = FrameSchedule.uniform(frame_count=4, stop_sample=10)

    progress = schedule.normalized_progress(
        stop_sample=10,
        dtype=torch.float64,
    )

    torch.testing.assert_close(
        progress,
        torch.tensor([0.0, 0.2, 0.5, 0.7], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_frame_schedule_normalized_progress_validates_endpoint_and_dtype() -> None:
    schedule = FrameSchedule.from_samples([0, 5])
    with pytest.raises(ValueError, match="before stop_sample"):
        schedule.normalized_progress(stop_sample=5)
    with pytest.raises(TypeError, match="stop_sample"):
        schedule.normalized_progress(stop_sample=cast(Any, True))
    with pytest.raises(TypeError, match="supported"):
        schedule.normalized_progress(
            stop_sample=6,
            dtype=torch.float8_e4m3fn,
        )
    with pytest.raises(ValueError, match="device type"):
        schedule.normalized_progress(stop_sample=6, device="meta")
    with pytest.raises(ValueError, match="int64"):
        schedule.normalized_progress(stop_sample=10**400)


def test_frame_schedule_normalized_progress_resolves_auto_device() -> None:
    progress = FrameSchedule.from_samples([0, 5]).normalized_progress(
        stop_sample=6,
        device="auto",
    )

    assert progress.device.type in {"cpu", "cuda", "mps"}


def test_frame_schedule_normalized_progress_avoids_low_precision_overflow() -> None:
    progress = FrameSchedule.from_samples([0, 70_000]).normalized_progress(
        stop_sample=80_000,
        dtype=torch.float16,
    )

    assert torch.all(torch.isfinite(progress))
    torch.testing.assert_close(
        progress,
        torch.tensor([0.0, 0.875], dtype=torch.float16),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("dtype", "stop_sample"),
    [
        (torch.float16, 4_096),
        (torch.bfloat16, 512),
        (torch.float32, 33_554_432),
        (torch.float64, 18_014_398_509_481_984),
    ],
)
def test_frame_schedule_progress_never_rounds_last_start_to_endpoint(
    dtype: torch.dtype,
    stop_sample: int,
) -> None:
    progress = FrameSchedule.from_samples([0, stop_sample - 1]).normalized_progress(
        stop_sample=stop_sample,
        dtype=dtype,
    )

    one = torch.tensor(1, dtype=dtype)
    assert progress[-1] < one
    assert progress[-1] == torch.nextafter(one, torch.zeros_like(one))


def test_frame_schedule_progress_rejects_dtype_collapsed_frames() -> None:
    stop_sample = 2**25
    schedule = FrameSchedule.from_samples([0, stop_sample - 2, stop_sample - 1])

    with pytest.raises(ValueError, match="not distinct"):
        schedule.normalized_progress(
            stop_sample=stop_sample,
            dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "time_device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param("mps", marks=pytest.mark.mps),
    ],
)
def test_frame_schedule_is_independent_of_time_device(time_device: str) -> None:
    if time_device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if time_device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    times = torch.tensor([0.0, 0.25, 0.5], device=time_device)
    schedule = FrameSchedule.from_seconds(times, sample_rate=8)
    assert schedule.starts.device.type == "cpu"
    torch.testing.assert_close(schedule.starts, torch.tensor([0, 2, 4]))


@pytest.mark.parametrize(
    ("factory", "error", "message"),
    [
        (
            lambda: FrameSchedule.from_samples([]),
            ValueError,
            "non-empty",
        ),
        (
            lambda: FrameSchedule.from_samples(torch.tensor([0.0, 1.0])),
            TypeError,
            "integer",
        ),
        (
            lambda: FrameSchedule.from_samples([0, True]),
            TypeError,
            "integer",
        ),
        (
            lambda: FrameSchedule.from_samples([0, 2**63]),
            ValueError,
            "int64",
        ),
        (
            lambda: FrameSchedule.from_samples(
                torch.empty(2, dtype=torch.int64, device="meta")
            ),
            ValueError,
            "CPU, CUDA, or MPS",
        ),
        (
            lambda: FrameSchedule.from_samples(
                torch.sparse_coo_tensor(
                    torch.tensor([[0, 1]]),
                    torch.tensor([0, 1]),
                    size=(2,),
                )
            ),
            TypeError,
            "dense strided",
        ),
        (
            lambda: FrameSchedule.from_samples(
                torch.quantize_per_tensor(
                    torch.tensor([0.0, 1.0]),
                    scale=0.1,
                    zero_point=0,
                    dtype=torch.qint8,
                )
            ),
            TypeError,
            "quantized",
        ),
        (
            lambda: FrameSchedule.from_samples([1, 2]),
            ValueError,
            "first frame start",
        ),
        (
            lambda: FrameSchedule.from_samples([0, 2, 2]),
            ValueError,
            "strictly increasing",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 0.01], sample_rate=10),
            ValueError,
            "map to strictly increasing",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 1.0e20], sample_rate=1),
            ValueError,
            "int64",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 10**400], sample_rate=1),
            ValueError,
            "representable as float64",
        ),
        (
            lambda: FrameSchedule.from_seconds(
                torch.empty(2, device="meta"), sample_rate=10
            ),
            ValueError,
            "CPU, CUDA, or MPS",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 0.1], sample_rate=0),
            ValueError,
            "sample_rate",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 0.1], sample_rate=cast(Any, True)),
            TypeError,
            "sample_rate",
        ),
        (
            lambda: FrameSchedule.from_seconds(
                [0.0, 0.1], sample_rate=cast(Any, np.bool_(True))
            ),
            TypeError,
            "sample_rate",
        ),
        (
            lambda: FrameSchedule.from_seconds(
                [0.0, 0.1], sample_rate=cast(Any, torch.tensor(10.0))
            ),
            TypeError,
            "sample_rate",
        ),
        (
            lambda: FrameSchedule.from_seconds([0.0, 0.1], sample_rate=10**400),
            ValueError,
            "sample_rate",
        ),
        (
            lambda: FrameSchedule.from_seconds(cast(Any, [False, 0.1]), sample_rate=10),
            TypeError,
            "booleans",
        ),
        (
            lambda: FrameSchedule.uniform(frame_count=3, stop_sample=2),
            ValueError,
            "cannot exceed",
        ),
        (
            lambda: FrameSchedule.fixed_hop(stop_sample=8, hop_size=0),
            ValueError,
            "hop_size",
        ),
    ],
)
def test_frame_schedule_rejects_invalid_values(
    factory: Any,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        factory()


@pytest.mark.numerical
def test_dynamic_fixed_hop_with_constant_rir_equals_static_convolution() -> None:
    generator = torch.Generator().manual_seed(10)
    signal = torch.randn(23, generator=generator, dtype=torch.float64)
    rir = torch.randn(5, generator=generator, dtype=torch.float64)
    dynamic_rirs = rir.repeat(4, 1)
    schedule = FrameSchedule.fixed_hop(stop_sample=signal.numel(), hop_size=7)

    actual = DynamicConvolver(time_reference="emission").convolve(
        signal,
        dynamic_rirs,
        schedule=schedule,
    )
    expected = fft_convolve(signal, rir).unsqueeze(0)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
def test_convolve_rir_multi_source_multi_mic_matches_numpy_sum() -> None:
    generator = torch.Generator().manual_seed(20)
    signal = torch.randn(2, 31, generator=generator, dtype=torch.float64)
    rirs = torch.randn(2, 3, 7, generator=generator, dtype=torch.float64)
    actual = convolve_rir(signal, rirs)
    expected = np.stack(
        [
            sum(
                np.convolve(signal[source].numpy(), rirs[source, mic].numpy())
                for source in range(2)
            )
            for mic in range(3)
        ]
    )
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=1e-12, atol=1e-12
    )


@pytest.mark.numerical
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
        pytest.param(
            "mps",
            marks=[
                pytest.mark.mps,
                pytest.mark.filterwarnings(
                    r"ignore:An output with one or more elements was resized since "
                    r"it had shape \[\], which does not match the required output "
                    r"shape .*:UserWarning"
                ),
                pytest.mark.filterwarnings(
                    r"ignore:MPS.*The constant padding of more than 3 dimensions is "
                    r"not currently supported natively\. It uses View Ops default "
                    r"implementation to run\. This may have performance "
                    r"implications\..*:UserWarning"
                ),
            ],
        ),
    ],
)
def test_dynamic_emission_batched_path_supports_autograd(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    generator = torch.Generator().manual_seed(21)
    base_signal = torch.randn(
        2,
        11,
        generator=generator,
        dtype=torch.float32,
    )
    base_rirs = torch.randn(
        3,
        2,
        2,
        5,
        generator=generator,
        dtype=torch.float32,
    )

    def run(target: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal = base_signal.to(device=target).detach().requires_grad_()
        rirs = base_rirs.to(device=target).detach().requires_grad_()
        output = DynamicConvolver(time_reference="emission").convolve(
            signal,
            rirs,
            schedule=FrameSchedule.from_samples([0, 4, 8]),
        )
        signal_grad, rir_grad = torch.autograd.grad(
            output.square().mean(),
            (signal, rirs),
        )
        return (
            output.detach().cpu(),
            signal_grad.detach().cpu(),
            rir_grad.detach().cpu(),
        )

    actual = run(device)
    for value in actual:
        assert torch.all(torch.isfinite(value))
        assert torch.count_nonzero(value) > 0
    if device != "cpu":
        reference = run("cpu")
        for actual_value, reference_value in zip(actual, reference, strict=True):
            torch.testing.assert_close(
                actual_value,
                reference_value,
                rtol=3e-4,
                atol=3e-5,
            )


@pytest.mark.numerical
def test_static_fft_paths_support_autograd_on_cpu() -> None:
    generator = torch.Generator().manual_seed(211)
    signal_1d = torch.randn(13, generator=generator, requires_grad=True)
    rir_1d = torch.randn(5, generator=generator, requires_grad=True)
    output_1d = fft_convolve(signal_1d, rir_1d)
    signal_grad_1d, rir_grad_1d = torch.autograd.grad(
        output_1d.square().mean(),
        (signal_1d, rir_1d),
    )

    signal = torch.randn(2, 13, generator=generator, requires_grad=True)
    rirs = torch.randn(2, 3, 5, generator=generator, requires_grad=True)
    output = convolve_rir(signal, rirs)
    signal_grad, rir_grad = torch.autograd.grad(
        output.square().mean(),
        (signal, rirs),
    )

    for value in (signal_grad_1d, rir_grad_1d, signal_grad, rir_grad):
        assert torch.all(torch.isfinite(value))
        assert torch.count_nonzero(value) > 0


@pytest.mark.numerical
def test_dynamic_observation_path_supports_autograd_on_cpu() -> None:
    generator = torch.Generator().manual_seed(212)
    signal = torch.randn(2, 11, generator=generator, requires_grad=True)
    rirs = torch.randn(3, 2, 2, 5, generator=generator, requires_grad=True)
    output = DynamicConvolver(time_reference="observation").convolve(
        signal,
        rirs,
        schedule=FrameSchedule.from_samples([0, 4, 8]),
    )

    signal_grad, rir_grad = torch.autograd.grad(
        output.square().mean(),
        (signal, rirs),
    )

    assert torch.all(torch.isfinite(signal_grad))
    assert torch.all(torch.isfinite(rir_grad))
    assert torch.count_nonzero(signal_grad) > 0
    assert torch.count_nonzero(rir_grad) > 0


@pytest.mark.numerical
def test_dynamic_emission_chunk_boundary_passes_gradcheck() -> None:
    generator = torch.Generator().manual_seed(213)
    signal = (
        torch.randn(1, 11, generator=generator, dtype=torch.float64) / 10
    ).requires_grad_()
    rirs = (
        torch.randn(9, 1, 1, 3, generator=generator, dtype=torch.float64) / 10
    ).requires_grad_()
    schedule = FrameSchedule.from_samples(range(9))
    convolver = DynamicConvolver(time_reference="emission")

    assert torch.autograd.gradcheck(
        lambda signal_value, rir_value: convolver.convolve(
            signal_value,
            rir_value,
            schedule=schedule,
        ),
        (signal, rirs),
    )


@pytest.mark.numerical
def test_dynamic_observation_passes_gradcheck() -> None:
    generator = torch.Generator().manual_seed(214)
    signal = (
        torch.randn(1, 5, generator=generator, dtype=torch.float64) / 10
    ).requires_grad_()
    rirs = (
        torch.randn(3, 1, 1, 3, generator=generator, dtype=torch.float64) / 10
    ).requires_grad_()
    schedule = FrameSchedule.from_samples([0, 2, 5])
    convolver = DynamicConvolver(time_reference="observation")

    assert torch.autograd.gradcheck(
        lambda signal_value, rir_value: convolver.convolve(
            signal_value,
            rir_value,
            schedule=schedule,
        ),
        (signal, rirs),
    )


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_static_fft_paths_promote_low_precision_on_cpu(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(22)
    signal = torch.randn(2, 17, generator=generator).to(dtype)
    rirs = torch.randn(2, 3, 6, generator=generator).to(dtype)
    tolerance = 4 * torch.finfo(dtype).eps

    actual_1d = fft_convolve(signal[0], rirs[0, 0])
    expected_1d = fft_convolve(signal[0].float(), rirs[0, 0].float())
    assert actual_1d.dtype == dtype
    torch.testing.assert_close(
        actual_1d.float(),
        expected_1d,
        rtol=tolerance,
        atol=tolerance,
    )

    actual_batched = convolve_rir(signal, rirs)
    expected_batched = convolve_rir(signal.float(), rirs.float())
    assert actual_batched.dtype == dtype
    torch.testing.assert_close(
        actual_batched.float(),
        expected_batched,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("time_reference", ["emission", "observation"])
def test_dynamic_fft_paths_promote_low_precision_on_cpu(
    dtype: torch.dtype,
    time_reference: str,
) -> None:
    generator = torch.Generator().manual_seed(23)
    signal = torch.randn(2, 11, generator=generator).to(dtype)
    rirs = torch.randn(3, 2, 2, 5, generator=generator).to(dtype)
    schedule = FrameSchedule.from_samples([0, 4, 8])
    convolver = DynamicConvolver(time_reference=cast(Any, time_reference))
    tolerance = 4 * torch.finfo(dtype).eps

    actual = convolver.convolve(signal, rirs, schedule=schedule)
    expected = convolver.convolve(signal.float(), rirs.float(), schedule=schedule)

    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.numerical
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dynamic_emission_low_precision_casts_only_final_sum(
    dtype: torch.dtype,
) -> None:
    generator = torch.Generator().manual_seed(24)
    signal = torch.randn(32, 127, generator=generator).to(dtype)
    rirs = torch.randn(5, 32, 2, 19, generator=generator).to(dtype)
    schedule = FrameSchedule.from_samples([0, 17, 43, 71, 103])
    convolver = DynamicConvolver(time_reference="emission")

    actual = convolver.convolve(signal, rirs, schedule=schedule)
    float32_reference = convolver.convolve(
        signal.float(),
        rirs.float(),
        schedule=schedule,
    ).to(dtype)

    torch.testing.assert_close(actual, float32_reference, rtol=0, atol=0)


@pytest.mark.numerical
def test_dynamic_observation_low_precision_casts_only_final_result() -> None:
    tiny = torch.finfo(torch.float16).smallest_normal * torch.finfo(torch.float16).eps
    signal = torch.tensor([tiny], dtype=torch.float16)
    rirs = torch.tensor([[-tiny]], dtype=torch.float16)
    schedule = FrameSchedule.from_samples([0])
    convolver = DynamicConvolver(time_reference="observation")

    actual = convolver.convolve(signal, rirs, schedule=schedule)
    float32_reference = convolver.convolve(
        signal.float(),
        rirs.float(),
        schedule=schedule,
    ).to(torch.float16)

    torch.testing.assert_close(actual, float32_reference, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(float32_reference))


@pytest.mark.numerical
def test_dynamic_emission_matches_segment_reference() -> None:
    signal = torch.arange(1, 13, dtype=torch.float64).repeat(2, 1)
    rirs = torch.tensor(
        [
            [
                [[1.0, 0.5], [0.0, 1.0]],
                [[-1.0, 0.0], [0.5, -0.5]],
            ],
            [
                [[2.0, 0.0], [0.0, -1.0]],
                [[0.0, 1.0], [1.0, 0.5]],
            ],
            [
                [[-0.5, 0.25], [1.0, 0.0]],
                [[0.25, 0.25], [-1.0, 1.0]],
            ],
        ],
        dtype=torch.float64,
    )
    schedule = FrameSchedule.fixed_hop(stop_sample=12, hop_size=4)
    actual = DynamicConvolver(time_reference="emission").convolve(
        signal,
        rirs,
        schedule=schedule,
    )
    expected = np.zeros((2, signal.shape[-1] + rirs.shape[-1] - 1))
    for frame in range(3):
        start = frame * 4
        for source in range(2):
            for mic in range(2):
                convolution = np.convolve(
                    signal[source, start : start + 4].numpy(),
                    rirs[frame, source, mic].numpy(),
                )
                expected[mic, start : start + convolution.size] += convolution
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=1e-12, atol=1e-12
    )


@pytest.mark.numerical
def test_dynamic_emission_handles_chunk_boundary_and_uneven_frames() -> None:
    generator = torch.Generator().manual_seed(25)
    signal = torch.randn(2, 109, generator=generator, dtype=torch.float64)
    rirs = torch.randn(9, 2, 2, 7, generator=generator, dtype=torch.float64)
    starts = [0, 1, 2, 3, 4, 5, 6, 7, 107]
    schedule = FrameSchedule.from_samples(starts)

    actual = DynamicConvolver(time_reference="emission").convolve(
        signal,
        rirs,
        schedule=schedule,
    )

    expected = np.zeros((2, signal.shape[-1] + rirs.shape[-1] - 1))
    boundaries = [*starts, signal.shape[-1]]
    for frame, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:], strict=True)
    ):
        for source in range(signal.shape[0]):
            for microphone in range(rirs.shape[2]):
                convolution = np.convolve(
                    signal[source, start:end].numpy(),
                    rirs[frame, source, microphone].numpy(),
                )
                expected[
                    microphone,
                    start : start + convolution.size,
                ] += convolution

    torch.testing.assert_close(
        actual,
        torch.from_numpy(expected),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.numerical
def test_dynamic_emission_uses_schedule_boundaries() -> None:
    signal = torch.arange(1, 9, dtype=torch.float64)
    rirs = torch.tensor([[[[1.0]]], [[[2.0]]], [[[-1.0]]]], dtype=torch.float64)
    schedule = FrameSchedule.from_seconds([0.0, 0.3, 0.6], sample_rate=10)
    actual = DynamicConvolver(time_reference="emission").convolve(
        signal,
        rirs,
        schedule=schedule,
    )
    expected = torch.tensor(
        [[1.0, 2.0, 3.0, 8.0, 10.0, 12.0, -7.0, -8.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.numerical
def test_dynamic_observation_selects_rir_at_each_output_sample() -> None:
    signal = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    rirs = torch.tensor(
        [
            [[[1.0, 10.0]]],
            [[[2.0, 20.0]]],
        ],
        dtype=torch.float64,
    )
    actual = DynamicConvolver(time_reference="observation").convolve(
        signal,
        rirs,
        schedule=FrameSchedule.from_samples([0, 2]),
    )

    expected = torch.tensor([[1.0, 12.0, 46.0, 68.0, 80.0]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.numerical
def test_observation_schedule_allows_frame_start_in_convolution_tail() -> None:
    signal = torch.tensor([1.0, 1.0], dtype=torch.float64)
    rirs = torch.tensor(
        [
            [[[1.0, 1.0, 1.0, 1.0]]],
            [[[2.0, 2.0, 2.0, 2.0]]],
        ],
        dtype=torch.float64,
    )
    schedule = FrameSchedule.from_samples([0, 3])

    actual = DynamicConvolver(time_reference="observation").convolve(
        signal,
        rirs,
        schedule=schedule,
    )
    torch.testing.assert_close(
        actual,
        torch.tensor([[1.0, 2.0, 2.0, 4.0, 2.0]], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="input timeline endpoint"):
        DynamicConvolver(time_reference="emission").convolve(
            signal,
            rirs,
            schedule=schedule,
        )


def test_observation_schedule_rejects_start_at_output_endpoint() -> None:
    with pytest.raises(ValueError, match="output timeline endpoint"):
        DynamicConvolver(time_reference="observation").convolve(
            torch.ones(2),
            torch.ones(2, 1, 1, 4),
            schedule=FrameSchedule.from_samples([0, 5]),
        )


def test_dynamic_convolver_requires_explicit_schedule_for_raw_rirs() -> None:
    with pytest.raises(ValueError, match="required for raw"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(8),
            torch.ones(2, 1, 1, 3),
        )


def test_dynamic_convolver_requires_schedule_frame_count_to_match() -> None:
    convolver = DynamicConvolver(time_reference="emission")
    signal = torch.ones(8)
    rirs = torch.ones(2, 1, 1, 3)

    for schedule in (
        FrameSchedule.fixed_hop(stop_sample=8, hop_size=2),
        FrameSchedule.from_samples([0]),
    ):
        with pytest.raises(ValueError, match="schedule has .* but rirs has 2"):
            convolver.convolve(signal, rirs, schedule=schedule)


def test_dynamic_convolver_rejects_ambiguous_3d_rirs_for_multi_source() -> None:
    signal = torch.randn(2, 256)
    ambiguous_rirs = torch.randn(5, 2, 64)
    with pytest.raises(ValueError, match="Use 4D"):
        DynamicConvolver(time_reference="emission").convolve(
            signal,
            ambiguous_rirs,
            schedule=FrameSchedule.uniform(frame_count=5, stop_sample=256),
        )


def test_dynamic_convolver_validates_configuration() -> None:
    with pytest.raises(ValueError, match="time_reference"):
        DynamicConvolver(time_reference=cast(Any, "invalid"))
    convolver = DynamicConvolver(time_reference="emission")
    with pytest.raises(TypeError, match="FrameSchedule"):
        convolver.convolve(
            torch.ones(8),
            torch.ones(1, 1, 1, 3),
            schedule=cast(Any, [0]),
        )


def test_dynamic_convolver_rejects_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(8, dtype=torch.float32),
            torch.ones(2, 1, 1, 3, dtype=torch.float64),
            schedule=FrameSchedule.from_samples([0, 4]),
        )


@pytest.mark.parametrize(
    ("signal", "rir", "error", "message"),
    [
        (torch.ones(2, 2), torch.ones(2), ValueError, "expects 1D"),
        (torch.ones(2), torch.ones(2, 2), ValueError, "expects 1D"),
        (torch.tensor([]), torch.ones(2), ValueError, "non-empty"),
        (torch.ones(2), torch.tensor([]), ValueError, "non-empty"),
        (torch.ones(2, dtype=torch.int64), torch.ones(2), TypeError, "floating-point"),
        (torch.ones(2), torch.ones(2, dtype=torch.int64), TypeError, "floating-point"),
        (
            torch.ones(2, dtype=torch.float8_e4m3fn),
            torch.ones(2, dtype=torch.float8_e4m3fn),
            TypeError,
            "supported",
        ),
        (
            torch.ones(2, dtype=torch.float32),
            torch.ones(2, dtype=torch.float64),
            ValueError,
            "same dtype",
        ),
    ],
)
def test_fft_convolve_validates_inputs(
    signal: torch.Tensor, rir: torch.Tensor, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        fft_convolve(signal, rir)


def test_public_convolution_rejects_non_tensor_payloads_cleanly() -> None:
    with pytest.raises(TypeError, match="must be Tensors"):
        fft_convolve(cast(Any, [1.0]), cast(Any, [1.0]))
    with pytest.raises(TypeError, match="signal must be a Tensor"):
        convolve_rir(cast(Any, [1.0]), torch.ones(1))
    with pytest.raises(TypeError, match="rirs must be a Tensor"):
        convolve_rir(torch.ones(1), cast(Any, [1.0]))
    with pytest.raises(TypeError, match="signal must be a Tensor"):
        DynamicConvolver(time_reference="emission").convolve(
            cast(Any, [1.0]),
            torch.ones(1, 1),
            schedule=FrameSchedule.from_samples([0]),
        )
    with pytest.raises(TypeError, match="rirs must be a Tensor"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(1),
            cast(Any, [[1.0]]),
            schedule=FrameSchedule.from_samples([0]),
        )


def test_convolution_rejects_sparse_and_meta_tensors_before_shape_kernels() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]]),
        torch.tensor([1.0, 2.0]),
        size=(3,),
    )
    with pytest.raises(TypeError, match="dense strided"):
        fft_convolve(sparse, torch.ones(2))
    with pytest.raises(TypeError, match="dense strided"):
        convolve_rir(torch.ones(3), sparse)
    with pytest.raises(TypeError, match="dense strided"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(3),
            sparse.unsqueeze(0),
            schedule=FrameSchedule.from_samples([0]),
        )
    with pytest.raises(ValueError, match="CPU, CUDA, or MPS"):
        fft_convolve(torch.ones(3, device="meta"), torch.ones(2, device="meta"))


def test_static_convolver_validates_broadcasts_and_retains_microphone_axis() -> None:
    signal = torch.tensor([1.0, 2.0])
    rirs = torch.tensor([[[1.0]], [[2.0]]])
    actual = convolve_rir(signal, rirs)
    assert actual.shape == (1, 2)
    torch.testing.assert_close(actual, torch.tensor([[3.0, 6.0]]))
    with pytest.raises(ValueError, match="source count"):
        convolve_rir(torch.ones(3, 4), torch.ones(2, 1, 3))
    with pytest.raises(TypeError, match="floating-point"):
        convolve_rir(torch.ones(4, dtype=torch.int64), torch.ones(1, 1, 3))
    with pytest.raises(TypeError, match="supported"):
        convolve_rir(
            torch.ones(4, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1, 3, dtype=torch.float8_e4m3fn),
        )
    with pytest.raises(ValueError, match="same dtype"):
        convolve_rir(
            torch.ones(4, dtype=torch.float32),
            torch.ones(1, 1, 3, dtype=torch.float64),
        )


def test_dynamic_convolver_rejects_float8_before_fft() -> None:
    with pytest.raises(TypeError, match="supported"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(4, dtype=torch.float8_e4m3fn),
            torch.ones(1, 1, 1, 3, dtype=torch.float8_e4m3fn),
            schedule=FrameSchedule.from_samples([0]),
        )


@pytest.mark.parametrize(
    "rirs",
    [
        torch.empty(0, 1, 3),
        torch.empty(1, 0, 3),
    ],
)
def test_static_convolver_rejects_empty_source_or_microphone_axis(
    rirs: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="at least one source and microphone"):
        convolve_rir(torch.ones(4), rirs)


@pytest.mark.parametrize(
    "rirs",
    [
        torch.empty(2, 0, 1, 3),
        torch.empty(2, 1, 0, 3),
    ],
)
def test_dynamic_convolver_rejects_empty_source_or_microphone_axis(
    rirs: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="at least one source and microphone"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(4),
            rirs,
            schedule=FrameSchedule.from_samples([0, 2]),
        )


def _dynamic_result(
    *,
    source_moves: bool = True,
    microphone_moves: bool = False,
    schedule: bool = True,
) -> RIRResult:
    room = Room.shoebox([4.0, 3.0], fs=10)
    sources = Source.from_positions([[1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]])
    source_end = [1.2, 1.0] if source_moves else [1.0, 1.0]
    microphone_end = [2.2, 1.0] if microphone_moves else [2.0, 1.0]
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=[[[1.0, 1.0]], [source_end]],
        mic_traj=[[[2.0, 1.0]], [microphone_end]],
        schedule=(FrameSchedule.from_samples([0, 4]) if schedule else None),
    )
    request = SimulationConfig(max_order=0, nsample=2)
    resolved = _resolve_simulation_config(
        request,
        fs=room.fs,
        room_dimension=int(room.size.numel()),
        tensor_values=(room.size, sources.positions, mics.positions),
    )
    return RIRResult(
        rirs=torch.ones(2, 1, 1, 2),
        scene=scene,
        config=resolved,
    )


def test_dynamic_convolver_consumes_result_frame_schedule() -> None:
    result = _dynamic_result()
    actual = DynamicConvolver(time_reference="emission").convolve(
        torch.ones(8),
        result,
    )
    expected = DynamicConvolver(time_reference="emission").convolve(
        torch.ones(8),
        result.rirs,
        schedule=FrameSchedule.from_samples([0, 4]),
    )
    torch.testing.assert_close(actual, expected)


def test_dynamic_convolver_rejects_result_with_resized_rirs() -> None:
    result = _dynamic_result()
    result.rirs.resize_(2, 1, 2, 1)

    with pytest.raises(ValueError, match="microphones"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(8),
            result,
        )


def test_dynamic_convolver_rejects_duplicate_result_schedule() -> None:
    result = _dynamic_result()
    with pytest.raises(ValueError, match="must be omitted"):
        DynamicConvolver(time_reference="emission").convolve(
            torch.ones(8),
            result,
            schedule=FrameSchedule.from_samples([0, 4]),
        )


def test_dynamic_result_without_frame_schedule_requires_explicit_schedule() -> None:
    result = _dynamic_result(schedule=False)
    convolver = DynamicConvolver(time_reference="emission")
    with pytest.raises(ValueError, match="has no frame schedule"):
        convolver.convolve(torch.ones(8), result)
    actual = convolver.convolve(
        torch.ones(8),
        result,
        schedule=FrameSchedule.uniform(
            frame_count=2,
            stop_sample=8,
        ),
    )
    assert actual.shape == (1, 9)


def test_emission_reference_rejects_moving_microphone_result() -> None:
    result = _dynamic_result(source_moves=False, microphone_moves=True)
    with pytest.raises(ValueError, match="does not support moving microphones"):
        DynamicConvolver(time_reference="emission").convolve(torch.ones(8), result)


def test_observation_reference_accepts_moving_microphone_result() -> None:
    result = _dynamic_result(source_moves=False, microphone_moves=True)
    actual = DynamicConvolver(time_reference="observation").convolve(
        torch.ones(8),
        result,
    )
    expected = DynamicConvolver(time_reference="observation").convolve(
        torch.ones(8),
        result.rirs,
        schedule=FrameSchedule.from_samples([0, 4]),
    )
    torch.testing.assert_close(actual, expected)


def test_observation_reference_rejects_moving_source_result() -> None:
    with pytest.raises(ValueError, match="requires fixed sources"):
        DynamicConvolver(time_reference="observation").convolve(
            torch.ones(8),
            _dynamic_result(source_moves=True),
        )


def test_dynamic_convolver_rejects_simultaneous_motion_result() -> None:
    result = _dynamic_result(source_moves=True, microphone_moves=True)
    with pytest.raises(ValueError, match="retarded-time"):
        DynamicConvolver(time_reference="observation").convolve(
            torch.ones(8),
            result,
        )


def test_dynamic_convolver_retains_single_microphone_axis() -> None:
    actual = DynamicConvolver(time_reference="emission")(
        torch.ones(6),
        torch.ones(2, 1, 1, 1),
        schedule=FrameSchedule.uniform(frame_count=2, stop_sample=6),
    )
    assert actual.shape == (1, 6)
