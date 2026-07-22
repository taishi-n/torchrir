import numpy as np
import pytest
import torch

from torchrir import DynamicScene, MicrophoneArray, RIRResult, Room, Source
from torchrir.config import SimulationConfig
from torchrir.signal import DynamicConvolver, convolve_rir, fft_convolve


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


@pytest.mark.numerical
def test_dynamic_hop_with_constant_rir_equals_static_convolution() -> None:
    generator = torch.Generator().manual_seed(10)
    signal = torch.randn(23, generator=generator, dtype=torch.float64)
    rir = torch.randn(5, generator=generator, dtype=torch.float64)
    dynamic_rirs = rir.repeat(4, 1)
    actual = DynamicConvolver(mode="hop", hop=7).convolve(signal, dynamic_rirs)
    expected = fft_convolve(signal, rir)
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
def test_dynamic_convolver_multi_mic_matches_segment_reference() -> None:
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
    hop = 4
    actual = DynamicConvolver(mode="hop", hop=hop).convolve(signal, rirs)
    expected = np.zeros((2, signal.shape[-1] + rirs.shape[-1] - 1))
    for frame in range(3):
        start = frame * hop
        for source in range(2):
            for mic in range(2):
                convolution = np.convolve(
                    signal[source, start : start + hop].numpy(),
                    rirs[frame, source, mic].numpy(),
                )
                expected[mic, start : start + convolution.size] += convolution
    torch.testing.assert_close(
        actual, torch.from_numpy(expected), rtol=1e-12, atol=1e-12
    )


@pytest.mark.numerical
def test_dynamic_trajectory_uses_timestamp_sample_boundaries() -> None:
    signal = torch.arange(1, 9, dtype=torch.float64)
    rirs = torch.tensor([[[[1.0]]], [[[2.0]]], [[[-1.0]]]], dtype=torch.float64)
    actual = DynamicConvolver(timestamps=torch.tensor([0.0, 0.3, 0.6]), fs=10).convolve(
        signal, rirs
    )
    expected = torch.tensor(
        [1.0, 2.0, 3.0, 8.0, 10.0, 12.0, -7.0, -8.0], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_dynamic_convolver_rejects_ambiguous_3d_rirs_for_multi_source():
    signal = torch.randn(2, 256)
    ambiguous_rirs = torch.randn(5, 2, 64)
    with pytest.raises(ValueError, match="Use 4D"):
        DynamicConvolver(mode="hop", hop=64).convolve(signal, ambiguous_rirs)


def test_dynamic_convolver_validates_configuration() -> None:
    with pytest.raises(ValueError, match="hop must be positive"):
        DynamicConvolver(mode="hop", hop=0)
    with pytest.raises(ValueError, match="strictly increasing"):
        DynamicConvolver(timestamps=torch.tensor([0.0, 0.5, 0.25]), fs=8)
    with pytest.raises(ValueError, match="fs must be positive"):
        DynamicConvolver(fs=float("nan"))


def test_dynamic_convolver_rejects_timestamp_outside_signal() -> None:
    convolver = DynamicConvolver(timestamps=torch.tensor([0.0, 2.0]), fs=8)
    with pytest.raises(ValueError, match="before the end"):
        convolver.convolve(torch.ones(8), torch.ones(2, 1, 1, 3))


def test_dynamic_convolver_rejects_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        DynamicConvolver(mode="hop", hop=4).convolve(
            torch.ones(8, dtype=torch.float32),
            torch.ones(2, 1, 1, 3, dtype=torch.float64),
        )


def test_hop_mode_requires_enough_rir_frames() -> None:
    with pytest.raises(ValueError, match="requires 4"):
        DynamicConvolver(mode="hop", hop=2).convolve(
            torch.ones(8), torch.ones(2, 1, 1, 3)
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


def test_static_convolver_validates_and_broadcasts_sources() -> None:
    signal = torch.tensor([1.0, 2.0])
    rirs = torch.tensor([[[1.0]], [[2.0]]])
    torch.testing.assert_close(convolve_rir(signal, rirs), torch.tensor([3.0, 6.0]))
    with pytest.raises(ValueError, match="source count"):
        convolve_rir(torch.ones(3, 4), torch.ones(2, 1, 3))
    with pytest.raises(TypeError, match="floating-point"):
        convolve_rir(torch.ones(4, dtype=torch.int64), torch.ones(1, 1, 3))
    with pytest.raises(ValueError, match="same dtype"):
        convolve_rir(
            torch.ones(4, dtype=torch.float32),
            torch.ones(1, 1, 3, dtype=torch.float64),
        )


def _dynamic_result() -> RIRResult:
    room = Room.shoebox([4.0, 3.0], fs=10)
    sources = Source.from_positions([[1.0, 1.0]])
    mics = MicrophoneArray.from_positions([[2.0, 1.0]])
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=[[[1.0, 1.0]], [[1.2, 1.0]]],
        mic_traj=[[[2.0, 1.0]], [[2.0, 1.0]]],
        timestamps=[0.0, 0.4],
    )
    return RIRResult(
        rirs=torch.ones(2, 1, 1, 2),
        scene=scene,
        config=SimulationConfig(max_order=0, nsample=2),
    )


def test_dynamic_convolver_consumes_result_metadata() -> None:
    result = _dynamic_result()
    actual = DynamicConvolver()(torch.ones(8), result)
    expected = DynamicConvolver(timestamps=result.timestamps, fs=10).convolve(
        torch.ones(8), result.rirs
    )
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="timestamps conflict"):
        DynamicConvolver(timestamps=torch.tensor([0.0, 0.5])).convolve(
            torch.ones(8), result
        )
    with pytest.raises(ValueError, match="sample rate"):
        DynamicConvolver(fs=20).convolve(torch.ones(8), result)


def test_dynamic_trajectory_without_timestamps_partitions_evenly() -> None:
    signal = torch.arange(1, 7, dtype=torch.float64)
    rirs = torch.tensor([[[[1.0]]], [[[2.0]]]], dtype=torch.float64)
    expected = torch.tensor([1.0, 2.0, 3.0, 8.0, 10.0, 12.0], dtype=torch.float64)
    torch.testing.assert_close(DynamicConvolver()(signal, rirs), expected)
    with pytest.raises(ValueError, match="cannot exceed"):
        DynamicConvolver().convolve(torch.ones(2), torch.ones(3, 1, 1, 1))
