"""Deterministic FFT work bounds, independent of machine speed."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from torchrir.signal import DynamicConvolver, FrameSchedule, convolve_rir


@pytest.fixture
def fft_work(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count transformed samples, including every batch/source/mic axis."""
    work = {"rfft": 0, "irfft": 0}

    def counted(
        name: str, original: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        def transform(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            length = kwargs.get("n", args[0] if args else None)
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            if length is None:
                length = tensor.shape[dim]
                if name == "irfft":
                    length = 2 * (length - 1)
            work[name] += tensor.numel() // tensor.shape[dim] * length
            return original(tensor, *args, **kwargs)

        return transform

    monkeypatch.setattr(torch.fft, "rfft", counted("rfft", torch.fft.rfft))
    monkeypatch.setattr(torch.fft, "irfft", counted("irfft", torch.fft.irfft))
    return work


@pytest.mark.unit
def test_static_fft_work_scales_inverse_with_microphones(
    fft_work: dict[str, int],
) -> None:
    sources, microphones, samples, taps = 3, 2, 31, 7
    signal = torch.ones(sources, samples, dtype=torch.float64)
    rirs = torch.ones(sources, microphones, taps, dtype=torch.float64)

    convolve_rir(signal, rirs)

    fft_length = 1 << (samples + taps - 2).bit_length()
    assert fft_work["rfft"] <= (sources + sources * microphones) * fft_length
    assert fft_work["irfft"] <= microphones * fft_length


@pytest.mark.unit
def test_emission_fft_work_scales_inverse_with_frames_and_microphones(
    fft_work: dict[str, int],
) -> None:
    frames, sources, microphones, frame_samples, taps = 9, 3, 2, 10, 7
    signal = torch.ones(sources, frames * frame_samples, dtype=torch.float64)
    rirs = torch.ones(frames, sources, microphones, taps, dtype=torch.float64)
    schedule = FrameSchedule.fixed_hop(
        stop_sample=signal.shape[-1], hop_size=frame_samples
    )

    DynamicConvolver(time_reference="emission").convolve(
        signal, rirs, schedule=schedule
    )

    fft_length = 1 << (frame_samples + taps - 2).bit_length()
    assert fft_work["rfft"] <= frames * (sources + sources * microphones) * fft_length
    assert fft_work["irfft"] <= frames * microphones * fft_length


@pytest.mark.unit
def test_observation_fft_work_is_bounded_by_history_plus_output_frames(
    fft_work: dict[str, int],
) -> None:
    sources, microphones, samples, taps = 3, 2, 71, 9
    starts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 18, 23, 27, 40, 56, 70, 75]
    signal = torch.ones(sources, samples, dtype=torch.float64)
    rirs = torch.ones(len(starts), sources, microphones, taps, dtype=torch.float64)

    DynamicConvolver(time_reference="observation").convolve(
        signal, rirs, schedule=FrameSchedule.from_samples(starts)
    )

    boundaries = [*starts, samples + taps - 1]
    fft_volume = sum(
        1 << (taps - 1 + end - start - 1).bit_length()
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    )
    assert fft_work["rfft"] <= (sources + sources * microphones) * fft_volume
    assert fft_work["irfft"] <= microphones * fft_volume
