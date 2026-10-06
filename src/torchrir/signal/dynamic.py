"""Piecewise time-varying RIR convolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..models import DynamicScene, RIRResult
from ..models.schedule import FrameSchedule, _validate_frame_starts
from ..models.scene import _validate_dynamic_time_reference
from .internal import (
    _ensure_dynamic_rirs,
    _ensure_signal,
    _fft_work_dtype,
    _validate_convolution_dtypes,
)


TimeReference = Literal["emission", "observation"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicConvolver:
    """Convolve signals with piecewise time-varying RIRs.

    ``time_reference="emission"`` selects an RIR frame from the input sample's
    emission time. It models moving sources observed by fixed microphones.
    ``time_reference="observation"`` selects one RIR frame from each output
    sample's observation time. It models fixed sources observed by moving
    microphones.

    Frame segmentation is independent of this physical convention and is
    supplied to each [convolve][torchrir.signal.DynamicConvolver.convolve]
    call as a [FrameSchedule][torchrir.signal.FrameSchedule]. A dynamic
    [RIRResult][torchrir.models.RIRResult] whose dynamic scene contains a frame
    schedule supplies it directly; passing another schedule in that case is
    rejected.

    Attributes:
        time_reference: ``"emission"`` or ``"observation"``. There is no
            implicit default because choosing the wrong reference changes the
            physical model.
    """

    time_reference: TimeReference

    def __post_init__(self) -> None:
        if self.time_reference not in ("emission", "observation"):
            raise ValueError("time_reference must be 'emission' or 'observation'")

    def __call__(
        self,
        signal: Tensor,
        rirs: Tensor | RIRResult,
        *,
        schedule: FrameSchedule | None = None,
    ) -> Tensor:
        return self.convolve(signal, rirs, schedule=schedule)

    def convolve(
        self,
        signal: Tensor,
        rirs: Tensor | RIRResult,
        *,
        schedule: FrameSchedule | None = None,
    ) -> Tensor:
        """Convolve dry signals with a dynamic RIR sequence.

        Args:
            signal: Dry signal with shape ``(samples,)`` or
                ``(n_sources, samples)``.
            rirs: Dynamic RIR tensor with shape ``(frames, rir_samples)``,
                ``(frames, n_microphones, rir_samples)`` for one source, or
                ``(frames, n_sources, n_microphones, rir_samples)``. A dynamic
                [RIRResult][torchrir.models.RIRResult] may be passed instead.
            schedule: One start sample per RIR frame. It is required for a raw
                tensor and for an ``RIRResult`` without a frame schedule. It
                must be omitted when the result's scene already contains one.

        Returns:
            Tensor with shape ``(n_microphones, output_samples)``, where
            ``output_samples = signal_samples + rir_samples - 1``. The output
            keeps the input signal's dtype and device, including for one
            microphone.
        """

        if schedule is not None and not isinstance(schedule, FrameSchedule):
            raise TypeError("schedule must be a FrameSchedule")

        if isinstance(rirs, RIRResult):
            rirs.validate()
            if not isinstance(rirs.scene, DynamicScene):
                raise ValueError("DynamicConvolver requires a dynamic RIRResult")
            _validate_dynamic_time_reference(
                rirs.scene,
                time_reference=self.time_reference,
            )
            if rirs.scene.schedule is not None:
                if schedule is not None:
                    raise ValueError(
                        "schedule must be omitted when DynamicScene contains a schedule"
                    )
                schedule = rirs.scene.schedule
            elif schedule is None:
                raise ValueError(
                    "schedule is required when RIRResult has no frame schedule"
                )
            rirs_tensor = rirs.rirs
            if (
                schedule.conversion_sample_rate is not None
                and schedule.conversion_sample_rate != float(rirs.scene.room.fs)
            ):
                raise ValueError(
                    "schedule seconds-conversion sample rate "
                    f"{schedule.conversion_sample_rate:g} conflicts with "
                    f"RIR sample rate {float(rirs.scene.room.fs):g}"
                )
        else:
            if schedule is None:
                raise ValueError("schedule is required for raw dynamic RIR tensors")
            rirs_tensor = rirs

        signal_tensor = _ensure_signal(signal)
        rirs_tensor = _ensure_dynamic_rirs(rirs_tensor, signal_tensor)
        signal_tensor, rirs_tensor = _validate_dynamic_inputs(
            signal_tensor,
            rirs_tensor,
        )
        starts = _validate_schedule_for_convolution(
            schedule,
            frame_count=int(rirs_tensor.shape[0]),
            signal_samples=int(signal_tensor.shape[1]),
            rir_samples=int(rirs_tensor.shape[-1]),
            time_reference=self.time_reference,
        )

        if self.time_reference == "emission":
            return _convolve_emission(signal_tensor, rirs_tensor, starts=starts)
        return _convolve_observation(signal_tensor, rirs_tensor, starts=starts)


def _convolve_emission(signal: Tensor, rirs: Tensor, *, starts: Tensor) -> Tensor:
    """Select RIR frames from emitted input samples."""

    n_samples = int(signal.shape[1])
    _, n_src, n_mic, rir_len = rirs.shape
    boundaries = torch.cat(
        [starts, torch.tensor([n_samples], dtype=torch.int64, device="cpu")]
    )
    lengths = boundaries[1:] - boundaries[:-1]

    return _convolve_emission_batched(
        signal,
        rirs,
        boundaries=boundaries,
        lengths=lengths,
    )


def _convolve_observation(signal: Tensor, rirs: Tensor, *, starts: Tensor) -> Tensor:
    """Select output-time RIR frames using batched overlap-save."""

    n_samples = int(signal.shape[1])
    _, n_src, n_mic, rir_len = rirs.shape
    out_len = n_samples + rir_len - 1
    boundaries = [*starts.tolist(), out_len]
    work_dtype = _fft_work_dtype(signal.dtype)
    out = torch.zeros(
        (n_mic, out_len),
        dtype=work_dtype,
        device=signal.device,
    )

    for chunk_start, chunk_end, fft_len in _observation_chunk_ranges(
        boundaries, rir_len=rir_len, max_frames=8
    ):
        segments = torch.zeros(
            (chunk_end - chunk_start, n_src, fft_len),
            dtype=work_dtype,
            device=signal.device,
        )
        for chunk_index, frame_index in enumerate(range(chunk_start, chunk_end)):
            input_base = boundaries[frame_index] - rir_len + 1
            input_start = max(0, input_base)
            input_end = min(n_samples, boundaries[frame_index + 1])
            local_start = input_start - input_base
            segments[
                chunk_index, :, local_start : local_start + input_end - input_start
            ] = signal[:, input_start:input_end]

        segment_spectrum = torch.fft.rfft(segments, n=fft_len, dim=-1)
        rir_spectrum = torch.fft.rfft(
            rirs[chunk_start:chunk_end].to(dtype=work_dtype), n=fft_len, dim=-1
        )
        convolution = torch.fft.irfft(
            (segment_spectrum[:, :, None, :] * rir_spectrum).sum(dim=1),
            n=fft_len,
            dim=-1,
        )
        # Circular aliasing is confined to the first rir_len - 1 samples.
        for chunk_index, frame_index in enumerate(range(chunk_start, chunk_end)):
            output_start = boundaries[frame_index]
            output_end = boundaries[frame_index + 1]
            out[:, output_start:output_end] = convolution[
                chunk_index, :, rir_len - 1 : rir_len - 1 + output_end - output_start
            ]
    return out.to(dtype=signal.dtype)


def _observation_chunk_ranges(
    boundaries: list[int], *, rir_len: int, max_frames: int
) -> list[tuple[int, int, int]]:
    """Batch adjacent frames of equal FFT length without padding inflation."""
    ranges: list[tuple[int, int, int]] = []
    frame_count = len(boundaries) - 1
    chunk_start = 0
    while chunk_start < frame_count:
        window_length = (
            rir_len - 1 + boundaries[chunk_start + 1] - boundaries[chunk_start]
        )
        fft_length = 1 << (window_length - 1).bit_length()
        chunk_end = chunk_start + 1
        limit = min(chunk_start + max_frames, frame_count)
        while chunk_end < limit:
            next_length = (
                rir_len - 1 + boundaries[chunk_end + 1] - boundaries[chunk_end]
            )
            if 1 << (next_length - 1).bit_length() != fft_length:
                break
            chunk_end += 1
        ranges.append((chunk_start, chunk_end, fft_length))
        chunk_start = chunk_end
    return ranges


def _validate_schedule_for_convolution(
    schedule: FrameSchedule | None,
    *,
    frame_count: int,
    signal_samples: int,
    rir_samples: int,
    time_reference: TimeReference,
) -> Tensor:
    assert schedule is not None
    starts = schedule.starts
    _validate_frame_starts(starts)
    if starts.numel() != frame_count:
        raise ValueError(
            f"schedule has {starts.numel()} frames, but rirs has {frame_count}"
        )
    endpoint = (
        signal_samples
        if time_reference == "emission"
        else signal_samples + rir_samples - 1
    )
    if starts[-1].item() >= endpoint:
        timeline = "input" if time_reference == "emission" else "output"
        raise ValueError(
            f"last frame start must be before the {timeline} timeline endpoint "
            f"({endpoint})"
        )
    return starts


def _validate_dynamic_inputs(signal: Tensor, rirs: Tensor) -> tuple[Tensor, Tensor]:
    if signal.numel() == 0 or signal.shape[-1] == 0:
        raise ValueError("signal must contain at least one sample")
    if rirs.shape[0] == 0 or rirs.shape[-1] == 0:
        raise ValueError("rirs must contain at least one frame and one sample")
    if rirs.shape[1] == 0 or rirs.shape[2] == 0:
        raise ValueError("rirs must contain at least one source and microphone")
    _validate_convolution_dtypes(signal, rirs)
    if signal.device != rirs.device:
        raise ValueError("signal and rirs must be on the same device")
    if signal.dtype != rirs.dtype:
        raise ValueError("signal and rirs must use the same dtype")
    n_src = int(rirs.shape[1])
    if signal.shape[0] == 1 and n_src > 1:
        signal = signal.expand(n_src, -1)
    elif signal.shape[0] != n_src:
        raise ValueError(f"signal has {signal.shape[0]} sources but rirs has {n_src}")
    return signal, rirs


def _convolve_emission_batched(
    signal: Tensor,
    rirs: Tensor,
    *,
    boundaries: Tensor,
    lengths: Tensor,
    chunk_size: int = 8,
) -> Tensor:
    """GPU-friendly batched emission-time convolution using FFT."""

    n_samples = int(signal.shape[1])
    _, n_src, n_mic, rir_len = rirs.shape
    work_dtype = _fft_work_dtype(signal.dtype)
    out = torch.zeros(
        (n_mic, n_samples + rir_len - 1),
        dtype=work_dtype,
        device=signal.device,
    )

    for chunk_start, chunk_end in _emission_chunk_ranges(
        lengths,
        rir_len=rir_len,
        max_frames=chunk_size,
    ):
        chunk_lengths = lengths[chunk_start:chunk_end]
        max_len = int(chunk_lengths.max().item())
        segments = torch.zeros(
            (chunk_end - chunk_start, n_src, max_len),
            dtype=work_dtype,
            device=signal.device,
        )
        for chunk_index, frame_index in enumerate(range(chunk_start, chunk_end)):
            start = int(boundaries[frame_index].item())
            end = int(boundaries[frame_index + 1].item())
            segments[chunk_index, :, : end - start] = signal[:, start:end]

        convolution_len = max_len + rir_len - 1
        fft_len = 1 << (convolution_len - 1).bit_length()
        segment_spectrum = torch.fft.rfft(segments, n=fft_len, dim=-1)
        rir_spectrum = torch.fft.rfft(
            rirs[chunk_start:chunk_end].to(dtype=work_dtype),
            n=fft_len,
            dim=-1,
        )
        source_sum = torch.fft.irfft(
            (segment_spectrum[:, :, None, :] * rir_spectrum).sum(dim=1),
            n=fft_len,
            dim=-1,
        )[..., :convolution_len]

        for chunk_index, frame_index in enumerate(range(chunk_start, chunk_end)):
            segment_len = int(chunk_lengths[chunk_index].item())
            start = int(boundaries[frame_index].item())
            selected_len = segment_len + rir_len - 1
            out[:, start : start + selected_len] += source_sum[
                chunk_index,
                :,
                :selected_len,
            ]
    return out.to(dtype=signal.dtype)


def _emission_chunk_ranges(
    lengths: Tensor,
    *,
    rir_len: int,
    max_frames: int,
) -> list[tuple[int, int]]:
    """Group adjacent frames while bounding padding waste to twofold."""

    ranges: list[tuple[int, int]] = []
    chunk_start = 0
    frame_count = int(lengths.numel())
    while chunk_start < frame_count:
        chunk_end = chunk_start
        length_sum = 0
        maximum_length = 0
        limit = min(chunk_start + max_frames, frame_count)
        while chunk_end < limit:
            frame_length = int(lengths[chunk_end].item())
            next_count = chunk_end - chunk_start + 1
            next_sum = length_sum + frame_length
            next_maximum = max(maximum_length, frame_length)
            padded = next_count * (next_maximum + rir_len - 1)
            useful = next_sum + next_count * (rir_len - 1)
            if next_count > 1 and padded > 2 * useful:
                break
            length_sum = next_sum
            maximum_length = next_maximum
            chunk_end += 1
        ranges.append((chunk_start, chunk_end))
        chunk_start = chunk_end
    return ranges
