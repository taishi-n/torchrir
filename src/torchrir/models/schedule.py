"""Exact sample-domain schedules for dynamic RIR frames."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral

import torch
from torch import Tensor

from ..util._dtypes import (
    validate_materialized_tensor,
    validate_supported_float_dtype,
)
from ..util.device import DeviceSpec
from ..util._time import frame_times_to_samples
from ..util._scalars import normalize_finite_real, normalize_integer


_MAX_SAMPLE_INDEX = torch.iinfo(torch.int64).max


@dataclass(frozen=True, slots=True, init=False)
class FrameSchedule:
    """Immutable start samples for a sequence of dynamic RIR frames.

    One start is stored for each frame. The first start is zero and subsequent
    starts increase strictly. ``starts`` returns a new CPU ``int64`` tensor on
    every access, so callers cannot mutate the schedule through that view.
    """

    _starts: tuple[int, ...] = field(repr=False)
    conversion_sample_rate: float | None = field(default=None, repr=False)

    def __init__(
        self,
        starts: Tensor | Sequence[int],
    ) -> None:
        if torch.is_tensor(starts):
            validate_materialized_tensor(starts, name="frame starts")
            raw = starts.detach().cpu()
            if raw.ndim != 1 or raw.numel() == 0:
                raise ValueError("frame starts must be a non-empty 1D sequence")
            if raw.is_floating_point() or raw.is_complex() or raw.dtype == torch.bool:
                raise TypeError("frame starts must contain integer sample indices")
            values = raw.tolist()
        else:
            try:
                values = list(starts)
            except TypeError as exc:
                raise TypeError("frame starts must be an integer sequence") from exc
            if not values:
                raise ValueError("frame starts must be a non-empty 1D sequence")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in values
        ):
            raise TypeError("frame starts must contain integer sample indices")
        normalized_values = [int(value) for value in values]
        if any(value < 0 or value > _MAX_SAMPLE_INDEX for value in normalized_values):
            raise ValueError("frame starts must fit in non-negative int64 samples")
        normalized = torch.tensor(normalized_values, dtype=torch.int64, device="cpu")
        _validate_frame_starts(normalized)
        object.__setattr__(self, "_starts", tuple(normalized.tolist()))
        object.__setattr__(self, "conversion_sample_rate", None)

    @property
    def starts(self) -> Tensor:
        """Return a mutation-safe CPU ``int64`` snapshot."""

        return torch.tensor(self._starts, dtype=torch.int64, device="cpu")

    def __len__(self) -> int:
        return len(self._starts)

    def normalized_progress(
        self,
        *,
        stop_sample: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> Tensor:
        """Return exact frame starts normalized to ``[0, stop_sample)``.

        This is the canonical interpolation grid for geometry whose nominal
        endpoint occurs at ``stop_sample``. The endpoint itself is not a frame
        start, so every returned value is strictly smaller than one.
        """

        stop_sample = _validate_positive_integer(stop_sample, name="stop_sample")
        if self._starts[-1] >= stop_sample:
            raise ValueError("last frame start must be before stop_sample")
        validate_supported_float_dtype(dtype)
        progress = torch.tensor(self._starts, dtype=torch.float64) / float(stop_sample)
        converted = progress.to(dtype=dtype)
        upper_bound = torch.nextafter(
            torch.tensor(1, dtype=dtype),
            torch.tensor(0, dtype=dtype),
        )
        converted = torch.clamp_max(converted, upper_bound)
        if converted.numel() > 1 and torch.any(converted[1:] <= converted[:-1]):
            raise ValueError(
                "frame starts are not distinct at the requested progress dtype"
            )
        resolved_device, _ = DeviceSpec(device=device, dtype=dtype).resolve()
        return converted.to(device=resolved_device)

    def __repr__(self) -> str:
        """Return an informative representation including conversion provenance."""

        if len(self._starts) <= 8:
            starts = repr(list(self._starts))
        else:
            head = ", ".join(str(value) for value in self._starts[:4])
            tail = ", ".join(str(value) for value in self._starts[-3:])
            starts = f"[{head}, ..., {tail}] ({len(self._starts)} frames)"
        return (
            f"{type(self).__name__}(starts={starts}, "
            f"conversion_sample_rate={self.conversion_sample_rate!r})"
        )

    @classmethod
    def from_samples(
        cls,
        starts: Tensor | Sequence[int],
    ) -> FrameSchedule:
        """Build a schedule from exact frame starts."""

        return cls(starts)

    @classmethod
    def from_seconds(
        cls,
        times: Tensor | Sequence[float],
        *,
        sample_rate: float,
    ) -> FrameSchedule:
        """Floor second-based frame times once into exact sample starts."""

        normalized_sample_rate = _normalize_sample_rate(sample_rate)
        schedule = cls(
            frame_times_to_samples(times, sample_rate=normalized_sample_rate)
        )
        object.__setattr__(
            schedule,
            "conversion_sample_rate",
            normalized_sample_rate,
        )
        return schedule

    @classmethod
    def uniform(
        cls,
        *,
        frame_count: int,
        stop_sample: int,
    ) -> FrameSchedule:
        """Partition ``[0, stop_sample)`` uniformly into ``frame_count`` frames."""

        frame_count = _validate_positive_integer(frame_count, name="frame_count")
        stop_sample = _validate_positive_integer(stop_sample, name="stop_sample")
        if frame_count > stop_sample:
            raise ValueError("frame_count cannot exceed stop_sample")
        starts = [(index * stop_sample) // frame_count for index in range(frame_count)]
        return cls(starts)

    @classmethod
    def fixed_hop(
        cls,
        *,
        stop_sample: int,
        hop_size: int,
    ) -> FrameSchedule:
        """Start frames every ``hop_size`` samples before ``stop_sample``."""

        stop_sample = _validate_positive_integer(stop_sample, name="stop_sample")
        hop_size = _validate_positive_integer(hop_size, name="hop_size")
        return cls(range(0, stop_sample, hop_size))


def _validate_frame_starts(starts: Tensor) -> None:
    if starts[0].item() != 0:
        raise ValueError("first frame start must be 0")
    if starts.numel() > 1 and torch.any(starts[1:] <= starts[:-1]):
        raise ValueError("frame starts must be strictly increasing")


def _validate_positive_integer(value: object, *, name: str) -> int:
    normalized = normalize_integer(value, name=name, minimum=1)
    if normalized > _MAX_SAMPLE_INDEX:
        raise ValueError(f"{name} must fit in a positive int64 sample index")
    return normalized


def _normalize_sample_rate(value: float) -> float:
    return normalize_finite_real(value, name="sample_rate", positive=True)


__all__ = ["FrameSchedule"]
