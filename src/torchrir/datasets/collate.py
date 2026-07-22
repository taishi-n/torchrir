"""Collate helpers for DataLoader usage."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .base import DatasetItem, _validate_sample_rate
from ..util._dtypes import (
    validate_materialized_tensor,
    validate_supported_float_tensor,
)
from ..util._scalars import normalize_finite_real


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class CollateBatch:
    """Collated batch of dataset items.

    Fields:
        - audio: Padded audio tensor of shape (batch, max_len).
        - lengths: Original lengths for each item.
        - sample_rate: Sample rate shared across the batch.
        - utterance_ids: Utterance IDs per item.
        - texts: Optional text per item.
        - speakers: Optional speaker IDs per item.
        - metadata: Optional per-item metadata (pass-through).
    """

    audio: Tensor
    lengths: Tensor
    sample_rate: int
    utterance_ids: Sequence[str]
    texts: Sequence[str | None]
    speakers: Sequence[str | None]
    metadata: Sequence[Any] | None = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.audio):
            raise TypeError("audio must be a Tensor")
        if self.audio.ndim != 2 or 0 in self.audio.shape:
            raise ValueError(
                "audio must have shape (batch, samples) with non-zero axes"
            )
        validate_supported_float_tensor(self.audio, name="audio")
        if not torch.all(torch.isfinite(self.audio)):
            raise ValueError("audio must contain finite values")
        if not torch.is_tensor(self.lengths):
            raise TypeError("lengths must be a Tensor")
        validate_materialized_tensor(self.lengths, name="lengths")
        if (
            self.lengths.ndim != 1
            or self.lengths.dtype != torch.int64
            or self.lengths.device.type != "cpu"
        ):
            raise ValueError("lengths must be a 1D CPU int64 Tensor")
        if self.lengths.shape[0] != self.audio.shape[0]:
            raise ValueError("lengths must contain one value per batch item")
        if torch.any(self.lengths <= 0) or torch.any(
            self.lengths > self.audio.shape[1]
        ):
            raise ValueError("lengths must lie in [1, audio.shape[1]]")
        object.__setattr__(self, "sample_rate", _validate_sample_rate(self.sample_rate))

        utterance_ids = _snapshot_sequence(
            self.utterance_ids,
            name="utterance_ids",
        )
        texts = _snapshot_sequence(self.texts, name="texts")
        speakers = _snapshot_sequence(self.speakers, name="speakers")
        batch_size = int(self.audio.shape[0])
        if not (len(utterance_ids) == len(texts) == len(speakers) == batch_size):
            raise ValueError("item metadata sequences must match the batch size")
        if any(
            not isinstance(value, str) or not value.strip() for value in utterance_ids
        ):
            raise ValueError("utterance_ids must contain non-empty strings")
        if any(value is not None and not isinstance(value, str) for value in texts):
            raise TypeError("texts must contain strings or None")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in speakers
        ):
            raise ValueError("speakers must contain non-empty strings or None")
        metadata = (
            None
            if self.metadata is None
            else _snapshot_sequence(self.metadata, name="metadata")
        )
        if metadata is not None and len(metadata) != batch_size:
            raise ValueError("metadata must contain one value per batch item")
        object.__setattr__(self, "utterance_ids", utterance_ids)
        object.__setattr__(self, "texts", texts)
        object.__setattr__(self, "speakers", speakers)
        object.__setattr__(self, "metadata", metadata)


def collate_dataset_items(
    items: Iterable[DatasetItem],
    *,
    pad_value: float = 0.0,
    keep_metadata: bool = False,
) -> CollateBatch:
    """Collate DatasetItem entries into a padded batch.

    Args:
        items: Iterable of DatasetItem.
        pad_value: Value used for padding.
        keep_metadata: Preserve item-level metadata field if present.

    Returns:
        CollateBatch with padded audio and immutable metadata tuples.
    """
    if not isinstance(keep_metadata, bool):
        raise TypeError("keep_metadata must be a bool")
    batch = list(items)
    if not batch:
        raise ValueError("collate_dataset_items received an empty batch")
    if any(not isinstance(item, DatasetItem) for item in batch):
        raise TypeError("items must contain only DatasetItem instances")
    for item in batch:
        item.validate()

    normalized_pad_value = normalize_finite_real(pad_value, name="pad_value")
    dtype_limit = torch.finfo(batch[0].audio.dtype)
    if not dtype_limit.min <= normalized_pad_value <= dtype_limit.max:
        raise ValueError(f"pad_value must be representable as {batch[0].audio.dtype}")

    sample_rate = batch[0].sample_rate
    for item in batch[1:]:
        if item.sample_rate != sample_rate:
            raise ValueError("sample_rate must be consistent within a batch")
        if item.audio.dtype != batch[0].audio.dtype:
            raise ValueError("audio dtype must be consistent within a batch")
        if item.audio.device != batch[0].audio.device:
            raise ValueError("audio device must be consistent within a batch")

    lengths = torch.tensor([item.audio.shape[0] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    audio = torch.full(
        (len(batch), max_len),
        normalized_pad_value,
        dtype=batch[0].audio.dtype,
        device=batch[0].audio.device,
    )

    for idx, item in enumerate(batch):
        audio[idx, : item.audio.shape[0]] = item.audio

    utterance_ids = tuple(item.utterance_id for item in batch)
    texts = tuple(item.text for item in batch)
    speakers = tuple(item.speaker for item in batch)

    metadata: tuple[Any, ...] | None = None
    if keep_metadata:
        metadata = tuple(item.metadata for item in batch)

    return CollateBatch(
        audio=audio,
        lengths=lengths,
        sample_rate=sample_rate,
        utterance_ids=utterance_ids,
        texts=texts,
        speakers=speakers,
        metadata=metadata,
    )


def _snapshot_sequence(value: object, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence")
    return tuple(value)
