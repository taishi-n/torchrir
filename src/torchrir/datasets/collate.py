from __future__ import annotations

"""Collate helpers for DataLoader usage."""

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
from torch import Tensor

from .base import DatasetItem


@dataclass(frozen=True)
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
    utterance_ids: list[str]
    texts: list[Optional[str]]
    speakers: list[Optional[str]]
    metadata: Optional[list[Any]] = None


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
        CollateBatch with padded audio and metadata lists.
    """
    batch = list(items)
    if not batch:
        raise ValueError("collate_dataset_items received an empty batch")

    sample_rate = batch[0].sample_rate
    for item in batch[1:]:
        if item.sample_rate != sample_rate:
            raise ValueError("sample_rate must be consistent within a batch")

    lengths = torch.tensor([item.audio.numel() for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    audio = torch.full(
        (len(batch), max_len),
        pad_value,
        dtype=batch[0].audio.dtype,
        device=batch[0].audio.device,
    )

    for idx, item in enumerate(batch):
        audio[idx, : item.audio.numel()] = item.audio

    utterance_ids = [item.utterance_id for item in batch]
    texts = [item.text for item in batch]
    speakers = [item.speaker for item in batch]

    metadata: Optional[list[Any]] = None
    if keep_metadata:
        metadata = [getattr(item, "metadata", None) for item in batch]

    return CollateBatch(
        audio=audio,
        lengths=lengths,
        sample_rate=sample_rate,
        utterance_ids=utterance_ids,
        texts=texts,
        speakers=speakers,
        metadata=metadata,
    )
