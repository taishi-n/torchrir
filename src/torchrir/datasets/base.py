from __future__ import annotations

"""Dataset protocol definitions."""

from typing import Protocol, Sequence, Tuple

import torch


class SentenceLike(Protocol):
    """Minimal sentence interface for dataset entries."""

    utterance_id: str
    text: str


class BaseDataset(Protocol):
    """Protocol for datasets used in torchrir examples and tools."""

    def list_speakers(self) -> list[str]:
        """Return available speaker IDs."""

    def available_sentences(self) -> Sequence[SentenceLike]:
        """Return sentence entries that have audio available."""

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio for an utterance and return (audio, sample_rate)."""
