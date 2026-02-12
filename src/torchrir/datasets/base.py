"""Dataset protocol definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .attribution import DatasetAttribution


class SentenceLike(Protocol):
    """Minimal sentence interface for dataset entries."""

    utterance_id: str
    text: str


@dataclass(frozen=True)
class DatasetItem:
    """Dataset item for DataLoader consumption."""

    audio: torch.Tensor
    sample_rate: int
    utterance_id: str
    text: Optional[str] = None
    speaker: Optional[str] = None


class BaseDataset(Dataset[DatasetItem]):
    """Base dataset class compatible with torch.utils.data.Dataset."""

    _sentences_cache: Optional[list[SentenceLike]] = None

    def list_speakers(self) -> list[str]:
        """Return available speaker IDs."""
        raise NotImplementedError

    def available_sentences(self) -> Sequence[SentenceLike]:
        """Return sentence entries that have audio available."""
        raise NotImplementedError

    def load_audio(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio for an utterance and return (audio, sample_rate)."""
        raise NotImplementedError

    def attribution_info(self) -> DatasetAttribution:
        """Return attribution and license information for this dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._get_sentences())

    def __getitem__(self, idx) -> DatasetItem:  # ty: ignore[invalid-method-override]
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int, got {type(idx)!r}")
        sentences = self._get_sentences()
        sentence = sentences[idx]
        audio, sample_rate = self.load_audio(sentence.utterance_id)
        speaker = getattr(self, "speaker", None)
        text = getattr(sentence, "text", None)
        return DatasetItem(
            audio=audio,
            sample_rate=sample_rate,
            utterance_id=sentence.utterance_id,
            text=text,
            speaker=speaker,
        )

    def _get_sentences(self) -> list[SentenceLike]:
        if self._sentences_cache is None:
            self._sentences_cache = list(self.available_sentences())
        return self._sentences_cache
