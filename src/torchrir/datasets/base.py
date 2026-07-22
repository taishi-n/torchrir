"""Dataset protocol definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch
from torch.utils.data import Dataset

from .attribution import DatasetAttribution
from ..util._dtypes import validate_supported_float_tensor
from ..util._scalars import normalize_integer, normalize_sample_rate


class SentenceLike(Protocol):
    """Minimal sentence interface for dataset entries."""

    utterance_id: str
    text: str


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class DatasetItem:
    """Validated mono dataset item for DataLoader consumption."""

    audio: torch.Tensor
    sample_rate: int
    utterance_id: str
    text: str | None = None
    speaker: str | None = None
    metadata: Any = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate the shallow-mutable tensor payload."""

        _validate_mono_audio(self.audio)
        object.__setattr__(
            self,
            "sample_rate",
            _validate_sample_rate(self.sample_rate),
        )
        if not isinstance(self.utterance_id, str):
            raise TypeError("utterance_id must be a string")
        if not self.utterance_id.strip():
            raise ValueError("utterance_id must be a non-empty string")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or None")
        if self.speaker is not None and not isinstance(self.speaker, str):
            raise TypeError("speaker must be a string or None")
        if isinstance(self.speaker, str) and not self.speaker.strip():
            raise ValueError("speaker must be a non-empty string or None")


class BaseDataset(Dataset[DatasetItem], ABC):
    """Base dataset class compatible with torch.utils.data.Dataset."""

    _sentences_cache: list[SentenceLike] | None = None

    @abstractmethod
    def list_speakers(self) -> list[str]:
        """Return available speaker IDs."""
        raise NotImplementedError

    @abstractmethod
    def available_sentences(self) -> Sequence[SentenceLike]:
        """Return sentence entries that have audio available."""
        raise NotImplementedError

    @abstractmethod
    def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
        """Load audio for an utterance and return (audio, sample_rate)."""
        raise NotImplementedError

    @abstractmethod
    def attribution_info(self) -> DatasetAttribution:
        """Return attribution and license information for this dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._get_sentences())

    def __getitem__(self, idx: int) -> DatasetItem:  # ty: ignore[invalid-method-override]
        idx = normalize_integer(idx, name="index")
        sentences = self._get_sentences()
        if idx < -len(sentences) or idx >= len(sentences):
            raise IndexError("dataset index out of range")
        sentence = sentences[idx]
        audio, sample_rate = self.load_audio(sentence.utterance_id)
        speaker = getattr(sentence, "speaker_id", None)
        if speaker is None:
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


def _validate_mono_audio(audio: object) -> None:
    if not torch.is_tensor(audio):
        raise TypeError("audio must be a Tensor")
    if audio.ndim != 1:
        raise ValueError("DatasetItem.audio must be mono with shape (samples,)")
    if audio.numel() == 0:
        raise ValueError("DatasetItem.audio must contain at least one sample")
    validate_supported_float_tensor(audio, name="DatasetItem.audio")
    if not torch.all(torch.isfinite(audio)):
        raise ValueError("DatasetItem.audio must contain finite values")


def _validate_sample_rate(sample_rate: object) -> int:
    return normalize_sample_rate(sample_rate)
