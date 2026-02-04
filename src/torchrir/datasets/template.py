from __future__ import annotations

"""Dataset template for future extensions.

This is a placeholder stub intended to be expanded as new datasets are added.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from .base import BaseDataset, SentenceLike


@dataclass
class TemplateSentence:
    """Minimal sentence metadata for a template dataset."""

    utterance_id: str
    text: str


class TemplateDataset(BaseDataset):
    """Template dataset implementation (placeholder for future extension)."""

    def __init__(self, root: Path, speaker: str = "default", download: bool = False) -> None:
        self.root = Path(root)
        self.speaker = speaker
        if download:
            raise NotImplementedError("download is not implemented for TemplateDataset")

    def list_speakers(self) -> List[str]:
        """Return available speaker IDs."""
        return ["default"]

    def available_sentences(self) -> Sequence[SentenceLike]:
        """Return sentence entries that have audio available."""
        raise NotImplementedError("available_sentences is not implemented for TemplateDataset")

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio for an utterance and return (audio, sample_rate)."""
        raise NotImplementedError("load_wav is not implemented for TemplateDataset")
