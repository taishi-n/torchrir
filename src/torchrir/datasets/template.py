from __future__ import annotations

"""Dataset template for future extensions.

Work in progress:
    This module is a placeholder for future dataset integrations. The goal is
    to provide a consistent interface for downloading, caching, enumerating
    speakers/utterances, and loading audio in a reproducible way.
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
    """Work in progress template dataset implementation.

    Goal:
        Implement concrete dataset handlers by filling in download logic,
        metadata parsing, and audio loading while keeping the BaseDataset
        protocol intact.
    """

    def __init__(
        self, root: Path, speaker: str = "default", download: bool = False
    ) -> None:
        self.root = Path(root)
        self.speaker = speaker
        if download:
            raise NotImplementedError(
                "download is not implemented yet. Intended to fetch and cache "
                "dataset archives under root."
            )

    def list_speakers(self) -> List[str]:
        """Return available speaker IDs."""
        return ["default"]

    def available_sentences(self) -> Sequence[SentenceLike]:
        """Return sentence entries that have audio available.

        Work in progress:
            Intended to parse dataset metadata and filter to utterances that
            have corresponding audio files on disk.
        """
        raise NotImplementedError("available_sentences is not implemented yet")

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load audio for an utterance and return (audio, sample_rate).

        Work in progress:
            Intended to load audio from local cache and return mono float32.
        """
        raise NotImplementedError("load_wav is not implemented yet")
