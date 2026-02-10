"""Experimental dataset stubs for future integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..datasets.base import BaseDataset, DatasetItem


@dataclass(frozen=True)
class TemplateSentence:
    """Template for dataset sentences (work in progress)."""

    utterance_id: str
    speaker: str
    text: str
    wav_path: Path


class TemplateDataset(BaseDataset):
    """Template dataset stub for future integrations.

    This class is a placeholder to document the expected dataset API surface.
    It will be replaced with concrete dataset loaders in future releases.
    """

    def __len__(self) -> int:
        raise NotImplementedError("TemplateDataset is not implemented yet")

    def __getitem__(self, idx) -> DatasetItem:
        raise NotImplementedError("TemplateDataset is not implemented yet")
