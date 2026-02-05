"""Dataset helpers for torchrir.

Includes CMU ARCTIC and LibriSpeech dataset wrappers, a template stub for new
integrations, and collate utilities for DataLoader usage. Use
``load_dataset_sources`` to build fixed-length source signals from random
utterances.

Example:
    >>> from torch.utils.data import DataLoader
    >>> from torchrir import CmuArcticDataset, collate_dataset_items
    >>> dataset = CmuArcticDataset("datasets/cmu_arctic", speaker="bdl", download=True)
    >>> loader = DataLoader(dataset, batch_size=4, collate_fn=collate_dataset_items)

    >>> from pathlib import Path
    >>> from torchrir import LibriSpeechDataset
    >>> librispeech = LibriSpeechDataset(Path("datasets/librispeech"), subset="train-clean-100")
"""

from .base import BaseDataset, DatasetItem, SentenceLike
from .utils import choose_speakers, load_dataset_sources
from ..io.audio import load_wav_mono, save_wav
from .collate import CollateBatch, collate_dataset_items
from .template import TemplateDataset, TemplateSentence
from .librispeech import LibriSpeechDataset, LibriSpeechSentence

from .cmu_arctic import CmuArcticDataset, CmuArcticSentence, list_cmu_arctic_speakers

__all__ = [
    "BaseDataset",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "choose_speakers",
    "DatasetItem",
    "CollateBatch",
    "collate_dataset_items",
    "list_cmu_arctic_speakers",
    "SentenceLike",
    "load_dataset_sources",
    "load_wav_mono",
    "save_wav",
    "TemplateDataset",
    "TemplateSentence",
    "LibriSpeechDataset",
    "LibriSpeechSentence",
]
