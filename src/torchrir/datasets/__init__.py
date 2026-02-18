"""Dataset helpers for torchrir.

Includes CMU ARCTIC and LibriSpeech dataset wrappers plus collate utilities for
DataLoader usage. Experimental dataset stubs live under
``torchrir.experimental``. Use ``load_dataset_sources`` to build fixed-length
source signals from random utterances. Dynamic CMU ARCTIC scene generation is
available via ``build_dynamic_cmu_arctic_dataset``.

Examples:
    ```python
    from torch.utils.data import DataLoader
    from torchrir.datasets import CmuArcticDataset, collate_dataset_items
    dataset = CmuArcticDataset("datasets/cmu_arctic", speaker="bdl", download=True)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_dataset_items)
    ```

    ```python
    from pathlib import Path
    from torchrir.datasets import LibriSpeechDataset
    librispeech = LibriSpeechDataset(Path("datasets/librispeech"), subset="train-clean-100")
    ```
"""

from .base import BaseDataset, DatasetItem, SentenceLike
from .attribution import DatasetAttribution, attribution_for, default_modification_notes
from .utils import choose_speakers, load_dataset_sources
from ..io.audio import load, save
from .collate import CollateBatch, collate_dataset_items
from .librispeech import LibriSpeechDataset, LibriSpeechSentence

from .cmu_arctic import CmuArcticDataset, CmuArcticSentence, cmu_arctic_speakers
from .dynamic_cmu_arctic import build_dynamic_cmu_arctic_dataset

__all__ = [
    "BaseDataset",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "choose_speakers",
    "DatasetItem",
    "DatasetAttribution",
    "CollateBatch",
    "default_modification_notes",
    "collate_dataset_items",
    "cmu_arctic_speakers",
    "build_dynamic_cmu_arctic_dataset",
    "attribution_for",
    "SentenceLike",
    "load_dataset_sources",
    "load",
    "save",
    "LibriSpeechDataset",
    "LibriSpeechSentence",
]
