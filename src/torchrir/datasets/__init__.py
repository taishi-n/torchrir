"""Dataset helpers for torchrir.

Includes CMU ARCTIC and LibriSpeech dataset wrappers plus collate utilities for
DataLoader usage. Use ``load_dataset_sources`` to build fixed-length source
signals from random utterances. Dynamic CMU ARCTIC scene generation is
available via ``build_dynamic_cmu_arctic``.

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

from typing import TYPE_CHECKING, Any

from .base import BaseDataset, DatasetItem, SentenceLike
from .attribution import DatasetAttribution, attribution_for, default_modification_notes
from .utils import choose_speakers, load_dataset_sources
from .collate import CollateBatch, collate_dataset_items
from .librispeech import LibriSpeechDataset, LibriSpeechSentence

from .cmu_arctic import CmuArcticDataset, CmuArcticSentence, cmu_arctic_speakers
from .dynamic_builder import DynamicCmuArcticBuildConfig, DynamicDatasetBuildResult

if TYPE_CHECKING:
    from .dynamic_cmu_arctic import build_dynamic_cmu_arctic


def __getattr__(name: str) -> Any:
    """Load the executable dynamic builder only when its API is requested."""

    if name == "build_dynamic_cmu_arctic":
        from .dynamic_cmu_arctic import build_dynamic_cmu_arctic

        globals()[name] = build_dynamic_cmu_arctic
        return build_dynamic_cmu_arctic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "build_dynamic_cmu_arctic",
    "attribution_for",
    "SentenceLike",
    "load_dataset_sources",
    "LibriSpeechDataset",
    "LibriSpeechSentence",
    "DynamicCmuArcticBuildConfig",
    "DynamicDatasetBuildResult",
]
