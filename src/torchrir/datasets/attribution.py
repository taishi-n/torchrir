"""Attribution metadata for supported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class DatasetAttribution:
    """Structured attribution info used for redistribution notices."""

    dataset_key: str
    dataset: str
    source: str
    license_name: str
    license_url: str
    required_attribution: str
    attribution_required: bool = True
    subset: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def attribution_for(dataset: str, subset: Optional[str] = None) -> DatasetAttribution:
    """Return attribution info for a supported dataset key."""
    key = dataset.lower()
    if key == "cmu_arctic":
        return DatasetAttribution(
            dataset_key="cmu_arctic",
            dataset="CMU ARCTIC",
            source="http://www.festvox.org/cmu_arctic/",
            license_name="Permissive (attribution required; see upstream COPYING)",
            license_url="http://www.festvox.org/cmu_arctic/",
            required_attribution=(
                "Carnegie Mellon University, Language Technologies Institute (CMU ARCTIC)"
            ),
        )
    if key == "librispeech":
        return DatasetAttribution(
            dataset_key="librispeech",
            dataset="LibriSpeech (SLR12)",
            source="https://www.openslr.org/12",
            license_name="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            license_url="https://creativecommons.org/licenses/by/4.0/",
            required_attribution=(
                "Vassil Panayotov, Guoguo Chen, Daniel Povey, and "
                "Sanjeev Khudanpur (LibriSpeech, 2015)"
            ),
            subset=subset,
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def default_modification_notes(*, dynamic: bool) -> list[str]:
    """Return concise modification notes for generated outputs."""
    notes = [
        "Utterances are concatenated and trimmed to a fixed duration per source.",
        "Outputs are derived mixtures and per-source convolved references.",
    ]
    if dynamic:
        notes.insert(
            1,
            "Dynamic room impulse responses are simulated with ISM over trajectories.",
        )
    else:
        notes.insert(
            1,
            "Static room impulse responses are simulated with ISM at fixed geometry.",
        )
    return notes
