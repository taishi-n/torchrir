"""Attribution metadata for supported datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetAttribution:
    """Structured attribution info used for redistribution notices."""

    dataset_key: str
    dataset: str
    source: str
    license_name: str
    license_url: str
    required_attribution: str
    attribution_required: bool = True
    subset: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "dataset_key",
            "dataset",
            "source",
            "license_name",
            "license_url",
            "required_attribution",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
            if "\n" in value or "\r" in value:
                raise ValueError(f"{name} must be a single-line string")
        if not isinstance(self.attribution_required, bool):
            raise TypeError("attribution_required must be a bool")
        if self.subset is not None:
            if not isinstance(self.subset, str):
                raise TypeError("subset must be a string or None")
            if not self.subset.strip():
                raise ValueError("subset must be non-empty when provided")
            if "\n" in self.subset or "\r" in self.subset:
                raise ValueError("subset must be a single-line string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping."""
        return asdict(self)


def attribution_for(dataset: str, subset: str | None = None) -> DatasetAttribution:
    """Return attribution info for a supported dataset key."""
    if not isinstance(dataset, str):
        raise TypeError("dataset must be a string")
    if subset is not None:
        if not isinstance(subset, str):
            raise TypeError("subset must be a string or None")
        if not subset.strip():
            raise ValueError("subset must be non-empty when provided")
    key = dataset.strip().lower()
    if key == "cmu_arctic":
        if subset is not None:
            raise ValueError("subset is only valid for LibriSpeech attribution")
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
    if not isinstance(dynamic, bool):
        raise TypeError("dynamic must be a bool")
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
