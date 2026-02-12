"""Output helpers for saving audio and metadata."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING
import logging

from torch import Tensor

from .audio import save
from .metadata import build_metadata, save_metadata_json

if TYPE_CHECKING:
    from ..models import MicrophoneArray, Room, Source


def save_scene_audio(
    *,
    out_dir: Path,
    audio: Tensor,
    fs: int,
    audio_name: str,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save scene audio to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / audio_name
    save(out_path, audio, fs)
    if logger is not None:
        logger.info("saved: %s", out_path)
    return out_path


def save_attribution_file(
    *,
    out_dir: Path,
    dataset_attribution: Mapping[str, Any] | Any,
    modifications: list[str],
    attribution_name: str = "ATTRIBUTION.txt",
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save dataset attribution and modification notes to a text file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = _coerce_attribution_mapping(dataset_attribution)

    lines = [
        "TorchRIR Dataset Attribution",
        "",
        "This directory contains derived audio generated with TorchRIR.",
        "",
        f"Dataset: {info['dataset']}",
        f"Source: {info['source']}",
        f"License: {info['license_name']}",
        f"License URL: {info['license_url']}",
        f"Required attribution: {info['required_attribution']}",
    ]
    subset = info.get("subset")
    if subset is not None:
        lines.append(f"Subset: {subset}")
    lines.extend(
        [
            "",
            "Modifications applied in this output:",
            *[f"- {note}" for note in modifications],
            "",
            "When redistributing these derived files, keep this attribution file",
            "and include the upstream dataset license terms.",
            "",
            "See repository notice: THIRD_PARTY_DATASETS.md",
        ]
    )
    out_path = out_dir / attribution_name
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if logger is not None:
        logger.info("saved: %s", out_path)
    return out_path


def save_scene_metadata(
    *,
    out_dir: Path,
    metadata_name: str,
    room: "Room",
    sources: "Source",
    mics: "MicrophoneArray",
    rirs: Tensor,
    src_traj: Optional[Tensor] = None,
    mic_traj: Optional[Tensor] = None,
    timestamps: Optional[Tensor] = None,
    signal_len: Optional[int] = None,
    source_info: Optional[Any] = None,
    extra: Optional[dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Build and save scene metadata JSON to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=src_traj,
        mic_traj=mic_traj,
        timestamps=timestamps,
        signal_len=signal_len,
        source_info=source_info,
        extra=extra,
    )
    meta_path = out_dir / metadata_name
    save_metadata_json(meta_path, metadata)
    if logger is not None:
        logger.info("saved: %s", meta_path)
    return metadata


def _coerce_attribution_mapping(
    dataset_attribution: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    if isinstance(dataset_attribution, Mapping):
        info = dict(dataset_attribution)
    elif hasattr(dataset_attribution, "to_dict") and callable(
        dataset_attribution.to_dict
    ):
        info = dict(dataset_attribution.to_dict())
    elif is_dataclass(dataset_attribution):
        info = asdict(dataset_attribution)
    else:
        raise TypeError(
            "dataset_attribution must be a mapping, dataclass, or expose to_dict()."
        )
    required = (
        "dataset",
        "source",
        "license_name",
        "license_url",
        "required_attribution",
    )
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"dataset_attribution missing required keys: {missing}")
    return info
