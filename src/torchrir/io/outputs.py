"""Output helpers for saving audio and metadata."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Optional, TYPE_CHECKING
import logging

from torch import Tensor

from .metadata import (
    TimeReference,
    build_metadata,
    build_result_metadata,
    save_metadata_json,
)
from ..logging import _validate_info_logger

if TYPE_CHECKING:
    from ..models import MicrophoneArray, RIRResult, Room, Source
    from ..models.schedule import FrameSchedule


_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)
_WINDOWS_INVALID_FILENAME_CHARS = frozenset('<>:"/\\|?*')


def save_scene_audio(
    *,
    out_dir: Path,
    audio: Tensor,
    fs: int,
    audio_name: str,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Save scene audio without applying implicit gain normalization."""
    _validate_info_logger(logger)
    audio_name = _validate_output_name(audio_name, name="audio_name")
    from . import save_wav

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / audio_name
    save_wav(out_path, audio, fs)
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
    _validate_info_logger(logger)
    attribution_name = _validate_output_name(
        attribution_name,
        name="attribution_name",
    )
    info = _coerce_attribution_mapping(dataset_attribution)
    notes = _validate_modifications(modifications)
    out_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "TorchRIR Dataset Attribution",
        "",
        "This directory contains derived audio generated with TorchRIR.",
        "",
        f"Dataset: {info['dataset']}",
        f"Source: {info['source']}",
        f"License: {info['license_name']}",
        f"License URL: {info['license_url']}",
        "Attribution required: " + ("yes" if info["attribution_required"] else "no"),
        f"Required attribution: {info['required_attribution']}",
    ]
    subset = info.get("subset")
    if subset is not None:
        lines.append(f"Subset: {subset}")
    lines.extend(
        [
            "",
            "Modifications applied in this output:",
            *[f"- {note}" for note in notes],
            "",
        ]
    )
    if info["attribution_required"]:
        lines.extend(
            [
                "When redistributing these derived files, keep this attribution file",
                "and include the upstream dataset license terms.",
            ]
        )
    else:
        lines.append(
            "Upstream attribution is not required; this file records provenance."
        )
    lines.extend(["", "See repository notice: THIRD_PARTY_DATASETS.md"])
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
    schedule: Optional["FrameSchedule"] = None,
    time_reference: TimeReference | None = None,
    signal_len: Optional[int] = None,
    source_info: Optional[Any] = None,
    extra: Mapping[str, Any] | None = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Build and save scene metadata JSON to the output directory."""
    _validate_info_logger(logger)
    metadata_name = _validate_output_name(metadata_name, name="metadata_name")
    metadata = build_metadata(
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=schedule,
        time_reference=time_reference,
        signal_len=signal_len,
        source_info=source_info,
        extra=extra,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / metadata_name
    save_metadata_json(meta_path, metadata)
    if logger is not None:
        logger.info("saved: %s", meta_path)
    return metadata


def save_result_metadata(
    *,
    out_dir: Path,
    result: "RIRResult",
    metadata_name: str = "metadata.json",
    schedule: Optional["FrameSchedule"] = None,
    time_reference: TimeReference | None = None,
    signal_len: Optional[int] = None,
    source_info: Optional[Any] = None,
    extra: Mapping[str, Any] | None = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Build and save metadata from an RIRResult."""

    _validate_info_logger(logger)
    metadata_name = _validate_output_name(metadata_name, name="metadata_name")
    metadata = build_result_metadata(
        result,
        schedule=schedule,
        time_reference=time_reference,
        signal_len=signal_len,
        source_info=source_info,
        extra=extra,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save_metadata_json(out_dir / metadata_name, metadata)
    if logger is not None:
        logger.info("saved: %s", out_dir / metadata_name)
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
    for key in required:
        info[key] = _validate_single_line_string(
            info[key],
            name=f"dataset_attribution[{key!r}]",
        )
    subset = info.get("subset")
    if subset is not None:
        info["subset"] = _validate_single_line_string(
            subset,
            name="dataset_attribution['subset']",
        )
    attribution_required = info.get("attribution_required", True)
    if not isinstance(attribution_required, bool):
        raise TypeError("dataset_attribution['attribution_required'] must be a bool")
    info["attribution_required"] = attribution_required
    return info


def _validate_modifications(modifications: object) -> tuple[str, ...]:
    if not isinstance(modifications, list):
        raise TypeError("modifications must be a list of strings")
    return tuple(
        _validate_single_line_string(note, name="modifications entries")
        for note in modifications
    )


def _validate_single_line_string(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")
    if "\n" in value or "\r" in value or "\x00" in value:
        raise ValueError(f"{name} must be a single-line string")
    return value


def _validate_output_name(value: object, *, name: str) -> str:
    normalized = _validate_single_line_string(value, name=name)
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(normalized)
    if (
        len(posix.parts) != 1
        or len(windows.parts) != 1
        or posix.name != normalized
        or windows.name != normalized
        or any(character in _WINDOWS_INVALID_FILENAME_CHARS for character in normalized)
        or any(ord(character) < 32 for character in normalized)
        or normalized.split(".", 1)[0].rstrip(" .").upper()
        in _WINDOWS_RESERVED_BASENAMES
        or normalized.rstrip(" .") != normalized
        or normalized in {".", ".."}
    ):
        raise ValueError(f"{name} must be a single file name")
    return normalized
