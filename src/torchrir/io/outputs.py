from __future__ import annotations

"""Output helpers for saving audio and metadata."""

from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
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
