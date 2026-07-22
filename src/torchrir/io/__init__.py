"""I/O helpers for audio files and metadata serialization."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor

from .audio import (
    AudioData,
    AudioInfo,
    _load_audio,
    _save_audio,
    info_audio,
    load_audio,
    load_audio_data,
    save_audio,
    save_audio_data,
)
from .metadata import build_metadata, build_result_metadata, save_metadata_json
from .outputs import (
    save_attribution_file,
    save_result_metadata,
    save_scene_audio,
    save_scene_metadata,
)


def _validate_wav_path(path: Path, operation: str) -> None:
    if not isinstance(path, Path):
        raise TypeError("wav path must be a pathlib.Path")
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".wave"}:
        raise ValueError(
            f"{operation} expects a wav file, got '{path.name}'. "
            f"Use torchrir.io.{operation}_audio for other formats."
        )


def load_wav(path: Path) -> tuple[Tensor, int]:
    """Load a wav file and return mono audio and sample rate.

    This entry point is wav-only. For non-wav formats, use
    ``torchrir.io.load_audio``.
    """

    _validate_wav_path(path, "load")
    return _load_audio(path, caller="load_wav")


def save_wav(
    path: Path,
    audio: Tensor,
    sample_rate: int,
    *,
    normalize: bool = False,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a wav file without changing its gain.

    With no explicit ``subtype``, WAV output uses 32-bit floating-point samples
    so values outside ``[-1, 1]`` are not clipped. Set ``normalize=True``
    explicitly to peak-normalize. This entry point is wav-only; for non-wav
    formats, use ``torchrir.io.save_audio``.
    """

    _validate_wav_path(path, "save")
    _save_audio(
        path,
        audio,
        sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


def info_wav(path: Path) -> AudioInfo:
    """Return metadata for a wav file.

    This entry point is wav-only. For non-wav formats, use
    ``torchrir.io.info_audio``.
    """

    _validate_wav_path(path, "info")
    return info_audio(path)


__all__ = [
    "AudioData",
    "AudioInfo",
    "build_metadata",
    "build_result_metadata",
    "info_audio",
    "info_wav",
    "load_audio",
    "load_audio_data",
    "load_wav",
    "save_attribution_file",
    "save_audio",
    "save_audio_data",
    "save_metadata_json",
    "save_scene_audio",
    "save_scene_metadata",
    "save_result_metadata",
    "save_wav",
]
