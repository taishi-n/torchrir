"""I/O helpers for audio files and metadata serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

from torch import Tensor

from .audio import AudioData, AudioInfo, _load_audio, _save_audio, info_audio
from .metadata import build_metadata, save_metadata_json
from .outputs import save_scene_audio, save_scene_metadata


@dataclass(frozen=True)
class AudioBackend:
    """Audio I/O backend definition."""

    name: str
    load: Callable[[Path, str], Tuple[Tensor, int]]
    save: Callable[[Path, Tensor, int, bool, float, str | None], None]
    info: Callable[[Path], AudioInfo]


def _soundfile_load(path: Path, caller: str) -> Tuple[Tensor, int]:
    return _load_audio(path, caller=caller)


def _soundfile_save(
    path: Path,
    audio: Tensor,
    sample_rate: int,
    normalize: bool,
    peak: float,
    subtype: str | None,
) -> None:
    _save_audio(
        path,
        audio,
        sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


_AUDIO_BACKENDS = {
    "soundfile": AudioBackend(
        name="soundfile",
        load=_soundfile_load,
        save=_soundfile_save,
        info=info_audio,
    )
}
_DEFAULT_AUDIO_BACKEND = "soundfile"


def list_audio_backends() -> list[str]:
    """Return the available audio backends."""

    return sorted(_AUDIO_BACKENDS.keys())


def get_audio_backend() -> str:
    """Return the current default audio backend."""

    return _DEFAULT_AUDIO_BACKEND


def set_audio_backend(name: str) -> None:
    """Set the default audio backend."""

    if name not in _AUDIO_BACKENDS:
        raise ValueError(
            f"Unknown audio backend '{name}'. Available: {sorted(_AUDIO_BACKENDS)}"
        )
    global _DEFAULT_AUDIO_BACKEND
    _DEFAULT_AUDIO_BACKEND = name


def _resolve_backend(name: str | None) -> AudioBackend:
    backend_name = name or _DEFAULT_AUDIO_BACKEND
    if backend_name not in _AUDIO_BACKENDS:
        raise ValueError(
            f"Unknown audio backend '{backend_name}'. Available: {sorted(_AUDIO_BACKENDS)}"
        )
    return _AUDIO_BACKENDS[backend_name]


def _normalize_format(path: Path, fmt: str | None) -> str:
    fmt = (fmt or path.suffix.lstrip(".")).lower()
    if not fmt:
        raise ValueError(
            "Audio format could not be inferred from the path. "
            "Pass format='wav' or use torchrir.io.audio.load_audio/save_audio."
        )
    return fmt


def load(
    path: Path,
    *,
    backend: str | None = None,
    format: str | None = None,
) -> Tuple[Tensor, int]:
    """Load a wav file and return mono audio and sample rate.

    This entry point is wav-only. For non-wav formats, use
    ``torchrir.io.audio.load_audio``.
    """

    fmt = _normalize_format(path, format)
    if fmt not in {"wav", "wave"}:
        raise ValueError(
            f"load expects a wav file, got format '{fmt}'. "
            "Use torchrir.io.audio.load_audio for non-wav formats."
        )
    backend_impl = _resolve_backend(backend)
    return backend_impl.load(path, "load")


def save(
    path: Path,
    audio: Tensor,
    sample_rate: int,
    *,
    backend: str | None = None,
    format: str | None = None,
    normalize: bool = True,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a wav file to disk.

    This entry point is wav-only. For non-wav formats, use
    ``torchrir.io.audio.save_audio``.
    """

    fmt = _normalize_format(path, format)
    if fmt not in {"wav", "wave"}:
        raise ValueError(
            f"save expects a wav file, got format '{fmt}'. "
            "Use torchrir.io.audio.save_audio for non-wav formats."
        )
    backend_impl = _resolve_backend(backend)
    backend_impl.save(path, audio, sample_rate, normalize, peak, subtype)


def info(
    path: Path,
    *,
    backend: str | None = None,
    format: str | None = None,
) -> AudioInfo:
    """Return metadata for a wav file.

    This entry point is wav-only. For non-wav formats, use
    ``torchrir.io.audio.info_audio``.
    """

    fmt = _normalize_format(path, format)
    if fmt not in {"wav", "wave"}:
        raise ValueError(
            f"info expects a wav file, got format '{fmt}'. "
            "Use torchrir.io.audio.info_audio for non-wav formats."
        )
    backend_impl = _resolve_backend(backend)
    return backend_impl.info(path)


__all__ = [
    "AudioData",
    "AudioBackend",
    "build_metadata",
    "get_audio_backend",
    "info",
    "list_audio_backends",
    "load",
    "save_scene_audio",
    "save_scene_metadata",
    "save_metadata_json",
    "save",
    "set_audio_backend",
]
