"""Audio file utilities (dataset-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import warnings

import torch


@dataclass(frozen=True)
class AudioInfo:
    """Basic audio file metadata."""

    sample_rate: int
    num_frames: int
    num_channels: int
    format: str
    subtype: str
    duration: float


def _load_audio(path: Path, *, caller: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return mono audio and sample rate."""
    import soundfile as sf

    info = sf.info(str(path))
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio_t = torch.from_numpy(audio)
    if audio_t.shape[1] > 1:
        warnings.warn(
            f"{caller} received {audio_t.shape[1]} channels; using channel 0 only.",
            RuntimeWarning,
        )
        audio_t = audio_t[:, 0]
    else:
        audio_t = audio_t.squeeze(1)
    setattr(audio_t, "_torchrir_subtype", info.subtype)
    setattr(audio_t, "_torchrir_format", info.format)
    return audio_t, sample_rate


def load(path: Path) -> Tuple[torch.Tensor, int]:
    """Load a wav file and return mono audio and sample rate.

    Notes:
        - Multichannel input uses channel 0 only (warns).
        - The original file subtype/format are stored on the returned tensor
          as `_torchrir_subtype` and `_torchrir_format` for reuse by save.
        - For non-wav formats, use ``torchrir.io.audio.load_audio``.

    Example:
        >>> audio, fs = load(Path("datasets/cmu_arctic/.../arctic_a0001.wav"))
    """
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".wave"}:
        raise ValueError(
            f"load expects a wav file, got '{path.name}'. "
            "Use torchrir.io.audio.load_audio for non-wav formats."
        )
    return _load_audio(path, caller="load")


def load_audio(path: Path) -> Tuple[torch.Tensor, int]:
    """Load an audio file (wav/flac/other supported by soundfile).

    Notes:
        - Multichannel input uses channel 0 only (warns).
        - The original file subtype/format are stored on the returned tensor
          as `_torchrir_subtype` and `_torchrir_format` for reuse by save_audio.
    """
    return _load_audio(path, caller="load_audio")


def _save_audio(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = True,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel audio file to disk."""
    import soundfile as sf

    audio = audio.detach().cpu().to(torch.float32)
    if normalize:
        if peak <= 0:
            raise ValueError("peak must be positive when normalize=True")
        max_val = float(audio.abs().max().item()) if audio.numel() else 0.0
        if max_val > 0:
            audio = audio / max_val * peak
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.transpose(0, 1)
    if subtype is None:
        subtype = getattr(audio, "_torchrir_subtype", None)
    sf.write(str(path), audio.numpy(), sample_rate, subtype=subtype)


def save(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = True,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel wav to disk.

    By default this normalizes to the specified peak and preserves the input
    file subtype when `subtype=None` and the tensor came from `load`.
    Values outside [-1, 1] are preserved when normalization is disabled.
    For non-wav formats, use ``torchrir.io.audio.save_audio``.

    Example:
        >>> save(Path("outputs/example.wav"), audio, sample_rate)
    """
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".wave"}:
        raise ValueError(
            f"save expects a wav file, got '{path.name}'. "
            "Use torchrir.io.audio.save_audio for non-wav formats."
        )
    _save_audio(
        path,
        audio,
        sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


def save_audio(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = True,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel audio file to disk.

    Use this for non-wav formats supported by soundfile (e.g., FLAC).
    """
    _save_audio(
        path,
        audio,
        sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


def info_audio(path: Path) -> AudioInfo:
    """Return metadata for an audio file (wav/flac/other supported by soundfile)."""
    import soundfile as sf

    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=info.samplerate,
        num_frames=info.frames,
        num_channels=info.channels,
        format=info.format,
        subtype=info.subtype,
        duration=float(info.duration),
    )
