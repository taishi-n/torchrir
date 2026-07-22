"""Audio file utilities (dataset-agnostic)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Optional, Tuple, TypeAlias, cast
import warnings

import torch

from ..util._dtypes import validate_supported_float_tensor
from ..util._scalars import (
    normalize_finite_real,
    normalize_integer,
    normalize_sample_rate,
)


AudioSource: TypeAlias = Path | BinaryIO


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class AudioData:
    """Channel-preserving audio plus sample-rate and file metadata.

    ``save_audio_data`` reuses ``subtype`` only when the destination container
    matches ``format``. ``format`` describes a loaded file, while the
    destination path selects the saved container format.
    """

    audio: torch.Tensor
    sample_rate: int
    format: Optional[str] = None
    subtype: Optional[str] = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.audio):
            raise TypeError("AudioData.audio must be a Tensor")
        if self.audio.ndim not in (1, 2):
            raise ValueError(
                "AudioData.audio must have shape (samples,) or (channels, samples)"
            )
        if self.audio.numel() == 0:
            raise ValueError(
                "AudioData.audio must contain at least one channel and sample"
            )
        validate_supported_float_tensor(self.audio, name="AudioData.audio")
        if not torch.all(torch.isfinite(self.audio)):
            raise ValueError("AudioData.audio must contain finite values")
        object.__setattr__(
            self,
            "sample_rate",
            _validate_sample_rate(self.sample_rate, owner="AudioData"),
        )
        for name in ("format", "subtype"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise TypeError(f"AudioData.{name} must be a string or None")
            if isinstance(value, str) and not value.strip():
                raise ValueError(f"AudioData.{name} must be a non-empty string or None")


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioInfo:
    """Basic audio file metadata."""

    sample_rate: int
    num_frames: int
    num_channels: int
    format: str
    subtype: str
    duration: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sample_rate",
            _validate_sample_rate(self.sample_rate, owner="AudioInfo"),
        )
        object.__setattr__(
            self,
            "num_frames",
            normalize_integer(
                self.num_frames,
                name="AudioInfo.num_frames",
                minimum=0,
                maximum=torch.iinfo(torch.int64).max,
            ),
        )
        object.__setattr__(
            self,
            "num_channels",
            normalize_integer(
                self.num_channels,
                name="AudioInfo.num_channels",
                minimum=1,
                maximum=torch.iinfo(torch.int32).max,
            ),
        )
        for name in ("format", "subtype"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"AudioInfo.{name} must be a string")
            if not value.strip():
                raise ValueError(f"AudioInfo.{name} must be a non-empty string")
        object.__setattr__(
            self,
            "duration",
            normalize_finite_real(
                self.duration,
                name="AudioInfo.duration",
                non_negative=True,
            ),
        )


def _load_audio_data(source: AudioSource) -> AudioData:
    """Load all channels from one path or caller-owned binary stream."""

    _validate_audio_source(source)
    sf = _soundfile()
    if isinstance(source, Path):
        sound_file_context = sf.SoundFile(str(source), mode="r")
    else:
        sound_file_context = sf.SoundFile(source, mode="r", closefd=False)
    with sound_file_context as sound_file:
        audio = sound_file.read(dtype="float32", always_2d=True)
        sample_rate = sound_file.samplerate
        format_name = sound_file.format
        subtype = sound_file.subtype
    audio_t = torch.from_numpy(audio).transpose(0, 1).contiguous()
    if audio_t.shape[0] == 1:
        audio_t = audio_t.squeeze(0)
    return AudioData(
        audio=audio_t,
        sample_rate=sample_rate,
        format=format_name,
        subtype=subtype,
    )


def _load_audio(source: AudioSource, *, caller: str) -> Tuple[torch.Tensor, int]:
    """Load an audio file and return mono audio and sample rate."""
    data = _load_audio_data(source)
    audio = data.audio
    if audio.ndim == 2:
        warnings.warn(
            f"{caller} received {audio.shape[0]} channels; using channel 0 only. "
            "Use load_audio_data() to preserve all channels.",
            RuntimeWarning,
            stacklevel=3,
        )
        audio = audio[0]
    return audio, data.sample_rate


def load_audio_data(source: AudioSource) -> AudioData:
    """Load audio and metadata without closing a caller-owned stream."""

    return _load_audio_data(source)


def load_audio(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Load an audio file (wav/flac/other supported by soundfile).

    Notes:
        - Multichannel input uses channel 0 only (warns).
        - A caller-owned binary stream remains open after this function returns.
    """
    return _load_audio(source, caller="load_audio")


def _save_audio(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = False,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel audio file to disk."""
    _validate_audio_path(path)
    if not isinstance(normalize, bool):
        raise TypeError("normalize must be a bool")
    if not torch.is_tensor(audio):
        raise TypeError("audio must be a Tensor")
    if audio.ndim not in (1, 2):
        raise ValueError(
            "audio must have shape (samples,) or channel-first (channels, samples)"
        )
    if audio.numel() == 0:
        raise ValueError("audio must contain at least one channel and sample")
    validate_supported_float_tensor(audio, name="audio")
    sample_rate = _validate_sample_rate(sample_rate, owner="audio")
    if not torch.all(torch.isfinite(audio)):
        raise ValueError("audio must contain finite values")
    if subtype is not None:
        if not isinstance(subtype, str):
            raise TypeError("subtype must be a string or None")
        if not subtype.strip():
            raise ValueError("subtype must be a non-empty string or None")
    resolved_subtype = subtype
    if resolved_subtype is None and path.suffix.lower() in {".wav", ".wave"}:
        resolved_subtype = "FLOAT"
    subtype_name = resolved_subtype.upper() if resolved_subtype is not None else None
    storage_dtype = torch.float64 if subtype_name == "DOUBLE" else torch.float32
    audio = audio.detach().cpu()
    storage_limit = torch.finfo(storage_dtype).max
    if normalize:
        normalized_peak = normalize_finite_real(
            peak,
            name="peak when normalize=True",
            positive=True,
        )
        if normalized_peak > storage_limit:
            raise ValueError(
                f"peak when normalize=True must be representable as {storage_dtype}"
            )
        audio = audio.to(torch.float64)
        max_val = float(audio.abs().max().item())
        if max_val > 0:
            audio = audio / max_val * normalized_peak
    else:
        max_val = float(audio.abs().max().item())
        if max_val > storage_limit:
            raise ValueError(f"audio must be representable as {storage_dtype}")
    audio = audio.to(storage_dtype)
    if not torch.all(torch.isfinite(audio)):
        raise ValueError(f"audio must remain finite as {storage_dtype}")
    output_peak = float(audio.abs().max().item())
    if subtype_name not in {"FLOAT", "DOUBLE"} and output_peak > 1.0:
        raise ValueError(
            "audio exceeds the normalized [-1, 1] range of integer/compressed "
            "formats; apply one common scale, pass normalize=True, or save WAV "
            "with subtype='FLOAT' or 'DOUBLE'"
        )
    if audio.ndim == 2:
        audio = audio.transpose(0, 1)
    sf = _soundfile()
    sf.write(
        str(path),
        audio.numpy(),
        sample_rate,
        format=_container_format_from_path(path),
        subtype=resolved_subtype,
    )


def save_audio(
    path: Path,
    audio: torch.Tensor,
    sample_rate: int,
    *,
    normalize: bool = False,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save a mono or multi-channel audio file without changing its gain.

    Use ``normalize=True`` only when independent peak normalization is intended.
    WAV destinations default to the ``FLOAT`` subtype. Other formats reject
    out-of-range samples before their integer/compressed encoding can clip.
    For related stems and mixtures, apply one common scale before saving.
    """
    _save_audio(
        path,
        audio,
        sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


def save_audio_data(
    path: Path,
    data: AudioData,
    *,
    normalize: bool = False,
    peak: float = 1.0,
    subtype: str | None = None,
) -> None:
    """Save audio, preserving subtype only when the container is unchanged."""

    _validate_audio_path(path)
    if not isinstance(data, AudioData):
        raise TypeError("data must be an AudioData instance")
    inherited_subtype = None
    if subtype is None and data.subtype is not None:
        destination_format = _container_format_from_path(path)
        if (
            data.format is not None
            and destination_format is not None
            and data.format.upper() == destination_format
        ):
            inherited_subtype = data.subtype
    _save_audio(
        path,
        data.audio,
        data.sample_rate,
        normalize=normalize,
        peak=peak,
        subtype=inherited_subtype if subtype is None else subtype,
    )


def info_audio(path: Path) -> AudioInfo:
    """Return metadata for an audio file (wav/flac/other supported by soundfile)."""
    _validate_audio_path(path)
    sf = _soundfile()

    info = sf.info(str(path))
    return AudioInfo(
        sample_rate=info.samplerate,
        num_frames=info.frames,
        num_channels=info.channels,
        format=info.format,
        subtype=info.subtype,
        duration=float(info.duration),
    )


def _soundfile() -> Any:
    try:
        import soundfile
    except ImportError as exc:
        raise ImportError(
            "Audio I/O requires the 'audio' extra: pip install torchrir[audio]"
        ) from exc
    return soundfile


def _container_format_from_path(path: Path) -> str | None:
    suffix = path.suffix.lower()
    aliases = {
        ".aif": "AIFF",
        ".aifc": "AIFF",
        ".aiff": "AIFF",
        ".oga": "OGG",
        ".snd": "AU",
        ".wav": "WAV",
        ".wave": "WAV",
    }
    if suffix in aliases:
        return aliases[suffix]
    if len(suffix) <= 1:
        return None
    return suffix[1:].upper()


def _validate_sample_rate(value: object, *, owner: str) -> int:
    return normalize_sample_rate(value, name=f"{owner}.sample_rate")


def _validate_audio_path(path: object) -> None:
    if not isinstance(path, Path):
        raise TypeError("audio path must be a pathlib.Path")


def _validate_audio_source(source: object) -> None:
    if isinstance(source, Path):
        return
    required_methods = ("read", "seek", "tell", "seekable")
    if not all(callable(getattr(source, name, None)) for name in required_methods):
        raise TypeError("audio source must be a pathlib.Path or seekable binary stream")
    if bool(getattr(source, "closed", False)):
        raise ValueError("audio source stream must be open")
    stream = cast(BinaryIO, source)
    try:
        seekable = stream.seekable()
    except (OSError, ValueError) as exc:
        raise ValueError("audio source stream must be seekable") from exc
    if seekable is not True:
        raise ValueError("audio source stream must be seekable")
    try:
        position = stream.tell()
        probe = stream.read(0)
        stream.seek(position)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("audio source stream must support binary seek/read") from exc
    if not isinstance(probe, (bytes, bytearray, memoryview)):
        raise TypeError("audio source must be a pathlib.Path or seekable binary stream")
