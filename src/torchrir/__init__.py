"""TorchRIR public API."""

from pathlib import Path
from typing import Tuple
import warnings

from torch import Tensor

from . import io
from .models import (
    DynamicScene,
    MicrophoneArray,
    RIRResult,
    Room,
    Scene,
    Source,
    StaticScene,
)


def load(
    path: Path, *, backend: str | None = None, format: str | None = None
) -> Tuple[Tensor, int]:
    """Deprecated top-level loader. Use `torchrir.io.load_wav`/`torchrir.io.load_audio`."""

    warnings.warn(
        "torchrir.load is deprecated. Use torchrir.io.load_wav or torchrir.io.load_audio.",
        DeprecationWarning,
        stacklevel=2,
    )
    return io.load_wav(path, backend=backend, format=format)


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
    """Deprecated top-level saver. Use `torchrir.io.save_wav`/`torchrir.io.save_audio`."""

    warnings.warn(
        "torchrir.save is deprecated. Use torchrir.io.save_wav or torchrir.io.save_audio.",
        DeprecationWarning,
        stacklevel=2,
    )
    io.save_wav(
        path,
        audio,
        sample_rate,
        backend=backend,
        format=format,
        normalize=normalize,
        peak=peak,
        subtype=subtype,
    )


__all__ = [
    "DynamicScene",
    "Room",
    "Source",
    "MicrophoneArray",
    "Scene",
    "StaticScene",
    "RIRResult",
    "load",
    "save",
]
