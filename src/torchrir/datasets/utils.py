"""Dataset-agnostic utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
import random

import torch

from .base import BaseDataset, SentenceLike, _validate_sample_rate
from ..util._dtypes import validate_supported_float_tensor
from ..util._scalars import normalize_finite_real


_INT64_MAX = torch.iinfo(torch.int64).max


def choose_speakers(
    speakers: Sequence[str], num_sources: int, rng: random.Random
) -> list[str]:
    """Select unique speakers for the requested number of sources.

    Examples:
        ```python
        rng = random.Random(0)
        speakers = choose_speakers(available_speakers, num_sources=2, rng=rng)
        ```
    """
    if isinstance(speakers, (str, bytes)) or not isinstance(speakers, Sequence):
        raise TypeError("speakers must be a sequence of speaker IDs")
    _validate_positive_integer(num_sources, name="num_sources")
    _validate_rng(rng)
    normalized_speakers = list(speakers)
    if not normalized_speakers:
        raise RuntimeError("no speakers available")
    if any(
        not isinstance(speaker, str) or not speaker.strip()
        for speaker in normalized_speakers
    ):
        raise ValueError("speakers must contain non-empty strings")
    if len(set(normalized_speakers)) != len(normalized_speakers):
        raise ValueError("speakers must be unique")
    if num_sources > len(normalized_speakers):
        raise ValueError(
            f"num_sources must be <= {len(normalized_speakers)} for unique speakers"
        )
    return rng.sample(normalized_speakers, num_sources)


def load_dataset_sources(
    *,
    dataset_factory: Callable[[str], BaseDataset],
    speakers: Sequence[str],
    num_sources: int,
    duration_s: float,
    rng: random.Random,
) -> tuple[torch.Tensor, int, list[tuple[str, list[str]]]]:
    """Load and concatenate utterances for each speaker into fixed-length signals.

    Examples:
        ```python
        from pathlib import Path
        from torchrir.datasets import CmuArcticDataset, cmu_arctic_speakers
        rng = random.Random(0)
        root = Path("datasets/cmu_arctic")
        signals, fs, info = load_dataset_sources(
            dataset_factory=lambda spk: CmuArcticDataset(root, speaker=spk, download=True),
            speakers=cmu_arctic_speakers(),
            num_sources=2,
            duration_s=10.0,
            rng=rng,
        )
        ```
    """
    if not callable(dataset_factory):
        raise TypeError("dataset_factory must be callable")
    _validate_positive_integer(num_sources, name="num_sources")
    duration = _validate_positive_duration(duration_s)
    _validate_rng(rng)

    selected_speakers = choose_speakers(speakers, num_sources, rng)
    signals: list[torch.Tensor] = []
    info: list[tuple[str, list[str]]] = []
    fs: int | None = None
    target_samples: int | None = None
    common_dtype: torch.dtype | None = None
    common_device: torch.device | None = None

    for speaker in selected_speakers:
        dataset = dataset_factory(speaker)
        if not isinstance(dataset, BaseDataset):
            raise TypeError("dataset_factory must return a BaseDataset")
        sentences: Sequence[SentenceLike] = dataset.available_sentences()
        if not sentences:
            raise RuntimeError(f"no sentences found for speaker {speaker}")

        utterance_ids: list[str] = []
        segments: list[torch.Tensor] = []
        total = 0
        sentences = list(sentences)
        rng.shuffle(sentences)
        idx = 0

        while target_samples is None or total < target_samples:
            if idx >= len(sentences):
                rng.shuffle(sentences)
                idx = 0
            sentence = sentences[idx]
            idx += 1
            audio, sample_rate = dataset.load_audio(sentence.utterance_id)
            _validate_loaded_audio(audio, sample_rate)
            if fs is None:
                fs = sample_rate
                target_samples = _ceil_dataset_sample_count(duration, fs)
                common_dtype = audio.dtype
                common_device = audio.device
            elif sample_rate != fs:
                raise ValueError(
                    f"sample rate mismatch: expected {fs}, got {sample_rate} for {speaker}"
                )
            if audio.dtype != common_dtype:
                raise ValueError(
                    "audio dtype must be consistent across dataset sources"
                )
            if audio.device != common_device:
                raise ValueError(
                    "audio device must be consistent across dataset sources"
                )
            segments.append(audio)
            utterance_ids.append(sentence.utterance_id)
            total += int(audio.shape[0])

        assert target_samples is not None
        signal = torch.cat(segments, dim=0)[:target_samples]
        signals.append(signal)
        info.append((speaker, utterance_ids))

    stacked = torch.stack(signals, dim=0)
    if fs is None:
        raise RuntimeError("no audio loaded from dataset sources")
    return stacked, int(fs), info


def _validate_positive_integer(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_positive_duration(value: object) -> float:
    return normalize_finite_real(value, name="duration_s", positive=True)


def _ceil_dataset_sample_count(duration_s: float, sample_rate: int) -> int:
    try:
        product = duration_s * sample_rate
    except OverflowError as exc:
        raise ValueError(
            "duration_s and sample_rate must produce a sample count in the "
            "positive int64 range"
        ) from exc
    if not math.isfinite(product) or product <= 0.0:
        raise ValueError(
            "duration_s and sample_rate must produce a sample count in the "
            "positive int64 range"
        )
    count = math.ceil(product)
    if count > _INT64_MAX:
        raise ValueError(
            "duration_s and sample_rate must produce a sample count in the "
            "positive int64 range"
        )
    return count


def _validate_rng(value: object) -> None:
    if not isinstance(value, random.Random):
        raise TypeError("rng must be a random.Random instance")


def _validate_loaded_audio(audio: object, sample_rate: object) -> None:
    if not torch.is_tensor(audio):
        raise TypeError("dataset audio must be a Tensor")
    if audio.ndim != 1:
        raise ValueError("dataset audio must be mono with shape (samples,)")
    if audio.numel() == 0:
        raise ValueError("dataset audio must contain at least one sample")
    validate_supported_float_tensor(audio, name="dataset audio")
    if not torch.all(torch.isfinite(audio)):
        raise ValueError("dataset audio must contain finite values")
    _validate_sample_rate(sample_rate)
