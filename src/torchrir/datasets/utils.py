from __future__ import annotations

"""Dataset-agnostic utilities."""

import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .base import BaseDataset, SentenceLike


def choose_speakers(
    dataset: BaseDataset, num_sources: int, rng: random.Random
) -> List[str]:
    """Select unique speakers for the requested number of sources.

    Example:
        >>> rng = random.Random(0)
        >>> speakers = choose_speakers(dataset, num_sources=2, rng=rng)
    """
    speakers = dataset.list_speakers()
    if not speakers:
        raise RuntimeError("no speakers available")
    if num_sources > len(speakers):
        raise ValueError(f"num_sources must be <= {len(speakers)} for unique speakers")
    return rng.sample(speakers, num_sources)


def load_dataset_sources(
    *,
    dataset_factory: Callable[[Optional[str]], BaseDataset],
    num_sources: int,
    duration_s: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, int, List[Tuple[str, List[str]]]]:
    """Load and concatenate utterances for each speaker into fixed-length signals.

    Example:
        >>> from pathlib import Path
        >>> from torchrir import CmuArcticDataset
        >>> rng = random.Random(0)
        >>> root = Path("datasets/cmu_arctic")
        >>> signals, fs, info = load_dataset_sources(
        ...     dataset_factory=lambda spk: CmuArcticDataset(root, speaker=spk, download=True),
        ...     num_sources=2,
        ...     duration_s=10.0,
        ...     rng=rng,
        ... )
    """
    dataset0 = dataset_factory(None)
    speakers = choose_speakers(dataset0, num_sources, rng)
    signals: List[torch.Tensor] = []
    info: List[Tuple[str, List[str]]] = []
    fs: int | None = None
    target_samples: int | None = None

    for speaker in speakers:
        dataset = dataset_factory(speaker)
        sentences: Sequence[SentenceLike] = dataset.available_sentences()
        if not sentences:
            raise RuntimeError(f"no sentences found for speaker {speaker}")

        utterance_ids: List[str] = []
        segments: List[torch.Tensor] = []
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
            audio, sample_rate = dataset.load_wav(sentence.utterance_id)
            if fs is None:
                fs = sample_rate
                target_samples = int(duration_s * fs)
            elif sample_rate != fs:
                raise ValueError(
                    f"sample rate mismatch: expected {fs}, got {sample_rate} for {speaker}"
                )
            segments.append(audio)
            utterance_ids.append(sentence.utterance_id)
            total += audio.numel()

        signal = torch.cat(segments, dim=0)[:target_samples]
        signals.append(signal)
        info.append((speaker, utterance_ids))

    stacked = torch.stack(signals, dim=0)
    if fs is None:
        raise RuntimeError("no audio loaded from dataset sources")
    return stacked, int(fs), info
