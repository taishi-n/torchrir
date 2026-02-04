from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import torch

from torchrir import CmuArcticDataset, list_cmu_arctic_speakers


def choose_speakers(num_sources: int, rng: random.Random) -> List[str]:
    speakers = list_cmu_arctic_speakers()
    if not speakers:
        raise RuntimeError("no CMU ARCTIC speakers available")
    if num_sources > len(speakers):
        raise ValueError(f"num_sources must be <= {len(speakers)} for unique speakers")
    return rng.sample(speakers, num_sources)


def load_cmu_arctic_sources(
    *,
    root: Path,
    num_sources: int,
    duration_s: float,
    rng: random.Random,
    download: bool = True,
) -> Tuple[torch.Tensor, int, List[Tuple[str, List[str]]]]:
    speakers = choose_speakers(num_sources, rng)
    signals: List[torch.Tensor] = []
    info: List[Tuple[str, List[str]]] = []
    fs: int | None = None
    target_samples: int | None = None

    for speaker in speakers:
        dataset = CmuArcticDataset(root, speaker=speaker, download=download)
        sentences = dataset.available_sentences()
        if not sentences:
            raise RuntimeError(f"no sentences found for speaker {speaker}")

        utterance_ids: List[str] = []
        segments: List[torch.Tensor] = []
        total = 0
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
    return stacked, int(fs), info


def sample_positions(
    *,
    num: int,
    room_size: torch.Tensor,
    rng: random.Random,
    margin: float = 0.5,
) -> torch.Tensor:
    dim = room_size.numel()
    low = [margin] * dim
    high = [float(room_size[i].item()) - margin for i in range(dim)]
    coords = []
    for _ in range(num):
        point = [rng.uniform(low[i], high[i]) for i in range(dim)]
        coords.append(point)
    return torch.tensor(coords, dtype=torch.float32)


def linear_trajectory(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    return torch.stack(
        [start + (end - start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    )


def binaural_mic_positions(center: torch.Tensor, offset: float = 0.08) -> torch.Tensor:
    dim = center.numel()
    offset_vec = torch.zeros((dim,), dtype=torch.float32)
    offset_vec[0] = offset
    left = center - offset_vec
    right = center + offset_vec
    return torch.stack([left, right], dim=0)


def clamp_positions(positions: torch.Tensor, room_size: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    min_v = torch.full_like(room_size, margin)
    max_v = room_size - margin
    return torch.max(torch.min(positions, max_v), min_v)
