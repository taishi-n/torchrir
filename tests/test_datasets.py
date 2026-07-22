"""Public dataset API tests using small, local fixtures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from torchrir.datasets import (
    BaseDataset,
    CmuArcticDataset,
    DatasetItem,
    LibriSpeechDataset,
    choose_speakers,
    collate_dataset_items,
    load_dataset_sources,
)
from torchrir.datasets.attribution import attribution_for


@dataclass(frozen=True)
class _Sentence:
    utterance_id: str
    text: str


class _MemoryDataset(BaseDataset):
    def __init__(self, speaker: str | None = None, sample_rate: int = 10) -> None:
        self.speaker = speaker
        self.sample_rate = sample_rate

    def list_speakers(self) -> list[str]:
        return ["alice", "bob", "charlie"]

    def available_sentences(self) -> list[_Sentence]:
        return [_Sentence("utt-1", "one"), _Sentence("utt-2", "two")]

    def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
        value = 1.0 if utterance_id == "utt-1" else 2.0
        return torch.full((3,), value), self.sample_rate

    def attribution_info(self):
        return attribution_for("cmu_arctic")


def test_base_dataset_caches_sentences_and_builds_items() -> None:
    dataset = _MemoryDataset("alice")
    assert len(dataset) == 2
    item = dataset[1]
    assert item.utterance_id == "utt-2"
    assert item.text == "two"
    assert item.speaker == "alice"
    assert item.sample_rate == 10
    torch.testing.assert_close(item.audio, torch.full((3,), 2.0))
    with pytest.raises(TypeError, match="Index must be int"):
        dataset["1"]


def test_collate_dataset_items_pads_and_preserves_metadata() -> None:
    items = [
        DatasetItem(torch.tensor([1.0, 2.0]), 8000, "a", "text-a", "alice"),
        DatasetItem(torch.tensor([3.0]), 8000, "b", None, "bob"),
    ]
    batch = collate_dataset_items(items, pad_value=-1.0, keep_metadata=True)
    torch.testing.assert_close(batch.audio, torch.tensor([[1.0, 2.0], [3.0, -1.0]]))
    assert batch.lengths.tolist() == [2, 1]
    assert batch.sample_rate == 8000
    assert batch.utterance_ids == ["a", "b"]
    assert batch.texts == ["text-a", None]
    assert batch.speakers == ["alice", "bob"]
    assert batch.metadata == [None, None]


def test_collate_dataset_items_rejects_invalid_batches() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_dataset_items([])
    with pytest.raises(ValueError, match="sample_rate"):
        collate_dataset_items(
            [
                DatasetItem(torch.ones(2), 8000, "a"),
                DatasetItem(torch.ones(2), 16000, "b"),
            ]
        )


def test_choose_speakers_is_unique_and_seeded() -> None:
    dataset = _MemoryDataset()
    first = choose_speakers(dataset, 2, random.Random(12))
    repeated = choose_speakers(dataset, 2, random.Random(12))
    assert first == repeated
    assert len(set(first)) == 2
    with pytest.raises(ValueError, match="num_sources"):
        choose_speakers(dataset, 4, random.Random(0))


def test_load_dataset_sources_concatenates_and_trims() -> None:
    signals, sample_rate, info = load_dataset_sources(
        dataset_factory=lambda speaker: _MemoryDataset(speaker),
        num_sources=2,
        duration_s=0.5,
        rng=random.Random(4),
    )
    assert signals.shape == (2, 5)
    assert sample_rate == 10
    assert len(info) == 2
    assert len({speaker for speaker, _ in info}) == 2
    assert all(len(utterance_ids) == 2 for _, utterance_ids in info)


def _write_cmu_fixture(root: Path) -> None:
    dataset_root = root / "ARCTIC" / "cmu_us_bdl_arctic"
    wav_root = dataset_root / "wav"
    etc_root = dataset_root / "etc"
    wav_root.mkdir(parents=True)
    etc_root.mkdir()
    (etc_root / "txt.done.data").write_text(
        '( arctic_a0001 "available" )\n( arctic_a0002 "missing" )\n',
        encoding="utf-8",
    )
    sf.write(wav_root / "arctic_a0001.wav", np.linspace(-0.1, 0.1, 16), 8000)


def test_cmu_arctic_discovers_only_available_audio(tmp_path: Path) -> None:
    _write_cmu_fixture(tmp_path)
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    sentences = dataset.available_sentences()
    assert [(item.utterance_id, item.text) for item in sentences] == [
        ("arctic_a0001", "available")
    ]
    audio, sample_rate = dataset.load_audio("arctic_a0001")
    assert audio.shape == (16,)
    assert sample_rate == 8000
    assert dataset.attribution_info().dataset_key == "cmu_arctic"


def test_cmu_arctic_rejects_unknown_or_missing_speaker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported speaker"):
        CmuArcticDataset(tmp_path, speaker="unknown")
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        CmuArcticDataset(tmp_path, speaker="bdl")


def _write_librispeech_fixture(root: Path) -> None:
    chapter = root / "LibriSpeech" / "dev-clean" / "103" / "1240"
    chapter.mkdir(parents=True)
    (chapter / "103-1240.trans.txt").write_text(
        "103-1240-0000 AVAILABLE TEXT\n103-1240-0001 MISSING AUDIO\n",
        encoding="utf-8",
    )
    sf.write(chapter / "103-1240-0000.flac", np.linspace(-0.1, 0.1, 16), 16000)


def test_librispeech_discovers_transcripts_and_audio(tmp_path: Path) -> None:
    _write_librispeech_fixture(tmp_path)
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean")
    assert dataset.list_speakers() == ["103"]
    sentences = dataset.available_sentences()
    assert len(sentences) == 1
    assert sentences[0].utterance_id == "103-1240-0000"
    assert sentences[0].text == "AVAILABLE TEXT"
    audio, sample_rate = dataset.load_audio("103-1240-0000")
    assert audio.shape == (16,)
    assert sample_rate == 16000
    assert dataset.attribution_info().subset == "dev-clean"


def test_librispeech_validates_subset_and_speaker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported subset"):
        LibriSpeechDataset(tmp_path, subset="not-a-subset")
    _write_librispeech_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="speaker directory"):
        LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="999")
