"""Public dataset API tests using small, local fixtures."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass
import os
from pathlib import Path
import random
from typing import Any, cast

import numpy as np
import pytest
import soundfile as sf
import torch

import torchrir.datasets.cmu_arctic as cmu_arctic_module
import torchrir.datasets.librispeech as librispeech_module
from torchrir.datasets import (
    BaseDataset,
    CmuArcticDataset,
    CmuArcticSentence,
    CollateBatch,
    DatasetItem,
    LibriSpeechDataset,
    LibriSpeechSentence,
    choose_speakers,
    cmu_arctic_speakers,
    collate_dataset_items,
    load_dataset_sources,
)
from torchrir.datasets.attribution import (
    DatasetAttribution,
    attribution_for,
    default_modification_notes,
)


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
    with pytest.raises(TypeError, match="index must be an integer"):
        dataset[cast(Any, "1")]
    with pytest.raises(TypeError, match="index must be an integer"):
        dataset[True]


def test_base_dataset_is_abstract() -> None:
    with pytest.raises(TypeError, match="abstract"):
        BaseDataset()


def test_dataset_attribution_contracts_are_explicit() -> None:
    attribution = attribution_for(" LibriSpeech ", subset="dev-clean")
    assert attribution.dataset_key == "librispeech"
    assert attribution.subset == "dev-clean"
    with pytest.raises(TypeError, match="dataset must be a string"):
        attribution_for(cast(Any, 1))
    with pytest.raises(ValueError, match="only valid for LibriSpeech"):
        attribution_for("cmu_arctic", subset="unused")
    with pytest.raises(ValueError, match="dataset_key must be non-empty"):
        DatasetAttribution(
            dataset_key="",
            dataset="dataset",
            source="https://example.com",
            license_name="license",
            license_url="https://example.com/license",
            required_attribution="credit",
        )
    with pytest.raises(TypeError, match="attribution_required"):
        DatasetAttribution(
            dataset_key="dataset",
            dataset="dataset",
            source="https://example.com",
            license_name="license",
            license_url="https://example.com/license",
            required_attribution="credit",
            attribution_required=cast(Any, 1),
        )
    with pytest.raises(TypeError, match="dynamic"):
        default_modification_notes(dynamic=cast(Any, 1))


def test_sentence_records_are_validated_keyword_only_values() -> None:
    cmu = CmuArcticSentence(utterance_id="arctic_a0001", text="text")
    libri = LibriSpeechSentence(
        utterance_id="103-1240-0000",
        text="text",
        speaker_id="103",
        chapter_id="1240",
    )
    assert cmu.utterance_id == "arctic_a0001"
    assert libri.speaker_id == "103"
    with pytest.raises(TypeError):
        CmuArcticSentence("arctic_a0001", "text")  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonical"):
        CmuArcticSentence(utterance_id="invalid", text="text")
    with pytest.raises(ValueError, match="agree"):
        LibriSpeechSentence(
            utterance_id="103-1240-0000",
            text="text",
            speaker_id="104",
            chapter_id="1240",
        )


def test_collate_dataset_items_pads_and_preserves_metadata() -> None:
    items = [
        DatasetItem(
            audio=torch.tensor([1.0, 2.0]),
            sample_rate=8000,
            utterance_id="a",
            text="text-a",
            speaker="alice",
            metadata={"index": 0},
        ),
        DatasetItem(
            audio=torch.tensor([3.0]),
            sample_rate=8000,
            utterance_id="b",
            speaker="bob",
            metadata={"index": 1},
        ),
    ]
    batch = collate_dataset_items(items, pad_value=-1.0, keep_metadata=True)
    torch.testing.assert_close(batch.audio, torch.tensor([[1.0, 2.0], [3.0, -1.0]]))
    assert batch.lengths.tolist() == [2, 1]
    assert batch.sample_rate == 8000
    assert batch.utterance_ids == ("a", "b")
    assert batch.texts == ("text-a", None)
    assert batch.speakers == ("alice", "bob")
    assert batch.metadata == ({"index": 0}, {"index": 1})

    without_metadata = collate_dataset_items(items)
    assert without_metadata.metadata is None


def test_collate_dataset_items_rejects_invalid_batches() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        collate_dataset_items([])
    with pytest.raises(ValueError, match="sample_rate"):
        collate_dataset_items(
            [
                DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="a"),
                DatasetItem(audio=torch.ones(2), sample_rate=16000, utterance_id="b"),
            ]
        )
    with pytest.raises(TypeError, match="keep_metadata must be a bool"):
        collate_dataset_items(
            [DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="a")],
            keep_metadata=1,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("audio", "error", "message"),
    [
        (torch.empty(0), ValueError, "at least one sample"),
        (torch.ones(2, 3), ValueError, "mono"),
        (torch.ones(2, dtype=torch.int64), TypeError, "floating-point"),
        (torch.ones(2, dtype=torch.float8_e4m3fn), TypeError, "supported"),
        (torch.tensor([float("nan")]), ValueError, "finite"),
    ],
)
def test_dataset_item_rejects_invalid_audio(
    audio: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        DatasetItem(audio=audio, sample_rate=8000, utterance_id="a")


@pytest.mark.parametrize(
    ("updates", "error", "message"),
    [
        ({"sample_rate": True}, TypeError, "integer"),
        ({"sample_rate": 0}, ValueError, "positive"),
        ({"sample_rate": 2**31}, ValueError, "at most"),
        ({"utterance_id": ""}, ValueError, "non-empty"),
        ({"utterance_id": 1}, TypeError, "must be a string"),
        ({"text": 1}, TypeError, "string or None"),
        ({"speaker": ""}, ValueError, "non-empty"),
        ({"speaker": 1}, TypeError, "string or None"),
    ],
)
def test_dataset_item_rejects_invalid_metadata_fields(
    updates: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "audio": torch.ones(2),
        "sample_rate": 8000,
        "utterance_id": "a",
    }
    values.update(updates)
    with pytest.raises(error, match=message):
        DatasetItem(**values)  # type: ignore[arg-type]


def test_dataset_records_are_keyword_only_frozen_and_identity_comparing() -> None:
    with pytest.raises(TypeError):
        DatasetItem(torch.ones(2), 8000, "a")  # type: ignore[misc]

    first = DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="a")
    second = DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="a")
    assert first == first
    assert first != second
    with pytest.raises(FrozenInstanceError):
        first.sample_rate = 16000  # type: ignore[misc]


def test_collate_dataset_items_revalidates_and_rejects_mixed_layout() -> None:
    first = DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="a")
    second = DatasetItem(
        audio=torch.ones(2, dtype=torch.float64),
        sample_rate=8000,
        utterance_id="b",
    )
    with pytest.raises(ValueError, match="dtype"):
        collate_dataset_items([first, second])

    first.audio[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        collate_dataset_items([first])

    with pytest.raises(ValueError, match="pad_value"):
        collate_dataset_items([second], pad_value=float("nan"))
    with pytest.raises(ValueError, match="pad_value"):
        collate_dataset_items([second], pad_value=10**400)
    half = DatasetItem(
        audio=torch.ones(2, dtype=torch.float16),
        sample_rate=8000,
        utterance_id="half",
    )
    with pytest.raises(ValueError, match="representable"):
        collate_dataset_items([half], pad_value=65_505.0)
    single = DatasetItem(
        audio=torch.ones(2, dtype=torch.float32),
        sample_rate=8000,
        utterance_id="single",
    )
    with pytest.raises(ValueError, match="representable"):
        collate_dataset_items([single], pad_value=1.0e40)


def test_collate_dataset_items_rejects_mixed_devices_when_available() -> None:
    accelerator: str | None = None
    if torch.cuda.is_available():
        accelerator = "cuda"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
    if accelerator is None:
        pytest.skip("no accelerator available for a mixed-device batch")

    cpu = DatasetItem(audio=torch.ones(2), sample_rate=8000, utterance_id="cpu")
    other = DatasetItem(
        audio=torch.ones(2, device=accelerator),
        sample_rate=8000,
        utterance_id="accelerator",
    )
    with pytest.raises(ValueError, match="device"):
        collate_dataset_items([cpu, other])


def test_collate_batch_normalizes_sequences_and_validates_lengths() -> None:
    batch = CollateBatch(
        audio=torch.zeros(2, 3),
        lengths=torch.tensor([3, 2]),
        sample_rate=8000,
        utterance_ids=["a", "b"],
        texts=[None, "text"],
        speakers=["alice", None],
        metadata=[{"index": 0}, {"index": 1}],
    )
    assert batch.utterance_ids == ("a", "b")
    assert batch.texts == (None, "text")
    assert batch.speakers == ("alice", None)
    assert batch.metadata == ({"index": 0}, {"index": 1})
    assert batch == batch
    other = CollateBatch(
        audio=torch.zeros(2, 3),
        lengths=torch.tensor([3, 2]),
        sample_rate=8000,
        utterance_ids=["a", "b"],
        texts=[None, "text"],
        speakers=["alice", None],
    )
    assert batch != other
    with pytest.raises(FrozenInstanceError):
        batch.sample_rate = 16000  # type: ignore[misc]
    with pytest.raises(TypeError):
        cast(Any, CollateBatch)(
            torch.zeros(1, 2),
            torch.tensor([2]),
            8000,
            ["a"],
            [None],
            [None],
        )

    with pytest.raises(ValueError, match="one value per batch item"):
        CollateBatch(
            audio=torch.zeros(2, 3),
            lengths=torch.tensor([3]),
            sample_rate=8000,
            utterance_ids=["a", "b"],
            texts=[None, None],
            speakers=[None, None],
        )
    with pytest.raises(ValueError, match="metadata"):
        CollateBatch(
            audio=torch.zeros(2, 3),
            lengths=torch.tensor([3, 2]),
            sample_rate=8000,
            utterance_ids=["a", "b"],
            texts=[None, None],
            speakers=[None, None],
            metadata=[None],
        )
    with pytest.raises(TypeError, match="floating-point"):
        CollateBatch(
            audio=torch.zeros(1, 2, dtype=torch.int64),
            lengths=torch.tensor([2]),
            sample_rate=8000,
            utterance_ids=["a"],
            texts=[None],
            speakers=[None],
        )
    with pytest.raises(TypeError, match="utterance_ids must be a sequence"):
        CollateBatch(
            audio=torch.zeros(1, 2),
            lengths=torch.tensor([2]),
            sample_rate=8000,
            utterance_ids=cast(Any, None),
            texts=[None],
            speakers=[None],
        )
    with pytest.raises(ValueError, match="CPU int64"):
        CollateBatch(
            audio=torch.zeros(1, 2),
            lengths=torch.tensor([2], dtype=torch.int32),
            sample_rate=8000,
            utterance_ids=["a"],
            texts=[None],
            speakers=[None],
        )
    sparse_lengths = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([2]),
        size=(1,),
    )
    with pytest.raises(TypeError, match="strided"):
        CollateBatch(
            audio=torch.zeros(1, 2),
            lengths=sparse_lengths,
            sample_rate=8000,
            utterance_ids=["a"],
            texts=[None],
            speakers=[None],
        )


def test_choose_speakers_is_unique_and_seeded() -> None:
    speakers = ["alice", "bob", "charlie"]
    first = choose_speakers(speakers, 2, random.Random(12))
    repeated = choose_speakers(speakers, 2, random.Random(12))
    assert first == repeated
    assert len(set(first)) == 2
    with pytest.raises(ValueError, match="num_sources"):
        choose_speakers(speakers, 4, random.Random(0))
    with pytest.raises(ValueError, match="positive"):
        choose_speakers(speakers, 0, random.Random(0))
    with pytest.raises(TypeError, match="integer"):
        choose_speakers(speakers, True, random.Random(0))


def test_load_dataset_sources_concatenates_and_trims() -> None:
    signals, sample_rate, info = load_dataset_sources(
        dataset_factory=lambda speaker: _MemoryDataset(speaker),
        speakers=["alice", "bob", "charlie"],
        num_sources=2,
        duration_s=0.5,
        rng=random.Random(4),
    )
    assert signals.shape == (2, 5)
    assert sample_rate == 10
    assert len(info) == 2
    assert len({speaker for speaker, _ in info}) == 2
    assert all(len(utterance_ids) == 2 for _, utterance_ids in info)


def test_load_dataset_sources_uses_ceil_for_requested_duration() -> None:
    signals, sample_rate, _ = load_dataset_sources(
        dataset_factory=lambda speaker: _MemoryDataset(speaker),
        speakers=["alice", "bob", "charlie"],
        num_sources=1,
        duration_s=0.51,
        rng=random.Random(4),
    )
    assert sample_rate == 10
    assert signals.shape == (1, 6)


@pytest.mark.parametrize(
    "duration",
    [0.0, -1.0, float("nan"), float("inf"), 10**400],
)
def test_load_dataset_sources_rejects_invalid_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="duration_s"):
        load_dataset_sources(
            dataset_factory=lambda speaker: _MemoryDataset(speaker),
            speakers=["alice", "bob", "charlie"],
            num_sources=1,
            duration_s=duration,
            rng=random.Random(4),
        )


@pytest.mark.parametrize(
    ("duration_s", "sample_rate"),
    [
        (1.0e308, 10),
        (1.0e10, 1_000_000_000),
    ],
)
def test_load_dataset_sources_rejects_sample_count_overflow(
    duration_s: float,
    sample_rate: int,
) -> None:
    with pytest.raises(ValueError, match="duration_s.*sample_rate.*int64"):
        load_dataset_sources(
            dataset_factory=lambda speaker: _MemoryDataset(speaker, sample_rate),
            speakers=["alice"],
            num_sources=1,
            duration_s=duration_s,
            rng=random.Random(4),
        )


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (torch.empty(0), "at least one sample"),
        (torch.ones(2, 3), "mono"),
        (torch.ones(2, dtype=torch.int64), "floating-point"),
        (torch.ones(2, dtype=torch.float8_e4m3fn), "supported"),
        (torch.tensor([float("nan")]), "finite"),
    ],
)
def test_load_dataset_sources_rejects_invalid_loaded_audio(
    audio: torch.Tensor,
    message: str,
) -> None:
    class _InvalidAudioDataset(_MemoryDataset):
        def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
            del utterance_id
            return audio, 10

    with pytest.raises((TypeError, ValueError), match=message):
        load_dataset_sources(
            dataset_factory=lambda speaker: _InvalidAudioDataset(speaker),
            speakers=["alice", "bob", "charlie"],
            num_sources=1,
            duration_s=0.5,
            rng=random.Random(4),
        )


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
    (tmp_path / "ARCTIC" / "cmu_us_bdl_arctic" / "wav" / "arctic_a0002.wav").mkdir()
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    sentences = dataset.available_sentences()
    assert [(item.utterance_id, item.text) for item in sentences] == [
        ("arctic_a0001", "available")
    ]
    audio, sample_rate = dataset.load_audio("arctic_a0001")
    assert audio.shape == (16,)
    assert sample_rate == 8000
    assert dataset.list_speakers() == ["bdl"]
    assert cmu_arctic_speakers(tmp_path) == ["bdl"]
    assert dataset.attribution_info().dataset_key == "cmu_arctic"


@pytest.mark.skipif(os.name == "nt", reason="requires replacing an open file path")
def test_cmu_transcript_and_audio_are_read_from_verified_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cmu_fixture(tmp_path)
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    transcript = dataset.text_path
    replacement_transcript = transcript.parent / "replacement.txt"
    replacement_transcript.write_text(
        '( arctic_a0001 "MALICIOUS" )\n',
        encoding="utf-8",
    )
    audio = dataset.audio_path("arctic_a0001")
    replacement_audio = audio.parent / "replacement.wav"
    sf.write(replacement_audio, np.ones(16, dtype=np.float32), 8000)
    original_open = cmu_arctic_module.open_regular_file_below
    swapped: set[Path] = set()

    @contextmanager
    def swap_after_open(path: Path, *, root: Path):
        with original_open(path, root=root) as opened:
            if path == transcript and path not in swapped:
                path.replace(transcript.parent / "original.txt")
                replacement_transcript.replace(path)
                swapped.add(path)
            elif path == audio and path not in swapped:
                path.replace(audio.parent / "original.wav")
                replacement_audio.replace(path)
                swapped.add(path)
            yield opened

    monkeypatch.setattr(cmu_arctic_module, "open_regular_file_below", swap_after_open)

    assert [sentence.text for sentence in dataset.sentences()] == [
        "available",
        "missing",
    ]
    loaded, sample_rate = dataset.load_audio("arctic_a0001")
    assert sample_rate == 8000
    assert not torch.allclose(loaded, torch.ones_like(loaded))


def test_dataset_identifiers_reject_non_string_values_before_membership(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError, match="speaker must be a string"):
        CmuArcticDataset(tmp_path, speaker=cast(Any, []))
    with pytest.raises(TypeError, match="subset must be a string"):
        LibriSpeechDataset(tmp_path, subset=cast(Any, []))
    with pytest.raises(TypeError, match="speaker must be a string"):
        LibriSpeechDataset(tmp_path, speaker=cast(Any, []))
    with pytest.raises(TypeError, match="utterance_id must be a string"):
        CmuArcticSentence(utterance_id=cast(Any, 1), text="text")

    _write_cmu_fixture(tmp_path)
    cmu = CmuArcticDataset(tmp_path, speaker="bdl")
    with pytest.raises(TypeError, match="utterance_id must be a string"):
        cmu.audio_path(cast(Any, 1))


def test_cmu_arctic_rejects_invalid_transcript_id(tmp_path: Path) -> None:
    _write_cmu_fixture(tmp_path)
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    dataset.text_path.write_text('( invalid_id "bad" )\n', encoding="utf-8")

    with pytest.raises(ValueError, match="transcript ID"):
        dataset.available_sentences()


@pytest.mark.parametrize(
    "malformed_line",
    [")", '( arctic_a0001 "text" garbage )'],
)
def test_cmu_arctic_rejects_malformed_transcript_lines_cleanly(
    tmp_path: Path,
    malformed_line: str,
) -> None:
    _write_cmu_fixture(tmp_path)
    transcript = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic" / "etc" / "txt.done.data"
    transcript.write_text(
        transcript.read_text(encoding="utf-8") + malformed_line + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="incomplete"):
        CmuArcticDataset(tmp_path, speaker="bdl")

    transcript.write_text(
        '( arctic_a0001 "available" )\n( arctic_a0002 "missing" )\n',
        encoding="utf-8",
    )
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    transcript.write_text(malformed_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="transcript line"):
        dataset.available_sentences()


def test_cmu_arctic_rejects_unknown_or_missing_speaker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported speaker"):
        CmuArcticDataset(tmp_path, speaker="unknown")
    with pytest.raises(FileNotFoundError, match="missing or incomplete"):
        CmuArcticDataset(tmp_path, speaker="bdl")


def test_cmu_arctic_ready_tree_skips_archive_and_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_cmu_fixture(tmp_path)
    monkeypatch.setattr(
        cmu_arctic_module,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready dataset must not download"),
    )

    dataset = CmuArcticDataset(tmp_path, speaker="bdl", download=True)

    assert dataset.list_speakers() == ["bdl"]


def test_cmu_arctic_rejects_incomplete_tree_and_non_boolean_download(
    tmp_path: Path,
) -> None:
    (tmp_path / "ARCTIC" / "cmu_us_bdl_arctic").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        CmuArcticDataset(tmp_path, speaker="bdl")
    with pytest.raises(TypeError, match="download must be a bool"):
        CmuArcticDataset(tmp_path, speaker="bdl", download=1)  # type: ignore[arg-type]


def test_cmu_arctic_rejects_transcript_without_matching_canonical_audio(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic"
    (dataset_root / "etc").mkdir(parents=True)
    (dataset_root / "wav").mkdir()
    (dataset_root / "etc" / "txt.done.data").write_text(
        '( arctic_a0001 "missing audio" )\n',
        encoding="utf-8",
    )
    (dataset_root / "wav" / "garbage.wav").write_bytes(b"not audio")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        CmuArcticDataset(tmp_path, speaker="bdl")
    assert cmu_arctic_speakers(tmp_path) == []


def test_cmu_arctic_rejects_external_audio_symlink(tmp_path: Path) -> None:
    _write_cmu_fixture(tmp_path)
    audio_path = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic" / "wav" / "arctic_a0001.wav"
    outside = tmp_path / "outside.wav"
    audio_path.replace(outside)
    audio_path.symlink_to(outside)

    with pytest.raises(FileNotFoundError, match="incomplete"):
        CmuArcticDataset(tmp_path, speaker="bdl")


def test_cmu_arctic_rejects_utterance_path_traversal(tmp_path: Path) -> None:
    _write_cmu_fixture(tmp_path)
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    with pytest.raises(ValueError, match="utterance_id"):
        dataset.load_audio("../../../outside")


def test_cmu_arctic_load_audio_requires_a_regular_file(tmp_path: Path) -> None:
    _write_cmu_fixture(tmp_path)
    dataset = CmuArcticDataset(tmp_path, speaker="bdl")
    (dataset.audio_dir / "arctic_a0002.wav").mkdir()

    with pytest.raises(FileNotFoundError, match="regular file"):
        dataset.load_audio("arctic_a0002")


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
    (
        tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240" / "103-1240-0001.flac"
    ).mkdir()
    (tmp_path / "LibriSpeech" / "dev-clean" / "metadata").mkdir()
    (tmp_path / "LibriSpeech" / "dev-clean" / "999").mkdir()
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean")
    assert dataset.list_speakers() == ["103"]
    sentences = dataset.available_sentences()
    assert len(sentences) == 1
    assert sentences[0].utterance_id == "103-1240-0000"
    assert sentences[0].text == "AVAILABLE TEXT"
    audio, sample_rate = dataset.load_audio("103-1240-0000")
    assert audio.shape == (16,)
    assert sample_rate == 16000
    assert dataset[0].speaker == "103"
    assert dataset.attribution_info().subset == "dev-clean"


@pytest.mark.skipif(os.name == "nt", reason="requires replacing an open file path")
def test_librispeech_transcript_and_audio_use_verified_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_librispeech_fixture(tmp_path)
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="103")
    chapter = dataset._subset_dir / "103" / "1240"
    transcript = chapter / "103-1240.trans.txt"
    replacement_transcript = chapter / "replacement.tmp"
    replacement_transcript.write_text(
        "103-1240-0000 MALICIOUS\n",
        encoding="utf-8",
    )
    audio = chapter / "103-1240-0000.flac"
    replacement_audio = chapter / "replacement.flac"
    sf.write(replacement_audio, np.ones(16, dtype=np.float32), 16000)
    original_open = librispeech_module.open_regular_file_below
    swapped: set[Path] = set()

    @contextmanager
    def swap_after_open(path: Path, *, root: Path):
        with original_open(path, root=root) as opened:
            if path == transcript and path not in swapped:
                path.replace(chapter / "original.trans.txt")
                replacement_transcript.replace(path)
                swapped.add(path)
            elif path == audio and path not in swapped:
                path.replace(chapter / "original.flac")
                replacement_audio.replace(path)
                swapped.add(path)
            yield opened

    monkeypatch.setattr(librispeech_module, "open_regular_file_below", swap_after_open)

    assert [sentence.text for sentence in dataset.available_sentences()] == [
        "AVAILABLE TEXT"
    ]
    loaded, sample_rate = dataset.load_audio("103-1240-0000")
    assert sample_rate == 16000
    assert not torch.allclose(loaded, torch.ones_like(loaded))


def test_librispeech_ignores_transcripts_outside_usable_speaker_trees(
    tmp_path: Path,
) -> None:
    _write_librispeech_fixture(tmp_path)
    subset = tmp_path / "LibriSpeech" / "dev-clean"
    metadata = subset / "metadata"
    metadata.mkdir()
    (metadata / "notes.trans.txt").write_text(
        "NOT-A-LIBRISPEECH-ID stray metadata\n",
        encoding="utf-8",
    )
    unusable_chapter = subset / "999" / "999"
    unusable_chapter.mkdir(parents=True)
    (unusable_chapter / "999-999.trans.txt").write_text(
        "ALSO-NOT-A-LIBRISPEECH-ID stray speaker\n",
        encoding="utf-8",
    )

    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean")

    assert dataset.list_speakers() == ["103"]
    assert [item.utterance_id for item in dataset.available_sentences()] == [
        "103-1240-0000"
    ]


def test_librispeech_validates_subset_and_speaker(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported subset"):
        LibriSpeechDataset(tmp_path, subset="not-a-subset")
    _write_librispeech_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="speaker directory"):
        LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="999")
    with pytest.raises(ValueError, match="numeric"):
        LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="../103")


def test_librispeech_ready_tree_skips_archive_and_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_librispeech_fixture(tmp_path)
    monkeypatch.setattr(
        librispeech_module,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready dataset must not download"),
    )

    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean", download=True)

    assert dataset.list_speakers() == ["103"]


def test_librispeech_rejects_incomplete_tree_and_non_boolean_download(
    tmp_path: Path,
) -> None:
    (tmp_path / "LibriSpeech" / "dev-clean").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="incomplete"):
        LibriSpeechDataset(tmp_path, subset="dev-clean")
    with pytest.raises(TypeError, match="download must be a bool"):
        LibriSpeechDataset(
            tmp_path,
            subset="dev-clean",
            download="false",  # type: ignore[arg-type]
        )


def test_librispeech_rejects_transcript_without_matching_canonical_audio(
    tmp_path: Path,
) -> None:
    chapter = tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240"
    chapter.mkdir(parents=True)
    (chapter / "103-1240.trans.txt").write_text(
        "103-1240-0000 MISSING AUDIO\n",
        encoding="utf-8",
    )
    (chapter / "garbage.flac").write_bytes(b"not audio")

    with pytest.raises(FileNotFoundError, match="incomplete"):
        LibriSpeechDataset(tmp_path, subset="dev-clean")


def test_librispeech_ready_check_rejects_trailing_invalid_transcript_id(
    tmp_path: Path,
) -> None:
    _write_librispeech_fixture(tmp_path)
    transcript = (
        tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240" / "103-1240.trans.txt"
    )
    transcript.write_text(
        transcript.read_text(encoding="utf-8") + "INVALID trailing line\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="incomplete"):
        LibriSpeechDataset(tmp_path, subset="dev-clean")


def test_librispeech_rejects_external_audio_symlink(tmp_path: Path) -> None:
    _write_librispeech_fixture(tmp_path)
    audio_path = (
        tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240" / "103-1240-0000.flac"
    )
    outside = tmp_path / "outside.flac"
    audio_path.replace(outside)
    audio_path.symlink_to(outside)

    with pytest.raises(FileNotFoundError, match="incomplete"):
        LibriSpeechDataset(tmp_path, subset="dev-clean")


def test_librispeech_rejects_utterance_path_traversal_and_other_speaker(
    tmp_path: Path,
) -> None:
    _write_librispeech_fixture(tmp_path)
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="103")
    with pytest.raises(ValueError, match="utterance_id"):
        dataset.load_audio("../../../outside")
    with pytest.raises(ValueError, match="configured speaker"):
        dataset.load_audio("104-1240-0000")


def test_librispeech_load_audio_requires_a_regular_file(tmp_path: Path) -> None:
    _write_librispeech_fixture(tmp_path)
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="103")
    chapter = tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240"
    (chapter / "103-1240-0001.flac").mkdir()

    with pytest.raises(FileNotFoundError, match="regular file"):
        dataset.load_audio("103-1240-0001")


@pytest.mark.parametrize("directory", ["speaker", "chapter"])
def test_librispeech_load_audio_rejects_directory_symlink_added_after_init(
    tmp_path: Path,
    directory: str,
) -> None:
    _write_librispeech_fixture(tmp_path)
    dataset = LibriSpeechDataset(tmp_path, subset="dev-clean", speaker="103")
    subset = tmp_path / "LibriSpeech" / "dev-clean"
    speaker = subset / "103"
    chapter = speaker / "1240"
    if directory == "speaker":
        replacement = subset / "speaker-storage"
        speaker.replace(replacement)
        speaker.symlink_to(replacement, target_is_directory=True)
    else:
        replacement = speaker / "chapter-storage"
        chapter.replace(replacement)
        chapter.symlink_to(replacement, target_is_directory=True)

    with pytest.raises(ValueError, match="directories must not be symlinks"):
        dataset.load_audio("103-1240-0000")
