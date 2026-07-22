"""LibriSpeech dataset helpers."""

from __future__ import annotations

from contextlib import ExitStack
from io import TextIOWrapper
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .attribution import DatasetAttribution, attribution_for
from ._archive import (
    cleanup_stale_extraction_directories,
    extract_or_download_archive,
    extraction_temporary_prefix,
    is_regular_file_below,
    managed_temporary_directory,
    open_regular_file_below,
    open_verified_tar,
    publish_staged_directory,
    recover_staged_publication,
    safe_extractall,
)
from .base import BaseDataset
from ._download import (
    DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    cleanup_stale_download_directories,
    file_matches_digest,
    retry_download,
    stream_verified_download,
)
from ._locking import dataset_write_lock
from ..io.audio import load_audio

BASE_URL = "https://www.openslr.org/resources/12"
VALID_SUBSETS = {
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
}
ARCHIVE_MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}
_SPEAKER_ID = re.compile(r"\d+\Z")
_UTTERANCE_ID = re.compile(r"(\d+)-(\d+)-(\d+)\Z")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class LibriSpeechSentence:
    """Sentence metadata from LibriSpeech."""

    utterance_id: str
    text: str
    speaker_id: str
    chapter_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if not isinstance(self.speaker_id, str):
            raise TypeError("speaker_id must be a string")
        if not isinstance(self.chapter_id, str):
            raise TypeError("chapter_id must be a string")
        utterance_speaker, utterance_chapter, _ = _parse_utterance_id(self.utterance_id)
        if (
            _SPEAKER_ID.fullmatch(self.speaker_id) is None
            or _SPEAKER_ID.fullmatch(self.chapter_id) is None
            or utterance_speaker != self.speaker_id
            or utterance_chapter != self.chapter_id
        ):
            raise ValueError(
                "utterance_id must agree with numeric speaker_id and chapter_id"
            )


class LibriSpeechDataset(BaseDataset):
    """LibriSpeech dataset loader.

    Examples:
        ```python
        dataset = LibriSpeechDataset(Path("datasets/librispeech"), subset="train-clean-100", download=True)
        audio, fs = dataset.load_audio("103-1240-0000")
        ```
    """

    def __init__(
        self,
        root: Path,
        subset: str = "train-clean-100",
        speaker: str | None = None,
        download: bool = False,
    ) -> None:
        """Initialize a LibriSpeech dataset handle.

        Args:
            root: Root directory where the dataset is stored.
            subset: LibriSpeech subset name (e.g., "train-clean-100").
            speaker: Optional speaker ID directory name (e.g., "103").
                If provided, restrict loading to that speaker.
            download: Download and extract if missing.
        """
        if not isinstance(download, bool):
            raise TypeError("download must be a bool")
        if not isinstance(subset, str):
            raise TypeError("subset must be a string")
        if subset not in VALID_SUBSETS:
            raise ValueError(f"unsupported subset: {subset}")
        if speaker is not None and not isinstance(speaker, str):
            raise TypeError("speaker must be a string or None")
        if isinstance(speaker, str) and _SPEAKER_ID.fullmatch(speaker) is None:
            raise ValueError("speaker must be a numeric LibriSpeech speaker ID")
        self.root = Path(root)
        self.subset = subset
        self.speaker = speaker
        self._archive_name = f"{subset}.tar.gz"
        self._base_dir = self.root / "LibriSpeech"
        self._subset_dir = self._base_dir / subset
        self._speaker_dir = self._subset_dir / speaker if speaker else None
        recover_staged_publication(self._subset_dir)

        if download:
            self._download_and_extract()

        if not _librispeech_subset_ready(self._subset_dir):
            raise FileNotFoundError(
                "dataset is missing or incomplete; run with download=True or place "
                f"a complete archive under {self.root}"
            )
        if self._speaker_dir is not None and not _librispeech_speaker_ready(
            self._speaker_dir
        ):
            raise FileNotFoundError(
                f"speaker directory is missing or incomplete: {self._speaker_dir}"
            )

    def list_speakers(self) -> list[str]:
        """Return available speaker IDs."""
        if self.speaker is not None:
            return [self.speaker]
        if not self._subset_dir.exists():
            return []
        return sorted(
            p.name
            for p in self._subset_dir.iterdir()
            if _SPEAKER_ID.fullmatch(p.name) is not None
            and _librispeech_speaker_ready(p)
        )

    def available_sentences(self) -> list[LibriSpeechSentence]:
        """Return sentences that have a corresponding audio file."""
        sentences: list[LibriSpeechSentence] = []
        search_root = (
            self._speaker_dir if self._speaker_dir is not None else self._subset_dir
        )
        if search_root.is_symlink():
            raise FileNotFoundError("LibriSpeech search root must not be a symlink")
        speaker_dirs = [
            self._subset_dir / speaker_id for speaker_id in self.list_speakers()
        ]
        for speaker_dir in speaker_dirs:
            for trans_path in _canonical_transcript_paths(speaker_dir):
                chapter_dir = trans_path.parent
                speaker_id = speaker_dir.name
                chapter_id = chapter_dir.name
                with open_regular_file_below(
                    trans_path,
                    root=speaker_dir,
                ) as (binary, _):
                    with TextIOWrapper(binary, encoding="utf-8") as transcript:
                        for line in transcript:
                            line = line.strip()
                            if not line:
                                continue
                            utt_id, text = _parse_text_line(line)
                            utt_speaker, utt_chapter, _ = _parse_utterance_id(utt_id)
                            if utt_speaker != speaker_id or utt_chapter != chapter_id:
                                raise ValueError(
                                    "transcript utterance ID conflicts with its "
                                    "directory"
                                )
                            wav_path = chapter_dir / f"{utt_id}.flac"
                            if is_regular_file_below(wav_path, speaker_dir):
                                sentences.append(
                                    LibriSpeechSentence(
                                        utterance_id=utt_id,
                                        text=text,
                                        speaker_id=speaker_id,
                                        chapter_id=chapter_id,
                                    )
                                )
        return sentences

    def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
        """Load mono audio for the given utterance ID."""
        speaker_id, chapter_id, _ = _parse_utterance_id(utterance_id)
        if self.speaker is not None and speaker_id != self.speaker:
            raise ValueError("utterance_id does not belong to the configured speaker")
        if self._subset_dir.is_symlink():
            raise ValueError("LibriSpeech subset directory must not be a symlink")
        root = self._subset_dir.resolve()
        speaker_dir = root / speaker_id
        chapter_dir = speaker_dir / chapter_id
        if speaker_dir.is_symlink() or chapter_dir.is_symlink():
            raise ValueError(
                "LibriSpeech speaker and chapter directories must not be symlinks"
            )
        candidate = chapter_dir / f"{utterance_id}.flac"
        if candidate.is_symlink():
            raise ValueError("LibriSpeech audio file must not be a symlink")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ValueError("utterance path escapes the LibriSpeech subset directory")
        stack = ExitStack()
        try:
            binary, _ = stack.enter_context(open_regular_file_below(path, root=root))
        except ValueError as exc:
            stack.close()
            raise FileNotFoundError(
                "LibriSpeech audio must be a regular file below the subset tree"
            ) from exc
        with stack:
            return load_audio(binary)

    def _download_and_extract(self) -> None:
        """Download and extract the subset archive if needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / f".{self.subset}.torchrir.lock"
        with dataset_write_lock(lock_path):
            archive_path = self.root / self._archive_name
            cleanup_stale_download_directories(archive_path)
            cleanup_stale_extraction_directories(
                parent=self.root,
                target=self._subset_dir,
            )
            if self._requested_tree_ready():
                return
            self._download_and_extract_locked()

    def _requested_tree_ready(self) -> bool:
        return (
            _librispeech_speaker_ready(self._speaker_dir)
            if self._speaker_dir is not None
            else _librispeech_subset_ready(self._subset_dir)
        )

    def _download_and_extract_locked(self) -> None:
        """Download and publish while holding this subset's writer lock."""

        archive_path = self.root / self._archive_name
        url = f"{BASE_URL}/{self._archive_name}"
        checksum = ARCHIVE_MD5[self.subset]

        def download_archive() -> None:
            logger.info("Downloading %s", url)
            _download(url, archive_path, md5=checksum)

        def extract_archive() -> None:
            logger.info("Extracting %s", archive_path)
            _extract_librispeech_archive_staged(
                archive_path,
                root=self.root,
                subset_dir=self._subset_dir,
                md5=checksum,
            )

        extract_or_download_archive(
            archive_path,
            root=self.root,
            download=download_archive,
            extract=extract_archive,
            operation_logger=logger,
        )

    def attribution_info(self) -> DatasetAttribution:
        """Return attribution and license information for LibriSpeech."""
        return attribution_for("librispeech", subset=self.subset)


def _download(url: str, dest: Path, *, md5: str, retries: int = 1) -> None:
    """Download an archive, retrying only transport/integrity failures."""

    retry_download(
        lambda: _stream_download(url, dest, md5=md5),
        retries=retries,
        logger=logger,
    )


def _stream_download(
    url: str,
    dest: Path,
    *,
    md5: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
) -> None:
    """Stream and verify one archive through a unique sibling file."""

    stream_verified_download(
        url,
        dest,
        algorithm="md5",
        expected_digest=md5,
        timeout=timeout,
        total_timeout=total_timeout,
    )


def _has_md5(path: Path, expected: str) -> bool:
    return file_matches_digest(path, algorithm="md5", expected_digest=expected)


def _librispeech_subset_ready(subset_dir: Path) -> bool:
    if subset_dir.is_symlink() or not subset_dir.is_dir():
        return False
    for speaker_dir in subset_dir.iterdir():
        if not speaker_dir.is_dir() or _SPEAKER_ID.fullmatch(speaker_dir.name) is None:
            continue
        if _librispeech_speaker_ready(speaker_dir):
            return True
    return False


def _librispeech_speaker_ready(speaker_dir: Path) -> bool:
    has_audio = False
    for transcript in _canonical_transcript_paths(speaker_dir):
        chapter_dir = transcript.parent
        try:
            with open_regular_file_below(transcript, root=speaker_dir) as (binary, _):
                with TextIOWrapper(binary, encoding="utf-8") as transcript_file:
                    for raw_line in transcript_file:
                        line = raw_line.strip()
                        if not line:
                            continue
                        utterance_id, _ = _parse_text_line(line)
                        try:
                            utterance_speaker, utterance_chapter, _ = (
                                _parse_utterance_id(utterance_id)
                            )
                        except (TypeError, ValueError):
                            return False
                        if (
                            utterance_speaker != speaker_dir.name
                            or utterance_chapter != chapter_dir.name
                        ):
                            return False
                        if is_regular_file_below(
                            chapter_dir / f"{utterance_id}.flac",
                            speaker_dir,
                        ):
                            has_audio = True
        except (OSError, UnicodeError, ValueError):
            return False
    return has_audio


def _canonical_transcript_paths(speaker_dir: Path) -> list[Path]:
    """Return deterministic transcripts from direct numeric chapter directories."""

    if (
        speaker_dir.is_symlink()
        or not speaker_dir.is_dir()
        or _SPEAKER_ID.fullmatch(speaker_dir.name) is None
    ):
        return []
    transcripts: list[Path] = []
    for chapter_dir in sorted(speaker_dir.iterdir(), key=lambda path: path.name):
        if (
            chapter_dir.is_symlink()
            or not chapter_dir.is_dir()
            or _SPEAKER_ID.fullmatch(chapter_dir.name) is None
        ):
            continue
        transcripts.extend(
            sorted(
                chapter_dir.glob("*.trans.txt"),
                key=lambda path: path.name,
            )
        )
    return transcripts


def _extract_librispeech_archive_staged(
    archive_path: Path,
    *,
    root: Path,
    subset_dir: Path,
    md5: str,
) -> None:
    with managed_temporary_directory(
        parent=root,
        prefix=extraction_temporary_prefix(subset_dir),
    ) as workspace:
        staging_root = workspace / "payload"
        staging_root.mkdir()
        with open_verified_tar(
            archive_path,
            root=root,
            algorithm="md5",
            expected_digest=md5,
            mode="r:gz",
        ) as tar:
            safe_extractall(tar, staging_root)
        staged_subset = staging_root / "LibriSpeech" / subset_dir.name
        if not _librispeech_subset_ready(staged_subset):
            raise ValueError("LibriSpeech archive contains an incomplete subset tree")
        publish_staged_directory(staged_subset, subset_dir)


def _parse_text_line(line: str) -> tuple[str, str]:
    """Parse a LibriSpeech transcript line into (utterance_id, text)."""
    left, _, right = line.partition(" ")
    return left, right.strip()


def _parse_utterance_id(utterance_id: str) -> tuple[str, str, str]:
    if not isinstance(utterance_id, str):
        raise TypeError("utterance_id must be a string")
    match = _UTTERANCE_ID.fullmatch(utterance_id)
    if match is None:
        raise ValueError("invalid LibriSpeech utterance_id")
    return match.group(1), match.group(2), match.group(3)
