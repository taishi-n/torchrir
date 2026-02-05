from __future__ import annotations

"""LibriSpeech dataset helpers."""

import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from .base import BaseDataset
from .utils import load_wav_mono

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

logger = logging.getLogger(__name__)


@dataclass
class LibriSpeechSentence:
    """Sentence metadata from LibriSpeech."""

    utterance_id: str
    text: str
    speaker_id: str
    chapter_id: str


class LibriSpeechDataset(BaseDataset):
    """LibriSpeech dataset loader.

    Example:
        >>> dataset = LibriSpeechDataset(Path("datasets/librispeech"), subset="train-clean-100", download=True)
        >>> audio, fs = dataset.load_wav("103-1240-0000")
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
            download: Download and extract if missing.
        """
        if subset not in VALID_SUBSETS:
            raise ValueError(f"unsupported subset: {subset}")
        self.root = Path(root)
        self.subset = subset
        self.speaker = speaker
        self._archive_name = f"{subset}.tar.gz"
        self._base_dir = self.root / "LibriSpeech"
        self._subset_dir = self._base_dir / subset
        self._speaker_dir = self._subset_dir / speaker if speaker else None

        if download:
            self._download_and_extract()

        if not self._subset_dir.exists():
            raise FileNotFoundError(
                "dataset not found; run with download=True or place the archive under "
                f"{self.root}"
            )
        if self._speaker_dir is not None and not self._speaker_dir.exists():
            raise FileNotFoundError(
                f"speaker directory not found: {self._speaker_dir}"
            )

    def list_speakers(self) -> List[str]:
        """Return available speaker IDs."""
        if self.speaker is not None:
            return [self.speaker]
        if not self._subset_dir.exists():
            return []
        return sorted([p.name for p in self._subset_dir.iterdir() if p.is_dir()])

    def available_sentences(self) -> List[LibriSpeechSentence]:
        """Return sentences that have a corresponding audio file."""
        sentences: List[LibriSpeechSentence] = []
        search_root = (
            self._speaker_dir if self._speaker_dir is not None else self._subset_dir
        )
        for trans_path in search_root.rglob("*.trans.txt"):
            chapter_dir = trans_path.parent
            speaker_id = chapter_dir.parent.name
            chapter_id = chapter_dir.name
            with trans_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, text = _parse_text_line(line)
                    wav_path = chapter_dir / f"{utt_id}.flac"
                    if wav_path.exists():
                        sentences.append(
                            LibriSpeechSentence(
                                utterance_id=utt_id,
                                text=text,
                                speaker_id=speaker_id,
                                chapter_id=chapter_id,
                            )
                        )
        return sentences

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load a mono wav for the given utterance ID."""
        speaker_id, chapter_id, _ = utterance_id.split("-", 2)
        path = self._subset_dir / speaker_id / chapter_id / f"{utterance_id}.flac"
        return load_wav_mono(path)

    def _download_and_extract(self) -> None:
        """Download and extract the subset archive if needed."""
        self.root.mkdir(parents=True, exist_ok=True)
        archive_path = self.root / self._archive_name
        url = f"{BASE_URL}/{self._archive_name}"

        if not archive_path.exists():
            logger.info("Downloading %s", url)
            _download(url, archive_path)
        if not self._subset_dir.exists():
            logger.info("Extracting %s", archive_path)
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(self.root)
            except (tarfile.ReadError, EOFError, OSError) as exc:
                logger.warning("Extraction failed (%s); re-downloading.", exc)
                if archive_path.exists():
                    archive_path.unlink()
                _download(url, archive_path)
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(self.root)


def _download(url: str, dest: Path, retries: int = 1) -> None:
    """Download a file with retry and resume-safe temp file."""
    for attempt in range(retries + 1):
        try:
            _stream_download(url, dest)
            return
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            if attempt >= retries:
                raise
            logger.warning("Download failed (%s); retrying...", exc)


def _stream_download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a progress indicator."""
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    if tmp_path.exists():
        tmp_path.unlink()

    with urllib.request.urlopen(url) as response:
        total = response.length or 0
        downloaded = 0
        chunk_size = 1024 * 1024
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
    if total > 0 and downloaded != total:
        raise IOError(f"incomplete download: {downloaded} of {total} bytes")
    tmp_path.replace(dest)


def _parse_text_line(line: str) -> Tuple[str, str]:
    """Parse a LibriSpeech transcript line into (utterance_id, text)."""
    left, _, right = line.partition(" ")
    return left, right.strip()
