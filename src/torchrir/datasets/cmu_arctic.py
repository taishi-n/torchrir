from __future__ import annotations

"""CMU ARCTIC dataset helpers."""

import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from .base import BaseDataset
from .utils import load_wav_mono

BASE_URL = "http://www.festvox.org/cmu_arctic/packed"
VALID_SPEAKERS = {
    "aew",
    "ahw",
    "aup",
    "awb",
    "axb",
    "bdl",
    "clb",
    "eey",
    "fem",
    "gka",
    "jmk",
    "ksp",
    "ljm",
    "lnh",
    "rms",
    "rxr",
    "slp",
    "slt",
}

logger = logging.getLogger(__name__)


def list_cmu_arctic_speakers() -> List[str]:
    """Return supported CMU ARCTIC speaker IDs."""
    return sorted(VALID_SPEAKERS)


@dataclass
class CmuArcticSentence:
    """Sentence metadata from CMU ARCTIC."""

    utterance_id: str
    text: str


class CmuArcticDataset(BaseDataset):
    """CMU ARCTIC dataset loader.

    Example:
        >>> dataset = CmuArcticDataset(Path("datasets/cmu_arctic"), speaker="bdl", download=True)
        >>> audio, fs = dataset.load_wav("arctic_a0001")
    """

    def __init__(
        self, root: Path, speaker: str = "bdl", download: bool = False
    ) -> None:
        """Initialize a CMU ARCTIC dataset handle.

        Args:
            root: Root directory where the dataset is stored.
            speaker: Speaker ID (e.g., "bdl").
            download: Download and extract if missing.
        """
        if speaker not in VALID_SPEAKERS:
            raise ValueError(f"unsupported speaker: {speaker}")
        self.root = Path(root)
        self.speaker = speaker
        self._base_dir = self.root / "ARCTIC"
        self._archive_name = f"cmu_us_{speaker}_arctic.tar.bz2"
        self._dataset_dir = self._base_dir / f"cmu_us_{speaker}_arctic"

        if download:
            self._download_and_extract()

        if not self._dataset_dir.exists():
            raise FileNotFoundError(
                "dataset not found; run with download=True or place the archive under "
                f"{self._base_dir}"
            )

    @property
    def wav_dir(self) -> Path:
        """Return the directory containing wav files."""
        return self._dataset_dir / "wav"

    @property
    def text_path(self) -> Path:
        """Return the path to txt.done.data."""
        return self._dataset_dir / "etc" / "txt.done.data"

    def _download_and_extract(self) -> None:
        """Download and extract the speaker archive if needed."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._base_dir / self._archive_name
        url = f"{BASE_URL}/{self._archive_name}"

        if not archive_path.exists():
            logger.info("Downloading %s", url)
            _download(url, archive_path)
        if not self._dataset_dir.exists():
            logger.info("Extracting %s", archive_path)
            try:
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self._base_dir)
            except (tarfile.ReadError, EOFError, OSError) as exc:
                logger.warning("Extraction failed (%s); re-downloading.", exc)
                if archive_path.exists():
                    archive_path.unlink()
                _download(url, archive_path)
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self._base_dir)

    def sentences(self) -> List[CmuArcticSentence]:
        """Parse all sentence metadata."""
        sentences: List[CmuArcticSentence] = []
        with self.text_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt, text = _parse_text_line(line)
                sentences.append(CmuArcticSentence(utterance_id=utt, text=text))
        return sentences

    def available_sentences(self) -> List[CmuArcticSentence]:
        """Return sentences that have a corresponding wav file."""
        wav_ids = {p.stem for p in self.wav_dir.glob("*.wav")}
        return [s for s in self.sentences() if s.utterance_id in wav_ids]

    def list_speakers(self) -> List[str]:
        """Return available speaker IDs."""
        return list_cmu_arctic_speakers()

    def wav_path(self, utterance_id: str) -> Path:
        """Return the wav path for an utterance ID."""
        return self.wav_dir / f"{utterance_id}.wav"

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        """Load a mono wav for the given utterance ID."""
        path = self.wav_path(utterance_id)
        return load_wav_mono(path)


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
    """Parse a txt.done.data line into (utterance_id, text)."""
    left, _, right = line.partition('"')
    utterance = left.replace("(", "").strip().split()[0]
    text = right.rsplit('"', 1)[0]
    return utterance, text


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    """Save a mono or multi-channel wav to disk.

    Example:
        >>> save_wav(Path("outputs/example.wav"), audio, sample_rate)
    """
    import soundfile as sf

    audio = audio.detach().cpu().clamp(-1.0, 1.0).to(torch.float32)
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.transpose(0, 1)
    sf.write(str(path), audio.numpy(), sample_rate)
