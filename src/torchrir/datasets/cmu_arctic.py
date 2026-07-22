"""CMU ARCTIC dataset helpers."""

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

BASE_URL = "https://www.festvox.org/cmu_arctic/packed"
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
ARCHIVE_SHA256 = {
    "aew": "645cb33c0f0b2ce41384fdd8d3db2c3f5fc15c1e688baeb74d2e08cab18ab406",
    "ahw": "024664adeb892809d646a3efd043625b46b5bfa3e6189b3500b2d0d59dfab06c",
    "aup": "2c55bc3050caa996758869126ad10cf42e1441212111db034b3a45189c18b6fc",
    "awb": "d74a950c9739a65f7bfc4dfa6187f2730fa03de5b8eb3f2da97a51b74df64d3c",
    "axb": "dd65c3d2907d1ee52f86e44f578319159e60f4bf722a9142be01161d84e330ff",
    "bdl": "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
    "clb": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
    "eey": "8a0ee4e5acbd4b2f61a4fb947c1730ab3adcc9dc50b195981d99391d29928e8a",
    "fem": "3fcff629412b57233589cdb058f730594a62c4f3a75c20de14afe06621ef45e2",
    "gka": "dc82e7967cbd5eddbed33074b0699128dbd4482b41711916d58103707e38c67f",
    "jmk": "3a37c0e1dfc91e734fdbc88b562d9e2ebca621772402cdc693bbc9b09b211d73",
    "ksp": "8029cafce8296f9bed3022c44ef1e7953332b6bf6943c14b929f468122532717",
    "ljm": "b23993765cbf2b9e7bbc3c85b6c56eaf292ac81ee4bb887b638a24d104f921a0",
    "lnh": "4faf34d71aa7112813252fb20c5433e2fdd9a9de55a00701ffcbf05f24a5991a",
    "rms": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
    "rxr": "1fa4271c393e5998d200e56c102ff46fcfea169aaa2148ad9e9469616fbfdd9b",
    "slp": "54345ed55e45c23d419e9a823eef427f1cc93c83a710735ec667d068c916abf1",
    "slt": "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
}
_UTTERANCE_ID = re.compile(r"arctic_[a-z]\d{4}\Z")

logger = logging.getLogger(__name__)


def cmu_arctic_speakers(root: Path | None = None) -> list[str]:
    """Return supported speakers, or locally installed speakers below ``root``."""

    if root is None:
        return sorted(VALID_SPEAKERS)
    base_dir = Path(root) / "ARCTIC"
    return sorted(
        speaker
        for speaker in VALID_SPEAKERS
        if _cmu_dataset_ready(base_dir / f"cmu_us_{speaker}_arctic")
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CmuArcticSentence:
    """Sentence metadata from CMU ARCTIC."""

    utterance_id: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.utterance_id, str):
            raise TypeError("utterance_id must be a string")
        if _UTTERANCE_ID.fullmatch(self.utterance_id) is None:
            raise ValueError("utterance_id must be a canonical CMU ARCTIC ID")
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


class CmuArcticDataset(BaseDataset):
    """CMU ARCTIC dataset loader.

    Examples:
        ```python
        dataset = CmuArcticDataset(Path("datasets/cmu_arctic"), speaker="bdl", download=True)
        audio, fs = dataset.load_audio("arctic_a0001")
        ```
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
        if not isinstance(download, bool):
            raise TypeError("download must be a bool")
        if not isinstance(speaker, str):
            raise TypeError("speaker must be a string")
        if speaker not in VALID_SPEAKERS:
            raise ValueError(f"unsupported speaker: {speaker}")
        self.root = Path(root)
        self.speaker = speaker
        self._base_dir = self.root / "ARCTIC"
        self._archive_name = f"cmu_us_{speaker}_arctic.tar.bz2"
        self._dataset_dir = self._base_dir / f"cmu_us_{speaker}_arctic"
        recover_staged_publication(self._dataset_dir)

        if download:
            self._download_and_extract()

        if not _cmu_dataset_ready(self._dataset_dir):
            raise FileNotFoundError(
                "dataset is missing or incomplete; run with download=True or place "
                "a complete archive under "
                f"{self._base_dir}"
            )

    @property
    def audio_dir(self) -> Path:
        """Return the directory containing audio files."""
        return self._dataset_dir / "wav"

    @property
    def text_path(self) -> Path:
        """Return the path to txt.done.data."""
        return self._dataset_dir / "etc" / "txt.done.data"

    def _download_and_extract(self) -> None:
        """Download and extract the speaker archive if needed."""
        self._base_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self._base_dir / f".{self._dataset_dir.name}.torchrir.lock"
        with dataset_write_lock(lock_path):
            archive_path = self._base_dir / self._archive_name
            cleanup_stale_download_directories(archive_path)
            cleanup_stale_extraction_directories(
                parent=self._base_dir,
                target=self._dataset_dir,
            )
            if _cmu_dataset_ready(self._dataset_dir):
                return
            self._download_and_extract_locked()

    def _download_and_extract_locked(self) -> None:
        """Download and publish while holding this speaker's writer lock."""

        archive_path = self._base_dir / self._archive_name
        url = f"{BASE_URL}/{self._archive_name}"
        checksum = ARCHIVE_SHA256[self.speaker]

        def download_archive() -> None:
            logger.info("Downloading %s", url)
            _download(url, archive_path, sha256=checksum)

        def extract_archive() -> None:
            logger.info("Extracting %s", archive_path)
            _extract_cmu_archive_staged(
                archive_path,
                base_dir=self._base_dir,
                dataset_dir=self._dataset_dir,
                sha256=checksum,
            )

        extract_or_download_archive(
            archive_path,
            root=self._base_dir,
            download=download_archive,
            extract=extract_archive,
            operation_logger=logger,
        )

    def sentences(self) -> list[CmuArcticSentence]:
        """Parse all sentence metadata."""
        if self._dataset_dir.is_symlink():
            raise FileNotFoundError("CMU ARCTIC transcript is missing or unsafe")
        sentences: list[CmuArcticSentence] = []
        try:
            with open_regular_file_below(
                self.text_path,
                root=self._dataset_dir,
            ) as (binary, _):
                with TextIOWrapper(binary, encoding="utf-8") as transcript:
                    for line in transcript:
                        line = line.strip()
                        if not line:
                            continue
                        utt, text = _parse_text_line(line)
                        if _UTTERANCE_ID.fullmatch(utt) is None:
                            raise ValueError(f"invalid CMU ARCTIC transcript ID: {utt}")
                        sentences.append(CmuArcticSentence(utterance_id=utt, text=text))
        except (OSError, UnicodeError) as exc:
            raise FileNotFoundError(
                "CMU ARCTIC transcript is missing or unsafe"
            ) from exc
        return sentences

    def available_sentences(self) -> list[CmuArcticSentence]:
        """Return sentences that have a corresponding wav file."""
        wav_ids = {
            path.stem
            for path in self.audio_dir.glob("*.wav")
            if is_regular_file_below(path, self._dataset_dir)
        }
        return [s for s in self.sentences() if s.utterance_id in wav_ids]

    def list_speakers(self) -> list[str]:
        """Return speaker IDs with a usable local dataset directory."""
        return cmu_arctic_speakers(self.root)

    def audio_path(self, utterance_id: str) -> Path:
        """Return the audio path for an utterance ID."""
        if not isinstance(utterance_id, str):
            raise TypeError("utterance_id must be a string")
        if _UTTERANCE_ID.fullmatch(utterance_id) is None:
            raise ValueError("invalid CMU ARCTIC utterance_id")
        if self._dataset_dir.is_symlink():
            raise ValueError("CMU ARCTIC dataset directory must not be a symlink")
        root = self._dataset_dir.resolve()
        candidate = self.audio_dir / f"{utterance_id}.wav"
        if candidate.is_symlink():
            raise ValueError("CMU ARCTIC audio file must not be a symlink")
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ValueError("utterance path escapes the CMU ARCTIC audio directory")
        return path

    def load_audio(self, utterance_id: str) -> tuple[torch.Tensor, int]:
        """Load mono audio for the given utterance ID."""
        path = self.audio_path(utterance_id)
        stack = ExitStack()
        try:
            binary, _ = stack.enter_context(
                open_regular_file_below(path, root=self._dataset_dir)
            )
        except ValueError as exc:
            stack.close()
            raise FileNotFoundError(
                "CMU ARCTIC audio must be a regular file below the dataset tree"
            ) from exc
        with stack:
            return load_audio(binary)

    def attribution_info(self) -> DatasetAttribution:
        """Return attribution and license information for CMU ARCTIC."""
        return attribution_for("cmu_arctic")


def _download(url: str, dest: Path, *, sha256: str, retries: int = 1) -> None:
    """Download an archive, retrying only transport/integrity failures."""

    retry_download(
        lambda: _stream_download(url, dest, sha256=sha256),
        retries=retries,
        logger=logger,
    )


def _stream_download(
    url: str,
    dest: Path,
    *,
    sha256: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
) -> None:
    """Stream and verify one archive through a unique sibling file."""

    stream_verified_download(
        url,
        dest,
        algorithm="sha256",
        expected_digest=sha256,
        timeout=timeout,
        total_timeout=total_timeout,
    )


def _has_sha256(path: Path, expected: str) -> bool:
    return file_matches_digest(path, algorithm="sha256", expected_digest=expected)


def _cmu_dataset_ready(dataset_dir: Path) -> bool:
    text_path = dataset_dir / "etc" / "txt.done.data"
    audio_dir = dataset_dir / "wav"
    if (
        dataset_dir.is_symlink()
        or not is_regular_file_below(text_path, dataset_dir)
        or not audio_dir.is_dir()
    ):
        return False
    has_audio = False
    try:
        with open_regular_file_below(text_path, root=dataset_dir) as (binary, _):
            with TextIOWrapper(binary, encoding="utf-8") as transcript:
                for raw_line in transcript:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        utterance_id, _ = _parse_text_line(line)
                    except ValueError:
                        return False
                    if _UTTERANCE_ID.fullmatch(utterance_id) is None:
                        return False
                    if is_regular_file_below(
                        audio_dir / f"{utterance_id}.wav",
                        dataset_dir,
                    ):
                        has_audio = True
    except (OSError, UnicodeError, ValueError):
        return False
    return has_audio


def _extract_cmu_archive_staged(
    archive_path: Path,
    *,
    base_dir: Path,
    dataset_dir: Path,
    sha256: str,
) -> None:
    with managed_temporary_directory(
        parent=base_dir,
        prefix=extraction_temporary_prefix(dataset_dir),
    ) as workspace:
        staging_root = workspace / "payload"
        staging_root.mkdir()
        with open_verified_tar(
            archive_path,
            root=base_dir,
            algorithm="sha256",
            expected_digest=sha256,
            mode="r:bz2",
        ) as tar:
            safe_extractall(tar, staging_root)
        staged_dataset = staging_root / dataset_dir.name
        if not _cmu_dataset_ready(staged_dataset):
            raise ValueError("CMU ARCTIC archive contains an incomplete dataset tree")
        publish_staged_directory(staged_dataset, dataset_dir)


def _parse_text_line(line: str) -> tuple[str, str]:
    """Parse a txt.done.data line into (utterance_id, text)."""
    stripped = line.strip()
    if not stripped.startswith("(") or not stripped.endswith(")"):
        raise ValueError("invalid CMU ARCTIC transcript line")
    left, opening_quote, remainder = stripped.partition('"')
    text, closing_quote, suffix = remainder.rpartition('"')
    tokens = left.removeprefix("(").strip().split()
    if (
        not opening_quote
        or not closing_quote
        or len(tokens) != 1
        or suffix.strip() != ")"
    ):
        raise ValueError("invalid CMU ARCTIC transcript line")
    utterance = tokens[0]
    return utterance, text
