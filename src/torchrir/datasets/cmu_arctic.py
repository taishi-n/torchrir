from __future__ import annotations

import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

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


def list_cmu_arctic_speakers() -> List[str]:
    return sorted(VALID_SPEAKERS)


@dataclass
class CmuArcticSentence:
    utterance_id: str
    text: str


class CmuArcticDataset:
    def __init__(self, root: Path, speaker: str = "bdl", download: bool = False) -> None:
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
        return self._dataset_dir / "wav"

    @property
    def text_path(self) -> Path:
        return self._dataset_dir / "etc" / "txt.done.data"

    def _download_and_extract(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self._base_dir / self._archive_name
        url = f"{BASE_URL}/{self._archive_name}"

        if not archive_path.exists():
            print(f"Downloading {url}")
            _download(url, archive_path)
        if not self._dataset_dir.exists():
            print(f"Extracting {archive_path}")
            try:
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self._base_dir)
            except (tarfile.ReadError, EOFError, OSError) as exc:
                print(f"Extraction failed ({exc}); re-downloading.")
                if archive_path.exists():
                    archive_path.unlink()
                _download(url, archive_path)
                with tarfile.open(archive_path, "r:bz2") as tar:
                    tar.extractall(self._base_dir)

    def sentences(self) -> List[CmuArcticSentence]:
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
        wav_ids = {p.stem for p in self.wav_dir.glob("*.wav")}
        return [s for s in self.sentences() if s.utterance_id in wav_ids]

    def wav_path(self, utterance_id: str) -> Path:
        return self.wav_dir / f"{utterance_id}.wav"

    def load_wav(self, utterance_id: str) -> Tuple[torch.Tensor, int]:
        path = self.wav_path(utterance_id)
        return load_wav_mono(path)


def _download(url: str, dest: Path, retries: int = 1) -> None:
    for attempt in range(retries + 1):
        try:
            _stream_download(url, dest)
            return
        except Exception as exc:
            if dest.exists():
                dest.unlink()
            if attempt >= retries:
                raise
            print(f"Download failed ({exc}); retrying...")


def _stream_download(url: str, dest: Path) -> None:
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
                if total > 0:
                    pct = (downloaded / total) * 100
                    sys.stdout.write(f"\rDownloading {dest.name}: {pct:5.1f}%")
                    sys.stdout.flush()
    if total > 0 and downloaded != total:
        raise IOError(f"incomplete download: {downloaded} of {total} bytes")
    tmp_path.replace(dest)
    if total > 0:
        sys.stdout.write("\n")


def _parse_text_line(line: str) -> Tuple[str, str]:
    left, _, right = line.partition('"')
    utterance = left.replace("(", "").strip().split()[0]
    text = right.rsplit('"', 1)[0]
    return utterance, text


def load_wav_mono(path: Path) -> Tuple[torch.Tensor, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio_t = torch.from_numpy(audio)
    if audio_t.shape[1] > 1:
        audio_t = audio_t.mean(dim=1)
    else:
        audio_t = audio_t.squeeze(1)
    return audio_t, sample_rate


def save_wav(path: Path, audio: torch.Tensor, sample_rate: int) -> None:
    import soundfile as sf

    audio = audio.detach().cpu().clamp(-1.0, 1.0).to(torch.float32)
    if audio.ndim == 2 and audio.shape[0] <= 8:
        audio = audio.transpose(0, 1)
    sf.write(str(path), audio.numpy(), sample_rate)
