from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torchrir.io import (
    get_audio_backend,
    info,
    list_audio_backends,
    load,
    save,
    set_audio_backend,
)
from torchrir.io.audio import load_audio, save_audio
from torchrir.io.audio import load_audio_data, save_audio_data


def _sine_wave(num_samples: int, fs: int) -> torch.Tensor:
    t = torch.arange(num_samples, dtype=torch.float32) / float(fs)
    return torch.sin(2.0 * torch.pi * 440.0 * t)


def test_save_load_wav_roundtrip(tmp_path: Path) -> None:
    fs = 16000
    audio = _sine_wave(2048, fs)
    path = tmp_path / "tone.wav"
    save(path, audio, fs, normalize=False)
    loaded, loaded_fs = load(path)
    assert loaded_fs == fs
    assert loaded.ndim == 1
    assert loaded.shape == audio.shape


def test_load_wav_uses_channel_zero(tmp_path: Path) -> None:
    fs = 8000
    left = _sine_wave(1024, fs)
    right = _sine_wave(1024, fs) * 0.5
    audio = torch.stack([left, right], dim=0)
    path = tmp_path / "stereo.wav"
    save(path, audio, fs, normalize=False)
    with pytest.warns(RuntimeWarning, match="channel 0 only"):
        loaded, loaded_fs = load(path)
    assert loaded_fs == fs
    assert loaded.ndim == 1
    assert torch.allclose(loaded, left, atol=1e-4, rtol=1e-4)


def test_load_save_wav_rejects_non_wav() -> None:
    with pytest.raises(ValueError, match="expects a wav file"):
        load(Path("not_audio.flac"))
    with pytest.raises(ValueError, match="expects a wav file"):
        save(Path("not_audio.flac"), torch.zeros(16), 16000)


def test_load_audio_accepts_optional_non_wav(tmp_path: Path) -> None:
    import soundfile as sf

    if "FLAC" not in sf.available_formats():
        pytest.skip("FLAC not available in libsndfile")
    fs = 16000
    audio = _sine_wave(1024, fs)
    path = tmp_path / "tone.flac"
    save_audio(path, audio, fs)
    loaded, loaded_fs = load_audio(path)
    assert loaded_fs == fs
    assert loaded.ndim == 1


def test_audio_backend_registry() -> None:
    backends = list_audio_backends()
    assert "soundfile" in backends
    assert get_audio_backend() in backends
    set_audio_backend("soundfile")
    with pytest.raises(ValueError, match="Unknown audio backend"):
        set_audio_backend("unknown-backend")


def test_info_wav(tmp_path: Path) -> None:
    fs = 22050
    audio = _sine_wave(2048, fs)
    path = tmp_path / "info.wav"
    save(path, audio, fs, normalize=False)
    meta = info(path)
    assert meta.sample_rate == fs
    assert meta.num_channels == 1
    assert meta.num_frames == audio.numel()


def test_load_audio_data_and_save_audio_data_roundtrip(tmp_path: Path) -> None:
    fs = 16000
    path = tmp_path / "tone.wav"
    out_path = tmp_path / "tone_copy.wav"
    audio = _sine_wave(512, fs)
    save(path, audio, fs, normalize=False)

    data = load_audio_data(path)
    assert data.sample_rate == fs
    assert data.audio.ndim == 1
    save_audio_data(out_path, data, normalize=False)

    loaded, loaded_fs = load(out_path)
    assert loaded_fs == fs
    assert loaded.shape == data.audio.shape
