from __future__ import annotations

from collections.abc import Callable
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

import torchrir
import torchrir.io as torchrir_io
import torchrir.io.audio as audio_module
from torchrir.io import (
    AudioData,
    AudioInfo,
    info_audio,
    info_wav,
    load_audio,
    load_audio_data,
    load_wav,
    save_audio,
    save_audio_data,
    save_wav,
)


def _sine_wave(num_samples: int, fs: int) -> torch.Tensor:
    t = torch.arange(num_samples, dtype=torch.float32) / float(fs)
    return torch.sin(2.0 * torch.pi * 440.0 * t)


@pytest.mark.parametrize("extension", [".wav", ".wave"])
def test_save_load_wav_roundtrip(tmp_path: Path, extension: str) -> None:
    fs = 16000
    audio = _sine_wave(2048, fs)
    path = tmp_path / f"tone{extension}"
    save_wav(path, audio, fs, normalize=False)
    loaded, loaded_fs = load_wav(path)
    assert loaded_fs == fs
    assert loaded.ndim == 1
    assert loaded.shape == audio.shape


def test_audio_loader_uses_and_preserves_caller_owned_stream(tmp_path: Path) -> None:
    path = tmp_path / "tone.wav"
    original = _sine_wave(256, 8000)
    save_audio(path, original, 8000)

    with path.open("rb") as stream:
        data = load_audio_data(stream)
        assert not stream.closed
        stream.seek(0)
        loaded, sample_rate = load_audio(stream)
        assert not stream.closed

    assert data.sample_rate == 8000
    assert data.format == "WAV"
    assert sample_rate == 8000
    torch.testing.assert_close(loaded, original)


def test_audio_loader_rejects_invalid_or_unseekable_sources() -> None:
    for source in ("audio.wav", b"audio.wav", object(), StringIO("not binary")):
        with pytest.raises(TypeError, match="Path or seekable binary stream"):
            load_audio_data(cast(Any, source))

    closed = BytesIO()
    closed.close()
    with pytest.raises(ValueError, match="must be open"):
        load_audio_data(closed)

    class NonSeekable(BytesIO):
        def seekable(self) -> bool:
            return False

    with pytest.raises(ValueError, match="must be seekable"):
        load_audio_data(NonSeekable())


def test_audio_paths_require_pathlib_path(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="pathlib.Path"):
        save_audio(cast(Any, str(tmp_path / "audio.wav")), torch.zeros(8), 8000)
    with pytest.raises(TypeError, match="pathlib.Path"):
        save_audio_data(
            cast(Any, str(tmp_path / "audio.wav")),
            AudioData(audio=torch.zeros(8), sample_rate=8000),
        )
    with pytest.raises(TypeError, match="pathlib.Path"):
        info_audio(cast(Any, str(tmp_path / "audio.wav")))
    with pytest.raises(TypeError, match="pathlib.Path"):
        load_wav(cast(Any, "audio.wav"))


@pytest.mark.parametrize("writer", [save_wav, save_audio])
def test_audio_writers_preserve_gain_by_default(
    tmp_path: Path,
    writer: Callable[..., None],
) -> None:
    path = tmp_path / "quiet.wav"
    audio = torch.full((64,), 1.5)
    writer(path, audio, 8000)
    loaded, _ = load_wav(path)
    assert loaded.abs().max().item() == pytest.approx(1.5, abs=1e-6)
    assert info_wav(path).subtype == "FLOAT"


def test_integer_wav_subtype_rejects_samples_that_would_clip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"exceeds.*\[-1, 1\]"):
        save_wav(
            tmp_path / "clipped.wav",
            torch.tensor([0.0, 1.01]),
            8000,
            subtype="PCM_16",
        )


def test_audio_writer_rejects_invalid_save_options(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="normalize must be a bool"):
        save_audio(
            tmp_path / "invalid.wav",
            torch.zeros(8),
            8000,
            normalize=cast(Any, 1),
        )
    with pytest.raises(ValueError, match="non-empty"):
        save_audio(tmp_path / "invalid.wav", torch.zeros(8), 8000, subtype="")


def test_load_wav_uses_channel_zero(tmp_path: Path) -> None:
    fs = 8000
    left = _sine_wave(1024, fs)
    right = _sine_wave(1024, fs) * 0.5
    audio = torch.stack([left, right], dim=0)
    path = tmp_path / "stereo.wav"
    save_wav(path, audio, fs, normalize=False)
    with pytest.warns(RuntimeWarning, match="channel 0 only"):
        loaded, loaded_fs = load_wav(path)
    assert loaded_fs == fs
    assert loaded.ndim == 1
    assert torch.allclose(loaded, left, atol=1e-4, rtol=1e-4)


def test_wav_entry_points_reject_non_wav() -> None:
    with pytest.raises(ValueError, match="expects a wav file"):
        load_wav(Path("not_audio.flac"))
    with pytest.raises(ValueError, match="expects a wav file"):
        save_wav(Path("not_audio.flac"), torch.zeros(16), 16000)
    with pytest.raises(ValueError, match="expects a wav file"):
        info_wav(Path("not_audio.flac"))


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


def test_info_wav(tmp_path: Path) -> None:
    fs = 22050
    audio = _sine_wave(2048, fs)
    path = tmp_path / "info.wav"
    save_wav(path, audio, fs, normalize=False)
    meta = info_wav(path)
    assert meta.sample_rate == fs
    assert meta.num_channels == 1
    assert meta.num_frames == audio.numel()


def test_load_audio_data_and_save_audio_data_roundtrip(tmp_path: Path) -> None:
    fs = 16000
    path = tmp_path / "tone.wav"
    out_path = tmp_path / "tone_copy.wav"
    audio = _sine_wave(512, fs)
    save_wav(path, audio, fs, normalize=False)

    data = load_audio_data(path)
    assert data.sample_rate == fs
    assert data.audio.ndim == 1
    save_audio_data(out_path, data, normalize=False)

    loaded, loaded_fs = load_wav(out_path)
    assert loaded_fs == fs
    assert loaded.shape == data.audio.shape


def test_info_audio_accepts_non_wav(tmp_path: Path) -> None:
    import soundfile as sf

    if "FLAC" not in sf.available_formats():
        pytest.skip("FLAC not available in libsndfile")
    fs = 16000
    audio = _sine_wave(256, fs)
    path = tmp_path / "tone.flac"
    save_audio(path, audio, fs)
    meta = info_audio(path)
    assert meta.sample_rate == fs


@pytest.mark.parametrize("channels", [2, 32, 64])
def test_audio_data_preserves_channel_first_audio(
    tmp_path: Path, channels: int
) -> None:
    fs = 8000
    audio = torch.linspace(-0.25, 0.25, 128).repeat(channels, 1)
    path = tmp_path / f"channels_{channels}.wav"
    save_audio_data(path, AudioData(audio=audio, sample_rate=fs))
    loaded = load_audio_data(path)
    assert loaded.audio.shape == (channels, 128)
    assert torch.allclose(loaded.audio, audio, atol=5e-5, rtol=5e-5)


def test_audio_data_save_does_not_normalize_by_default(tmp_path: Path) -> None:
    audio = torch.full((64,), 0.25)
    path = tmp_path / "not_normalized.wav"
    save_audio_data(path, AudioData(audio=audio, sample_rate=8000))
    loaded = load_audio_data(path)
    assert torch.max(torch.abs(loaded.audio)).item() == pytest.approx(0.25, abs=5e-5)


def test_audio_data_rejects_invalid_tensor_state() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        AudioData(audio=torch.ones(8, dtype=torch.int64), sample_rate=8000)
    with pytest.raises(ValueError, match="finite"):
        AudioData(audio=torch.tensor([float("nan")]), sample_rate=8000)
    with pytest.raises(TypeError, match="sample_rate must be an integer"):
        AudioData(audio=torch.ones(8), sample_rate=True)
    with pytest.raises(ValueError, match="sample_rate must be at most"):
        AudioData(audio=torch.ones(8), sample_rate=2**31)
    with pytest.raises(ValueError, match="channel and sample"):
        AudioData(audio=torch.empty(0, 8), sample_rate=8000)
    with pytest.raises(TypeError, match="supported"):
        AudioData(
            audio=torch.ones(8, dtype=torch.float8_e4m3fn),
            sample_rate=8000,
        )


def test_audio_info_normalizes_integer_metadata_and_rejects_invalid_values() -> None:
    info = AudioInfo(
        sample_rate=cast(Any, np.int64(8000)),
        num_frames=cast(Any, np.int64(16)),
        num_channels=cast(Any, np.int64(2)),
        format="WAV",
        subtype="FLOAT",
        duration=0.002,
    )
    assert type(info.sample_rate) is int
    assert type(info.num_frames) is int
    assert type(info.num_channels) is int

    base: dict[str, object] = {
        "sample_rate": 8000,
        "num_frames": 16,
        "num_channels": 2,
        "format": "WAV",
        "subtype": "FLOAT",
        "duration": 0.002,
    }
    for name, value, error, message in (
        ("num_frames", True, TypeError, "integer"),
        ("num_frames", 2**63, ValueError, "at most"),
        ("num_channels", 0, ValueError, "positive"),
        ("num_channels", 2**31, ValueError, "at most"),
        ("format", 1, TypeError, "must be a string"),
        ("subtype", "", ValueError, "non-empty"),
    ):
        values = dict(base)
        values[name] = value
        with pytest.raises(error, match=message):
            AudioInfo(**values)


def test_audio_writer_rejects_float8_before_soundfile(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="supported"):
        save_audio(
            tmp_path / "unsupported.wav",
            torch.ones(8, dtype=torch.float8_e4m3fn),
            8000,
        )


def test_audio_writer_rejects_sample_rate_outside_backend_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sample-rate.wav"
    with pytest.raises(ValueError, match="sample_rate must be at most"):
        save_audio(path, torch.ones(8), 10**400)
    assert not path.exists()


def test_audio_writer_normalization_rejects_unrepresentable_peak(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="peak.*positive"):
        save_audio(
            tmp_path / "peak.wav",
            torch.ones(8),
            8000,
            normalize=True,
            peak=10**400,
        )


def test_audio_writer_rejects_peak_outside_storage_dtype_before_writing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "peak.wav"
    with pytest.raises(ValueError, match="representable as torch.float32"):
        save_audio(
            path,
            torch.ones(8),
            8000,
            normalize=True,
            peak=1.0e300,
        )
    assert not path.exists()


def test_audio_writer_normalizes_before_narrowing_storage_dtype(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normalized.wav"
    source = torch.tensor([1.0e300, -5.0e299], dtype=torch.float64)

    save_audio(path, source, 8000, normalize=True, peak=0.5)

    loaded, sample_rate = load_audio(path)
    assert sample_rate == 8000
    assert torch.all(torch.isfinite(loaded))
    assert loaded.abs().max().item() == pytest.approx(0.5)


def test_audio_writer_rejects_unrepresentable_unnormalized_audio(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unrepresentable.wav"
    with pytest.raises(ValueError, match="audio must be representable"):
        save_audio(path, torch.tensor([1.0e300], dtype=torch.float64), 8000)
    assert not path.exists()


@pytest.mark.parametrize("writer", [save_wav, save_audio])
def test_audio_writers_reject_zero_channel_audio(
    tmp_path: Path,
    writer: Callable[..., None],
) -> None:
    with pytest.raises(ValueError, match="channel and sample"):
        writer(tmp_path / "empty.wav", torch.empty(0, 8), 8000)


def test_save_audio_data_inherits_subtype_only_for_same_container(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def _capture(*args: object, **kwargs: Any) -> None:
        calls.append(dict(kwargs))

    monkeypatch.setattr(audio_module, "_save_audio", _capture)
    data = AudioData(
        audio=torch.zeros(8),
        sample_rate=8000,
        format="FLAC",
        subtype="PCM_16",
    )

    audio_module.save_audio_data(tmp_path / "same.flac", data)
    audio_module.save_audio_data(tmp_path / "different.ogg", data)
    audio_module.save_audio_data(
        tmp_path / "explicit.ogg",
        data,
        subtype="VORBIS",
    )

    assert [call["subtype"] for call in calls] == ["PCM_16", None, "VORBIS"]


@pytest.mark.parametrize(
    ("extension", "format_name", "subtype"),
    [
        (".wav", "WAV", "PCM_16"),
        (".au", "AU", "FLOAT"),
        (".aif", "AIFF", "PCM_16"),
    ],
)
def test_save_audio_data_preserves_subtype_for_same_real_container(
    tmp_path: Path,
    extension: str,
    format_name: str,
    subtype: str,
) -> None:
    import soundfile as sf

    if not sf.check_format(format_name, subtype):
        pytest.skip(f"libsndfile does not support {format_name}/{subtype}")
    input_path = tmp_path / f"input{extension}"
    output_path = tmp_path / f"output{extension}"
    save_audio(input_path, torch.linspace(-0.5, 0.5, 16), 8000, subtype=subtype)

    data = load_audio_data(input_path)
    save_audio_data(output_path, data)

    assert info_audio(output_path).format == format_name
    assert info_audio(output_path).subtype == subtype


def test_save_audio_data_does_not_reuse_cross_container_subtype(
    tmp_path: Path,
) -> None:
    import soundfile as sf

    if not sf.check_format("FLAC", "PCM_16"):
        pytest.skip("libsndfile does not support FLAC/PCM_16")
    input_path = tmp_path / "input.flac"
    output_path = tmp_path / "output.wav"
    save_audio(input_path, torch.linspace(-0.5, 0.5, 16), 8000, subtype="PCM_16")

    save_audio_data(output_path, load_audio_data(input_path))

    assert info_audio(output_path).format == "WAV"
    assert info_audio(output_path).subtype == "FLOAT"


def test_audio_data_equality_uses_identity_semantics() -> None:
    first = AudioData(audio=torch.zeros(8), sample_rate=8000)
    second = AudioData(audio=torch.zeros(8), sample_rate=8000)
    assert first == first
    assert first != second


def test_audio_public_exports_are_complete() -> None:
    assert AudioInfo.__name__ in torchrir_io.__all__
    assert "save_attribution_file" in torchrir_io.__all__
    for name in (
        "AudioBackend",
        "get_audio_backend",
        "info",
        "list_audio_backends",
        "load",
        "save",
        "set_audio_backend",
    ):
        assert name not in torchrir_io.__all__
        assert not hasattr(torchrir_io, name)
    assert not hasattr(torchrir, "load")
    assert not hasattr(torchrir, "save")
