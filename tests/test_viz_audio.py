"""The samples delivered to the media encoder follow an explicit level policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from torchrir.viz import animation


@pytest.mark.parametrize("channels", [1, 3])
def test_mux_preserves_selected_channels_in_double_wav(tmp_path, monkeypatch, channels):
    data = np.arange(12 * channels, dtype=np.float64).reshape(12, channels) * 1.0e-8
    source = tmp_path / "source.wav"
    sf.write(source, data, 8000, subtype="DOUBLE")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original video")
    selection = (2, 0) if channels == 3 else (0, 1)
    expected = data[:, selection] if channels == 3 else np.repeat(data, 2, axis=1)
    seen = []

    def mux(command, **kwargs):
        intermediate = Path(command[command.index("-i", command.index("-i") + 1) + 1])
        actual, rate = sf.read(intermediate, always_2d=True)
        assert rate == 8000
        assert sf.info(intermediate).subtype == "DOUBLE"
        np.testing.assert_array_equal(actual, expected)
        seen.append(actual)
        Path(command[-1]).write_bytes(b"muxed video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(animation.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(animation.subprocess, "run", mux)
    animation._add_stereo_audio_to_mp4(
        video_path=video, mixture_path=source, audio_channels=selection
    )
    assert len(seen) == 1
    assert video.read_bytes() == b"muxed video"


@pytest.mark.parametrize(
    "data",
    [
        np.full((16, 2), 1.5),
        np.full((16, 2), np.nan),
        np.full((16, 2), np.inf),
        np.empty((0, 2)),
    ],
)
def test_mux_rejects_invalid_levels_before_encoding(tmp_path, monkeypatch, data):
    source = tmp_path / "source.wav"
    sf.write(source, data, 8000, subtype="DOUBLE")
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    monkeypatch.setattr(animation.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        animation.subprocess, "run", lambda *args, **kwargs: pytest.fail("encoder ran")
    )
    with pytest.raises(ValueError, match="audio"):
        animation._add_stereo_audio_to_mp4(video_path=video, mixture_path=source)
    assert video.read_bytes() == b"original"
    assert set(tmp_path.iterdir()) == {source, video}


@pytest.mark.parametrize(
    "indices,error",
    [
        ((-1, 0), ValueError),
        ((0, 3), ValueError),
        ((True, 0), TypeError),
        ((0.0, 1), TypeError),
        ((0,), ValueError),
    ],
)
def test_mux_rejects_invalid_channel_selection(tmp_path, monkeypatch, indices, error):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((8, 2)), 8000, subtype="FLOAT")
    video = tmp_path / "video.mp4"
    video.touch()
    monkeypatch.setattr(animation.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(
        animation.subprocess, "run", lambda *args, **kwargs: pytest.fail("encoder ran")
    )
    with pytest.raises(error, match="audio_channels"):
        animation._add_stereo_audio_to_mp4(
            video_path=video, mixture_path=source, audio_channels=indices
        )
