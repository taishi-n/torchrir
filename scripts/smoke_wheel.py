"""Run outside the checkout with isolated Python from a wheel-only environment."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torchrir
from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.signal import DynamicConvolver, FrameSchedule, convolve_rir
from torchrir.sim import simulate


def check_base() -> None:
    assert Path(torchrir.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
    scene = StaticScene(
        room=Room.shoebox(size=[4.0, 4.0, 3.0], fs=3430, beta=[0.0] * 6),
        sources=Source.from_positions([[1.0, 1.0, 1.0]]),
        mics=MicrophoneArray.from_positions([[2.0, 1.0, 1.0], [3.0, 1.0, 1.0]]),
    )
    result = simulate(
        scene,
        SimulationConfig(
            max_order=0, nsample=64, use_lut=False, device="cpu", dtype=torch.float64
        ),
    )
    expected = torch.zeros((1, 2, 64), dtype=torch.float64)
    expected[0, 0, 10] = 1.0
    expected[0, 1, 20] = 0.5
    torch.testing.assert_close(result.rirs, expected, rtol=1e-12, atol=1e-12)

    x = torch.tensor(
        [[1.0, -2, 0, 3, 0.5, -1], [0.0, 2, 1, -0.5, 4, 1]], dtype=torch.float64
    )
    h = torch.arange(24, dtype=torch.float64).reshape(2, 2, 2, 3) / 10 - 0.8
    schedule = FrameSchedule.from_samples([0, 3])
    for mode in ("static", "emission", "observation"):
        expected = np.zeros((2, 8))
        for microphone in range(2):
            for output in range(8):
                for source in range(2):
                    for tap in range(3):
                        emitted = output - tap
                        if 0 <= emitted < 6:
                            frame = (
                                0
                                if mode == "static"
                                else int(
                                    (emitted if mode == "emission" else output) >= 3
                                )
                            )
                            expected[microphone, output] += float(
                                x[source, emitted] * h[frame, source, microphone, tap]
                            )
        actual = (
            convolve_rir(x, h[0])
            if mode == "static"
            else DynamicConvolver(time_reference=mode).convolve(x, h, schedule=schedule)
        )
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-12)
    print(
        "Installed base wheel: direct-path arrival/gain and three convolution modes passed"
    )


def check_extras() -> None:
    from torchrir.io import info_audio, load_audio_data, save_audio

    samples = torch.tensor([[1.5, -1.5, 1e-8, 0], [0.25, -0.5, 0.75, 1]])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "multichannel.wav"
        save_audio(path, samples, 8000)
        loaded = load_audio_data(path)
        assert loaded.sample_rate == 8000
        assert info_audio(path).subtype == "FLOAT"
        torch.testing.assert_close(loaded.audio, samples, rtol=0, atol=0)
        command = Path(sys.executable).parent / "torchrir-build-dynamic-cmu-arctic"
        completed = subprocess.run(
            [str(command), "--help"],
            cwd=directory,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "usage:" in completed.stdout
    print(
        "Installed extras wheel: FLOAT multichannel WAV and builder entry point passed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", action="store_true")
    args = parser.parse_args()
    check_base()
    if args.extras:
        check_extras()
