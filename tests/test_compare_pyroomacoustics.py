"""Cross-implementation RIR tests against pyroomacoustics.

The waveforms are compared on their native sample axes.  Cross-correlation is
used only to report/assert lag; it is never used to shift or crop either RIR.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from torchrir import MicrophoneArray, Room, Source
from torchrir.config import SimulationConfig
from torchrir.signal import fft_convolve
from torchrir.sim import simulate_rir

try:
    import pyroomacoustics as pra
except ImportError:
    if os.environ.get("TORCHRIR_REQUIRE_COMPARISON") == "1":
        raise
    pra = pytest.importorskip("pyroomacoustics")


_CASES = [
    pytest.param(
        [6.0, 4.0],
        [[1.0, 1.5], [2.0, 0.8]],
        [[3.0, 2.0], [5.0, 3.0]],
        [0.2, 0.3, 0.4, 0.5],
        id="2d",
    ),
    pytest.param(
        [6.0, 4.0, 3.0],
        [[1.0, 1.5, 1.2], [2.0, 0.8, 2.0]],
        [[3.0, 2.0, 1.2], [5.0, 3.0, 2.3]],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        id="3d",
    ),
]


def _materials(beta: list[float]):
    absorption = {
        "west": 1.0 - beta[0] ** 2,
        "east": 1.0 - beta[1] ** 2,
        "south": 1.0 - beta[2] ** 2,
        "north": 1.0 - beta[3] ** 2,
    }
    if len(beta) == 6:
        absorption.update(
            floor=1.0 - beta[4] ** 2,
            ceiling=1.0 - beta[5] ** 2,
        )
    return pra.make_materials(**absorption)


def _pyroom_rirs(
    *,
    room_size: list[float],
    sources: list[list[float]],
    microphones: list[list[float]],
    beta: list[float],
    fs: int,
    max_order: int,
) -> list[list[np.ndarray]]:
    previous_rir_hpf_enable = pra.constants.get("rir_hpf_enable")
    pra.constants.set("rir_hpf_enable", False)
    try:
        room = pra.ShoeBox(
            room_size,
            fs=fs,
            max_order=max_order,
            materials=_materials(beta),
        )
        for source in sources:
            room.add_source(source)
        room.add_microphone_array(np.asarray(microphones, dtype=np.float64).T)
        room.compute_rir()
        assert room.rir is not None
        return [
            [np.asarray(room.rir[mic][source], dtype=np.float32) for mic in range(2)]
            for source in range(2)
        ]
    finally:
        pra.constants.set("rir_hpf_enable", previous_rir_hpf_enable)


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-12))


def _lag_samples(actual: np.ndarray, expected: np.ndarray) -> int:
    correlation = np.correlate(actual, expected, mode="full")
    return int(np.argmax(np.abs(correlation)) - (expected.size - 1))


@pytest.mark.comparison
@pytest.mark.numerical
@pytest.mark.parametrize("max_order", [0, 1, 3])
@pytest.mark.parametrize("room_size,sources,microphones,beta", _CASES)
def test_rir_matches_pyroomacoustics_without_free_alignment(
    room_size: list[float],
    sources: list[list[float]],
    microphones: list[list[float]],
    beta: list[float],
    max_order: int,
) -> None:
    fs = 16000
    reference = _pyroom_rirs(
        room_size=room_size,
        sources=sources,
        microphones=microphones,
        beta=beta,
        fs=fs,
        max_order=max_order,
    )
    nsample = max(rir.size for per_source in reference for rir in per_source)
    torch_room = Room.shoebox(room_size, fs=fs, beta=beta)
    actual = simulate_rir(
        room=torch_room,
        sources=Source.from_positions(sources),
        mics=MicrophoneArray.from_positions(microphones),
        config=SimulationConfig(
            max_order=max_order,
            nsample=nsample,
            directivity="omni",
            use_lut=False,
            rir_hpf_enable=False,
        ),
    ).cpu()

    generator = torch.Generator().manual_seed(100 + max_order + len(room_size))
    test_signal = torch.randn(257, generator=generator)
    for source in range(2):
        for microphone in range(2):
            expected_rir = np.pad(
                reference[source][microphone],
                (0, nsample - reference[source][microphone].size),
            )
            actual_rir = actual[source, microphone].numpy()

            assert _lag_samples(actual_rir, expected_rir) == 0
            assert int(np.argmax(np.abs(actual_rir))) == int(
                np.argmax(np.abs(expected_rir))
            )
            assert _relative_l2(actual_rir, expected_rir) < 2e-3

            actual_signal = fft_convolve(
                test_signal, actual[source, microphone]
            ).numpy()
            expected_signal = np.convolve(test_signal.numpy(), expected_rir)
            assert _lag_samples(actual_signal, expected_signal) == 0
            assert _relative_l2(actual_signal, expected_signal) < 2e-3
