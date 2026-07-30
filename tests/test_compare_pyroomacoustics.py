"""Cross-implementation RIR tests against pyroomacoustics.

Pyroomacoustics 0.9.0 exposes the fixed 40-sample group delay of its 81-tap
fractional-delay filter.  The reference is cropped by that documented constant
before comparison with TorchRIR's physical sample axis.  No delay or scale is
estimated from the generated waveforms.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os

import numpy as np
import pytest
import torch

from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import RIRHighPassConfig, SimulationConfig
from torchrir.signal import fft_convolve
from torchrir.sim import simulate

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

_DIRECTIONAL_ROOM_SIZE = [6.0, 4.0, 3.0]
_DIRECTIONAL_SOURCE = [1.0, 1.5, 1.2]
_DIRECTIONAL_MICROPHONES = [[3.0, 2.0, 1.2], [5.0, 3.0, 2.3]]
_DIRECTIONAL_BETA = [0.65, 0.85, 0.55, 0.95, 0.75, 0.9]
_PYROOM_FRACTIONAL_DELAY = 81
_PYROOM_FRACTIONAL_DELAY_OFFSET = (_PYROOM_FRACTIONAL_DELAY - 1) // 2


@contextmanager
def _pyroom_hpf(enabled: bool) -> Iterator[None]:
    """Set pyroomacoustics' process-global HPF without leaking state."""
    previous_rir_hpf_enable = pra.constants.get("rir_hpf_enable")
    pra.constants.set("rir_hpf_enable", enabled)
    try:
        yield
    finally:
        pra.constants.set("rir_hpf_enable", previous_rir_hpf_enable)


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
    hpf_enabled: bool = False,
) -> list[list[np.ndarray]]:
    with _pyroom_hpf(hpf_enabled):
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


def _pyroom_analytic_directivity(pattern: str, *, target: str):
    # PyPI pyroomacoustics 0.9.0 has a broken ndarray-orientation constructor.
    # DirectionVector is its supported, deterministic path and avoids that
    # external reference bug.
    if target == "source":
        direction = pra.DirectionVector(azimuth=90.0, colatitude=90.0, degrees=True)
    elif target == "microphone":
        direction = pra.DirectionVector(azimuth=0.0, colatitude=90.0, degrees=True)
    else:
        raise ValueError(f"unsupported directivity target: {target}")

    directivity_type = {
        "cardioid": pra.Cardioid,
        "hypercardioid": pra.HyperCardioid,
        "bidir": pra.FigureEight,
    }[pattern]
    return directivity_type(orientation=direction)


def _pyroom_directional_rirs(
    *,
    fs: int,
    max_order: int,
    pattern: str,
    target: str,
) -> list[np.ndarray]:
    directivity = _pyroom_analytic_directivity(pattern, target=target)
    with _pyroom_hpf(False):
        room = pra.ShoeBox(
            _DIRECTIONAL_ROOM_SIZE,
            fs=fs,
            max_order=max_order,
            materials=_materials(_DIRECTIONAL_BETA),
        )
        room.add_source(
            _DIRECTIONAL_SOURCE,
            directivity=directivity if target == "source" else None,
        )
        room.add_microphone_array(
            np.asarray(_DIRECTIONAL_MICROPHONES, dtype=np.float64).T,
            directivity=directivity if target == "microphone" else None,
        )
        room.compute_rir()
        assert room.rir is not None
        return [
            np.asarray(room.rir[microphone][0], dtype=np.float32)
            for microphone in range(len(_DIRECTIONAL_MICROPHONES))
        ]


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-12))


def _lag_samples(actual: np.ndarray, expected: np.ndarray) -> int:
    correlation = np.correlate(actual, expected, mode="full")
    return int(np.argmax(np.abs(correlation)) - (expected.size - 1))


def _to_physical_time_axis(reference: np.ndarray, *, nsample: int) -> np.ndarray:
    physical = reference[_PYROOM_FRACTIONAL_DELAY_OFFSET:]
    if physical.size > nsample:
        return physical[:nsample]
    return np.pad(physical, (0, nsample - physical.size))


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
    nsample = max(
        rir.size - _PYROOM_FRACTIONAL_DELAY_OFFSET
        for per_source in reference
        for rir in per_source
    )
    torch_room = Room.shoebox(room_size, fs=fs, beta=beta)
    actual = simulate(
        StaticScene(
            room=torch_room,
            sources=Source.from_positions(sources),
            mics=MicrophoneArray.from_positions(microphones),
        ),
        SimulationConfig(
            max_order=max_order,
            nsample=nsample,
            use_lut=False,
        ),
    ).rirs.cpu()

    generator = torch.Generator().manual_seed(100 + max_order + len(room_size))
    test_signal = torch.randn(257, generator=generator)
    for source in range(2):
        for microphone in range(2):
            expected_rir = _to_physical_time_axis(
                reference[source][microphone], nsample=nsample
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


@pytest.mark.comparison
@pytest.mark.numerical
def test_zero_phase_hpf_matches_pyroomacoustics_natural_rir_horizons() -> None:
    fs = 16000
    room_size = [6.0, 4.0, 3.0]
    sources = [[1.0, 1.5, 1.2], [2.0, 0.8, 2.0]]
    microphones = [[3.0, 2.0, 1.2], [5.0, 3.0, 2.3]]
    beta = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    max_order = 3
    reference = _pyroom_rirs(
        room_size=room_size,
        sources=sources,
        microphones=microphones,
        beta=beta,
        fs=fs,
        max_order=max_order,
        hpf_enabled=True,
    )
    nsample = (
        max(
            rir.size - _PYROOM_FRACTIONAL_DELAY_OFFSET
            for per_source in reference
            for rir in per_source
        )
        + 256
    )
    actual = simulate(
        StaticScene(
            room=Room.shoebox(room_size, fs=fs, beta=beta),
            sources=Source.from_positions(sources),
            mics=MicrophoneArray.from_positions(microphones),
        ),
        SimulationConfig(
            max_order=max_order,
            nsample=nsample,
            use_lut=True,
            high_pass=RIRHighPassConfig(),
        ),
    ).rirs.cpu()

    for source in range(2):
        for microphone in range(2):
            expected = _to_physical_time_axis(
                reference[source][microphone],
                nsample=nsample,
            )
            assert _lag_samples(actual[source, microphone].numpy(), expected) == 0
            assert _relative_l2(actual[source, microphone].numpy(), expected) < 2e-3
            assert (
                torch.count_nonzero(
                    actual[
                        source,
                        microphone,
                        reference[source][microphone].size
                        - _PYROOM_FRACTIONAL_DELAY_OFFSET :,
                    ]
                )
                == 0
            )


@pytest.mark.comparison
@pytest.mark.numerical
@pytest.mark.parametrize("max_order", [0, 1, 3])
@pytest.mark.parametrize("pattern", ["cardioid", "hypercardioid", "bidir"])
@pytest.mark.parametrize("target", ["source", "microphone"])
def test_analytic_directivity_matches_pyroomacoustics(
    max_order: int,
    pattern: str,
    target: str,
) -> None:
    """Compare direct, odd-reflection, and mixed-parity image paths."""
    fs = 16000
    reference = _pyroom_directional_rirs(
        fs=fs,
        max_order=max_order,
        pattern=pattern,
        target=target,
    )
    nsample = max(rir.size - _PYROOM_FRACTIONAL_DELAY_OFFSET for rir in reference)
    source_orientation = [0.0, 1.0, 0.0] if target == "source" else None
    microphone_orientation = [1.0, 0.0, 0.0] if target == "microphone" else None
    source_directivity = pattern if target == "source" else "omni"
    microphone_directivity = pattern if target == "microphone" else "omni"
    actual = (
        simulate(
            StaticScene(
                room=Room.shoebox(
                    _DIRECTIONAL_ROOM_SIZE,
                    fs=fs,
                    beta=_DIRECTIONAL_BETA,
                ),
                sources=Source.from_positions(
                    [_DIRECTIONAL_SOURCE],
                    orientation=source_orientation,
                    directivity=source_directivity,
                ),
                mics=MicrophoneArray.from_positions(
                    _DIRECTIONAL_MICROPHONES,
                    orientation=microphone_orientation,
                    directivity=microphone_directivity,
                ),
            ),
            SimulationConfig(
                max_order=max_order,
                nsample=nsample,
                use_lut=False,
            ),
        )
        .rirs[0]
        .cpu()
        .numpy()
    )

    for microphone, reference_rir in enumerate(reference):
        expected = _to_physical_time_axis(reference_rir, nsample=nsample)
        assert _lag_samples(actual[microphone], expected) == 0
        assert int(np.argmax(np.abs(actual[microphone]))) == int(
            np.argmax(np.abs(expected))
        )
        assert _relative_l2(actual[microphone], expected) < 2e-3
