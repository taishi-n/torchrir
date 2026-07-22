"""Cross-implementation RIR tests against rir-generator 0.3.0.

The comparison applies only documented, fixed convention transforms.  It does
not estimate a scale or delay from the generated waveforms.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, cast

import numpy as np
import pytest

from torchrir import MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.sim import simulate


_REQUIRE_RIR_GENERATOR = (
    os.environ.get("TORCHRIR_REQUIRE_RIR_GENERATOR_COMPARISON") == "1"
)


def _import_rir_generator() -> Any | None:
    try:
        module = importlib.import_module("rir_generator")
    except Exception as exc:
        if _REQUIRE_RIR_GENERATOR:
            raise ImportError(
                "rir-generator 0.3.0 is required for this comparison job"
            ) from exc
        return None

    version = getattr(module, "__version__", None)
    if version != "0.3.0":
        message = f"rir-generator 0.3.0 is required, found {version!r}"
        if _REQUIRE_RIR_GENERATOR:
            raise RuntimeError(message)
        return None
    return module


rir_generator = _import_rir_generator()
pytestmark = pytest.mark.skipif(
    rir_generator is None,
    reason="rir-generator 0.3.0 is not installed",
)

_RIR_GENERATOR_TO_TORCHRIR_AMP_SCALE = float(4.0 * np.pi)
_MATCHED_FRACTIONAL_DELAY_LENGTH = 129

_ROOM_SIZE = [6.0, 4.0, 3.0]
_SOURCE = [1.0, 1.5, 1.2]
_MICROPHONES = [[3.0, 2.0, 1.2], [5.0, 3.0, 2.3]]
_ASYMMETRIC_BETA = [0.65, 0.85, 0.55, 0.95, 0.75, 0.9]


def _relative_l2(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / (np.linalg.norm(expected) + 1e-12))


def _lag_samples(actual: np.ndarray, expected: np.ndarray) -> int:
    correlation = np.correlate(actual, expected, mode="full")
    return int(np.argmax(np.abs(correlation)) - (expected.size - 1))


def _rir_generator_microphone_type(pattern: str):
    if rir_generator is None:
        raise RuntimeError("rir-generator is not installed")
    module = cast(Any, rir_generator)
    return {
        "omni": module.mtype.omnidirectional,
        "cardioid": module.mtype.cardioid,
        "bidir": module.mtype.bidirectional,
    }[pattern]


@pytest.mark.comparison
@pytest.mark.numerical
@pytest.mark.parametrize("max_order", [0, 1, 3])
@pytest.mark.parametrize("microphone_pattern", ["omni", "cardioid", "bidir"])
def test_rir_matches_rir_generator_with_explicit_conventions(
    max_order: int,
    microphone_pattern: str,
) -> None:
    if rir_generator is None:
        pytest.skip("rir-generator 0.3.0 is not installed")
    module = cast(Any, rir_generator)
    fs = 16000
    nsample = 4096
    microphone_orientation_angles = None if microphone_pattern == "omni" else [0.0, 0.0]

    reference = module.generate(
        c=343.0,
        fs=fs,
        r=_MICROPHONES,
        s=_SOURCE,
        L=_ROOM_SIZE,
        beta=_ASYMMETRIC_BETA,
        nsample=nsample,
        mtype=_rir_generator_microphone_type(microphone_pattern),
        order=max_order,
        dim=3,
        orientation=microphone_orientation_angles,
        hp_filter=False,
    )
    # rir-generator uses 1/(4*pi*r), whereas TorchRIR uses 1/r.
    expected = (
        np.asarray(reference, dtype=np.float64) * _RIR_GENERATOR_TO_TORCHRIR_AMP_SCALE
    )
    assert expected.shape == (nsample, len(_MICROPHONES))

    microphone_orientation = None if microphone_pattern == "omni" else [1.0, 0.0, 0.0]
    actual = (
        simulate(
            StaticScene(
                room=Room.shoebox(
                    _ROOM_SIZE,
                    fs=fs,
                    c=343.0,
                    beta=_ASYMMETRIC_BETA,
                ),
                sources=Source.from_positions([_SOURCE]),
                mics=MicrophoneArray.from_positions(
                    _MICROPHONES,
                    orientation=microphone_orientation,
                    directivity=microphone_pattern,
                ),
            ),
            SimulationConfig(
                max_order=max_order,
                nsample=nsample,
                frac_delay_length=_MATCHED_FRACTIONAL_DELAY_LENGTH,
                use_lut=False,
            ),
        )
        .rirs[0]
        .cpu()
        .numpy()
    )

    # At 16 kHz rir-generator uses Tw=128 without a global delay.  TorchRIR's
    # nearest odd support is 129 taps and its public axis is also physical, so
    # no delay transform is necessary.  The Hann-window conventions still
    # differ slightly.
    for microphone in range(len(_MICROPHONES)):
        expected_rir = expected[:, microphone]
        actual_rir = actual[microphone]
        assert _lag_samples(actual_rir, expected_rir) == 0
        assert int(np.argmax(np.abs(actual_rir))) == int(
            np.argmax(np.abs(expected_rir))
        )
        assert _relative_l2(actual_rir, expected_rir) < 2e-3
