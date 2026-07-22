"""Physical contracts for diffuse handoff and RIR high-pass filtering."""

from __future__ import annotations

import math

import pytest
import torch

from torchrir import Room
from torchrir.config import RIRHighPassConfig
from torchrir.sim.ism.diffuse import _apply_diffuse_tail
from torchrir.sim.ism.hpf import apply_rir_hpf


def _room(*, beta: list[float] | None = None) -> Room:
    return Room.shoebox(
        [5.0, 4.0, 3.0],
        fs=1000.0,
        beta=[0.8] * 6 if beta is None else beta,
        dtype=torch.float64,
    )


def _early_rir() -> torch.Tensor:
    rir = torch.zeros((1, 1, 200), dtype=torch.float64)
    rir[..., 10:20] = 1.0
    rir[..., 19] = 0.0
    return rir


def test_diffuse_handoff_uses_window_rms_not_previous_sample() -> None:
    room = _room()
    assert room.beta is not None
    output = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=4,
    )
    # The old one-sample estimator saw sample 19 == 0 and effectively silenced
    # the tail. The 10 ms RMS window preserves the preceding early-field power.
    assert torch.linalg.vector_norm(output[..., 30:]).item() > 1.0
    torch.testing.assert_close(output[..., :20], _early_rir()[..., :20])


def test_diffuse_tail_rejects_zero_time_handoff() -> None:
    room = _room()
    assert room.beta is not None
    with pytest.raises(ValueError, match="tdiff must be positive"):
        _apply_diffuse_tail(
            _early_rir(),
            room.size,
            room.beta,
            tdiff=0.0,
            tmax=0.2,
            fs=room.fs,
            c=room.c,
            seed=1,
        )


def test_diffuse_tail_rejects_handoff_window_without_energy() -> None:
    room = _room()
    assert room.beta is not None
    with pytest.raises(ValueError, match="increase max_order/nb_img"):
        _apply_diffuse_tail(
            torch.zeros_like(_early_rir()),
            room.size,
            room.beta,
            tdiff=0.02,
            tmax=0.2,
            fs=room.fs,
            c=room.c,
            seed=1,
        )


@pytest.mark.parametrize("level", [1.0e200, 1.0e-300])
def test_diffuse_handoff_rms_is_stable_for_extreme_finite_levels(
    level: float,
) -> None:
    room = _room(beta=[1.0] * 6)
    assert room.beta is not None
    rir = _early_rir() * level

    output = _apply_diffuse_tail(
        rir,
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=5,
    )

    assert torch.all(torch.isfinite(output))
    torch.testing.assert_close(output[..., :20], rir[..., :20], rtol=0, atol=0)
    assert torch.any(output[..., 25:] != 0)


def test_diffuse_tail_normalizes_finite_sample_index_overflow() -> None:
    room = _room()
    assert room.beta is not None

    with pytest.raises(ValueError, match="tmax.*sample range"):
        _apply_diffuse_tail(
            torch.ones((1, 1, 4), dtype=torch.float64),
            room.size,
            room.beta,
            tdiff=1.0,
            tmax=2.0,
            fs=1.0e308,
            c=room.c,
        )


def test_diffuse_tail_rejects_unrepresentable_finite_output() -> None:
    room = _room(beta=[1.0] * 6)
    assert room.beta is not None
    rir = torch.full((1, 1, 200), 1.0e308, dtype=torch.float64)

    with pytest.raises(ValueError, match="diffuse tail is not representable"):
        _apply_diffuse_tail(
            rir,
            room.size,
            room.beta,
            tdiff=0.02,
            tmax=0.2,
            fs=room.fs,
            c=room.c,
            seed=5,
        )


def test_perfect_reflection_has_nondecaying_diffuse_envelope() -> None:
    room = _room(beta=[1.0] * 6)
    assert room.beta is not None
    output = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=11,
    )
    generator = torch.Generator().manual_seed(11)
    noise = torch.randn((1, 1, 180), dtype=torch.float64, generator=generator)
    scale = math.sqrt(9.0 / 10.0)
    # The 5 ms crossfade occupies samples 20:25; thereafter beta=1 means that
    # the generated carrier is not assigned an arbitrary fallback decay.
    torch.testing.assert_close(output[..., 25:], noise[..., 5:] * scale)


def test_dynamic_diffuse_frames_share_a_coherent_stochastic_carrier() -> None:
    room = _room()
    assert room.beta is not None
    static = _early_rir()
    dynamic = static.unsqueeze(0).repeat(3, 1, 1, 1)
    static_output = _apply_diffuse_tail(
        static,
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=8,
    )
    dynamic_output = _apply_diffuse_tail(
        dynamic,
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=8,
    )
    torch.testing.assert_close(
        dynamic_output, static_output.unsqueeze(0).expand_as(dynamic_output)
    )


def test_seeded_diffuse_carriers_are_prefix_and_batch_invariant() -> None:
    room = _room()
    assert room.beta is not None
    long_rir = _early_rir().expand(2, 2, -1).clone()
    short_rir = long_rir[..., :100].clone()
    short = _apply_diffuse_tail(
        short_rir,
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.1,
        fs=room.fs,
        c=room.c,
        seed=17,
    )
    long = _apply_diffuse_tail(
        long_rir,
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=17,
    )
    single = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=17,
    )
    torch.testing.assert_close(short, long[..., :100], rtol=0, atol=0)
    torch.testing.assert_close(single[0, 0], long[0, 0], rtol=0, atol=0)


def test_diffuse_decay_uses_the_room_speed_of_sound() -> None:
    room = _room()
    assert room.beta is not None
    slow = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=100.0,
        seed=3,
    )
    fast = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=500.0,
        seed=3,
    )
    assert torch.linalg.vector_norm(slow[..., -50:]) > torch.linalg.vector_norm(
        fast[..., -50:]
    )


def test_causal_hpf_is_prefix_invariant() -> None:
    pytest.importorskip("scipy.signal")
    generator = torch.Generator().manual_seed(5)
    long = torch.randn((1, 1, 512), dtype=torch.float64, generator=generator)
    config = RIRHighPassConfig(phase="causal")
    short_output = apply_rir_hpf(long[..., :256], 16000.0, config)
    long_output = apply_rir_hpf(long, 16000.0, config)
    torch.testing.assert_close(short_output, long_output[..., :256], rtol=0, atol=0)


def test_causal_hpf_does_not_leak_seed_changes_before_handoff() -> None:
    pytest.importorskip("scipy.signal")
    room = _room()
    assert room.beta is not None
    first = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=1,
    )
    second = _apply_diffuse_tail(
        _early_rir(),
        room.size,
        room.beta,
        tdiff=0.02,
        tmax=0.2,
        fs=room.fs,
        c=room.c,
        seed=2,
    )
    config = RIRHighPassConfig(phase="causal")
    first = apply_rir_hpf(first, room.fs, config)
    second = apply_rir_hpf(second, room.fs, config)
    torch.testing.assert_close(first[..., :20], second[..., :20], rtol=0, atol=0)
    assert not torch.equal(first[..., 21:], second[..., 21:])


def test_zero_phase_hpf_is_an_explicit_phase_choice() -> None:
    pytest.importorskip("scipy.signal")
    impulse = torch.zeros((1, 1, 128), dtype=torch.float64)
    impulse[..., 32] = 1.0
    causal = apply_rir_hpf(
        impulse,
        16000.0,
        RIRHighPassConfig(phase="causal"),
    )
    zero_phase = apply_rir_hpf(
        impulse,
        16000.0,
        RIRHighPassConfig(phase="zero_phase"),
    )
    torch.testing.assert_close(causal[..., :32], torch.zeros_like(causal[..., :32]))
    assert torch.any(zero_phase[..., :32] != 0)
