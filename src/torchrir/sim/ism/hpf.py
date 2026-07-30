"""Optional RIR high-pass filtering."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ...config import RIRHighPassConfig


def _design_hpf_sos(fs: float, config: RIRHighPassConfig) -> np.ndarray:
    try:
        from scipy.signal import iirfilter
    except ImportError as exc:
        raise ImportError(
            "scipy is required when SimulationConfig.high_pass is enabled; "
            "install it with `pip install torchrir[hpf]`"
        ) from exc

    normalized_cutoff = 2.0 * config.cutoff_hz / fs
    if not 0.0 < normalized_cutoff < 1.0:
        raise ValueError("high-pass cutoff_hz must satisfy 0 < cutoff_hz < fs/2")
    return iirfilter(
        config.order,
        Wn=normalized_cutoff,
        rp=config.passband_ripple_db,
        rs=config.stopband_attenuation_db,
        btype="highpass",
        output="sos",
        ftype=config.filter_family,
    )


def apply_rir_hpf(
    rir: Tensor,
    fs: float,
    config: RIRHighPassConfig | None,
    *,
    zero_phase_lengths: Tensor | None = None,
) -> Tensor:
    """Apply an explicitly requested IIR high-pass filter to an RIR tensor.

    ``zero_phase_lengths`` selects the finite prefix filtered independently for
    every RIR. Samples after each prefix are zero-filled. The simulator supplies
    pyroomacoustics-compatible natural RIR lengths; direct callers may omit the
    argument to filter the complete final axis.
    """

    if config is None:
        return rir

    try:
        from scipy.signal import sosfilt, sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "scipy is required when SimulationConfig.high_pass is enabled; "
            "install it with `pip install torchrir[hpf]`"
        ) from exc

    sos = _design_hpf_sos(fs, config)
    rir_np = rir.detach().cpu().to(torch.float64).numpy()
    if config.phase == "causal":
        filtered = sosfilt(sos, rir_np, axis=-1)
    else:
        try:
            if zero_phase_lengths is None:
                filtered = sosfiltfilt(sos, rir_np, axis=-1)
            else:
                lengths_np = zero_phase_lengths.detach().cpu().to(torch.int64).numpy()
                if lengths_np.shape != rir_np.shape[:-1]:
                    raise ValueError(
                        "zero_phase_lengths must match the RIR leading dimensions"
                    )
                if np.any(lengths_np < 1) or np.any(lengths_np > rir_np.shape[-1]):
                    raise ValueError(
                        "zero_phase_lengths must lie within the RIR sample axis"
                    )
                filtered = np.zeros_like(rir_np)
                flat_input = rir_np.reshape(-1, rir_np.shape[-1])
                flat_output = filtered.reshape(-1, filtered.shape[-1])
                for index, length in enumerate(lengths_np.reshape(-1)):
                    prefix_length = int(length)
                    flat_output[index, :prefix_length] = sosfiltfilt(
                        sos,
                        flat_input[index, :prefix_length],
                    )
        except ValueError as exc:
            if "padlen" not in str(exc):
                raise
            raise ValueError(
                "RIR sample count is too short for zero-phase high-pass filtering; "
                "increase nsample/tmax, select causal phase, or disable high_pass"
            ) from exc
    return torch.as_tensor(
        np.ascontiguousarray(filtered),
        device=rir.device,
        dtype=rir.dtype,
    )
