from __future__ import annotations

"""RIR high-pass filtering utilities."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from ...config import SimulationConfig


def _design_hpf_sos(fs: float, fc: float, kwargs: dict[str, Any]) -> np.ndarray:
    try:
        from scipy.signal import iirfilter
    except ImportError as exc:
        raise ImportError(
            "scipy is required when rir_hpf_enable=True. "
            "Install scipy or disable the HPF."
        ) from exc

    wc = 2.0 * fc / fs
    if not 0.0 < wc < 1.0:
        raise ValueError("rir_hpf_fc must satisfy 0 < rir_hpf_fc < fs/2")

    hpf_kwargs = dict(kwargs)
    n = int(hpf_kwargs.pop("n", 2))
    if "type" in hpf_kwargs and "ftype" not in hpf_kwargs:
        hpf_kwargs["ftype"] = hpf_kwargs.pop("type")
    return iirfilter(
        n,
        Wn=wc,
        btype="highpass",
        output="sos",
        **hpf_kwargs,
    )


def apply_rir_hpf(rir: Tensor, fs: float, cfg: SimulationConfig) -> Tensor:
    """Apply pyroomacoustics-style IIR high-pass filtering to RIRs."""
    if not cfg.rir_hpf_enable:
        return rir

    try:
        from scipy.signal import sosfiltfilt
    except ImportError as exc:
        raise ImportError(
            "scipy is required when rir_hpf_enable=True. "
            "Install scipy or disable the HPF."
        ) from exc

    sos = _design_hpf_sos(fs, cfg.rir_hpf_fc, cfg.rir_hpf_kwargs)
    rir_np = rir.detach().cpu().to(torch.float64).numpy()
    filtered = sosfiltfilt(sos, rir_np, axis=-1)
    filtered = np.ascontiguousarray(filtered)
    return torch.as_tensor(filtered, device=rir.device, dtype=rir.dtype)
