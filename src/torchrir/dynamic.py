from __future__ import annotations

"""Dynamic convolution utilities.

DynamicConvolver is the public API for time-varying convolution. Lower-level
helpers live in signal.py and are not part of the stable surface.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .signal import _ensure_dynamic_rirs, _ensure_signal


@dataclass(frozen=True)
class DynamicConvolver:
    """Convolver for time-varying RIRs.

    Example:
        >>> convolver = DynamicConvolver(mode="trajectory")
        >>> y = convolver.convolve(signal, rirs)
    """

    mode: str = "trajectory"
    hop: Optional[int] = None
    timestamps: Optional[Tensor] = None
    fs: Optional[float] = None

    def __call__(self, signal: Tensor, rirs: Tensor) -> Tensor:
        return self.convolve(signal, rirs)

    def convolve(self, signal: Tensor, rirs: Tensor) -> Tensor:
        """Convolve signals with time-varying RIRs.

        Example:
            >>> y = DynamicConvolver(mode="hop", hop=1024).convolve(signal, rirs)
        """
        if self.mode not in ("trajectory", "hop"):
            raise ValueError("mode must be 'trajectory' or 'hop'")
        if self.mode == "hop":
            if self.hop is None:
                raise ValueError("hop must be provided for hop mode")
            return _convolve_dynamic_hop(signal, rirs, self.hop)
        return _convolve_dynamic_trajectory(
            signal, rirs, timestamps=self.timestamps, fs=self.fs
        )


def _convolve_dynamic_hop(signal: Tensor, rirs: Tensor, hop: int) -> Tensor:
    from .signal import _convolve_dynamic_rir_hop

    signal = _ensure_signal(signal)
    rirs = _ensure_dynamic_rirs(rirs, signal)
    return _convolve_dynamic_rir_hop(signal, rirs, hop)


def _convolve_dynamic_trajectory(
    signal: Tensor,
    rirs: Tensor,
    *,
    timestamps: Optional[Tensor],
    fs: Optional[float],
) -> Tensor:
    from .signal import _convolve_dynamic_rir_trajectory

    signal = _ensure_signal(signal)
    rirs = _ensure_dynamic_rirs(rirs, signal)
    return _convolve_dynamic_rir_trajectory(signal, rirs, timestamps=timestamps, fs=fs)
