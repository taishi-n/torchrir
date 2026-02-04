from __future__ import annotations

"""Simulation configuration for torchrir."""

from dataclasses import dataclass, replace
from typing import Optional

import torch


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration values for RIR simulation and convolution.

    Example:
        >>> cfg = SimulationConfig(max_order=6, tmax=0.3, device="auto")
        >>> cfg.validate()
    """

    fs: Optional[float] = None
    max_order: Optional[int] = None
    tmax: Optional[float] = None
    directivity: Optional[str | tuple[str, str]] = None
    device: Optional[torch.device | str] = None
    seed: Optional[int] = None
    use_lut: bool = True
    mixed_precision: bool = False
    frac_delay_length: int = 81
    sinc_lut_granularity: int = 20
    image_chunk_size: int = 2048
    accumulate_chunk_size: int = 4096
    use_compile: bool = False

    def validate(self) -> None:
        """Validate configuration values."""
        if self.fs is not None and self.fs <= 0:
            raise ValueError("fs must be positive")
        if self.max_order is not None and self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.tmax is not None and self.tmax <= 0:
            raise ValueError("tmax must be positive")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.frac_delay_length <= 0 or self.frac_delay_length % 2 == 0:
            raise ValueError("frac_delay_length must be a positive odd integer")
        if self.sinc_lut_granularity <= 0:
            raise ValueError("sinc_lut_granularity must be positive")
        if self.image_chunk_size <= 0:
            raise ValueError("image_chunk_size must be positive")
        if self.accumulate_chunk_size <= 0:
            raise ValueError("accumulate_chunk_size must be positive")

    def replace(self, **kwargs) -> "SimulationConfig":
        """Return a new config with updated fields."""
        new_cfg = replace(self, **kwargs)
        new_cfg.validate()
        return new_cfg


def default_config() -> SimulationConfig:
    """Return the default simulation configuration.

    Example:
        >>> cfg = default_config()
    """
    cfg = SimulationConfig()
    cfg.validate()
    return cfg
