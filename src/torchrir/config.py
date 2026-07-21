"""Configuration objects for torchrir."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Mapping, Optional

import torch


@dataclass(frozen=True)
class RIRHighPassConfig:
    """High-pass filter settings applied to simulated RIRs."""

    enabled: bool = True
    cutoff_hz: float = 10.0
    order: int = 2
    rp: float = 5.0
    rs: float = 60.0
    filter_type: str = "butter"

    def validate(self) -> None:
        if not math.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0:
            raise ValueError("RIR high-pass cutoff_hz must be positive")
        if self.order <= 0:
            raise ValueError("RIR high-pass order must be positive")
        if not math.isfinite(self.rp) or self.rp < 0:
            raise ValueError("RIR high-pass rp must be non-negative")
        if not math.isfinite(self.rs) or self.rs < 0:
            raise ValueError("RIR high-pass rs must be non-negative")
        if not self.filter_type:
            raise ValueError("RIR high-pass filter_type must be non-empty")

    def as_scipy_kwargs(self) -> dict[str, float | int | str]:
        return {
            "n": self.order,
            "rp": self.rp,
            "rs": self.rs,
            "type": self.filter_type,
        }


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration values for RIR simulation and convolution.

    Examples:
        ```python
        cfg = SimulationConfig(max_order=6, tmax=0.3, device="auto")
        cfg.validate()
        ```
    """

    fs: Optional[float] = None
    max_order: Optional[int] = None
    tmax: Optional[float] = None
    nsample: Optional[int] = None
    tdiff: Optional[float] = None
    directivity: Optional[str | tuple[str, str]] = None
    device: Optional[torch.device | str] = None
    dtype: Optional[torch.dtype] = None
    nb_img: Optional[torch.Tensor | tuple[int, ...]] = None
    seed: Optional[int] = None
    use_lut: bool = True
    mixed_precision: bool = False
    frac_delay_length: int = 81
    sinc_lut_granularity: int = 20
    image_chunk_size: int = 2048
    accumulate_chunk_size: int = 4096
    use_compile: bool = False
    rir_hpf_enable: bool = True
    rir_hpf_fc: float = 10.0
    rir_hpf_kwargs: Mapping[str, float | int | str] = field(
        default_factory=lambda: {"n": 2, "rp": 5.0, "rs": 60.0, "type": "butter"}
    )
    rir_hpf: Optional[RIRHighPassConfig] = None

    def __post_init__(self) -> None:
        # A frozen dataclass is not useful when it exposes a mutable dictionary.
        # Copy the legacy mapping and expose a read-only view instead.
        object.__setattr__(
            self, "rir_hpf_kwargs", MappingProxyType(dict(self.rir_hpf_kwargs))
        )
        self.validate()

    def validate(self) -> None:
        """Validate configuration values."""
        if self.fs is not None and (not math.isfinite(self.fs) or self.fs <= 0):
            raise ValueError("fs must be positive")
        if self.max_order is not None and self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.tmax is not None and (not math.isfinite(self.tmax) or self.tmax <= 0):
            raise ValueError("tmax must be positive")
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError("nsample must be positive")
        if self.tmax is not None and self.nsample is not None:
            raise ValueError("tmax and nsample are mutually exclusive")
        if self.tdiff is not None and (not math.isfinite(self.tdiff) or self.tdiff < 0):
            raise ValueError("tdiff must be non-negative")
        if self.tdiff is not None and self.tmax is not None and self.tdiff >= self.tmax:
            raise ValueError("tdiff must be smaller than tmax")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if (
            self.dtype is not None
            and not torch.empty((), dtype=self.dtype).is_floating_point()
        ):
            raise TypeError("dtype must be a real floating-point dtype")
        if self.frac_delay_length <= 0 or self.frac_delay_length % 2 == 0:
            raise ValueError("frac_delay_length must be a positive odd integer")
        if self.sinc_lut_granularity <= 0:
            raise ValueError("sinc_lut_granularity must be positive")
        if self.image_chunk_size <= 0:
            raise ValueError("image_chunk_size must be positive")
        if self.accumulate_chunk_size <= 0:
            raise ValueError("accumulate_chunk_size must be positive")
        if not math.isfinite(self.rir_hpf_fc) or self.rir_hpf_fc <= 0:
            raise ValueError("rir_hpf_fc must be positive")
        if "n" in self.rir_hpf_kwargs and int(self.rir_hpf_kwargs["n"]) <= 0:
            raise ValueError("rir_hpf_kwargs['n'] must be positive")
        self.high_pass.validate()

    def replace(self, **kwargs) -> "SimulationConfig":
        """Return a new config with updated fields."""
        new_cfg = replace(self, **kwargs)
        new_cfg.validate()
        return new_cfg

    @property
    def high_pass(self) -> RIRHighPassConfig:
        """Return normalized high-pass settings."""

        if self.rir_hpf is not None:
            return self.rir_hpf
        return RIRHighPassConfig(
            enabled=self.rir_hpf_enable,
            cutoff_hz=self.rir_hpf_fc,
            order=int(self.rir_hpf_kwargs.get("n", 2)),
            rp=float(self.rir_hpf_kwargs.get("rp", 5.0)),
            rs=float(self.rir_hpf_kwargs.get("rs", 60.0)),
            filter_type=str(self.rir_hpf_kwargs.get("type", "butter")),
        )


@dataclass(frozen=True)
class ResolvedSimulationConfig:
    """Effective, fully resolved settings used for one simulation."""

    fs: float
    max_order: int
    nsample: int
    tmax: float
    tdiff: float | None
    directivity: str | tuple[str, str]
    device: torch.device
    dtype: torch.dtype
    nb_img: torch.Tensor | tuple[int, ...] | None
    seed: int | None
    use_lut: bool
    frac_delay_length: int
    sinc_lut_granularity: int
    image_chunk_size: int
    accumulate_chunk_size: int
    use_compile: bool
    high_pass: RIRHighPassConfig

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        *,
        fs: float,
        max_order: int,
        nsample: int,
        directivity: str | tuple[str, str],
        device: torch.device,
        dtype: torch.dtype,
        tdiff: float | None = None,
    ) -> "ResolvedSimulationConfig":
        return cls(
            fs=float(fs),
            max_order=int(max_order),
            nsample=int(nsample),
            tmax=float(nsample) / float(fs),
            tdiff=tdiff,
            directivity=directivity,
            device=device,
            dtype=dtype,
            nb_img=config.nb_img,
            seed=config.seed,
            use_lut=config.use_lut,
            frac_delay_length=config.frac_delay_length,
            sinc_lut_granularity=config.sinc_lut_granularity,
            image_chunk_size=config.image_chunk_size,
            accumulate_chunk_size=config.accumulate_chunk_size,
            use_compile=config.use_compile,
            high_pass=config.high_pass,
        )


def default_config() -> SimulationConfig:
    """Return the default simulation configuration.

    Examples:
        ```python
        cfg = default_config()
        ```
    """
    cfg = SimulationConfig()
    cfg.validate()
    return cfg


__all__ = [
    "RIRHighPassConfig",
    "ResolvedSimulationConfig",
    "SimulationConfig",
    "default_config",
]
