"""Simulation strategy interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import warnings

import torch

from ..config import SimulationConfig, default_config
from .ism import simulate_dynamic_rir, simulate_rir
from ..models import DynamicScene, RIRResult, Scene, SceneLike, StaticScene


class RIRSimulator(Protocol):
    """Strategy interface for RIR simulation backends."""

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        """Run a simulation and return the result."""


@dataclass(frozen=True)
class ISMSimulator:
    """ISM-based simulator using the current core implementation.

    Examples:
        ```python
        result = ISMSimulator(max_order=6, tmax=0.3).simulate(scene, config)
        ```
    """

    max_order: int
    tmax: float | None = None
    nsample: int | None = None
    directivity: str | tuple[str, str] | None = "omni"
    nb_img: torch.Tensor | tuple[int, ...] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        if self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.tmax is None and self.nsample is None:
            raise ValueError("tmax or nsample must be provided")
        if self.tmax is not None and self.tmax <= 0:
            raise ValueError("tmax must be positive")
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError("nsample must be positive")

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        normalized_scene = _normalize_scene(scene)
        normalized_scene.validate()
        cfg = config or default_config()
        _ensure_no_conflict(
            field="max_order",
            simulator_value=self.max_order,
            config_value=cfg.max_order,
        )
        _ensure_no_conflict(
            field="tmax",
            simulator_value=self.tmax,
            config_value=cfg.tmax,
        )
        if isinstance(normalized_scene, DynamicScene):
            rirs = simulate_dynamic_rir(
                room=normalized_scene.room,
                src_traj=normalized_scene.src_traj,
                mic_traj=normalized_scene.mic_traj,
                max_order=self.max_order,
                nb_img=self.nb_img,
                nsample=self.nsample,
                tmax=self.tmax,
                directivity=self.directivity,
                config=cfg,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            rirs = simulate_rir(
                room=normalized_scene.room,
                sources=normalized_scene.sources,
                mics=normalized_scene.mics,
                max_order=self.max_order,
                nb_img=self.nb_img,
                nsample=self.nsample,
                tmax=self.tmax,
                directivity=self.directivity,
                config=cfg,
                device=self.device,
                dtype=self.dtype,
            )
        return RIRResult(rirs=rirs, scene=normalized_scene, config=cfg, seed=cfg.seed)


def _normalize_scene(scene: SceneLike) -> StaticScene | DynamicScene:
    if isinstance(scene, (StaticScene, DynamicScene)):
        return scene
    if isinstance(scene, Scene):
        warnings.warn(
            "Passing Scene to ISMSimulator is deprecated. "
            "Use StaticScene or DynamicScene.",
            DeprecationWarning,
            stacklevel=3,
        )
        if scene.is_dynamic():
            return scene.to_dynamic_scene()
        return scene.to_static_scene()
    raise TypeError("scene must be StaticScene, DynamicScene, or Scene")


def _ensure_no_conflict(
    *,
    field: str,
    simulator_value: int | float | None,
    config_value: int | float | None,
) -> None:
    if simulator_value is None or config_value is None:
        return
    if simulator_value != config_value:
        raise ValueError(
            f"conflicting '{field}' values: "
            f"ISMSimulator has {simulator_value}, config has {config_value}"
        )
