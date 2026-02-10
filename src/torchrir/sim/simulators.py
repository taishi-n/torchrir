"""Simulation strategy interfaces and implementations."""

from __future__ import annotations

from typing import Protocol

from ..config import SimulationConfig, default_config
from .ism import simulate_dynamic_rir, simulate_rir
from ..models import RIRResult, Scene


class RIRSimulator(Protocol):
    """Strategy interface for RIR simulation backends."""

    def simulate(
        self, scene: Scene, config: SimulationConfig | None = None
    ) -> RIRResult:
        """Run a simulation and return the result."""


class ISMSimulator:
    """ISM-based simulator using the current core implementation.

    Example:
        >>> result = ISMSimulator().simulate(scene, config)
    """

    def simulate(
        self, scene: Scene, config: SimulationConfig | None = None
    ) -> RIRResult:
        scene.validate()
        cfg = config or default_config()
        if scene.is_dynamic():
            if scene.src_traj is None or scene.mic_traj is None:
                raise ValueError("dynamic scene requires both src_traj and mic_traj")
            rirs = simulate_dynamic_rir(
                room=scene.room,
                src_traj=scene.src_traj,
                mic_traj=scene.mic_traj,
                max_order=None,
                nsample=None,
                tmax=None,
                directivity=None,
                config=cfg,
            )
        else:
            rirs = simulate_rir(
                room=scene.room,
                sources=scene.sources,
                mics=scene.mics,
                max_order=None,
                nsample=None,
                tmax=None,
                directivity=None,
                config=cfg,
            )
        return RIRResult(rirs=rirs, scene=scene, config=cfg, seed=cfg.seed)
