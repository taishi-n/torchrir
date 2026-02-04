from __future__ import annotations

"""Simulation strategy interfaces and implementations."""

from dataclasses import dataclass
from typing import Protocol

from .config import SimulationConfig, default_config
from .core import simulate_dynamic_rir, simulate_rir
from .results import RIRResult
from .scene import Scene


class RIRSimulator(Protocol):
    """Strategy interface for RIR simulation backends."""

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
        """Run a simulation and return the result."""


@dataclass(frozen=True)
class ISMSimulator:
    """ISM-based simulator using the current core implementation."""

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
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


@dataclass(frozen=True)
class RayTracingSimulator:
    """Placeholder for future ray tracing simulation."""

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
        raise NotImplementedError("RayTracingSimulator is not implemented yet")


@dataclass(frozen=True)
class FDTDSimulator:
    """Placeholder for future FDTD simulation."""

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
        raise NotImplementedError("FDTDSimulator is not implemented yet")
