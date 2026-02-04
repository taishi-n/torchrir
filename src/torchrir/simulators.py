from __future__ import annotations

"""Simulation strategy interfaces and implementations.

Note:
    RayTracingSimulator and FDTDSimulator are work in progress placeholders.
"""

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
    """Work in progress placeholder for ray tracing simulation.

    Goal:
        Provide a geometric acoustics backend that traces specular/diffuse
        reflection paths, supports frequency-dependent absorption/scattering,
        and returns a RIRResult compatible with the ISM path. The intent is to
        reuse Scene/SimulationConfig for inputs and keep output shape parity.
    """

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
        raise NotImplementedError("RayTracingSimulator is not implemented yet")


@dataclass(frozen=True)
class FDTDSimulator:
    """Work in progress placeholder for FDTD simulation.

    Goal:
        Provide a wave-based solver (finite-difference time-domain) with
        configurable grid resolution, boundary conditions, and stability
        constraints. The solver should target CPU/GPU execution and return
        RIRResult with the same metadata contract as ISM.
    """

    def simulate(self, scene: Scene, config: SimulationConfig | None = None) -> RIRResult:
        raise NotImplementedError("FDTDSimulator is not implemented yet")
