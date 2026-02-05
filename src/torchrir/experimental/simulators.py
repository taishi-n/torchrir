from __future__ import annotations

"""Experimental simulation backends (placeholders)."""

from dataclasses import dataclass

from ..models import RIRResult, Scene
from ..sim.config import SimulationConfig


@dataclass(frozen=True)
class RayTracingSimulator:
    """Work in progress placeholder for ray tracing simulation.

    Goal:
        Provide a geometric acoustics backend that traces specular/diffuse
        reflection paths, supports frequency-dependent absorption/scattering,
        and returns a RIRResult compatible with the ISM path. The intent is to
        reuse Scene/SimulationConfig for inputs and keep output shape parity.
    """

    def simulate(
        self, scene: Scene, config: SimulationConfig | None = None
    ) -> RIRResult:
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

    def simulate(
        self, scene: Scene, config: SimulationConfig | None = None
    ) -> RIRResult:
        raise NotImplementedError("FDTDSimulator is not implemented yet")
