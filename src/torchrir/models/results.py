"""Result containers for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from .scene import Scene

if TYPE_CHECKING:
    from ..config import SimulationConfig


@dataclass(frozen=True)
class RIRResult:
    """Container for RIRs with metadata.

    Example:
        >>> from torchrir.sim import ISMSimulator
        >>> result = ISMSimulator().simulate(scene, config)
        >>> rirs = result.rirs
    """

    rirs: Tensor
    scene: Scene
    config: "SimulationConfig"
    timestamps: Optional[Tensor] = None
    seed: Optional[int] = None
