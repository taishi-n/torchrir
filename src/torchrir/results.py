from __future__ import annotations

"""Result containers for simulation outputs."""

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from .config import SimulationConfig
from .scene import Scene


@dataclass(frozen=True)
class RIRResult:
    """Container for RIRs with metadata.

    Example:
        >>> from torchrir import ISMSimulator
        >>> result = ISMSimulator().simulate(scene, config)
        >>> rirs = result.rirs
    """

    rirs: Tensor
    scene: Scene
    config: SimulationConfig
    timestamps: Optional[Tensor] = None
    seed: Optional[int] = None
