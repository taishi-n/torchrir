"""Result containers for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from torch import Tensor

from .scene import SceneLike

if TYPE_CHECKING:
    from ..config import SimulationConfig


@dataclass(frozen=True)
class RIRResult:
    """Container for RIRs with metadata.

    Examples:
        ```python
        from torchrir.sim import ISMSimulator
        result = ISMSimulator(max_order=6, tmax=0.3).simulate(scene, config)
        rirs = result.rirs
        ```
    """

    rirs: Tensor
    scene: SceneLike
    config: "SimulationConfig"
    timestamps: Optional[Tensor] = None
    seed: Optional[int] = None
