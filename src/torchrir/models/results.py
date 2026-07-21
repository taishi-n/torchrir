"""Result containers for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

from .scene import DynamicScene, Scene, SceneLike

if TYPE_CHECKING:
    from ..config import ResolvedSimulationConfig, SimulationConfig


@dataclass(frozen=True)
class RIRResult:
    """Container for RIRs with metadata.

    Examples:
        ```python
        from torchrir.config import SimulationConfig
        from torchrir.sim import simulate

        config = SimulationConfig(max_order=6, tmax=0.3)
        result = simulate(scene, config)
        rirs = result.rirs
        ```
    """

    rirs: Tensor
    scene: SceneLike
    config: "SimulationConfig | ResolvedSimulationConfig"
    timestamps: Optional[Tensor] = None
    seed: Optional[int] = None
    backend: str = "ism"

    def __post_init__(self) -> None:
        is_dynamic = isinstance(self.scene, DynamicScene) or (
            isinstance(self.scene, Scene) and self.scene.is_dynamic()
        )
        expected_ndim = 4 if is_dynamic else 3
        if self.rirs.ndim != expected_ndim:
            raise ValueError(
                f"rirs must be {expected_ndim}D for {type(self.scene).__name__}, "
                f"got shape {tuple(self.rirs.shape)}"
            )
        if not self.rirs.is_floating_point():
            raise TypeError("rirs must use a real floating-point dtype")
        n_sources = int(self.scene.sources.positions.shape[0])
        n_mics = int(self.scene.mics.positions.shape[0])
        source_axis = 1 if is_dynamic else 0
        mic_axis = 2 if is_dynamic else 1
        if self.rirs.shape[source_axis] != n_sources:
            raise ValueError(
                f"rirs has {self.rirs.shape[source_axis]} sources, "
                f"but scene has {n_sources}"
            )
        if self.rirs.shape[mic_axis] != n_mics:
            raise ValueError(
                f"rirs has {self.rirs.shape[mic_axis]} microphones, "
                f"but scene has {n_mics}"
            )
        if self.rirs.shape[-1] == 0:
            raise ValueError("rirs must contain at least one sample")
        if is_dynamic:
            if isinstance(self.scene, DynamicScene):
                time_steps = int(self.scene.src_traj.shape[0])
            else:
                assert isinstance(self.scene, Scene)
                assert self.scene.src_traj is not None
                time_steps = int(self.scene.src_traj.shape[0])
            if self.rirs.shape[0] != time_steps:
                raise ValueError(
                    f"rirs has {self.rirs.shape[0]} frames, but scene has {time_steps}"
                )
        if not self.backend:
            raise ValueError("backend must be non-empty")
        timestamps = self.timestamps
        if timestamps is None and isinstance(self.scene, DynamicScene):
            timestamps = self.scene.timestamps
            object.__setattr__(self, "timestamps", timestamps)
        if timestamps is not None:
            if not is_dynamic:
                raise ValueError("timestamps are only valid for dynamic RIR results")
            if not torch.is_tensor(timestamps):
                raise TypeError("timestamps must be a Tensor")
            if timestamps.ndim != 1 or timestamps.numel() != self.rirs.shape[0]:
                raise ValueError("timestamps must match the dynamic RIR frame count")
            if timestamps.is_complex() or not torch.all(torch.isfinite(timestamps)):
                raise ValueError("timestamps must contain finite real values")
            if timestamps[0].item() != 0.0:
                raise ValueError("first timestamp must be 0")
            if timestamps.numel() > 1 and torch.any(timestamps[1:] <= timestamps[:-1]):
                raise ValueError("timestamps must be strictly increasing")
