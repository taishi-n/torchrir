"""Result containers for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..config import ResolvedSimulationConfig
from ..util._dtypes import validate_supported_float_tensor
from .scene import DynamicScene, StaticScene


@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
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
    scene: StaticScene | DynamicScene
    config: ResolvedSimulationConfig
    _scene_tensor_ids: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _scene_tensor_snapshots: tuple[Tensor, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        self._validate_invariants()
        tensors = _scene_tensors(self.scene)
        object.__setattr__(self, "_scene_tensor_ids", tuple(map(id, tensors)))
        object.__setattr__(
            self,
            "_scene_tensor_snapshots",
            tuple(tensor.detach().clone() for tensor in tensors),
        )

    def validate(self) -> None:
        """Revalidate this shallow-immutable result at a consumer boundary."""

        self._validate_invariants()
        current_tensors = _scene_tensors(self.scene)
        modified = len(current_tensors) != len(self._scene_tensor_ids)
        if not modified:
            for current, identity, snapshot in zip(
                current_tensors,
                self._scene_tensor_ids,
                self._scene_tensor_snapshots,
                strict=True,
            ):
                if id(current) != identity or not torch.equal(current, snapshot):
                    modified = True
                    break
        if modified:
            raise ValueError(
                "scene tensors were modified after this RIRResult was created"
            )

    def _validate_invariants(self) -> None:
        if not isinstance(self.scene, (StaticScene, DynamicScene)):
            raise TypeError("scene must be StaticScene or DynamicScene")
        self.scene.validate()
        if not isinstance(self.config, ResolvedSimulationConfig):
            raise TypeError("config must be ResolvedSimulationConfig")
        if not torch.is_tensor(self.rirs):
            raise TypeError("rirs must be a Tensor")
        is_dynamic = isinstance(self.scene, DynamicScene)
        expected_ndim = 4 if is_dynamic else 3
        if self.rirs.ndim != expected_ndim:
            raise ValueError(
                f"rirs must be {expected_ndim}D for {type(self.scene).__name__}, "
                f"got shape {tuple(self.rirs.shape)}"
            )
        validate_supported_float_tensor(self.rirs, name="rirs")
        if not torch.all(torch.isfinite(self.rirs)):
            raise ValueError("rirs must contain finite values")
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
        if self.rirs.shape[-1] != self.config.nsample:
            raise ValueError(
                f"rirs has {self.rirs.shape[-1]} samples, but config resolves "
                f"to {self.config.nsample}"
            )
        if self.scene.room.fs != self.config.fs:
            raise ValueError(
                f"scene sampling rate {self.scene.room.fs} conflicts with "
                f"config sampling rate {self.config.fs}"
            )
        if self.config.nb_img is not None and len(self.config.nb_img) != int(
            self.scene.room.size.numel()
        ):
            raise ValueError("config nb_img dimension conflicts with the scene room")
        if self.rirs.device != self.config.device:
            raise ValueError(
                f"rirs device {self.rirs.device} conflicts with config device "
                f"{self.config.device}"
            )
        if self.rirs.dtype != self.config.dtype:
            raise ValueError(
                f"rirs dtype {self.rirs.dtype} conflicts with config dtype "
                f"{self.config.dtype}"
            )
        if is_dynamic:
            assert isinstance(self.scene, DynamicScene)
            time_steps = int(self.scene.src_traj.shape[0])
            if self.rirs.shape[0] != time_steps:
                raise ValueError(
                    f"rirs has {self.rirs.shape[0]} frames, but scene has {time_steps}"
                )


def _scene_tensors(scene: StaticScene | DynamicScene) -> tuple[Tensor, ...]:
    tensors = [
        scene.room.size,
        scene.sources.positions,
        scene.mics.positions,
    ]
    if scene.room.beta is not None:
        tensors.append(scene.room.beta)
    if scene.sources.orientation is not None:
        tensors.append(scene.sources.orientation)
    if scene.mics.orientation is not None:
        tensors.append(scene.mics.orientation)
    if isinstance(scene, DynamicScene):
        tensors.extend((scene.src_traj, scene.mic_traj))
    return tuple(tensors)
