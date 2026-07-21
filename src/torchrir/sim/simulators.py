"""Simulation strategy interfaces and the scene-oriented public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import warnings

import torch

from ..config import ResolvedSimulationConfig, SimulationConfig, default_config
from ..models import DynamicScene, RIRResult, Scene, SceneLike, StaticScene
from .ism import simulate_dynamic_rir, simulate_rir


class RIRSimulator(Protocol):
    """Strategy interface for RIR simulation backends."""

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        """Run a simulation and return the result."""


@dataclass(frozen=True)
class ISMSimulator:
    """Image-source-method backend.

    New code should construct an empty backend and put all simulation settings
    in :class:`SimulationConfig`. Constructor settings remain available as a
    compatibility layer until TorchRIR 1.0.
    """

    max_order: int | None = None
    tmax: float | None = None
    nsample: int | None = None
    tdiff: float | None = None
    directivity: str | tuple[str, str] | None = None
    nb_img: torch.Tensor | tuple[int, ...] | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None

    def __post_init__(self) -> None:
        supplied = any(
            value is not None
            for value in (
                self.max_order,
                self.tmax,
                self.nsample,
                self.tdiff,
                self.directivity,
                self.nb_img,
                self.device,
                self.dtype,
            )
        )
        if supplied:
            warnings.warn(
                "ISMSimulator constructor settings are deprecated and will be "
                "removed in TorchRIR 1.0. Pass SimulationConfig to simulate().",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.max_order is not None and self.max_order < 0:
            raise ValueError("max_order must be non-negative")
        if self.tmax is not None and self.tmax <= 0:
            raise ValueError("tmax must be positive")
        if self.nsample is not None and self.nsample <= 0:
            raise ValueError("nsample must be positive")
        if self.tmax is not None and self.nsample is not None:
            raise ValueError("tmax and nsample are mutually exclusive")

    def simulate(
        self, scene: SceneLike, config: SimulationConfig | None = None
    ) -> RIRResult:
        normalized_scene = _normalize_scene(scene)
        normalized_scene.validate()
        cfg = self._merge_legacy_settings(config or default_config())

        if isinstance(normalized_scene, DynamicScene):
            orientation = _scene_orientation(normalized_scene)
            rirs = simulate_dynamic_rir(
                room=normalized_scene.room,
                src_traj=normalized_scene.src_traj,
                mic_traj=normalized_scene.mic_traj,
                orientation=orientation,
                config=cfg,
            )
        else:
            rirs = simulate_rir(
                room=normalized_scene.room,
                sources=normalized_scene.sources,
                mics=normalized_scene.mics,
                config=cfg,
            )

        assert cfg.max_order is not None
        effective = ResolvedSimulationConfig.from_config(
            cfg,
            fs=normalized_scene.room.fs,
            max_order=cfg.max_order,
            nsample=int(rirs.shape[-1]),
            directivity=cfg.directivity or "omni",
            device=rirs.device,
            dtype=rirs.dtype,
            tdiff=cfg.tdiff,
        )
        timestamps = (
            normalized_scene.timestamps
            if isinstance(normalized_scene, DynamicScene)
            else None
        )
        return RIRResult(
            rirs=rirs,
            scene=normalized_scene,
            config=effective,
            timestamps=timestamps,
            seed=cfg.seed,
            backend="ism",
        )

    def _merge_legacy_settings(self, config: SimulationConfig) -> SimulationConfig:
        updates: dict[str, object] = {}
        for field in (
            "max_order",
            "tmax",
            "nsample",
            "tdiff",
            "directivity",
            "nb_img",
            "device",
            "dtype",
        ):
            legacy_value = getattr(self, field)
            if legacy_value is None:
                continue
            config_value = getattr(config, field)
            if config_value is not None and not _settings_equal(
                legacy_value, config_value
            ):
                raise ValueError(
                    f"conflicting '{field}' values: ISMSimulator has "
                    f"{legacy_value}, config has {config_value}"
                )
            updates[field] = legacy_value
        return config.replace(**updates) if updates else config


def simulate(
    scene: SceneLike,
    config: SimulationConfig,
    *,
    backend: RIRSimulator | None = None,
) -> RIRResult:
    """Simulate a scene with a backend and return RIRs plus effective metadata."""

    simulator = ISMSimulator() if backend is None else backend
    return simulator.simulate(scene, config)


def _scene_orientation(
    scene: DynamicScene,
) -> torch.Tensor | tuple[torch.Tensor | None, torch.Tensor | None] | None:
    src_orientation = scene.sources.orientation
    mic_orientation = scene.mics.orientation
    if src_orientation is None and mic_orientation is None:
        return None
    return src_orientation, mic_orientation


def _normalize_scene(scene: SceneLike) -> StaticScene | DynamicScene:
    if isinstance(scene, (StaticScene, DynamicScene)):
        return scene
    if isinstance(scene, Scene):
        warnings.warn(
            "Passing Scene to ISMSimulator is deprecated and will be removed in "
            "TorchRIR 1.0. Use StaticScene or DynamicScene.",
            DeprecationWarning,
            stacklevel=3,
        )
        if scene.is_dynamic():
            return scene.to_dynamic_scene()
        return scene.to_static_scene()
    raise TypeError("scene must be StaticScene, DynamicScene, or Scene")


def _settings_equal(left: object, right: object) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.equal(torch.as_tensor(left), torch.as_tensor(right))
    if isinstance(left, (str, torch.device)) and isinstance(right, (str, torch.device)):
        if left == "auto" or right == "auto":
            return left == right
        try:
            return torch.device(left) == torch.device(right)
        except (RuntimeError, TypeError):
            pass
    return left == right


__all__ = ["ISMSimulator", "RIRSimulator", "simulate"]
