"""Scene-oriented public simulation entry point."""

from __future__ import annotations

from ..config import SimulationConfig, _resolve_simulation_config
from ..models import DynamicScene, RIRResult, StaticScene
from .ism.api import _simulate_dynamic_rir, _simulate_static_rir


def simulate(
    scene: StaticScene | DynamicScene,
    config: SimulationConfig,
) -> RIRResult:
    """Simulate a static or dynamic scene with the image-source method."""

    if not isinstance(scene, (StaticScene, DynamicScene)):
        raise TypeError("scene must be StaticScene or DynamicScene")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be SimulationConfig")
    scene.validate()

    if isinstance(scene, DynamicScene):
        tensor_values = (scene.src_traj, scene.mic_traj, scene.room.size)
    else:
        tensor_values = (
            scene.sources.positions,
            scene.mics.positions,
            scene.room.size,
        )
    resolved = _resolve_simulation_config(
        config,
        fs=scene.room.fs,
        room_dimension=int(scene.room.size.numel()),
        tensor_values=tensor_values,
    )

    if isinstance(scene, DynamicScene):
        rirs = _simulate_dynamic_rir(scene, resolved)
    else:
        rirs = _simulate_static_rir(scene, resolved)
    return RIRResult(
        rirs=rirs,
        scene=scene,
        config=resolved,
    )


__all__ = ["simulate"]
