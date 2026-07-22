"""Core data models for rooms, sources, microphones, scenes, and results.

Examples:
    ```python
    from torchrir import DynamicScene
    from torchrir.config import SimulationConfig
    from torchrir.sim import simulate
    scene = DynamicScene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    result = simulate(scene, SimulationConfig(max_order=4, tmax=0.3))
    ```
"""

from .results import RIRResult
from .room import MicrophoneArray, Room, Source
from .scene import DynamicScene, StaticScene

__all__ = [
    "DynamicScene",
    "MicrophoneArray",
    "Room",
    "RIRResult",
    "StaticScene",
    "Source",
]
