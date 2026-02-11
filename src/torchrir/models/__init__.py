"""Core data models for rooms, sources, microphones, scenes, and results.

Examples:
    ```python
    from torchrir import DynamicScene, RIRResult
    scene = DynamicScene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    result = RIRResult(rirs=rirs, scene=scene, config=config)
    ```
"""

from .results import RIRResult
from .room import MicrophoneArray, Room, Source
from .scene import DynamicScene, Scene, SceneLike, StaticScene

__all__ = [
    "DynamicScene",
    "MicrophoneArray",
    "Room",
    "RIRResult",
    "Scene",
    "SceneLike",
    "StaticScene",
    "Source",
]
