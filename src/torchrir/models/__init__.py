"""Core data models for rooms, sources, microphones, scenes, and results.

Example:
    >>> from torchrir import RIRResult, Scene
    >>> scene = Scene(room=room, sources=sources, mics=mics, src_traj=src_traj, mic_traj=mic_traj)
    >>> scene.validate()
    >>> result = RIRResult(rirs=rirs, scene=scene, config=config)
"""

from .results import RIRResult
from .room import MicrophoneArray, Room, Source
from .scene import Scene

__all__ = [
    "MicrophoneArray",
    "Room",
    "RIRResult",
    "Scene",
    "Source",
]
