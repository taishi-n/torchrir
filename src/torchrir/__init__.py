"""TorchRIR public API."""

from .infra import LoggingConfig, get_logger, setup_logging
from .io import build_metadata, load_wav_mono, save_metadata_json, save_wav
from .models import MicrophoneArray, RIRResult, Room, Scene, Source
from .signal import DynamicConvolver, convolve_rir, fft_convolve
from .sim import SimulationConfig, simulate_dynamic_rir, simulate_rir

__all__ = [
    "Room",
    "Source",
    "MicrophoneArray",
    "Scene",
    "RIRResult",
    "simulate_rir",
    "simulate_dynamic_rir",
    "SimulationConfig",
    "convolve_rir",
    "DynamicConvolver",
    "fft_convolve",
    "get_logger",
    "LoggingConfig",
    "build_metadata",
    "load_wav_mono",
    "save_wav",
    "save_metadata_json",
    "setup_logging",
]
