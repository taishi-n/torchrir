"""TorchRIR public API."""

from .config import SimulationConfig, default_config
from .core import simulate_dynamic_rir, simulate_rir
from .dynamic import DynamicConvolver
from .logging_utils import LoggingConfig, get_logger, setup_logging
from .plotting import plot_scene_dynamic, plot_scene_static
from .plotting_utils import plot_scene_and_save
from .room import MicrophoneArray, Room, Source
from .scene import Scene
from .results import RIRResult
from .simulators import FDTDSimulator, ISMSimulator, RIRSimulator, RayTracingSimulator
from .signal import convolve_rir, fft_convolve
from .datasets import (
    BaseDataset,
    CmuArcticDataset,
    CmuArcticSentence,
    choose_speakers,
    list_cmu_arctic_speakers,
    SentenceLike,
    load_dataset_sources,
    TemplateDataset,
    TemplateSentence,
    load_wav_mono,
    save_wav,
)
from .scene_utils import binaural_mic_positions, clamp_positions, linear_trajectory, sample_positions
from .utils import (
    att2t_SabineEstimation,
    att2t_sabine_estimation,
    beta_SabineEstimation,
    DeviceSpec,
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    resolve_device,
    t2n,
)

__all__ = [
    "MicrophoneArray",
    "Room",
    "Source",
    "RIRResult",
    "RIRSimulator",
    "ISMSimulator",
    "RayTracingSimulator",
    "FDTDSimulator",
    "convolve_rir",
    "att2t_SabineEstimation",
    "att2t_sabine_estimation",
    "beta_SabineEstimation",
    "DeviceSpec",
    "BaseDataset",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "choose_speakers",
    "DynamicConvolver",
    "estimate_beta_from_t60",
    "estimate_t60_from_beta",
    "fft_convolve",
    "get_logger",
    "list_cmu_arctic_speakers",
    "LoggingConfig",
    "resolve_device",
    "SentenceLike",
    "load_dataset_sources",
    "load_wav_mono",
    "TemplateDataset",
    "TemplateSentence",
    "binaural_mic_positions",
    "clamp_positions",
    "linear_trajectory",
    "sample_positions",
    "plot_scene_dynamic",
    "plot_scene_and_save",
    "plot_scene_static",
    "save_wav",
    "Scene",
    "setup_logging",
    "SimulationConfig",
    "default_config",
    "simulate_dynamic_rir",
    "simulate_rir",
    "t2n",
]
