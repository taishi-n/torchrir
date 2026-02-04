"""TorchRIR public API."""

from .config import (
    activate_lut,
    activate_mixed_precision,
    get_config,
    set_frac_delay_length,
    set_sinc_lut_granularity,
)
from .core import simulate_dynamic_rir, simulate_rir
from .plotting import plot_scene_dynamic, plot_scene_static
from .plotting_utils import plot_scene_and_save
from .room import MicrophoneArray, Room, Source
from .signal import convolve_dynamic_rir, convolve_rir, dynamic_convolve, fft_convolve
from .datasets import (
    CmuArcticDataset,
    CmuArcticSentence,
    list_cmu_arctic_speakers,
    load_wav_mono,
    save_wav,
)
from .utils import (
    att2t_SabineEstimation,
    att2t_sabine_estimation,
    beta_SabineEstimation,
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    resolve_device,
    t2n,
)

__all__ = [
    "MicrophoneArray",
    "Room",
    "Source",
    "convolve_dynamic_rir",
    "convolve_rir",
    "dynamic_convolve",
    "activate_lut",
    "activate_mixed_precision",
    "att2t_SabineEstimation",
    "att2t_sabine_estimation",
    "beta_SabineEstimation",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "estimate_beta_from_t60",
    "estimate_t60_from_beta",
    "fft_convolve",
    "get_config",
    "list_cmu_arctic_speakers",
    "resolve_device",
    "load_wav_mono",
    "plot_scene_dynamic",
    "plot_scene_and_save",
    "plot_scene_static",
    "save_wav",
    "set_frac_delay_length",
    "set_sinc_lut_granularity",
    "simulate_dynamic_rir",
    "simulate_rir",
    "t2n",
]
