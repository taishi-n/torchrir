"""TorchRIR public API."""

from .config import (
    activate_lut,
    activate_compile,
    activate_mixed_precision,
    get_config,
    set_accumulate_chunk_size,
    set_frac_delay_length,
    set_image_chunk_size,
    set_sinc_lut_granularity,
)
from .core import simulate_dynamic_rir, simulate_rir
from .dynamic import DynamicConvolver
from .plotting import plot_scene_dynamic, plot_scene_static
from .plotting_utils import plot_scene_and_save
from .room import MicrophoneArray, Room, Source
from .signal import convolve_dynamic_rir, convolve_rir, dynamic_convolve, fft_convolve
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
    "activate_compile",
    "activate_mixed_precision",
    "att2t_SabineEstimation",
    "att2t_sabine_estimation",
    "beta_SabineEstimation",
    "BaseDataset",
    "CmuArcticDataset",
    "CmuArcticSentence",
    "choose_speakers",
    "DynamicConvolver",
    "estimate_beta_from_t60",
    "estimate_t60_from_beta",
    "fft_convolve",
    "get_config",
    "list_cmu_arctic_speakers",
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
    "set_accumulate_chunk_size",
    "set_frac_delay_length",
    "set_image_chunk_size",
    "set_sinc_lut_granularity",
    "simulate_dynamic_rir",
    "simulate_rir",
    "t2n",
]
