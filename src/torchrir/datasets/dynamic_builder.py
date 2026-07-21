"""Configuration and result models for dynamic dataset generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class DynamicCmuArcticBuildConfig:
    """All inputs needed to build a dynamic CMU ARCTIC dataset."""

    cmu_root: Path
    dataset_root: Path = Path("outputs/cmu_arctic_torchrir_dynamic_dataset")
    speakers: Sequence[str] = ("bdl", "slt", "clb", "rms", "jmk", "awb")
    n_scenes: int = 10
    n_sources: int = 3
    n_moving_sources: int = 1
    duration_sec: float = 20.0
    room_size: Sequence[float] = (8.0, 6.0, 3.0)
    mic_center: Sequence[float] = (4.0, 3.0, 1.5)
    octa_edge_m: float = 1.0
    source_margin: Sequence[float] = (0.5, 0.5, 0.3)
    min_source_distance_m: float = 1.8
    trajectory_steps: int = 1024
    rir_samples: int = 4096
    rt60: float = 0.3
    sound_speed: float = 343.0
    max_order: int = 6
    seed: int = 42
    download_cmu: bool = False
    overwrite: bool = False
    randomize_mic_center: bool = True
    move_start_ratio: float = 0.35
    move_end_ratio: float = 0.65
    moving_speed_min: float = 0.3
    moving_speed_max: float = 0.8
    save_layout_mp4: bool = True
    save_layout_mp4_3d: bool = True
    layout_video_fps: float | None = None
    layout_video_mux_audio: bool = True
    save_layout_images: bool = True
    save_layout_images_3d: bool = True
    annotate_source_indices: bool = True


@dataclass(frozen=True)
class DynamicDatasetBuildResult:
    """Summary of a completed dataset build."""

    dataset_root: Path
    sample_rate: int
    n_mics: int
    n_scenes: int
    scene_dirs: tuple[Path, ...]


__all__ = ["DynamicCmuArcticBuildConfig", "DynamicDatasetBuildResult"]
