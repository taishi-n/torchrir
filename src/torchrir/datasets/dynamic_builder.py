"""Configuration and result models for dynamic dataset generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import torch

from ..config import SimulationConfig
from ..util._scalars import (
    normalize_finite_real,
    normalize_integer,
    normalize_sample_rate,
)
from ..util.acoustics import estimate_beta_from_t60


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicCmuArcticBuildConfig:
    """Validated, immutable request accepted by ``build_dynamic_cmu_arctic``.

    Paths are normalized to `pathlib.Path`. Every sequence field is
    copied into a tuple, so mutating a caller-owned list after construction
    cannot change a build request.
    """

    cmu_root: Path | str
    dataset_root: Path | str = Path("outputs/cmu_arctic_torchrir_dynamic_dataset")
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
    trajectory_steps: int = 256
    simulation: SimulationConfig = field(
        default_factory=lambda: SimulationConfig(
            max_order=6,
            nsample=4096,
            dtype=torch.float32,
        )
    )
    rt60: float = 0.3
    sound_speed: float = 343.0
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

    def __post_init__(self) -> None:
        cmu_root = Path(self.cmu_root)
        dataset_root = Path(self.dataset_root)
        object.__setattr__(self, "cmu_root", cmu_root)
        object.__setattr__(self, "dataset_root", dataset_root)
        _validate_non_overlapping_roots(cmu_root, dataset_root)

        speakers = _normalize_speakers(self.speakers)
        room_size = _normalize_triplet(self.room_size, name="room_size")
        mic_center = _normalize_triplet(self.mic_center, name="mic_center")
        source_margin = _normalize_triplet(
            self.source_margin,
            name="source_margin",
        )
        object.__setattr__(self, "speakers", speakers)
        object.__setattr__(self, "room_size", room_size)
        object.__setattr__(self, "mic_center", mic_center)
        object.__setattr__(self, "source_margin", source_margin)

        for name in ("n_scenes", "n_sources", "trajectory_steps"):
            object.__setattr__(
                self,
                name,
                _validate_integer(getattr(self, name), name=name, minimum=1),
            )
        object.__setattr__(
            self,
            "n_moving_sources",
            _validate_integer(
                self.n_moving_sources,
                name="n_moving_sources",
                minimum=0,
            ),
        )
        object.__setattr__(
            self,
            "seed",
            normalize_integer(
                self.seed,
                name="seed",
                minimum=0,
                maximum=torch.iinfo(torch.int64).max,
            ),
        )
        if not isinstance(self.simulation, SimulationConfig):
            raise TypeError("simulation must be a SimulationConfig")

        if self.trajectory_steps < 2:
            raise ValueError("trajectory_steps must be at least 2")
        if self.n_moving_sources > self.n_sources:
            raise ValueError(
                "n_moving_sources must satisfy 0 <= n_moving_sources <= n_sources"
            )
        if self.n_sources > len(speakers):
            raise ValueError(
                f"n_sources ({self.n_sources}) must be <= number of speakers "
                f"({len(speakers)})"
            )
        positive_floats = (
            "duration_sec",
            "octa_edge_m",
            "min_source_distance_m",
            "rt60",
            "sound_speed",
            "moving_speed_min",
            "moving_speed_max",
        )
        for name in positive_floats:
            value = _normalize_finite_float(getattr(self, name), name=name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        for name in ("move_start_ratio", "move_end_ratio"):
            object.__setattr__(
                self,
                name,
                _normalize_finite_float(getattr(self, name), name=name),
            )
        if not (0.0 <= self.move_start_ratio < self.move_end_ratio <= 1.0):
            raise ValueError(
                "motion ratios must satisfy 0 <= move_start_ratio < move_end_ratio <= 1"
            )
        if self.moving_speed_max < self.moving_speed_min:
            raise ValueError(
                "moving speeds must satisfy 0 < moving_speed_min <= moving_speed_max"
            )

        if self.layout_video_fps is not None:
            layout_video_fps = _normalize_finite_float(
                self.layout_video_fps,
                name="layout_video_fps",
            )
            if layout_video_fps <= 0:
                raise ValueError("layout_video_fps must be positive")
            object.__setattr__(self, "layout_video_fps", layout_video_fps)

        for name in (
            "download_cmu",
            "overwrite",
            "randomize_mic_center",
            "save_layout_mp4",
            "save_layout_mp4_3d",
            "layout_video_mux_audio",
            "save_layout_images",
            "save_layout_images_3d",
            "annotate_source_indices",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        if any(value <= 0 for value in room_size):
            raise ValueError("room_size must contain positive values")
        if any(value < 0 for value in source_margin) or any(
            2.0 * margin >= size
            for margin, size in zip(source_margin, room_size, strict=True)
        ):
            raise ValueError("source_margin leaves no feasible room interior")
        if self.simulation.nb_img is not None and len(self.simulation.nb_img) != 3:
            raise ValueError("simulation nb_img must contain three values")
        estimate_beta_from_t60(
            torch.tensor(room_size, dtype=torch.float64),
            self.rt60,
            c=self.sound_speed,
        )

        array_radius = self.octa_edge_m / math.sqrt(2.0)
        if self.randomize_mic_center:
            _validate_random_center_geometry(
                room_size=room_size,
                source_margin=source_margin,
                min_source_distance=self.min_source_distance_m,
                array_radius=array_radius,
            )
        else:
            _validate_fixed_center_geometry(
                room_size=room_size,
                mic_center=mic_center,
                source_margin=source_margin,
                min_source_distance=self.min_source_distance_m,
                array_radius=array_radius,
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class DynamicDatasetBuildResult:
    """Published artifacts and dimensions of a completed dataset build."""

    dataset_root: Path
    sample_rate: int
    n_mics: int
    n_scenes: int
    scene_dirs: tuple[Path, ...]

    def __post_init__(self) -> None:
        root = Path(self.dataset_root)
        scene_dirs = tuple(Path(path) for path in self.scene_dirs)
        object.__setattr__(self, "dataset_root", root)
        object.__setattr__(self, "scene_dirs", scene_dirs)
        object.__setattr__(
            self,
            "sample_rate",
            normalize_sample_rate(self.sample_rate),
        )
        for name in ("n_mics", "n_scenes"):
            object.__setattr__(
                self,
                name,
                _validate_integer(getattr(self, name), name=name, minimum=1),
            )
        if len(scene_dirs) != self.n_scenes:
            raise ValueError("scene_dirs length must equal n_scenes")
        if any(path.parent != root for path in scene_dirs):
            raise ValueError(
                "every scene directory must be directly below dataset_root"
            )


def _validate_non_overlapping_roots(cmu_root: Path, dataset_root: Path) -> None:
    resolved_cmu = cmu_root.resolve(strict=False)
    resolved_dataset = dataset_root.resolve(strict=False)
    if (
        resolved_cmu == resolved_dataset
        or resolved_cmu.is_relative_to(resolved_dataset)
        or resolved_dataset.is_relative_to(resolved_cmu)
    ):
        raise ValueError("cmu_root and dataset_root must not overlap")


def _normalize_speakers(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("speakers must be a sequence of speaker IDs")
    speakers = tuple(values)
    if not speakers:
        raise ValueError("speakers must not be empty")
    if any(not isinstance(value, str) for value in speakers):
        raise TypeError("speakers must contain strings")
    if any(not value.strip() for value in speakers):
        raise ValueError("speakers must contain non-empty strings")
    normalized = tuple(value.strip() for value in speakers)
    if len(set(normalized)) != len(normalized):
        raise ValueError("speakers must not contain duplicates")
    return normalized


def _normalize_triplet(
    values: Sequence[float],
    *,
    name: str,
) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of three numbers")
    try:
        normalized = tuple(
            normalize_finite_real(value, name=f"{name} values") for value in values
        )
    except TypeError as exc:
        raise TypeError(f"{name} must contain real numbers") from exc
    if len(normalized) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return normalized[0], normalized[1], normalized[2]


def _validate_integer(value: object, *, name: str, minimum: int) -> int:
    return normalize_integer(
        value,
        name=name,
        minimum=minimum,
        maximum=sys.maxsize,
    )


def _normalize_finite_float(value: object, *, name: str) -> float:
    return normalize_finite_real(value, name=name)


def _validate_random_center_geometry(
    *,
    room_size: tuple[float, float, float],
    source_margin: tuple[float, float, float],
    min_source_distance: float,
    array_radius: float,
) -> None:
    low_x = max(array_radius, source_margin[0] + min_source_distance)
    low_y = max(array_radius, source_margin[1] + min_source_distance)
    high_x = min(
        room_size[0] - array_radius,
        room_size[0] - source_margin[0] - min_source_distance,
    )
    high_y = min(
        room_size[1] - array_radius,
        room_size[1] - source_margin[1] - min_source_distance,
    )
    if high_x <= low_x or high_y <= low_y:
        raise ValueError(
            "room geometry has no feasible randomized microphone-center range"
        )
    if room_size[2] - array_radius <= array_radius:
        raise ValueError("room height is too small for the microphone array")


def _validate_fixed_center_geometry(
    *,
    room_size: tuple[float, float, float],
    mic_center: tuple[float, float, float],
    source_margin: tuple[float, float, float],
    min_source_distance: float,
    array_radius: float,
) -> None:
    if any(
        center <= array_radius or center >= size - array_radius
        for center, size in zip(mic_center, room_size, strict=True)
    ):
        raise ValueError(
            "mic_center must keep the complete microphone array strictly "
            "inside the room"
        )
    max_source_radius = min(
        mic_center[0] - source_margin[0],
        room_size[0] - source_margin[0] - mic_center[0],
        mic_center[1] - source_margin[1],
        room_size[1] - source_margin[1] - mic_center[1],
    )
    if max_source_radius < min_source_distance:
        raise ValueError("mic_center and source_margin leave no feasible source radius")


__all__ = ["DynamicCmuArcticBuildConfig", "DynamicDatasetBuildResult"]
