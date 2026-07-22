"""Dynamic CMU ARCTIC dataset builder compatible with oobss loaders.

This module generates ``scene_xxxx`` directories with the file layout expected
by ``oobss.experiments.dataset.create_loader({"type": "torchrir_dynamic", ...})``:

- ``mixture.wav``
- ``source_00.wav``, ``source_01.wav``, ...
- ``metadata.json``
- ``source_info.json``
- ``room_layout_2d.png`` (optional)
- ``room_layout_3d.png`` (optional, 3D rooms)
- ``room_layout_2d.mp4`` (optional)
- ``room_layout_3d.mp4`` (optional, 3D rooms)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import random
from typing import Sequence

import numpy as np
import torch

from ._archive import (
    cleanup_stale_temporary_directories,
    managed_temporary_directory,
    publish_staged_directory,
    recover_staged_publication,
)
from ._locking import dataset_write_lock
from .cmu_arctic import CmuArcticDataset
from .dynamic_builder import DynamicCmuArcticBuildConfig, DynamicDatasetBuildResult
from .utils import _ceil_dataset_sample_count, _validate_loaded_audio
from ..config import SimulationConfig
from ..geometry import polyhedron_array
from ..io import save_result_metadata
from ..logging import _validate_info_logger
from ..models import DynamicScene, MicrophoneArray, Room, Source
from ..signal import DynamicConvolver, FrameSchedule
from ..sim import simulate
from ..viz import save_scene_layout_images, save_scene_videos

LOGGER = logging.getLogger(__name__)

DEFAULT_SPEAKERS = ["bdl", "slt", "clb", "rms", "jmk", "awb"]


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )


def _parse_triplet(text: str, *, name: str) -> np.ndarray:
    parts = [segment.strip() for segment in text.split(",") if segment.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must be comma-separated x,y,z")
    return np.array([float(segment) for segment in parts], dtype=np.float64)


def _as_triplet(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}")
    return arr


def _sample_random_mic_center(
    *,
    rng: np.random.Generator,
    room_size: np.ndarray,
    source_margin: np.ndarray,
    min_source_distance_m: float,
    array_radius_m: float,
) -> np.ndarray:
    xmin, ymin = float(source_margin[0]), float(source_margin[1])
    xmax = float(room_size[0] - source_margin[0])
    ymax = float(room_size[1] - source_margin[1])

    low_xy = np.array(
        [
            max(array_radius_m, xmin + min_source_distance_m),
            max(array_radius_m, ymin + min_source_distance_m),
        ],
        dtype=np.float64,
    )
    high_xy = np.array(
        [
            min(float(room_size[0] - array_radius_m), xmax - min_source_distance_m),
            min(float(room_size[1] - array_radius_m), ymax - min_source_distance_m),
        ],
        dtype=np.float64,
    )
    low_z = float(array_radius_m)
    high_z = float(room_size[2] - array_radius_m)

    if np.any(high_xy <= low_xy) or high_z <= low_z:
        raise ValueError(
            "No feasible mic-center sampling range. "
            "Try larger room, smaller source margin, or smaller minimum source distance."
        )

    low_xy_open = np.nextafter(low_xy, np.inf)
    high_xy_open = np.nextafter(high_xy, -np.inf)
    low_z_open = float(np.nextafter(low_z, np.inf))
    high_z_open = float(np.nextafter(high_z, -np.inf))
    if np.any(high_xy_open <= low_xy_open) or high_z_open <= low_z_open:
        raise ValueError("No strictly interior microphone-center sampling range")
    center_xy = rng.uniform(low_xy_open, high_xy_open)
    center_z = float(rng.uniform(low_z_open, high_z_open))
    return np.array([center_xy[0], center_xy[1], center_z], dtype=np.float64)


def _max_radius_for_azimuth(
    *,
    azimuth_rad: float,
    mic_center: np.ndarray,
    room_size: np.ndarray,
    margin: np.ndarray,
) -> float:
    cx, cy = float(mic_center[0]), float(mic_center[1])
    xmin, ymin = float(margin[0]), float(margin[1])
    xmax = float(room_size[0] - margin[0])
    ymax = float(room_size[1] - margin[1])

    c = float(np.cos(azimuth_rad))
    s = float(np.sin(azimuth_rad))
    bounds: list[float] = []
    eps = 1.0e-9

    if c > eps:
        bounds.append((xmax - cx) / c)
    elif c < -eps:
        bounds.append((xmin - cx) / c)
    elif cx < xmin or cx > xmax:
        return -1.0

    if s > eps:
        bounds.append((ymax - cy) / s)
    elif s < -eps:
        bounds.append((ymin - cy) / s)
    elif cy < ymin or cy > ymax:
        return -1.0

    if not bounds:
        return -1.0
    return float(min(bounds))


def _sample_position_on_azimuth(
    *,
    rng: np.random.Generator,
    azimuth_rad: float,
    mic_center: np.ndarray,
    room_size: np.ndarray,
    margin: np.ndarray,
    min_radius_m: float = 1.5,
    max_radius_m: float | None = None,
) -> np.ndarray:
    xmin, ymin = float(margin[0]), float(margin[1])
    xmax = float(room_size[0] - margin[0])
    ymax = float(room_size[1] - margin[1])
    z_min = float(margin[2])
    z_max = float(room_size[2] - margin[2])
    if z_max <= z_min:
        raise ValueError("Room is too small for the configured z-margin")

    r_max = _max_radius_for_azimuth(
        azimuth_rad=azimuth_rad,
        mic_center=mic_center,
        room_size=room_size,
        margin=margin,
    )
    if max_radius_m is not None:
        r_max = min(r_max, float(max_radius_m))
    if r_max <= float(min_radius_m):
        raise ValueError(
            "No feasible source radius for azimuth "
            f"{np.rad2deg(azimuth_rad):.2f} deg. "
            f"Need > {min_radius_m} m, got max {r_max:.3f} m."
        )

    radius_upper = float(np.nextafter(r_max, -np.inf))
    if radius_upper <= float(min_radius_m):
        raise ValueError("No strictly interior source-radius sampling range")
    radius = float(rng.uniform(float(min_radius_m), radius_upper))
    z_low_open = float(np.nextafter(z_min, np.inf))
    z_high_open = float(np.nextafter(z_max, -np.inf))
    if z_high_open <= z_low_open:
        raise ValueError("Room has no strictly interior source z-range")
    z = float(rng.uniform(z_low_open, z_high_open))
    x = float(mic_center[0] + radius * np.cos(azimuth_rad))
    y = float(mic_center[1] + radius * np.sin(azimuth_rad))
    x = float(np.clip(x, np.nextafter(xmin, np.inf), np.nextafter(xmax, -np.inf)))
    y = float(np.clip(y, np.nextafter(ymin, np.inf), np.nextafter(ymax, -np.inf)))
    return np.array([x, y, z], dtype=np.float64)


def _build_constrained_source_positions(
    *,
    rng: np.random.Generator,
    room_size: np.ndarray,
    mic_center: np.ndarray,
    margin: np.ndarray,
    n_sources: int,
    n_moving_sources: int,
    duration_sec: float,
    move_start_ratio: float,
    move_end_ratio: float,
    moving_speed_min: float,
    moving_speed_max: float,
    min_radius_m: float = 1.5,
    max_trials: int = 100,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[int],
]:
    if n_sources <= 0:
        raise ValueError("n_sources must be > 0")
    if n_moving_sources < 0 or n_moving_sources > n_sources:
        raise ValueError(
            "n_moving_sources must satisfy 0 <= n_moving_sources <= n_sources"
        )
    if not (0.0 <= move_start_ratio < move_end_ratio <= 1.0):
        raise ValueError(
            "Motion ratios must satisfy 0 <= move_start_ratio < move_end_ratio <= 1."
        )
    if moving_speed_min <= 0.0 or moving_speed_max < moving_speed_min:
        raise ValueError(
            "moving speed range must satisfy 0 < moving_speed_min <= moving_speed_max."
        )

    move_duration_sec = float(duration_sec) * (move_end_ratio - move_start_ratio)
    if move_duration_sec <= 0.0:
        raise ValueError("Move duration must be positive.")

    moving_indices = (
        sorted(rng.choice(n_sources, size=n_moving_sources, replace=False).tolist())
        if n_moving_sources > 0
        else []
    )
    azimuth_step = (2.0 * np.pi) / float(n_sources)
    xmin, ymin = float(margin[0]), float(margin[1])
    xmax = float(room_size[0] - margin[0])
    ymax = float(room_size[1] - margin[1])
    cx, cy = float(mic_center[0]), float(mic_center[1])
    max_uniform_radius = min(cx - xmin, xmax - cx, cy - ymin, ymax - cy)
    if max_uniform_radius < float(min_radius_m):
        raise ValueError(
            "No feasible radius for arc motion across all azimuths. "
            f"Need >= {min_radius_m} m, got max {max_uniform_radius:.3f} m."
        )
    moving_index_set = set(moving_indices)

    for _ in range(max_trials):
        base_azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
        slot_indices = rng.permutation(n_sources).astype(np.float64)

        start_azimuth = base_azimuth + slot_indices * azimuth_step
        end_azimuth = start_azimuth.copy()

        try:
            starts = np.stack(
                [
                    _sample_position_on_azimuth(
                        rng=rng,
                        azimuth_rad=float(start_azimuth[src_idx]),
                        mic_center=mic_center,
                        room_size=room_size,
                        margin=margin,
                        min_radius_m=min_radius_m,
                        max_radius_m=max_uniform_radius
                        if src_idx in moving_index_set
                        else None,
                    )
                    for src_idx in range(n_sources)
                ],
                axis=0,
            )
            ends = starts.copy()
            source_velocity_mps = np.zeros(n_sources, dtype=np.float64)
            angular_velocity_rad_s = np.zeros(n_sources, dtype=np.float64)
            turn_direction = np.zeros(n_sources, dtype=np.int64)

            for src_idx in moving_indices:
                velocity = float(rng.uniform(moving_speed_min, moving_speed_max))
                radius = float(np.linalg.norm(starts[src_idx, :2] - mic_center[:2]))
                if radius < float(min_radius_m):
                    raise ValueError(
                        "Sampled source radius is smaller than min_radius_m."
                    )
                if radius <= 1.0e-9:
                    raise ValueError("Invalid radius for moving source.")
                direction = int(rng.choice(np.array([-1, 1], dtype=np.int64)))
                angular_velocity = float(direction * velocity / radius)
                delta_azimuth = float(angular_velocity * move_duration_sec)
                end_azimuth[src_idx] = float(start_azimuth[src_idx] + delta_azimuth)
                ends[src_idx, 0] = float(
                    mic_center[0] + radius * np.cos(end_azimuth[src_idx])
                )
                ends[src_idx, 1] = float(
                    mic_center[1] + radius * np.sin(end_azimuth[src_idx])
                )
                ends[src_idx, 2] = float(starts[src_idx, 2])
                if not (
                    xmin <= float(ends[src_idx, 0]) <= xmax
                    and ymin <= float(ends[src_idx, 1]) <= ymax
                ):
                    raise ValueError("Arc endpoint is outside the allowed room area.")
                source_velocity_mps[src_idx] = velocity
                angular_velocity_rad_s[src_idx] = angular_velocity
                turn_direction[src_idx] = direction
            return (
                starts,
                ends,
                start_azimuth,
                end_azimuth,
                source_velocity_mps,
                angular_velocity_rad_s,
                turn_direction,
                moving_indices,
            )
        except ValueError:
            continue

    raise ValueError(
        "Failed to place constrained source positions. "
        "Try larger room, smaller source margin, or fewer sources."
    )


def _build_source_trajectory(
    *,
    starts: np.ndarray,
    mic_center: np.ndarray,
    start_azimuth: np.ndarray,
    end_azimuth: np.ndarray,
    moving_indices: list[int],
    timeline: np.ndarray,
    move_start_ratio: float,
    move_end_ratio: float,
) -> np.ndarray:
    timeline = np.asarray(timeline, dtype=np.float64)
    if (
        timeline.ndim != 1
        or timeline.size == 0
        or not np.all(np.isfinite(timeline))
        or timeline[0] != 0.0
        or np.any(timeline < 0.0)
        or np.any(timeline >= 1.0)
        or np.any(np.diff(timeline) <= 0.0)
    ):
        raise ValueError(
            "timeline must be finite, start at 0, increase strictly, and stay below 1"
        )
    n_steps = int(timeline.size)
    if n_steps <= 1:
        return starts[None, :, :]
    if not (0.0 <= move_start_ratio < move_end_ratio <= 1.0):
        raise ValueError(
            "Motion ratios must satisfy 0 <= move_start_ratio < move_end_ratio <= 1."
        )
    alpha = np.zeros(n_steps, dtype=np.float64)
    in_move = (timeline >= move_start_ratio) & (timeline <= move_end_ratio)
    alpha[timeline > move_end_ratio] = 1.0
    alpha[in_move] = (timeline[in_move] - move_start_ratio) / (
        move_end_ratio - move_start_ratio
    )
    trajectory = np.repeat(starts[None, :, :], n_steps, axis=0)
    cx, cy = float(mic_center[0]), float(mic_center[1])
    moving_index_set = set(moving_indices)
    for src_idx in moving_index_set:
        radius = float(np.linalg.norm(starts[src_idx, :2] - mic_center[:2]))
        theta = (
            start_azimuth[src_idx]
            + (end_azimuth[src_idx] - start_azimuth[src_idx]) * alpha
        )
        trajectory[:, src_idx, 0] = cx + radius * np.cos(theta)
        trajectory[:, src_idx, 1] = cy + radius * np.sin(theta)
        trajectory[:, src_idx, 2] = float(starts[src_idx, 2])
    return trajectory


def _build_layout_annotation_lines(
    *,
    scene_id: str,
    move_start_sec: float,
    move_end_sec: float,
    source_velocity_mps: np.ndarray,
) -> list[str]:
    moving_speed_items = [
        f"S{src_idx}={float(speed):.2f}m/s"
        for src_idx, speed in enumerate(source_velocity_mps.tolist())
        if float(speed) > 0.0
    ]
    moving_speed_text = ", ".join(moving_speed_items) if moving_speed_items else "none"
    return [
        f"scene:{scene_id}",
        f"move:{move_start_sec:.2f}-{move_end_sec:.2f} s",
        f"speed:{moving_speed_text}",
    ]


def _load_fixed_length_signal(
    dataset: CmuArcticDataset,
    *,
    target_samples: int,
    expected_sample_rate: int,
    rng_py: random.Random,
) -> tuple[torch.Tensor, list[str]]:
    sentences = list(dataset.available_sentences())
    if not sentences:
        raise RuntimeError("No sentences available in selected speaker dataset")

    utterance_ids: list[str] = []
    chunks: list[torch.Tensor] = []
    total = 0
    local = list(sentences)
    rng_py.shuffle(local)
    idx = 0

    while total < target_samples:
        if idx >= len(local):
            rng_py.shuffle(local)
            idx = 0
        sentence = local[idx]
        idx += 1
        waveform, sample_rate = dataset.load_audio(sentence.utterance_id)
        _validate_loaded_audio(waveform, sample_rate)
        if sample_rate != expected_sample_rate:
            raise ValueError(
                "sample rate mismatch while loading utterance "
                f"{sentence.utterance_id}: expected {expected_sample_rate}, "
                f"got {sample_rate}"
            )
        chunks.append(waveform)
        utterance_ids.append(sentence.utterance_id)
        total += int(chunks[-1].numel())

    signal = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    return signal[:target_samples], utterance_ids


def _normalize_sources(
    stems: list[np.ndarray], *, peak_limit: float = 0.99
) -> list[np.ndarray]:
    stacked = np.stack(stems, axis=0)
    mixture = np.sum(stacked, axis=0)
    peak = max(
        float(np.max(np.abs(stacked))),
        float(np.max(np.abs(mixture))),
    )
    if peak <= 0.0 or peak <= peak_limit:
        return stems
    scale = peak_limit / peak
    return [stem * scale for stem in stems]


def _to_time_channel_audio(
    audio: np.ndarray | torch.Tensor, *, n_mics: int
) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float64)
    data = np.squeeze(data)

    if data.ndim == 1:
        return data[:, None]
    if data.ndim != 2:
        raise ValueError(f"Unexpected audio ndim: {data.ndim}, shape={data.shape}")

    if data.shape[1] == n_mics:
        return data
    if data.shape[0] == n_mics:
        return data.T

    raise ValueError(
        "Could not infer time/channel axes for convolved signal with "
        f"shape={data.shape} and n_mics={n_mics}"
    )


def build_dynamic_cmu_arctic(
    config: DynamicCmuArcticBuildConfig,
    *,
    logger: logging.Logger | None = None,
) -> DynamicDatasetBuildResult:
    """Build and publish a staged dynamic CMU ARCTIC dataset with rollback.

    Builds for the same target are serialized before any expensive scene work.
    Every scene is written below an owned sibling workspace, and the target is
    created or replaced only after the complete build succeeds. If publication
    fails, the previous target is restored before the error is reported.
    """

    if not isinstance(config, DynamicCmuArcticBuildConfig):
        raise TypeError("config must be a DynamicCmuArcticBuildConfig")
    _validate_info_logger(logger)
    dataset_root = Path(config.dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    build_lock = dataset_root.parent / f".{dataset_root.name}.torchrir-build.lock"
    workspace_prefix = f".{dataset_root.name}.torchrir-build."
    with dataset_write_lock(build_lock):
        recover_staged_publication(dataset_root)
        if dataset_root.exists() and not config.overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {dataset_root}. Use overwrite=True."
            )
        cleanup_stale_temporary_directories(
            parent=dataset_root.parent,
            prefix=workspace_prefix,
        )
        with managed_temporary_directory(
            parent=dataset_root.parent,
            prefix=workspace_prefix,
        ) as staging_parent:
            staging_root = staging_parent / "dataset"
            sample_rate, n_mics = _build_dynamic_cmu_arctic_in_place(
                replace(config, dataset_root=staging_root, overwrite=False),
                logger=logger,
            )
            _commit_staged_dataset(
                staging_root,
                dataset_root,
                overwrite=config.overwrite,
            )
        scene_dirs = tuple(
            dataset_root / f"scene_{index:04d}" for index in range(config.n_scenes)
        )
        return DynamicDatasetBuildResult(
            dataset_root=dataset_root,
            sample_rate=sample_rate,
            n_mics=n_mics,
            n_scenes=config.n_scenes,
            scene_dirs=scene_dirs,
        )


def _commit_staged_dataset(
    staging_root: Path, dataset_root: Path, *, overwrite: bool
) -> None:
    publish_staged_directory(
        staging_root,
        dataset_root,
        replace_existing=overwrite,
    )


def _build_dynamic_cmu_arctic_in_place(
    config: DynamicCmuArcticBuildConfig,
    *,
    logger: logging.Logger | None = None,
) -> tuple[int, int]:
    """Build a dynamic CMU ARCTIC dataset in a new staging directory."""

    _validate_info_logger(logger)

    cmu_root = Path(config.cmu_root)
    dataset_root = Path(config.dataset_root)
    speakers = config.speakers
    n_scenes = config.n_scenes
    n_sources = config.n_sources
    n_moving_sources = config.n_moving_sources
    duration_sec = config.duration_sec
    room_size = config.room_size
    mic_center = config.mic_center
    octa_edge_m = config.octa_edge_m
    source_margin = config.source_margin
    min_source_distance_m = config.min_source_distance_m
    trajectory_steps = config.trajectory_steps
    simulation = config.simulation
    rt60 = config.rt60
    sound_speed = config.sound_speed
    seed = config.seed
    download_cmu = config.download_cmu
    randomize_mic_center = config.randomize_mic_center
    move_start_ratio = config.move_start_ratio
    move_end_ratio = config.move_end_ratio
    moving_speed_min = config.moving_speed_min
    moving_speed_max = config.moving_speed_max
    save_layout_mp4 = config.save_layout_mp4
    save_layout_mp4_3d = config.save_layout_mp4_3d
    layout_video_fps = config.layout_video_fps
    layout_video_mux_audio = config.layout_video_mux_audio
    save_layout_images = config.save_layout_images
    save_layout_images_3d = config.save_layout_images_3d
    annotate_source_indices = config.annotate_source_indices

    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Dataset building requires the 'datasets' extra: "
            "pip install torchrir[datasets]"
        ) from exc
    log = LOGGER if logger is None else logger
    room_size_arr = _as_triplet(room_size, name="room_size")
    mic_center_arr = _as_triplet(mic_center, name="mic_center")
    source_margin_arr = _as_triplet(source_margin, name="source_margin")
    speakers_list = [str(speaker) for speaker in speakers]
    dataset_root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    rng_py = random.Random(seed)

    dataset_cache: dict[str, CmuArcticDataset] = {}
    sample_rate: int | None = None
    for speaker in speakers_list:
        ds = CmuArcticDataset(root=cmu_root, speaker=speaker, download=download_cmu)
        dataset_cache[speaker] = ds
        test_ids = ds.available_sentences()
        if not test_ids:
            raise RuntimeError(f"No utterances found for speaker '{speaker}'")
        _, sr = ds.load_audio(test_ids[0].utterance_id)
        if sample_rate is None:
            sample_rate = int(sr)
        elif int(sr) != sample_rate:
            raise ValueError(
                f"Sample rate mismatch across speakers: {sample_rate} vs {int(sr)}"
            )

    assert sample_rate is not None
    target_samples = _ceil_dataset_sample_count(duration_sec, sample_rate)
    effective_duration_sec = target_samples / sample_rate
    move_start_sec = effective_duration_sec * move_start_ratio
    move_end_sec = effective_duration_sec * move_end_ratio
    if trajectory_steps > target_samples:
        raise ValueError(
            "trajectory_steps cannot exceed the number of dry-signal samples"
        )
    schedule = FrameSchedule.uniform(
        frame_count=trajectory_steps,
        stop_sample=target_samples,
    )
    trajectory_timeline = schedule.normalized_progress(
        stop_sample=target_samples,
        dtype=torch.float64,
    ).numpy()

    radius = float(octa_edge_m) / np.sqrt(2.0)
    scene_dtype = simulation.dtype or torch.float32
    base_array = polyhedron_array(
        center=[0.0, 0.0, 0.0],
        kind="octahedron",
        radius=radius,
        dtype=scene_dtype,
    )
    base_mic_positions = base_array.cpu().numpy().astype(np.float64)
    n_mics = int(base_mic_positions.shape[0])

    room = Room.shoebox(
        size=room_size_arr.tolist(),
        fs=float(sample_rate),
        c=float(sound_speed),
        t60=float(rt60),
        dtype=scene_dtype,
    )
    convolver = DynamicConvolver(time_reference="emission")

    for scene_idx in range(n_scenes):
        scene_id = f"scene_{scene_idx:04d}"
        scene_dir = dataset_root / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        if randomize_mic_center:
            scene_mic_center = _sample_random_mic_center(
                rng=rng,
                room_size=room_size_arr,
                source_margin=source_margin_arr,
                min_source_distance_m=min_source_distance_m,
                array_radius_m=radius,
            )
        else:
            scene_mic_center = np.asarray(mic_center_arr, dtype=np.float64)
        mic_positions = base_mic_positions + scene_mic_center[None, :]
        mics = MicrophoneArray.from_positions(mic_positions.tolist(), dtype=scene_dtype)
        mic_traj_np = np.repeat(mic_positions[None, :, :], trajectory_steps, axis=0)

        chosen_speakers = rng_py.sample(speakers_list, n_sources)
        source_signals: list[torch.Tensor] = []
        source_info: list[dict[str, object]] = []
        for speaker in chosen_speakers:
            dataset = dataset_cache[speaker]
            signal, utterance_ids = _load_fixed_length_signal(
                dataset,
                target_samples=target_samples,
                expected_sample_rate=sample_rate,
                rng_py=rng_py,
            )
            source_signals.append(signal)
            source_info.append({"speaker": speaker, "utterance_ids": utterance_ids})

        dry = torch.stack(source_signals, dim=0)

        (
            starts,
            _ends,
            start_azimuth,
            end_azimuth,
            source_velocity_mps,
            angular_velocity_rad_s,
            turn_direction,
            moving_indices,
        ) = _build_constrained_source_positions(
            rng=rng,
            room_size=room_size_arr,
            mic_center=scene_mic_center,
            margin=source_margin_arr,
            n_sources=n_sources,
            n_moving_sources=n_moving_sources,
            duration_sec=effective_duration_sec,
            move_start_ratio=move_start_ratio,
            move_end_ratio=move_end_ratio,
            moving_speed_min=moving_speed_min,
            moving_speed_max=moving_speed_max,
            min_radius_m=min_source_distance_m,
        )

        src_traj_np = _build_source_trajectory(
            starts=starts,
            mic_center=scene_mic_center,
            start_azimuth=start_azimuth,
            end_azimuth=end_azimuth,
            moving_indices=moving_indices,
            timeline=trajectory_timeline,
            move_start_ratio=move_start_ratio,
            move_end_ratio=move_end_ratio,
        )

        moving_index_set = set(moving_indices)
        for src_idx, item in enumerate(source_info):
            item["source_index"] = int(src_idx)
            item["is_moving"] = bool(src_idx in moving_index_set)
            item["velocity_mps"] = float(source_velocity_mps[src_idx])
            item["motion_type"] = "arc" if src_idx in moving_index_set else "static"
            item["angular_velocity_rad_s"] = float(angular_velocity_rad_s[src_idx])
            item["turn_direction"] = int(turn_direction[src_idx])
            item["move_start_sec"] = float(move_start_sec)
            item["move_end_sec"] = float(move_end_sec)

        src_traj = torch.tensor(src_traj_np, dtype=scene_dtype)
        mic_traj = torch.tensor(mic_traj_np, dtype=scene_dtype)
        sources = Source.from_positions(starts.tolist(), dtype=scene_dtype)
        scene = DynamicScene(
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            schedule=schedule,
        )
        rir_result = simulate(scene, simulation)
        rirs = rir_result.rirs
        dry = dry.to(device=rirs.device, dtype=rirs.dtype)
        stems: list[np.ndarray] = []
        for src_idx in range(n_sources):
            stem_mc = convolver.convolve(
                dry[src_idx : src_idx + 1],
                rirs[:, src_idx : src_idx + 1, :, :],
                schedule=schedule,
            )
            stems.append(_to_time_channel_audio(stem_mc.cpu().numpy(), n_mics=n_mics))

        stems = [
            np.asarray(stem, dtype=np.float32) for stem in _normalize_sources(stems)
        ]
        mix = np.sum(np.stack(stems, axis=0), axis=0, dtype=np.float32)

        for src_idx, stem in enumerate(stems):
            sf.write(
                scene_dir / f"source_{src_idx:02d}.wav",
                stem,
                sample_rate,
                subtype="FLOAT",
            )
        mixture_path = scene_dir / "mixture.wav"
        sf.write(mixture_path, mix, sample_rate, subtype="FLOAT")

        layout_annotation_lines = _build_layout_annotation_lines(
            scene_id=scene_id,
            move_start_sec=move_start_sec,
            move_end_sec=move_end_sec,
            source_velocity_mps=source_velocity_mps,
        )

        if save_layout_images:
            save_scene_layout_images(
                out_dir=scene_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                logger=log,
                src_traj=src_traj,
                mic_traj=mic_traj,
                save_2d=True,
                save_3d=save_layout_images_3d,
                annotate_sources=annotate_source_indices,
                annotation_lines=layout_annotation_lines,
            )

        if save_layout_mp4:
            save_scene_videos(
                out_dir=scene_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                signal_len=target_samples,
                fs=sample_rate,
                logger=log,
                mp4_fps=layout_video_fps,
                save_3d=save_layout_mp4_3d,
                mixture_path=mixture_path,
                mux_audio=layout_video_mux_audio,
                annotate_sources=annotate_source_indices,
                annotation_lines=layout_annotation_lines,
            )

        save_result_metadata(
            out_dir=scene_dir,
            metadata_name="metadata.json",
            result=rir_result,
            time_reference="emission",
            signal_len=target_samples,
            source_info=source_info,
            extra={
                "scene_id": scene_id,
                "n_sources": n_sources,
                "n_moving_sources": n_moving_sources,
                "octa_edge_m": float(octa_edge_m),
                "mic_center_xyz_m": scene_mic_center.tolist(),
                "randomize_mic_center": bool(randomize_mic_center),
                "min_source_distance_from_array_center_m": float(min_source_distance_m),
                "azimuth_step_deg": float(360.0 / n_sources),
                "moving_source_indices": [int(idx) for idx in moving_indices],
                "start_azimuth_deg": np.rad2deg(start_azimuth).tolist(),
                "end_azimuth_deg": np.rad2deg(end_azimuth).tolist(),
                "source_velocity_mps": source_velocity_mps.tolist(),
                "motion_type": "arc",
                "angular_velocity_rad_s": angular_velocity_rad_s.tolist(),
                "turn_direction": turn_direction.tolist(),
                "motion_profile": {
                    "pre_static_ratio": float(move_start_ratio),
                    "move_ratio": float(move_end_ratio - move_start_ratio),
                    "post_static_ratio": float(1.0 - move_end_ratio),
                },
                "motion_time_sec": {
                    "requested_total": float(duration_sec),
                    "effective_total": effective_duration_sec,
                    "move_start": float(move_start_sec),
                    "move_end": float(move_end_sec),
                },
            },
            logger=log,
        )

        with (scene_dir / "source_info.json").open("w", encoding="utf-8") as fh:
            json.dump(source_info, fh, indent=2)

        log.info(
            "Built %s | speakers=%s | sample_rate=%d",
            scene_id,
            chosen_speakers,
            sample_rate,
        )

    return sample_rate, n_mics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build dynamic CMU ARCTIC scenes with torchrir."
    )
    parser.add_argument(
        "--cmu-root", type=Path, required=True, help="CMU ARCTIC root directory."
    )
    parser.add_argument(
        "--download-cmu",
        dest="download_cmu",
        action="store_true",
        help="Download CMU ARCTIC speakers if they are missing.",
    )
    parser.add_argument(
        "--no-download-cmu",
        dest="download_cmu",
        action="store_false",
        help="Disable CMU ARCTIC download and require local data.",
    )
    parser.set_defaults(download_cmu=False)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("outputs/cmu_arctic_torchrir_dynamic_dataset"),
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=DEFAULT_SPEAKERS,
        help="Candidate speaker IDs.",
    )
    parser.add_argument("--n-scenes", type=int, default=10)
    parser.add_argument("--n-sources", type=int, default=3)
    parser.add_argument("--n-moving-sources", type=int, default=1)
    parser.add_argument("--duration-sec", type=float, default=20.0)
    parser.add_argument("--room-size", type=str, default="8.0,6.0,3.0")
    parser.add_argument("--mic-center", type=str, default="4.0,3.0,1.5")
    parser.add_argument(
        "--randomize-mic-center",
        dest="randomize_mic_center",
        action="store_true",
        help="Randomize microphone center independently for each scene.",
    )
    parser.add_argument(
        "--no-randomize-mic-center",
        dest="randomize_mic_center",
        action="store_false",
        help="Use a fixed microphone center from --mic-center.",
    )
    parser.set_defaults(randomize_mic_center=True)
    parser.add_argument("--octa-edge-m", type=float, default=1.0)
    parser.add_argument("--source-margin", type=str, default="0.5,0.5,0.3")
    parser.add_argument("--min-source-distance-m", type=float, default=1.8)
    parser.add_argument("--trajectory-steps", type=int, default=256)
    parser.add_argument("--rir-samples", type=int, default=4096)
    parser.add_argument("--rt60", type=float, default=0.3)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Simulation device: cpu, cuda, mps, or auto.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Simulation dtype.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-dataset", action="store_true")
    parser.add_argument(
        "--save-layout-images",
        dest="save_layout_images",
        action="store_true",
        help="Save static layout images (room_layout_2d.png and room_layout_3d.png).",
    )
    parser.add_argument(
        "--no-save-layout-images",
        dest="save_layout_images",
        action="store_false",
        help="Disable static layout image rendering.",
    )
    parser.set_defaults(save_layout_images=True)
    parser.add_argument(
        "--save-layout-images-3d",
        dest="save_layout_images_3d",
        action="store_true",
        help="Save room_layout_3d.png for 3D rooms.",
    )
    parser.add_argument(
        "--no-save-layout-images-3d",
        dest="save_layout_images_3d",
        action="store_false",
        help="Disable room_layout_3d.png output.",
    )
    parser.set_defaults(save_layout_images_3d=True)
    parser.add_argument(
        "--save-layout-mp4",
        dest="save_layout_mp4",
        action="store_true",
        help="Save room_layout_2d.mp4 (and 3d when enabled) for each scene.",
    )
    parser.add_argument(
        "--no-save-layout-mp4",
        dest="save_layout_mp4",
        action="store_false",
        help="Disable MP4 layout rendering.",
    )
    parser.set_defaults(save_layout_mp4=True)
    parser.add_argument(
        "--save-layout-mp4-3d",
        dest="save_layout_mp4_3d",
        action="store_true",
        help="Save room_layout_3d.mp4 for 3D rooms.",
    )
    parser.add_argument(
        "--no-save-layout-mp4-3d",
        dest="save_layout_mp4_3d",
        action="store_false",
        help="Disable room_layout_3d.mp4 output.",
    )
    parser.set_defaults(save_layout_mp4_3d=True)
    parser.add_argument(
        "--layout-video-fps",
        type=float,
        default=None,
        help="Override MP4 frame rate. Auto when omitted.",
    )
    parser.add_argument(
        "--layout-video-no-audio",
        action="store_true",
        help="Disable muxing mixture audio into MP4 videos.",
    )
    parser.add_argument(
        "--no-annotate-source-indices",
        action="store_true",
        help="Disable source index annotations (S0, S1, ...) in layout plots/videos.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for ``python -m torchrir.datasets.dynamic_cmu_arctic``."""
    args = _parse_args()
    _configure_logging(args.log_level)

    room_size = _parse_triplet(args.room_size, name="room-size")
    mic_center = _parse_triplet(args.mic_center, name="mic-center")
    source_margin = _parse_triplet(args.source_margin, name="source-margin")
    dataset_root = args.dataset_root.expanduser().resolve()

    config = DynamicCmuArcticBuildConfig(
        cmu_root=args.cmu_root.expanduser().resolve(),
        dataset_root=dataset_root,
        speakers=list(args.speakers),
        n_scenes=int(args.n_scenes),
        n_sources=int(args.n_sources),
        n_moving_sources=int(args.n_moving_sources),
        duration_sec=float(args.duration_sec),
        room_size=room_size.tolist(),
        mic_center=mic_center.tolist(),
        octa_edge_m=float(args.octa_edge_m),
        source_margin=source_margin.tolist(),
        min_source_distance_m=float(args.min_source_distance_m),
        trajectory_steps=int(args.trajectory_steps),
        simulation=SimulationConfig(
            max_order=int(args.max_order),
            nsample=int(args.rir_samples),
            device=args.device,
            dtype=torch.float32 if args.dtype == "float32" else torch.float64,
        ),
        rt60=float(args.rt60),
        sound_speed=float(args.sound_speed),
        seed=int(args.seed),
        download_cmu=bool(args.download_cmu),
        overwrite=bool(args.overwrite_dataset),
        randomize_mic_center=bool(args.randomize_mic_center),
        save_layout_mp4=bool(args.save_layout_mp4),
        save_layout_mp4_3d=bool(args.save_layout_mp4_3d),
        layout_video_fps=args.layout_video_fps,
        layout_video_mux_audio=not bool(args.layout_video_no_audio),
        save_layout_images=bool(args.save_layout_images),
        save_layout_images_3d=bool(args.save_layout_images_3d),
        annotate_source_indices=not bool(args.no_annotate_source_indices),
    )
    result = build_dynamic_cmu_arctic(config)
    LOGGER.info(
        "Dataset build done | sample_rate=%d | n_mics=%d | n_scenes=%d",
        result.sample_rate,
        result.n_mics,
        result.n_scenes,
    )
    LOGGER.info("Dataset root: %s", result.dataset_root)


if __name__ == "__main__":
    main()
