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
import json
import logging
from pathlib import Path
import random
import shutil
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

from .cmu_arctic import CmuArcticDataset
from ..geometry import polyhedron_array
from ..io import save_scene_metadata
from ..models import MicrophoneArray, Room, Source
from ..signal import DynamicConvolver
from ..sim import simulate_dynamic_rir
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

    center_xy = rng.uniform(low_xy, high_xy)
    center_z = float(rng.uniform(low_z, high_z))
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
    if r_max < float(min_radius_m):
        raise ValueError(
            "No feasible source radius for azimuth "
            f"{np.rad2deg(azimuth_rad):.2f} deg. "
            f"Need >= {min_radius_m} m, got max {r_max:.3f} m."
        )

    radius = float(rng.uniform(float(min_radius_m), r_max))
    z = float(rng.uniform(z_min, z_max))
    x = float(mic_center[0] + radius * np.cos(azimuth_rad))
    y = float(mic_center[1] + radius * np.sin(azimuth_rad))
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
    n_steps: int,
    move_start_ratio: float,
    move_end_ratio: float,
) -> np.ndarray:
    if n_steps <= 1:
        return starts[None, :, :]
    if not (0.0 <= move_start_ratio < move_end_ratio <= 1.0):
        raise ValueError(
            "Motion ratios must satisfy 0 <= move_start_ratio < move_end_ratio <= 1."
        )
    timeline = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
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
        waveform, _ = dataset.load_audio(sentence.utterance_id)
        chunks.append(torch.as_tensor(waveform).reshape(-1))
        utterance_ids.append(sentence.utterance_id)
        total += int(chunks[-1].numel())

    signal = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    return signal[:target_samples], utterance_ids


def _normalize_sources(
    stems: list[np.ndarray], *, peak_limit: float = 0.99
) -> list[np.ndarray]:
    peak = max(float(np.max(np.abs(stem))) for stem in stems)
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


def build_dynamic_cmu_arctic_dataset(
    *,
    cmu_root: Path,
    dataset_root: Path = Path("outputs/cmu_arctic_torchrir_dynamic_dataset"),
    speakers: Sequence[str] = DEFAULT_SPEAKERS,
    n_scenes: int = 10,
    n_sources: int = 3,
    n_moving_sources: int = 1,
    duration_sec: float = 20.0,
    room_size: Sequence[float] | np.ndarray = (8.0, 6.0, 3.0),
    mic_center: Sequence[float] | np.ndarray = (4.0, 3.0, 1.5),
    octa_edge_m: float = 1.0,
    source_margin: Sequence[float] | np.ndarray = (0.5, 0.5, 0.3),
    trajectory_steps: int = 1024,
    rir_samples: int = 4096,
    rt60: float = 0.3,
    sound_speed: float = 343.0,
    max_order: int = 6,
    seed: int = 42,
    download_cmu: bool = False,
    overwrite: bool = False,
    randomize_mic_center: bool = True,
    move_start_ratio: float = 0.35,
    move_end_ratio: float = 0.65,
    moving_speed_min: float = 0.3,
    moving_speed_max: float = 0.8,
    save_layout_mp4: bool = True,
    save_layout_mp4_3d: bool = True,
    layout_video_fps: float | None = None,
    layout_video_mux_audio: bool = True,
    save_layout_images: bool = True,
    save_layout_images_3d: bool = True,
    annotate_source_indices: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[int, int]:
    """Build a dynamic CMU ARCTIC dataset with oobss-compatible layout."""
    log = LOGGER if logger is None else logger
    room_size_arr = _as_triplet(room_size, name="room_size")
    mic_center_arr = _as_triplet(mic_center, name="mic_center")
    source_margin_arr = _as_triplet(source_margin, name="source_margin")
    speakers_list = [str(speaker) for speaker in speakers]

    if n_scenes <= 0:
        raise ValueError("n_scenes must be > 0")
    if n_sources <= 0:
        raise ValueError("n_sources must be > 0")
    if n_moving_sources < 0 or n_moving_sources > n_sources:
        raise ValueError(
            "n_moving_sources must satisfy 0 <= n_moving_sources <= n_sources"
        )
    if n_sources > len(speakers_list):
        raise ValueError(
            f"n_sources ({n_sources}) must be <= number of provided speakers ({len(speakers_list)})"
        )
    if trajectory_steps <= 0:
        raise ValueError("trajectory_steps must be > 0")
    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be > 0")
    if rir_samples <= 0:
        raise ValueError("rir_samples must be > 0")

    min_source_distance_m = 1.8
    move_start_sec = float(duration_sec) * move_start_ratio
    move_end_sec = float(duration_sec) * move_end_ratio

    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset root already exists: {dataset_root}. Use overwrite=True."
            )
        shutil.rmtree(dataset_root)
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
    target_samples = int(float(duration_sec) * sample_rate)
    if target_samples <= 0:
        raise ValueError("duration_sec is too small")

    radius = float(octa_edge_m) / np.sqrt(2.0)
    base_array = polyhedron_array(
        center=[0.0, 0.0, 0.0],
        kind="octahedron",
        radius=radius,
        dtype=torch.float64,
    )
    base_mic_positions = base_array.cpu().numpy().astype(np.float64)
    n_mics = int(base_mic_positions.shape[0])
    if n_sources > n_mics:
        raise ValueError(
            f"n_sources ({n_sources}) must be <= n_mics ({n_mics}) for AuxIVA-based evaluation"
        )

    room = Room.shoebox(
        size=room_size_arr.tolist(),
        fs=float(sample_rate),
        c=float(sound_speed),
        t60=float(rt60),
        dtype=torch.float64,
    )
    convolver = DynamicConvolver(mode="trajectory")

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
        mics = MicrophoneArray.from_positions(
            mic_positions.tolist(), dtype=torch.float64
        )
        mic_traj_np = np.repeat(mic_positions[None, :, :], trajectory_steps, axis=0)

        chosen_speakers = rng_py.sample(speakers_list, n_sources)
        source_signals: list[torch.Tensor] = []
        source_info: list[dict[str, object]] = []
        for speaker in chosen_speakers:
            dataset = dataset_cache[speaker]
            signal, utterance_ids = _load_fixed_length_signal(
                dataset,
                target_samples=target_samples,
                rng_py=rng_py,
            )
            source_signals.append(signal)
            source_info.append({"speaker": speaker, "utterance_ids": utterance_ids})

        dry = torch.stack(source_signals, dim=0).to(dtype=torch.float64)

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
            duration_sec=float(duration_sec),
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
            n_steps=trajectory_steps,
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

        src_traj = torch.tensor(src_traj_np, dtype=torch.float64)
        mic_traj = torch.tensor(mic_traj_np, dtype=torch.float64)

        rirs = simulate_dynamic_rir(
            room=room,
            src_traj=src_traj,
            mic_traj=mic_traj,
            max_order=max_order,
            nsample=rir_samples,
        )

        stems: list[np.ndarray] = []
        for src_idx in range(n_sources):
            stem_mc = convolver.convolve(
                dry[src_idx : src_idx + 1],
                rirs[:, src_idx : src_idx + 1, :, :],
            )
            stems.append(_to_time_channel_audio(stem_mc.cpu().numpy(), n_mics=n_mics))

        stems = _normalize_sources(stems)
        mix = np.sum(np.stack(stems, axis=0), axis=0)

        for src_idx, stem in enumerate(stems):
            sf.write(scene_dir / f"source_{src_idx:02d}.wav", stem, sample_rate)
        mixture_path = scene_dir / "mixture.wav"
        sf.write(mixture_path, mix, sample_rate)

        sources = Source.from_positions(starts.tolist(), dtype=torch.float64)
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

        save_scene_metadata(
            out_dir=scene_dir,
            metadata_name="metadata.json",
            room=room,
            sources=sources,
            mics=mics,
            rirs=rirs,
            src_traj=src_traj,
            mic_traj=mic_traj,
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
                    "total": float(duration_sec),
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
    parser.add_argument("--trajectory-steps", type=int, default=1024)
    parser.add_argument("--rir-samples", type=int, default=4096)
    parser.add_argument("--rt60", type=float, default=0.3)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--max-order", type=int, default=6)
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

    sample_rate, n_mics = build_dynamic_cmu_arctic_dataset(
        cmu_root=args.cmu_root.expanduser().resolve(),
        dataset_root=dataset_root,
        speakers=list(args.speakers),
        n_scenes=int(args.n_scenes),
        n_sources=int(args.n_sources),
        n_moving_sources=int(args.n_moving_sources),
        duration_sec=float(args.duration_sec),
        room_size=room_size,
        mic_center=mic_center,
        octa_edge_m=float(args.octa_edge_m),
        source_margin=source_margin,
        trajectory_steps=int(args.trajectory_steps),
        rir_samples=int(args.rir_samples),
        rt60=float(args.rt60),
        sound_speed=float(args.sound_speed),
        max_order=int(args.max_order),
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
    LOGGER.info("Dataset build done | sample_rate=%d | n_mics=%d", sample_rate, n_mics)
    LOGGER.info("Dataset root: %s", dataset_root)


if __name__ == "__main__":
    main()
