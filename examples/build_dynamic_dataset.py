from __future__ import annotations

"""Sample: build a small dynamic dataset with CMU ARCTIC or LibriSpeech.

This example mirrors the high-level idea in Cross3D: generate many dynamic
acoustic scenes with randomized source motion, then store the resulting
mixtures plus per-scene metadata.

Key characteristics:
    - Fixed room geometry and fixed binaural microphone layout across scenes.
    - Randomized source positions and motion patterns per scene.
    - Configurable dataset backend (CMU ARCTIC / LibriSpeech).
    - Dynamic RIR simulation via ISM + trajectory-mode convolution.
    - Per-scene WAV and JSON metadata outputs.

Outputs (per scene index k):
    - scene_k.wav
    - scene_k_refXX.wav (per-source convolved references)
    - scene_k_metadata.json
    - ATTRIBUTION.txt (dataset attribution and redistribution note)
    - scene_k_static_2d.png / scene_k_dynamic_2d.png (and 3D variants when enabled)

Run (CMU ARCTIC):
    uv run python examples/build_dynamic_dataset.py --dataset cmu_arctic --num-scenes 4

Run (LibriSpeech):
    uv run python examples/build_dynamic_dataset.py --dataset librispeech --subset train-clean-100

Notes:
    - Use --num-moving-sources to keep some sources fixed.
    - Plotting is opt-in via --plot.
    - Downloading is automatic if data is missing (can also be requested via --download).
    - Reference outputs are per-source, RIR-convolved signals (premix).
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

try:
    from torchrir import MicrophoneArray, Room, Source
    from torchrir.logging import LoggingConfig, get_logger, setup_logging
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import MicrophoneArray, Room, Source
    from torchrir.logging import LoggingConfig, get_logger, setup_logging

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
from torchrir.datasets import (
    CmuArcticDataset,
    LibriSpeechDataset,
    attribution_for,
    default_modification_notes,
    load_dataset_sources,
)
from torchrir.geometry import arrays, sampling, trajectories
from torchrir.io import save_attribution_file, save_scene_audio, save_scene_metadata
from torchrir.signal import DynamicConvolver
from torchrir.sim import simulate_dynamic_rir
from torchrir.util import add_output_args, resolve_device
from torchrir.viz import save_scene_gifs, save_scene_plots

MIC_SPACING = 0.08


def _serialize_args(args) -> dict[str, object]:
    return {
        "dataset": args.dataset,
        "dataset_dir": str(args.dataset_dir),
        "subset": args.subset,
        "download": args.download,
        "num_scenes": args.num_scenes,
        "num_sources": args.num_sources,
        "num_moving_sources": args.num_moving_sources,
        "num_mics": args.num_mics,
        "duration": args.duration,
        "seed": args.seed,
        "room": list(args.room),
        "steps": args.steps,
        "order": args.order,
        "tmax": args.tmax,
        "device": args.device,
        "out_dir": str(args.out_dir),
        "plot": args.plot,
        "log_level": args.log_level,
    }


def _dataset_factory(
    *,
    dataset: str,
    root: Path,
    subset: str,
    download: bool,
    speaker: Optional[str],
):
    if dataset == "cmu_arctic":
        spk = speaker or "bdl"
        try:
            return CmuArcticDataset(root, speaker=spk, download=download)
        except FileNotFoundError:
            if download:
                raise
            return CmuArcticDataset(root, speaker=spk, download=True)
    try:
        return LibriSpeechDataset(
            root, subset=subset, speaker=speaker, download=download
        )
    except FileNotFoundError:
        if download:
            raise
        return LibriSpeechDataset(root, subset=subset, speaker=speaker, download=True)


def _random_trajectory(
    *,
    start: torch.Tensor,
    room_size: torch.Tensor,
    steps: int,
    rng: random.Random,
) -> tuple[torch.Tensor, str]:
    # Randomize how each source moves so datasets have diverse motion patterns.
    mode = rng.choice(["linear", "zigzag"])
    if mode == "linear":
        # Straight line from start to a random end point.
        end = sampling.sample_positions_with_z_range(
            num=1, room_size=room_size, rng=rng
        ).squeeze(0)
        traj = trajectories.linear_trajectory(start, end, steps)
        return traj, mode
    # Zigzag motion via a random mid point.
    mid = sampling.sample_positions_with_z_range(
        num=1, room_size=room_size, rng=rng
    ).squeeze(0)
    end = sampling.sample_positions_with_z_range(
        num=1, room_size=room_size, rng=rng
    ).squeeze(0)
    split = max(2, steps // 2)
    first = trajectories.linear_trajectory(start, mid, split)
    second = trajectories.linear_trajectory(mid, end, steps - split + 1)
    traj = torch.cat([first[:-1], second], dim=0)
    return traj, mode


def _build_source_trajectories(
    *,
    num_sources: int,
    num_moving_sources: int,
    room_size: torch.Tensor,
    mic_center: torch.Tensor,
    steps: int,
    rng: random.Random,
) -> tuple[torch.Tensor, List[str], List[int]]:
    # Sample a start for each source, then generate a trajectory per source.
    starts = sampling.sample_positions_min_distance(
        num=num_sources,
        room_size=room_size,
        rng=rng,
        center=mic_center,
        min_distance=1.5,
    )
    num_moving = max(0, min(num_sources, num_moving_sources))
    moving_indices = (
        set(rng.sample(range(num_sources), k=num_moving)) if num_moving > 0 else set()
    )
    trajs: List[torch.Tensor] = []
    modes: List[str] = []
    for idx in range(num_sources):
        if idx in moving_indices:
            traj, mode = _random_trajectory(
                start=starts[idx],
                room_size=room_size,
                steps=steps,
                rng=rng,
            )
        else:
            traj = starts[idx].unsqueeze(0).repeat(steps, 1)
            mode = "static"
        trajs.append(traj)
        modes.append(mode)
    # Stack to (T, n_src, dim) and keep positions inside the room.
    src_traj = torch.stack(trajs, dim=1)
    src_traj = sampling.clamp_positions(src_traj, room_size)
    return src_traj, modes, sorted(moving_indices)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small dynamic dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cmu_arctic", "librispeech"],
        default="cmu_arctic",
        help="Dataset backend to use.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Root directory for the dataset (defaults by dataset).",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train-clean-100",
        help="LibriSpeech subset (e.g., train-clean-100).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Download the dataset if missing.",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=4,
        help="Number of scenes to generate.",
    )
    parser.add_argument(
        "--num-sources",
        type=int,
        default=2,
        help="Number of sources per scene.",
    )
    parser.add_argument(
        "--num-moving-sources",
        type=int,
        default=1,
        help="Number of sources that move (others stay fixed).",
    )
    parser.add_argument(
        "--num-mics",
        type=int,
        default=2,
        help="Number of microphones in the fixed array.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Target duration (seconds) for each source signal.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--room",
        type=float,
        nargs="+",
        default=[6.0, 4.0, 3.0],
        help="Room size (Lx Ly [Lz]).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of RIR time steps for trajectories.",
    )
    parser.add_argument("--order", type=int, default=6, help="ISM reflection order.")
    parser.add_argument(
        "--tmax", type=float, default=0.4, help="RIR length in seconds."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device (cpu/cuda/mps/auto).",
    )
    add_output_args(
        parser,
        out_dir_default="outputs/dynamic_dataset",
        plot_default=False,
        include_gif=False,
        include_show=False,
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + fixed room setup.
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.build_dynamic_dataset")

    dataset_root = args.dataset_dir
    if dataset_root is None:
        dataset_root = (
            Path("datasets/cmu_arctic")
            if args.dataset == "cmu_arctic"
            else Path("datasets/librispeech")
        )
    args.dataset_dir = dataset_root

    # Fixed room and fixed microphone layout across all scenes.
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)
    room = Room.shoebox(
        size=args.room, fs=16000, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    rng = random.Random(args.seed)
    mic_center = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(
        0
    )
    if args.num_mics <= 0:
        raise ValueError("num_mics must be positive")
    if args.num_mics == 2:
        mic_pos = arrays.binaural_array(mic_center, offset=MIC_SPACING)
    else:
        mic_pos = arrays.linear_array(
            mic_center, num=args.num_mics, spacing=MIC_SPACING, axis=0
        )
    mic_pos = sampling.clamp_positions(mic_pos, room_size)
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_attribution = attribution_for(
        args.dataset, args.subset if args.dataset == "librispeech" else None
    )
    dataset_license = dataset_attribution.to_dict()
    modifications = default_modification_notes(dynamic=True)
    attribution_path = save_attribution_file(
        out_dir=args.out_dir,
        dataset_attribution=dataset_attribution,
        modifications=modifications,
        logger=logger,
    )

    for idx in range(args.num_scenes):
        # Use a per-scene RNG so each scene has independent random motion + sources.
        scene_rng = random.Random(args.seed + idx)
        signals, fs, info = load_dataset_sources(
            dataset_factory=lambda speaker: _dataset_factory(
                dataset=args.dataset,
                root=dataset_root,
                subset=args.subset,
                download=args.download,
                speaker=speaker,
            ),
            num_sources=args.num_sources,
            duration_s=args.duration,
            rng=scene_rng,
        )
        signals = signals.to(device)

        # Build random trajectories for each source; mics stay fixed.
        steps = max(2, args.steps)
        src_traj, modes, moving = _build_source_trajectories(
            num_sources=args.num_sources,
            num_moving_sources=args.num_moving_sources,
            room_size=room_size,
            mic_center=mic_center,
            steps=steps,
            rng=scene_rng,
        )
        src_traj = src_traj.to(device)
        mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1).to(device)

        # Use the initial positions for scene bookkeeping (trajectory is used for RIRs).
        sources = Source.from_positions(src_traj[0].tolist())

        if args.plot:
            prefix = f"scene_{idx:03d}"
            save_scene_plots(
                out_dir=args.out_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                prefix=prefix,
                show=False,
                logger=logger,
            )
            save_scene_gifs(
                out_dir=args.out_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                prefix=prefix,
                signal_len=signals.shape[1],
                fs=fs,
                gif_fps=-1,
                logger=logger,
            )

        # ISM simulation + dynamic convolution.
        rirs = simulate_dynamic_rir(
            room=room,
            src_traj=src_traj,
            mic_traj=mic_traj,
            max_order=args.order,
            tmax=args.tmax,
            device=device,
        )
        convolver = DynamicConvolver(mode="trajectory")
        y = convolver.convolve(signals, rirs)

        # Save per-source reference audio before mixing.
        reference_audio = []
        for src_idx in range(args.num_sources):
            ref = convolver.convolve(signals[src_idx], rirs[:, src_idx : src_idx + 1])
            ref_name = f"scene_{idx:03d}_ref{src_idx + 1:02d}.wav"
            save_scene_audio(
                out_dir=args.out_dir,
                audio=ref,
                fs=fs,
                audio_name=ref_name,
                logger=logger,
            )
            speaker, utterances = info[src_idx]
            reference_audio.append(
                {
                    "index": src_idx,
                    "filename": ref_name,
                    "speaker": speaker,
                    "utterances": utterances,
                    "kind": "convolved",
                }
            )

        # Save mixture audio and JSON metadata per scene.
        save_scene_audio(
            out_dir=args.out_dir,
            audio=y,
            fs=fs,
            audio_name=f"scene_{idx:03d}.wav",
            logger=logger,
        )
        save_scene_metadata(
            out_dir=args.out_dir,
            metadata_name=f"scene_{idx:03d}_metadata.json",
            room=room,
            sources=sources,
            mics=mics,
            rirs=rirs,
            src_traj=src_traj,
            mic_traj=mic_traj,
            signal_len=signals.shape[1],
            source_info=info,
            extra={
                "mode": "dynamic_dataset",
                "dataset": args.dataset,
                "subset": args.subset if args.dataset == "librispeech" else None,
                "dataset_license": dataset_license,
                "modifications": modifications,
                "attribution_file": attribution_path.name,
                "motion_modes": modes,
                "moving_sources": moving,
                "reference_audio": reference_audio,
                "args": _serialize_args(args),
            },
            logger=logger,
        )

        logger.info("scene %d: sources=%s", idx, info)
        logger.info("scene %d: dynamic RIR shape=%s", idx, tuple(rirs.shape))
        logger.info("scene %d: output shape=%s", idx, tuple(y.shape))


if __name__ == "__main__":
    main()
