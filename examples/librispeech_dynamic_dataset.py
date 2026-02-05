from __future__ import annotations

"""Sample: build a small dynamic LibriSpeech dataset (fixed room/mics, moving sources).

This mirrors the CMU ARCTIC dynamic dataset example but uses LibriSpeech
utterances as sources.

Key characteristics:
    - Fixed room geometry and fixed binaural microphone layout across scenes.
    - Randomized source positions and motion patterns per scene.
    - LibriSpeech utterances as source signals.
    - Dynamic RIR simulation via ISM + trajectory-mode convolution.
    - Per-scene WAV and JSON metadata outputs.

Outputs (per scene index k):
    - scene_k.wav
    - scene_k_metadata.json
    - scene_k_static_2d.png / scene_k_dynamic_2d.png (and 3D variants when enabled)

Run:
    uv run python examples/librispeech_dynamic_dataset.py --subset train-clean-100 --num-scenes 4 --num-sources 2 --duration 6
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import torch

try:
    from torchrir import (
        DynamicConvolver,
        LibriSpeechDataset,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        load_dataset_sources,
        setup_logging,
        simulate_dynamic_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        DynamicConvolver,
        LibriSpeechDataset,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        load_dataset_sources,
        setup_logging,
        simulate_dynamic_rir,
    )

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from torchrir.geometry import arrays, sampling, trajectories
from torchrir.io import save_audio, save_metadata
from torchrir.util import add_output_args, resolve_device
from torchrir.viz import save_scene_plots


def _dataset_factory(
    root: Path, subset: str, download: bool, speaker: str | None
) -> LibriSpeechDataset:
    return LibriSpeechDataset(
        root,
        subset=subset,
        speaker=speaker,
        download=download,
    )


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
        end = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
        traj = trajectories.linear_trajectory(start, end, steps)
        return traj, mode
    # Zigzag motion via a random mid point.
    mid = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    end = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    split = max(2, steps // 2)
    first = trajectories.linear_trajectory(start, mid, split)
    second = trajectories.linear_trajectory(mid, end, steps - split + 1)
    traj = torch.cat([first[:-1], second], dim=0)
    return traj, mode


def _build_source_trajectories(
    *,
    num_sources: int,
    room_size: torch.Tensor,
    steps: int,
    rng: random.Random,
) -> tuple[torch.Tensor, List[str]]:
    # Sample a start for each source, then generate a trajectory per source.
    starts = sampling.sample_positions(num=num_sources, room_size=room_size, rng=rng)
    trajs: List[torch.Tensor] = []
    modes: List[str] = []
    for idx in range(num_sources):
        traj, mode = _random_trajectory(
            start=starts[idx],
            room_size=room_size,
            steps=steps,
            rng=rng,
        )
        trajs.append(traj)
        modes.append(mode)
    # Stack to (T, n_src, dim) and keep positions inside the room.
    src_traj = torch.stack(trajs, dim=1)
    src_traj = sampling.clamp_positions(src_traj, room_size)
    return src_traj, modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic LibriSpeech dataset sample")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/librispeech"),
        help="Root directory for LibriSpeech.",
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
        default=True,
        help="Download the dataset if missing.",
    )
    parser.add_argument(
        "--no-download",
        action="store_false",
        dest="download",
        help="Disable dataset download.",
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
        default=16,
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
        out_dir_default="outputs/librispeech_dynamic_dataset",
        plot_default=True,
        include_gif=False,
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + fixed room setup.
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.librispeech_dynamic_dataset")

    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)

    rng = random.Random(args.seed)
    mic_center = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(
        0
    )
    mic_pos = sampling.clamp_positions(arrays.binaural_array(mic_center), room_size)
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    room: Room | None = None

    for idx in range(args.num_scenes):
        # Use a per-scene RNG so each scene has independent random motion + sources.
        scene_rng = random.Random(args.seed + idx)
        signals, fs, info = load_dataset_sources(
            dataset_factory=lambda speaker: _dataset_factory(
                args.dataset_dir, args.subset, args.download, speaker
            ),
            num_sources=args.num_sources,
            duration_s=args.duration,
            rng=scene_rng,
        )
        signals = signals.to(device)

        if room is None:
            room = Room.shoebox(
                size=args.room,
                fs=fs,
                beta=[0.9] * (6 if len(args.room) == 3 else 4),
            )

        # Build random trajectories for each source; mics stay fixed.
        steps = max(2, args.steps)
        src_traj, modes = _build_source_trajectories(
            num_sources=args.num_sources,
            room_size=room_size,
            steps=steps,
            rng=scene_rng,
        )
        src_traj = src_traj.to(device)
        mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1).to(device)

        # Use the initial positions for scene bookkeeping (trajectory is used for RIRs).
        sources = Source.from_positions(src_traj[0].tolist())

        if args.plot:
            save_scene_plots(
                out_dir=args.out_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                prefix=f"scene_{idx:03d}",
                show=args.show,
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
        y = DynamicConvolver(mode="trajectory").convolve(signals, rirs)

        # Save mixture audio and JSON metadata per scene.
        save_audio(
            out_dir=args.out_dir,
            audio=y,
            fs=fs,
            audio_name=f"scene_{idx:03d}.wav",
            logger=logger,
        )
        metadata = save_metadata(
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
                "mode": "dynamic_src",
                "trajectory_modes": modes,
                "scene_index": idx,
                "seed": args.seed + idx,
                "num_sources": args.num_sources,
                "subset": args.subset,
            },
            logger=logger,
        )


if __name__ == "__main__":
    main()
