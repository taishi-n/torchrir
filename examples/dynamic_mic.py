from __future__ import annotations

"""Dynamic CMU ARCTIC example (fixed sources + moving binaural mic).

This script:
1) Loads random CMU ARCTIC utterances for multiple speakers.
2) Samples fixed source positions and generates a moving binaural mic trajectory.
3) Simulates dynamic RIRs with ISM and convolves the dry signals.
4) Saves the binaural mixture and JSON metadata, optionally plots/animates.

Outputs (default `--out-dir outputs`):
- dynamic_mic_binaural.wav
- dynamic_mic_binaural_metadata.json
- optional plots and GIFs under the same directory
"""

import argparse
import random
import sys
from pathlib import Path

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
from torchrir.datasets import CmuArcticDataset, load_dataset_sources
from torchrir.geometry import arrays, sampling, trajectories
from torchrir.io import save_scene_audio, save_scene_metadata
from torchrir.signal import DynamicConvolver
from torchrir.sim import simulate_dynamic_rir
from torchrir.util import add_output_args, resolve_device
from torchrir.viz import save_scene_gifs, save_scene_plots


def main() -> None:
    """Run the dynamic-mic CMU ARCTIC simulation and save outputs."""
    parser = argparse.ArgumentParser(
        description="Dynamic RIR: fixed sources, moving binaural mic"
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/cmu_arctic"),
        help="Root directory for the CMU ARCTIC dataset.",
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
        "--num-sources",
        type=int,
        default=2,
        help="Number of source speakers to mix.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
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
        help="Number of RIR time steps for the trajectory.",
    )
    parser.add_argument("--order", type=int, default=8, help="ISM reflection order.")
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
        out_dir_default="outputs",
        plot_default=False,
        include_gif=True,
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + RNG
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.dynamic_mic")

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)

    # Build dataset factory so each speaker loads from the same root.
    def dataset_factory(speaker: str | None):
        spk = speaker or "bdl"
        return CmuArcticDataset(args.dataset_dir, speaker=spk, download=args.download)

    # Load and concatenate utterances into fixed-length sources.
    signals, fs, info = load_dataset_sources(
        dataset_factory=dataset_factory,
        num_sources=args.num_sources,
        duration_s=args.duration,
        rng=rng,
    )
    signals = signals.to(device)
    # Room setup (moving mic, fixed sources).
    room = Room.shoebox(
        size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    sources_pos = sampling.sample_positions(
        num=args.num_sources, room_size=room_size, rng=rng
    )
    mic_center_start = sampling.sample_positions(
        num=1, room_size=room_size, rng=rng
    ).squeeze(0)
    mic_center_end = sampling.sample_positions(
        num=1, room_size=room_size, rng=rng
    ).squeeze(0)
    steps = max(2, args.steps)
    mic_center_traj = trajectories.linear_trajectory(
        mic_center_start, mic_center_end, steps
    )
    mic_traj = torch.stack(
        [arrays.binaural_array(center) for center in mic_center_traj],
        dim=0,
    )
    mic_traj = sampling.clamp_positions(mic_traj, room_size)

    # Fixed source positions; mic moves along linear trajectory.
    src_traj = sources_pos.unsqueeze(0).repeat(steps, 1, 1)

    sources = Source.from_positions(sources_pos.tolist())
    mics = MicrophoneArray.from_positions(mic_traj[0].tolist())

    src_traj = src_traj.to(device)
    mic_traj = mic_traj.to(device)

    # Optional plots/GIFs.
    if args.plot:
        save_scene_plots(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix="dynamic_mic",
            show=args.show,
            logger=logger,
        )
    if args.gif:
        save_scene_gifs(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix="dynamic_mic",
            signal_len=signals.shape[1],
            fs=fs,
            gif_fps=int(args.gif_fps),
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

    y_dynamic = DynamicConvolver(mode="trajectory").convolve(signals, rirs)

    # Save outputs (audio + metadata).
    save_scene_audio(
        out_dir=args.out_dir,
        audio=y_dynamic,
        fs=fs,
        audio_name="dynamic_mic_binaural.wav",
        logger=logger,
    )
    metadata = save_scene_metadata(
        out_dir=args.out_dir,
        metadata_name="dynamic_mic_binaural_metadata.json",
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=signals.shape[1],
        source_info=info,
        extra={"mode": "dynamic_mic"},
        logger=logger,
    )

    logger.info("sources: %s", info)
    logger.info("dynamic RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y_dynamic.shape))


if __name__ == "__main__":
    main()
