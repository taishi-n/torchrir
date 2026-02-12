from __future__ import annotations

"""Static CMU ARCTIC example (fixed sources + fixed mic array).

This script:
1) Loads random CMU ARCTIC utterances for multiple speakers.
2) Samples fixed source positions and a mic array in a shoebox room.
3) Simulates static RIRs with ISM and convolves the dry signals.
4) Saves the convolved mixture and JSON metadata, optionally plots the scene.

Outputs (default `--out-dir outputs`):
- static.wav
- static_ref01.wav, static_ref02.wav, ... (per-source convolved references)
- static_metadata.json
- ATTRIBUTION.txt
- optional plots and GIFs under the same directory
  - static_static_2d.png (and static_static_3d.png if 3D)
  - static.gif (and static_3d.gif if 3D)
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
from torchrir.datasets import (
    CmuArcticDataset,
    attribution_for,
    default_modification_notes,
    load_dataset_sources,
)
from torchrir.geometry import arrays, sampling
from torchrir.io import save_attribution_file, save_scene_audio, save_scene_metadata
from torchrir.signal import convolve_rir
from torchrir.sim import simulate_rir
from torchrir.util import add_output_args, resolve_device
from torchrir.viz import save_scene_gifs, save_scene_plots

MIC_SPACING = 0.08


def main() -> None:
    """Run the static CMU ARCTIC simulation and save audio + metadata."""
    parser = argparse.ArgumentParser(
        description="Static RIR: fixed sources and a fixed mic array"
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
        "--num-mics",
        type=int,
        default=2,
        help="Number of microphones in the fixed array.",
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
    parser.add_argument("--order", type=int, default=8, help="ISM reflection order.")
    parser.add_argument(
        "--tmax",
        type=float,
        default=0.4,
        help="RIR length in seconds.",
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
        include_gif=False,
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + RNG
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.static")
    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)
    dataset_attribution = attribution_for("cmu_arctic")
    dataset_license = dataset_attribution.to_dict()
    modifications = default_modification_notes(dynamic=False)
    attribution_path = save_attribution_file(
        out_dir=args.out_dir,
        dataset_attribution=dataset_attribution,
        modifications=modifications,
        logger=logger,
    )

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

    # Room and random geometry (fixed sources + fixed binaural mic).
    room = Room.shoebox(
        size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    mic_center = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(
        0
    )
    sources_pos = sampling.sample_positions_min_distance(
        num=args.num_sources,
        room_size=room_size,
        rng=rng,
        center=mic_center,
        min_distance=1.5,
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

    sources = Source.from_positions(sources_pos.tolist())
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    # Optional static plot/GIF.
    if args.plot:
        save_scene_plots(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            prefix="static",
            show=args.show,
            logger=logger,
        )
        src_traj = sources_pos.unsqueeze(0)
        mic_traj = mic_pos.unsqueeze(0)
        save_scene_gifs(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix="static",
            signal_len=signals.shape[1],
            fs=fs,
            gif_fps=-1,
            logger=logger,
        )

    # ISM simulation + convolution.
    rirs = simulate_rir(
        room=room,
        sources=sources,
        mics=mics,
        max_order=args.order,
        tmax=args.tmax,
        device=device,
    )

    y_static = convolve_rir(signals, rirs)

    # Save per-source reference audio (convolved with its own RIR).
    reference_audio = []
    for src_idx in range(args.num_sources):
        ref = convolve_rir(signals[src_idx], rirs[src_idx : src_idx + 1])
        ref_name = f"static_ref{src_idx + 1:02d}.wav"
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

    # Save outputs (audio + metadata).
    save_scene_audio(
        out_dir=args.out_dir,
        audio=y_static,
        fs=fs,
        audio_name="static.wav",
        logger=logger,
    )
    metadata = save_scene_metadata(
        out_dir=args.out_dir,
        metadata_name="static_metadata.json",
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=None,
        mic_traj=None,
        signal_len=signals.shape[1],
        source_info=info,
        extra={
            "mode": "static",
            "reference_audio": reference_audio,
            "dataset_license": dataset_license,
            "modifications": modifications,
            "attribution_file": attribution_path.name,
        },
        logger=logger,
    )

    logger.info("sources: %s", info)
    logger.info("RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y_static.shape))


if __name__ == "__main__":
    main()
