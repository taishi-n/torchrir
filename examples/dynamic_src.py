"""Dynamic CMU ARCTIC example (moving sources + fixed mic array).

This script:
1) Loads random CMU ARCTIC utterances for multiple speakers.
2) Samples a fixed mic array and generates moving source trajectories.
3) Simulates dynamic RIRs with ISM and convolves the dry signals.
4) Saves the mixture and JSON metadata, optionally plots/animates.

Outputs (default `--out-dir outputs`):
- dynamic_src.wav
- dynamic_src_ref01.wav, dynamic_src_ref02.wav, ... (per-source convolved references)
- dynamic_src_metadata.json
- ATTRIBUTION.txt
- optional plots and GIFs under the same directory
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source
from torchrir.config import SimulationConfig
from torchrir.datasets import (
    CmuArcticDataset,
    attribution_for,
    cmu_arctic_speakers,
    default_modification_notes,
    load_dataset_sources,
)
from torchrir.geometry import arrays, sampling, trajectories
from torchrir.io import save_attribution_file, save_result_metadata, save_scene_audio
from torchrir.logging import LoggingConfig, get_logger, setup_logging
from torchrir.signal import DynamicConvolver, FrameSchedule
from torchrir.sim import simulate
from torchrir.util import add_output_args, resolve_device
from torchrir.viz import save_scene_gifs, save_scene_plots

MIC_SPACING = 0.08


def main() -> None:
    """Run the dynamic-source CMU ARCTIC simulation."""
    parser = argparse.ArgumentParser(
        description="Dynamic RIR: moving sources, fixed mic array"
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
        "--num-moving-sources",
        type=int,
        default=1,
        help="Number of sources that move (others stay fixed).",
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
        include_gif=False,
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + RNG
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.dynamic_src")

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)
    dataset_attribution = attribution_for("cmu_arctic")
    dataset_license = dataset_attribution.to_dict()
    modifications = default_modification_notes(dynamic=True)
    attribution_path = save_attribution_file(
        out_dir=args.out_dir,
        dataset_attribution=dataset_attribution,
        modifications=modifications,
        logger=logger,
    )

    # Build dataset factory so each speaker loads from the same root.
    def dataset_factory(speaker: str):
        return CmuArcticDataset(
            args.dataset_dir,
            speaker=speaker,
            download=args.download,
        )

    # Load and concatenate utterances into fixed-length sources.
    signals, fs, info = load_dataset_sources(
        dataset_factory=dataset_factory,
        speakers=cmu_arctic_speakers(None if args.download else args.dataset_dir),
        num_sources=args.num_sources,
        duration_s=args.duration,
        rng=rng,
    )
    signals = signals.to(device)
    # Room setup (fixed mic, moving sources).
    room = Room.shoebox(
        size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    # Fixed binaural mic (trajectory is constant).
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
    steps = max(2, args.steps)
    schedule = FrameSchedule.uniform(
        frame_count=steps,
        stop_sample=signals.shape[-1],
    )
    progress = schedule.normalized_progress(
        stop_sample=signals.shape[-1],
        dtype=room_size.dtype,
        device=room_size.device,
    )
    mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1)

    src_start = sampling.sample_positions_min_distance(
        num=args.num_sources,
        room_size=room_size,
        rng=rng,
        center=mic_center,
        min_distance=1.5,
    )
    src_end = sampling.sample_positions_with_z_range(
        num=args.num_sources, room_size=room_size, rng=rng
    )
    if room_size.numel() == 3:
        src_end[:, 2] = src_start[:, 2]
    num_moving = min(args.num_sources, max(1, args.num_moving_sources))
    moving_indices = set(rng.sample(range(args.num_sources), k=num_moving))
    src_traj = torch.stack(
        [
            (
                trajectories.linear_trajectory(
                    src_start[i],
                    src_end[i],
                    progress=progress,
                )
                if i in moving_indices
                else src_start[i].unsqueeze(0).repeat(steps, 1)
            )
            for i in range(args.num_sources)
        ],
        dim=1,
    )
    src_traj = sampling.clamp_positions(src_traj, room_size)

    sources = Source.from_positions(src_start.tolist())
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    # Optional plots/GIFs.
    if args.plot:
        save_scene_plots(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix="dynamic_src",
            show=args.show,
            logger=logger,
        )
        save_scene_gifs(
            out_dir=args.out_dir,
            room=room.size,
            sources=sources,
            mics=mics,
            src_traj=src_traj,
            mic_traj=mic_traj,
            prefix="dynamic_src",
            schedule=schedule,
            stop_sample=signals.shape[1],
            fs=fs,
            gif_fps=-1,
            logger=logger,
        )

    # ISM simulation + dynamic convolution.
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=schedule,
    )
    result = simulate(
        scene,
        SimulationConfig(max_order=args.order, tmax=args.tmax, device=device),
    )
    rirs = result.rirs

    convolver = DynamicConvolver(time_reference="emission")
    y_dynamic = convolver.convolve(signals, result)

    # Save per-source reference audio (convolved with its own RIR).
    reference_audio = []
    for src_idx in range(args.num_sources):
        ref = convolver.convolve(
            signals[src_idx], rirs[:, src_idx : src_idx + 1], schedule=schedule
        )
        ref_name = f"dynamic_src_ref{src_idx + 1:02d}.wav"
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
        audio=y_dynamic,
        fs=fs,
        audio_name="dynamic_src.wav",
        logger=logger,
    )
    save_result_metadata(
        out_dir=args.out_dir,
        metadata_name="dynamic_src_metadata.json",
        result=result,
        time_reference="emission",
        signal_len=signals.shape[1],
        source_info=info,
        extra={
            "mode": "dynamic_src",
            "reference_audio": reference_audio,
            "dataset_license": dataset_license,
            "modifications": modifications,
            "attribution_file": attribution_path.name,
        },
        logger=logger,
    )

    logger.info("sources: %s", info)
    logger.info("dynamic RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y_dynamic.shape))


if __name__ == "__main__":
    main()
