from __future__ import annotations

"""Sample: build a small dynamic dataset (fixed room/mics, moving sources).

This example mirrors the high-level idea in Cross3D: generate many dynamic
acoustic scenes with randomized source motion, then store the resulting
mixtures plus per-scene metadata.

Key characteristics:
    - Fixed room geometry and fixed binaural microphone layout across scenes.
    - Randomized source positions and motion patterns per scene.
    - CMU ARCTIC utterances as source signals.
    - Dynamic RIR simulation via ISM + trajectory-mode convolution.
    - Per-scene WAV and JSON metadata outputs.

Outputs (per scene index k):
    - scene_k.wav
    - scene_k_metadata.json
    - scene_k_static_2d.png / scene_k_dynamic_2d.png (and 3D variants when enabled)

Run:
    uv run python examples/cmu_arctic_dynamic_dataset.py --num-scenes 4 --num-sources 2 --duration 6
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List

import torch

try:
    from torchrir import (
        CmuArcticDataset,
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        build_metadata,
        clamp_positions,
        get_logger,
        linear_trajectory,
        load_dataset_sources,
        plot_scene_and_save,
        resolve_device,
        save_metadata_json,
        save_wav,
        setup_logging,
        simulate_dynamic_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        CmuArcticDataset,
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        build_metadata,
        clamp_positions,
        get_logger,
        linear_trajectory,
        load_dataset_sources,
        plot_scene_and_save,
        resolve_device,
        save_metadata_json,
        save_wav,
        setup_logging,
        simulate_dynamic_rir,
    )

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from torchrir import sample_positions


def _dataset_factory(
    root: Path, download: bool, speaker: str | None
) -> CmuArcticDataset:
    spk = speaker or "bdl"
    return CmuArcticDataset(root, speaker=spk, download=download)


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
        end = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
        traj = linear_trajectory(start, end, steps)
        return traj, mode
    # Zigzag motion via a random mid point.
    mid = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    end = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    split = max(2, steps // 2)
    first = linear_trajectory(start, mid, split)
    second = linear_trajectory(mid, end, steps - split + 1)
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
    starts = sample_positions(num=num_sources, room_size=room_size, rng=rng)
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
    src_traj = clamp_positions(src_traj, room_size)
    return src_traj, modes


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic CMU ARCTIC dataset sample")
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/cmu_arctic"))
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--num-scenes", type=int, default=4)
    parser.add_argument("--num-sources", type=int, default=2)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room", type=float, nargs="+", default=[6.0, 4.0, 3.0])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dynamic_dataset"))
    parser.add_argument("--plot", action="store_true", default=True)
    parser.add_argument("--no-plot", action="store_false", dest="plot")
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.cmu_arctic_dynamic_dataset")

    # Fixed room and fixed binaural microphone layout across all scenes.
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)
    room = Room.shoebox(
        size=args.room, fs=16000, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    rng = random.Random(args.seed)
    mic_center = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_pos = clamp_positions(MicrophoneArray.binaural(mic_center).positions, room_size)
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(args.num_scenes):
        # Use a per-scene RNG so each scene has independent random motion + sources.
        scene_rng = random.Random(args.seed + idx)
        signals, fs, info = load_dataset_sources(
            dataset_factory=lambda speaker: _dataset_factory(
                args.dataset_dir, args.download, speaker
            ),
            num_sources=args.num_sources,
            duration_s=args.duration,
            rng=scene_rng,
        )
        signals = signals.to(device)

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
            try:
                plot_scene_and_save(
                    out_dir=args.out_dir,
                    room=room.size,
                    sources=sources,
                    mics=mics,
                    src_traj=src_traj,
                    mic_traj=mic_traj,
                    prefix=f"scene_{idx:03d}",
                    show=args.show,
                )
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.warning("Plot skipped for scene %03d: %s", idx, exc)

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
        out_audio = args.out_dir / f"scene_{idx:03d}.wav"
        save_wav(out_audio, y, fs)
        meta_path = args.out_dir / f"scene_{idx:03d}_metadata.json"
        metadata = build_metadata(
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
            },
        )
        save_metadata_json(meta_path, metadata)

        logger.info("scene %03d: saved %s", idx, out_audio)
        logger.info("scene %03d: saved %s", idx, meta_path)


if __name__ == "__main__":
    main()
