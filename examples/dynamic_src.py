from __future__ import annotations

"""Example: moving sources with a fixed binaural microphone."""

import argparse
import random
import sys
from pathlib import Path

import torch

try:
    from torchrir import (
        CmuArcticDataset,
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        animate_scene_gif,
        build_metadata,
        get_logger,
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
        animate_scene_gif,
        build_metadata,
        get_logger,
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
from torchrir import clamp_positions, linear_trajectory, load_dataset_sources, sample_positions


def main() -> None:
    """Run the dynamic-source CMU ARCTIC simulation."""
    parser = argparse.ArgumentParser(
        description="Dynamic RIR: moving sources, fixed binaural mic"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("datasets/cmu_arctic"))
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument("--num-sources", type=int, default=2)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--room", type=float, nargs="+", default=[6.0, 4.0, 3.0])
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--order", type=int, default=8)
    parser.add_argument("--tmax", type=float, default=0.4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--plot", action="store_true", help="plot room and trajectories"
    )
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    parser.add_argument(
        "--gif", action="store_true", help="save trajectory animation GIF"
    )
    parser.add_argument("--gif-fps", type=float, default=0.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.dynamic_src")

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)

    def dataset_factory(speaker: str | None):
        spk = speaker or "bdl"
        return CmuArcticDataset(args.dataset_dir, speaker=spk, download=args.download)

    signals, fs, info = load_dataset_sources(
        dataset_factory=dataset_factory,
        num_sources=args.num_sources,
        duration_s=args.duration,
        rng=rng,
    )
    signals = signals.to(device)
    room = Room.shoebox(
        size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    src_start = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    src_end = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    steps = max(2, args.steps)
    src_traj = torch.stack(
        [
            linear_trajectory(src_start[i], src_end[i], steps)
            for i in range(args.num_sources)
        ],
        dim=1,
    )
    src_traj = clamp_positions(src_traj, room_size)

    mic_center = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_pos = MicrophoneArray.binaural(mic_center).positions
    mic_pos = clamp_positions(mic_pos, room_size)
    mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1)

    sources = Source.from_positions(src_start.tolist())
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    src_traj = src_traj.to(device)
    mic_traj = mic_traj.to(device)

    if args.plot:
        try:
            plot_scene_and_save(
                out_dir=args.out_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                prefix="dynamic_src",
                show=args.show,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Plot skipped: %s", exc)
    if args.gif:
        try:
            gif_path = args.out_dir / "dynamic_src.gif"
            animate_scene_gif(
                out_path=gif_path,
                room=room.size,
                sources=sources,
                mics=mics,
                src_traj=src_traj,
                mic_traj=mic_traj,
                fps=args.gif_fps if args.gif_fps > 0 else None,
                signal_len=signals.shape[1],
                fs=fs,
            )
            logger.info("saved: %s", gif_path)
            if room.size.numel() == 3:
                gif_path_3d = args.out_dir / "dynamic_src_3d.gif"
                animate_scene_gif(
                    out_path=gif_path_3d,
                    room=room.size,
                    sources=sources,
                    mics=mics,
                    src_traj=src_traj,
                    mic_traj=mic_traj,
                    fps=args.gif_fps if args.gif_fps > 0 else None,
                    signal_len=signals.shape[1],
                    fs=fs,
                    plot_2d=False,
                    plot_3d=True,
                )
                logger.info("saved: %s", gif_path_3d)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("GIF skipped: %s", exc)

    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=args.order,
        tmax=args.tmax,
        device=device,
    )

    y_dynamic = DynamicConvolver(mode="trajectory").convolve(signals, rirs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "dynamic_src_binaural.wav"
    save_wav(out_path, y_dynamic, fs)
    meta_path = args.out_dir / "dynamic_src_binaural_metadata.json"
    metadata = build_metadata(
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=src_traj,
        mic_traj=mic_traj,
        signal_len=signals.shape[1],
        source_info=info,
        extra={"mode": "dynamic_src"},
    )
    save_metadata_json(meta_path, metadata)

    logger.info("sources: %s", info)
    logger.info("dynamic RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y_dynamic.shape))
    logger.info("saved: %s", out_path)
    logger.info("saved: %s", meta_path)


if __name__ == "__main__":
    main()
