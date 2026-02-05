from __future__ import annotations

"""Static CMU ARCTIC example (fixed sources + fixed binaural mic).

This script:
1) Loads random CMU ARCTIC utterances for multiple speakers.
2) Samples fixed source positions and a binaural microphone in a shoebox room.
3) Simulates static RIRs with ISM and convolves the dry signals.
4) Saves the binaural mixture and JSON metadata, optionally plots the scene.

Outputs (default `--out-dir outputs`):
- static_binaural.wav
- static_binaural_metadata.json
- optional plots under the same directory
"""

import argparse
import random
import sys
from pathlib import Path

import torch

try:
    from torchrir import (
        CmuArcticDataset,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        build_metadata,
        convolve_rir,
        get_logger,
        plot_scene_and_save,
        resolve_device,
        save_metadata_json,
        save_wav,
        setup_logging,
        simulate_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        CmuArcticDataset,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        build_metadata,
        convolve_rir,
        get_logger,
        plot_scene_and_save,
        resolve_device,
        save_metadata_json,
        save_wav,
        setup_logging,
        simulate_rir,
    )

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
from torchrir.geometry import arrays, sampling
from torchrir import load_dataset_sources


def main() -> None:
    """Run the static CMU ARCTIC simulation and save audio + metadata."""
    parser = argparse.ArgumentParser(
        description="Static RIR: fixed sources and binaural mic"
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for WAV/metadata/plots.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot the static room layout and save PNGs.",
    )
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + RNG
    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.static")
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

    # Room and random geometry (fixed sources + fixed binaural mic).
    room = Room.shoebox(
        size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4)
    )

    sources_pos = sampling.sample_positions(
        num=args.num_sources, room_size=room_size, rng=rng
    )
    mic_center = sampling.sample_positions(num=1, room_size=room_size, rng=rng).squeeze(
        0
    )
    mic_pos = arrays.binaural_array(mic_center)
    mic_pos = sampling.clamp_positions(mic_pos, room_size)

    sources = Source.from_positions(sources_pos.tolist())
    mics = MicrophoneArray.from_positions(mic_pos.tolist())

    # Optional static plot.
    if args.plot:
        try:
            plot_scene_and_save(
                out_dir=args.out_dir,
                room=room.size,
                sources=sources,
                mics=mics,
                prefix="static",
                show=args.show,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("Plot skipped: %s", exc)

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

    # Save outputs (audio + metadata).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "static_binaural.wav"
    save_wav(out_path, y_static, fs)
    meta_path = args.out_dir / "static_binaural_metadata.json"
    metadata = build_metadata(
        room=room,
        sources=sources,
        mics=mics,
        rirs=rirs,
        src_traj=None,
        mic_traj=None,
        signal_len=signals.shape[1],
        source_info=info,
        extra={"mode": "static"},
    )
    save_metadata_json(meta_path, metadata)

    logger.info("sources: %s", info)
    logger.info("RIR shape: %s", tuple(rirs.shape))
    logger.info("output shape: %s", tuple(y_static.shape))
    logger.info("saved: %s", out_path)
    logger.info("saved: %s", meta_path)


if __name__ == "__main__":
    main()
