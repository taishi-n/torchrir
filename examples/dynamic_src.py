from __future__ import annotations

"""Example: moving sources with a fixed binaural microphone."""

import argparse
import random
import sys
from pathlib import Path

import torch

try:
    from torchrir import (
        DynamicConvolver,
        MicrophoneArray,
        Room,
        Source,
        plot_scene_and_save,
        resolve_device,
        save_wav,
        simulate_dynamic_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        DynamicConvolver,
        MicrophoneArray,
        Room,
        Source,
        plot_scene_and_save,
        resolve_device,
        save_wav,
        simulate_dynamic_rir,
    )

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
from cmu_arctic_scene_utils import (
    binaural_mic_positions,
    clamp_positions,
    linear_trajectory,
    load_cmu_arctic_sources,
    sample_positions,
)


def main() -> None:
    """Run the dynamic-source CMU ARCTIC simulation."""
    parser = argparse.ArgumentParser(description="Dynamic RIR: moving sources, fixed binaural mic")
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
    parser.add_argument("--plot", action="store_true", help="plot room and trajectories")
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = resolve_device(args.device)
    room_size = torch.tensor(args.room, dtype=torch.float32)
    signals, fs, info = load_cmu_arctic_sources(
        root=args.dataset_dir,
        num_sources=args.num_sources,
        duration_s=args.duration,
        rng=rng,
        download=args.download,
    )
    signals = signals.to(device)
    room = Room.shoebox(size=args.room, fs=fs, beta=[0.9] * (6 if len(args.room) == 3 else 4))

    src_start = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    src_end = sample_positions(num=args.num_sources, room_size=room_size, rng=rng)
    steps = max(2, args.steps)
    src_traj = torch.stack(
        [linear_trajectory(src_start[i], src_end[i], steps) for i in range(args.num_sources)],
        dim=1,
    )
    src_traj = clamp_positions(src_traj, room_size)

    mic_center = sample_positions(num=1, room_size=room_size, rng=rng).squeeze(0)
    mic_pos = binaural_mic_positions(mic_center)
    mic_pos = clamp_positions(mic_pos, room_size)
    mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1)

    sources = Source.positions(src_start.tolist())
    mics = MicrophoneArray.positions(mic_pos.tolist())

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
            print(f"Plot skipped: {exc}")

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

    print("sources:", info)
    print("dynamic RIR shape:", tuple(rirs.shape))
    print("output shape:", tuple(y_dynamic.shape))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
