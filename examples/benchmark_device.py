from __future__ import annotations

"""Simple CPU vs GPU benchmark for RIR simulation."""

import argparse
import sys
import time
from pathlib import Path

import torch

try:
    from torchrir import (
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        resolve_device,
        setup_logging,
        simulate_dynamic_rir,
        simulate_rir,
    )
except ModuleNotFoundError:  # allow running without installation
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from torchrir import (
        DynamicConvolver,
        LoggingConfig,
        MicrophoneArray,
        Room,
        Source,
        get_logger,
        resolve_device,
        setup_logging,
        simulate_dynamic_rir,
        simulate_rir,
    )


def _bench_once(device: torch.device, repeats: int) -> float:
    room = Room.shoebox(size=[8.0, 6.0, 3.5], fs=16000, beta=[0.92] * 6)
    sources = Source.from_positions([[1.0, 1.5, 1.2], [4.0, 2.5, 1.4]])
    mic_grid = []
    for x in (2.0, 3.0, 4.0, 5.0):
        for y in (1.5, 2.5, 3.5, 4.5):
            mic_grid.append([x, y, 1.2])
    mics = MicrophoneArray.from_positions(mic_grid)

    # Warmup
    simulate_rir(
        room=room, sources=sources, mics=mics, max_order=12, tmax=0.8, device=device
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        simulate_rir(
            room=room, sources=sources, mics=mics, max_order=12, tmax=0.8, device=device
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) / repeats


def _bench_dynamic(device: torch.device, repeats: int) -> float:
    room = Room.shoebox(size=[8.0, 6.0, 3.5], fs=16000, beta=[0.92] * 6)
    sources = Source.from_positions([[1.0, 1.5, 1.2]])
    mic_grid = []
    for x in (2.0, 3.0, 4.0, 5.0):
        for y in (1.5, 2.5, 3.5, 4.5):
            mic_grid.append([x, y, 1.2])
    mics = MicrophoneArray.from_positions(mic_grid)
    steps = 24
    src_traj = sources.positions.unsqueeze(0).repeat(steps, 1, 1)
    mic_start = torch.tensor([2.0, 1.0, 1.2])
    mic_end = torch.tensor([4.0, 3.0, 1.2])
    mic_traj = torch.stack(
        [mic_start + (mic_end - mic_start) * t / (steps - 1) for t in range(steps)],
        dim=0,
    ).unsqueeze(1)
    signal = torch.randn(1, 16000, device=device)

    rirs = simulate_dynamic_rir(
        room=room,
        src_traj=src_traj,
        mic_traj=mic_traj,
        max_order=10,
        tmax=0.8,
        device=device,
    )
    DynamicConvolver(mode="trajectory").convolve(signal, rirs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        rirs = simulate_dynamic_rir(
            room=room,
            src_traj=src_traj,
            mic_traj=mic_traj,
            max_order=10,
            tmax=0.8,
            device=device,
        )
        DynamicConvolver(mode="trajectory").convolve(signal, rirs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU vs GPU benchmark for RIR generation"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gpu", type=str, default="auto", help="cuda/mps/auto")
    parser.add_argument(
        "--dynamic", action="store_true", help="benchmark dynamic trajectory path"
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    setup_logging(LoggingConfig(level=args.log_level))
    logger = get_logger("examples.benchmark_device")

    if args.dynamic:
        cpu_time = _bench_dynamic(resolve_device("cpu"), repeats=args.repeats)
        logger.info("cpu dynamic avg: %.2f ms", cpu_time * 1000)
    else:
        cpu_time = _bench_once(resolve_device("cpu"), repeats=args.repeats)
        logger.info("cpu avg: %.2f ms", cpu_time * 1000)

    gpu_device = resolve_device(args.gpu)
    if gpu_device.type == "cpu":
        logger.warning("gpu not available; skipping gpu benchmark")
        return

    if args.dynamic:
        gpu_time = _bench_dynamic(gpu_device, repeats=args.repeats)
        logger.info("%s dynamic avg: %.2f ms", gpu_device.type, gpu_time * 1000)
    else:
        gpu_time = _bench_once(gpu_device, repeats=args.repeats)
        logger.info("%s avg: %.2f ms", gpu_device.type, gpu_time * 1000)
    logger.info("speedup: %.2fx", cpu_time / gpu_time)


if __name__ == "__main__":
    main()
