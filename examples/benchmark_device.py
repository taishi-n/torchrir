"""CPU vs GPU benchmark for RIR simulation and dynamic convolution.

This script measures average per-run latency for:
1) Static RIR generation (ISM).
2) Optional moving-microphone simulation + observation-time convolution.

It prints average milliseconds and speedup ratios. Use --dynamic to benchmark
moving-microphone RIRs and observation-time convolution.
"""

from __future__ import annotations

import argparse
import time

import torch

from torchrir import DynamicScene, MicrophoneArray, Room, Source, StaticScene
from torchrir.config import SimulationConfig
from torchrir.geometry import linear_trajectory
from torchrir.logging import LoggingConfig, get_logger, setup_logging
from torchrir.signal import DynamicConvolver, FrameSchedule
from torchrir.sim import simulate
from torchrir.util import resolve_device


def _bench_once(device: torch.device, repeats: int) -> float:
    # Static ISM benchmark with multiple sources and a dense mic grid.
    room = Room.shoebox(size=[8.0, 6.0, 3.5], fs=16000, beta=[0.92] * 6)
    sources = Source.from_positions([[1.0, 1.5, 1.2], [4.0, 2.5, 1.4]])
    mic_grid = []
    for x in (2.0, 3.0, 4.0, 5.0):
        for y in (1.5, 2.5, 3.5, 4.5):
            mic_grid.append([x, y, 1.2])
    mics = MicrophoneArray.from_positions(mic_grid)
    scene = StaticScene(room=room, sources=sources, mics=mics)
    config = SimulationConfig(max_order=12, tmax=0.8, device=device)

    # Warmup
    simulate(scene, config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        simulate(scene, config)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()
    end = time.perf_counter()
    return (end - start) / repeats


def _bench_dynamic(device: torch.device, repeats: int) -> float:
    # Dynamic ISM benchmark with moving mic and convolution.
    room = Room.shoebox(size=[8.0, 6.0, 3.5], fs=16000, beta=[0.92] * 6)
    sources = Source.from_positions([[1.0, 1.5, 1.2]])
    mic_grid = []
    for x in (2.0, 3.0, 4.0, 5.0):
        for y in (1.5, 2.5, 3.5, 4.5):
            mic_grid.append([x, y, 1.2])
    mics = MicrophoneArray.from_positions(mic_grid)
    steps = 24
    signal = torch.randn(1, 16000, device=device)
    schedule = FrameSchedule.uniform(
        frame_count=steps,
        stop_sample=signal.shape[-1],
    )
    progress = schedule.normalized_progress(
        stop_sample=signal.shape[-1],
        dtype=mics.positions.dtype,
        device=mics.positions.device,
    )
    src_traj = sources.positions.unsqueeze(0).repeat(steps, 1, 1)
    displacement_end = torch.tensor([0.5, 0.5, 0.0])
    displacement = linear_trajectory(
        torch.zeros_like(displacement_end),
        displacement_end,
        progress=progress,
    )
    mic_traj = mics.positions.unsqueeze(0) + displacement[:, None, :]
    scene = DynamicScene(
        room=room,
        sources=sources,
        mics=mics,
        src_traj=src_traj,
        mic_traj=mic_traj,
        schedule=schedule,
    )
    config = SimulationConfig(max_order=10, tmax=0.8, device=device)

    result = simulate(scene, config)
    DynamicConvolver(time_reference="observation").convolve(signal, result)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        result = simulate(scene, config)
        DynamicConvolver(time_reference="observation").convolve(signal, result)
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of repeated runs (averaged).",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="auto",
        help="GPU device selection (cuda/mps/auto).",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Benchmark dynamic trajectory path (simulate + convolve).",
    )
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level.")
    args = parser.parse_args()

    # Logging + CPU baseline.
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
