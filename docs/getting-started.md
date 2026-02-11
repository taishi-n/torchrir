# Getting Started

This page shows the minimum workflow for the core TorchRIR APIs:

1. Define room / source / microphone geometry.
2. Simulate static or dynamic RIRs.
3. Convolve dry signals with the generated RIRs.

## Install

```bash
pip install torchrir
```

## 1) Static RIR + Convolution + Plot Saving

```python
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchrir import MicrophoneArray, Room, Source
from torchrir.signal import convolve_rir
from torchrir.sim import simulate_rir
from torchrir.viz import plot_scene_static

fs = 16000
out_dir = Path("outputs/getting_started")
out_dir.mkdir(parents=True, exist_ok=True)

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=fs, beta=[0.9] * 6)
sources = Source.from_positions([[1.0, 1.5, 1.2]])
mics = MicrophoneArray.from_positions([[2.5, 2.0, 1.2], [2.7, 2.0, 1.2]])

rirs = simulate_rir(
    room=room,
    sources=sources,
    mics=mics,
    max_order=6,
    tmax=0.3,
    directivity="omni",
    device="auto",
)
print("static RIR shape:", tuple(rirs.shape))  # (n_src, n_mic, rir_len)

dry = torch.randn(1, fs * 2, device=rirs.device, dtype=rirs.dtype)  # (n_src, n_samples)
wet = convolve_rir(dry, rirs)
print("convolved shape:", tuple(wet.shape))  # (n_mic, n_samples + rir_len - 1)

# Save source/mic layout plot (2D top view).
room_top = room.size[:2]
src_top = sources.positions[:, :2]
mic_top = mics.positions[:, :2]
ax = plot_scene_static(
    room=room_top,
    sources=src_top,
    mics=mic_top,
    title="Static scene (top view 2D)",
)
ax.figure.savefig(out_dir / "layout_static.png", dpi=150, bbox_inches="tight")
plt.close(ax.figure)

# Save waveform plot before/after convolution with a shared x-axis.
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(dry[0].cpu().numpy())
axes[0].set_title("Dry source signal")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Amplitude")

axes[1].plot(wet[0].cpu().numpy())  # first microphone output
axes[1].set_title("Convolved signal (mic 1)")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Amplitude")

fig.tight_layout()
fig.savefig(out_dir / "waveform_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

Expected plot outputs:

- `outputs/getting_started/layout_static.png`
- `outputs/getting_started/waveform_before_after.png`

## 2) Dynamic RIR + Trajectory Convolution + Plot Saving

```python
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from torchrir import Room
from torchrir.signal import DynamicConvolver
from torchrir.sim import simulate_dynamic_rir
from torchrir.viz import plot_scene_dynamic

fs = 16000
out_dir = Path("outputs/getting_started")
out_dir.mkdir(parents=True, exist_ok=True)

room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=fs, beta=[0.9] * 6)

steps = 16
# Same initial source/mic layout as the static example;
# only the source moves in this dynamic example.
src_start = torch.tensor([1.0, 1.5, 1.2], dtype=torch.float32)
src_end = torch.tensor([3.0, 1.5, 1.2], dtype=torch.float32)
alpha = torch.linspace(0.0, 1.0, steps, dtype=torch.float32).view(steps, 1, 1)
src_traj = src_start.view(1, 1, 3) + alpha * (src_end - src_start).view(1, 1, 3)

# Fixed microphones (same two-mic setup as the static example).
mic_pos = torch.tensor([[2.5, 2.0, 1.2], [2.7, 2.0, 1.2]], dtype=torch.float32)
mic_traj = mic_pos.unsqueeze(0).repeat(steps, 1, 1)

dynamic_rirs = simulate_dynamic_rir(
    room=room,
    src_traj=src_traj,
    mic_traj=mic_traj,
    max_order=6,
    tmax=0.3,
    directivity="omni",
    device="auto",
)
print("dynamic RIR shape:", tuple(dynamic_rirs.shape))  # (T, n_src, n_mic, rir_len)

dry = torch.randn(
    1, fs * 2, device=dynamic_rirs.device, dtype=dynamic_rirs.dtype
)  # (n_src, n_samples)
wet = DynamicConvolver(mode="trajectory").convolve(dry, dynamic_rirs)
print("dynamic convolved shape:", tuple(wet.shape))

# Save source/mic trajectory layout plot (2D top view).
room_top = room.size[:2]
src_traj_top = src_traj[:, :, :2]
mic_traj_top = mic_traj[:, :, :2]
ax = plot_scene_dynamic(
    room=room_top,
    src_traj=src_traj_top,
    mic_traj=mic_traj_top,
    title="Dynamic scene trajectories (top view 2D)",
)
ax.figure.savefig(out_dir / "layout_dynamic.png", dpi=150, bbox_inches="tight")
plt.close(ax.figure)

# Save waveform plot before/after convolution with a shared x-axis.
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(dry[0].cpu().numpy())
axes[0].set_title("Dry source signal")
axes[0].set_xlabel("Sample")
axes[0].set_ylabel("Amplitude")

axes[1].plot(wet[0].cpu().numpy())  # first microphone output
axes[1].set_title("Dynamic convolved signal (mic 1)")
axes[1].set_xlabel("Sample")
axes[1].set_ylabel("Amplitude")

fig.tight_layout()
fig.savefig(out_dir / "waveform_dynamic_before_after.png", dpi=150, bbox_inches="tight")
plt.close(fig)
```

Expected plot outputs:

- `outputs/getting_started/layout_dynamic.png`
- `outputs/getting_started/waveform_dynamic_before_after.png`

!!! note
    `device="auto"` may select `mps`/`cuda`, so the dry signal should be created on the same device as the RIR tensor.
    On some PyTorch+MPS versions, FFT convolution can emit internal resize warnings; if you want a warning-free tutorial run, use `device="cpu"`.

## Next Steps

- See [Examples](examples.md) for CLI workflows and dataset generation scripts.
- See [API documentation](api.md) for all options and full signatures.
