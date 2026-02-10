"""Shared helpers for visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch


def _to_cpu(value: Any) -> torch.Tensor:
    """Move a value to CPU as a tensor."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _positions_to_cpu(entity: torch.Tensor | object) -> torch.Tensor:
    """Extract positions from an entity and move to CPU."""
    pos = getattr(entity, "positions", entity)
    pos = _to_cpu(pos)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)
    return pos


def _traj_steps(
    src_traj: Optional[torch.Tensor | Sequence],
    mic_traj: Optional[torch.Tensor | Sequence],
) -> int:
    """Infer the number of trajectory steps."""
    if src_traj is not None:
        return int(_to_cpu(src_traj).shape[0])
    return int(_to_cpu(mic_traj).shape[0])


def _trajectory_to_cpu(
    traj: Optional[torch.Tensor | Sequence], fallback_pos: torch.Tensor, steps: int
) -> torch.Tensor:
    """Normalize trajectory to CPU tensor with shape (T, N, dim)."""
    if traj is None:
        return fallback_pos.unsqueeze(0).repeat(steps, 1, 1)
    traj = _to_cpu(traj)
    if traj.ndim != 3:
        raise ValueError("trajectory must be of shape (T, N, dim)")
    return traj


def _save_axes(ax: Any, path: Path, *, show: bool) -> None:
    """Save a matplotlib axis to disk."""
    import matplotlib.pyplot as plt

    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
