"""Shared helpers for visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import torch

_MPL_DEFAULT_STYLE_APPLIED = False
_MM_PER_INCH = 25.4
_STATIC_FIGSIZE_INCHES = (160.0 / _MM_PER_INCH, 100.0 / _MM_PER_INCH)
_STATIC_SAVE_DPI = 300


def _ensure_default_mpl_style() -> None:
    """Apply the default torchrir.viz matplotlib style once per process.

    The default style uses SciencePlots grid profile without LaTeX:
    ``plt.style.use(["science", "grid", "no-latex"])``.
    """
    global _MPL_DEFAULT_STYLE_APPLIED
    if _MPL_DEFAULT_STYLE_APPLIED:
        return

    try:
        import matplotlib.pyplot as plt
        import scienceplots  # noqa: F401
    except Exception:
        return

    plt.style.use(["science", "grid", "no-latex"])
    _MPL_DEFAULT_STYLE_APPLIED = True


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
    fig.set_size_inches(*_STATIC_FIGSIZE_INCHES)
    fig.tight_layout()
    fig.savefig(path, dpi=_STATIC_SAVE_DPI)
    if show:
        plt.show()
    plt.close(fig)


def _add_axes_annotation(
    ax: Any,
    annotation_lines: Optional[Sequence[str]] = None,
    *,
    x: float = 0.02,
    y: float = 0.98,
    fontsize: float | None = None,
) -> Any | None:
    """Add top-left annotation text to 2D/3D matplotlib axes."""
    if not annotation_lines:
        return None
    lines = [str(line).strip() for line in annotation_lines if str(line).strip()]
    if not lines:
        return None

    text = "\n".join(lines)
    if hasattr(ax, "text2D"):
        return ax.text2D(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsize,
        )
    return ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
    )
