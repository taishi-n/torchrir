"""Metadata helpers for simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import torch
from torch import Tensor

from ..models import MicrophoneArray, Room, Source


@dataclass(frozen=True)
class ArrayAttributes:
    """Structured description of a microphone array."""

    geometry_name: str
    positions: Tensor
    orientation: Optional[Tensor]
    center: Tensor
    normal: Optional[Tensor]
    spacing: Optional[float]


def build_metadata(
    *,
    room: Room,
    sources: Source,
    mics: MicrophoneArray,
    rirs: Tensor,
    src_traj: Optional[Tensor] = None,
    mic_traj: Optional[Tensor] = None,
    timestamps: Optional[Tensor] = None,
    signal_len: Optional[int] = None,
    source_info: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build JSON-serializable metadata for a simulation output.

    Examples:
        ```python
        metadata = build_metadata(
            room=room,
            sources=sources,
            mics=mics,
            rirs=rirs,
            src_traj=src_traj,
            mic_traj=mic_traj,
            signal_len=signal.shape[-1],
        )
        save_metadata_json(Path(\"outputs/scene_metadata.json\"), metadata)
        ```
    """
    nsample = int(rirs.shape[-1])
    fs = float(room.fs)
    time_axis = {
        "fs": fs,
        "nsample": nsample,
        "t": _to_serializable(torch.arange(nsample, dtype=torch.float32) / fs),
    }

    src_pos = sources.positions
    mic_pos = mics.positions
    dim = int(room.size.numel())
    src_traj_n = _normalize_traj(src_traj, src_pos, dim, "src_traj")
    mic_traj_n = _normalize_traj(mic_traj, mic_pos, dim, "mic_traj")

    t_steps = max(src_traj_n.shape[0], mic_traj_n.shape[0])
    if src_traj_n.shape[0] == 1 and t_steps > 1:
        src_traj_n = src_traj_n.expand(t_steps, -1, -1)
    if mic_traj_n.shape[0] == 1 and t_steps > 1:
        mic_traj_n = mic_traj_n.expand(t_steps, -1, -1)
    if src_traj_n.shape[0] != mic_traj_n.shape[0]:
        raise ValueError("src_traj and mic_traj must have matching time steps")

    azimuth, elevation = _compute_doa(src_traj_n, mic_traj_n)
    doa = {
        "frame": "world",
        "unit": "radians",
        "azimuth": _to_serializable(azimuth),
        "elevation": _to_serializable(elevation),
    }

    timestamps_out: Optional[Tensor] = None
    if timestamps is not None:
        timestamps_out = timestamps
    elif t_steps > 1 and signal_len is not None:
        duration = max(0.0, (float(signal_len) - 1.0) / fs)
        timestamps_out = torch.linspace(0.0, duration, t_steps, dtype=torch.float32)

    array_attrs = _array_attributes(mics)

    metadata: Dict[str, Any] = {
        "room": {
            "size": _to_serializable(room.size),
            "c": float(room.c),
            "beta": _to_serializable(room.beta) if room.beta is not None else None,
            "t60": float(room.t60) if room.t60 is not None else None,
            "fs": fs,
        },
        "sources": {
            "positions": _to_serializable(src_pos),
            "orientation": _to_serializable(sources.orientation),
        },
        "mics": {
            "positions": _to_serializable(mic_pos),
            "orientation": _to_serializable(mics.orientation),
        },
        "trajectories": {
            "sources": _to_serializable(src_traj_n if t_steps > 1 else None),
            "mics": _to_serializable(mic_traj_n if t_steps > 1 else None),
        },
        "array": {
            "geometry": array_attrs.geometry_name,
            "positions": _to_serializable(array_attrs.positions),
            "orientation": _to_serializable(array_attrs.orientation),
            "center": _to_serializable(array_attrs.center),
            "normal": _to_serializable(array_attrs.normal),
            "spacing": array_attrs.spacing,
        },
        "time_axis": time_axis,
        "doa": doa,
        "timestamps": _to_serializable(timestamps_out),
        "rirs_shape": list(rirs.shape),
        "dynamic": bool(t_steps > 1),
    }

    if source_info is not None:
        metadata["source_info"] = _to_serializable(source_info)
    if extra:
        metadata["extra"] = _to_serializable(extra)
    return metadata


def save_metadata_json(path: Path, metadata: Dict[str, Any]) -> None:
    """Save metadata as JSON to the given path.

    Examples:
        ```python
        save_metadata_json(Path(\"outputs/scene_metadata.json\"), metadata)
        ```
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def _normalize_traj(traj: Optional[Tensor], pos: Tensor, dim: int, name: str) -> Tensor:
    if traj is None:
        if pos.ndim != 2 or pos.shape[1] != dim:
            raise ValueError(f"{name} default positions must have shape (N, {dim})")
        return pos.unsqueeze(0)
    if not torch.is_tensor(traj):
        raise TypeError(f"{name} must be a Tensor")
    if traj.ndim == 2:
        if traj.shape[1] != dim:
            raise ValueError(f"{name} must have shape (T, {dim})")
        return traj.unsqueeze(1)
    if traj.ndim == 3:
        if traj.shape[2] != dim:
            raise ValueError(f"{name} must have shape (T, N, {dim})")
        return traj
    raise ValueError(f"{name} must have shape (T, N, {dim})")


def _compute_doa(src_traj: Tensor, mic_traj: Tensor) -> tuple[Tensor, Tensor]:
    vec = src_traj[:, :, None, :] - mic_traj[:, None, :, :]
    x = vec[..., 0]
    y = vec[..., 1]
    azimuth = torch.atan2(y, x)
    if vec.shape[-1] < 3:
        elevation = torch.zeros_like(azimuth)
    else:
        z = vec[..., 2]
        r_xy = torch.sqrt(x**2 + y**2)
        elevation = torch.atan2(z, r_xy)
    return azimuth, elevation


def _array_attributes(mics: MicrophoneArray) -> ArrayAttributes:
    pos = mics.positions
    n_mic = pos.shape[0]
    if n_mic == 1:
        geometry = "single"
    elif n_mic == 2:
        geometry = "binaural"
    else:
        geometry = "custom"
    center = pos.mean(dim=0)
    spacing = None
    if n_mic >= 2:
        dists = torch.cdist(pos, pos)
        dists = dists[dists > 0]
        if dists.numel() > 0:
            spacing = float(dists.min().item())
    return ArrayAttributes(
        geometry_name=geometry,
        positions=pos,
        orientation=mics.orientation,
        center=center,
        normal=None,
        spacing=spacing,
    )


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value
