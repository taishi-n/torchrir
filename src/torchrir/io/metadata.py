"""Metadata helpers for simulation outputs."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from importlib.metadata import PackageNotFoundError, version
import math
from pathlib import Path
import stat
import tempfile
from typing import Any, Literal, Mapping

import json
import numpy as np
import torch
from torch import Tensor

from ..models import (
    DynamicScene,
    MicrophoneArray,
    RIRResult,
    Room,
    Source,
    StaticScene,
)
from ..models.schedule import FrameSchedule
from ..models.scene import _validate_dynamic_time_reference
from ..util._dtypes import validate_supported_float_tensor
from ..util._scalars import normalize_integer
from ..util.tensor import stable_vector_norm


TimeReference = Literal["emission", "observation"]

_SCHEMA_NAME = "torchrir.scene"
_SCHEMA_VERSION = 1
_INT64_MAX = torch.iinfo(torch.int64).max


def build_metadata(
    *,
    room: Room,
    sources: Source,
    mics: MicrophoneArray,
    rirs: Tensor,
    src_traj: Tensor | None = None,
    mic_traj: Tensor | None = None,
    schedule: FrameSchedule | None = None,
    time_reference: TimeReference | None = None,
    signal_len: int | None = None,
    source_info: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
    is_dynamic = src_traj is not None or mic_traj is not None
    dim = int(room.size.numel())
    src_traj_n = _normalize_traj(src_traj, sources.positions, dim, "src_traj")
    mic_traj_n = _normalize_traj(mic_traj, mics.positions, dim, "mic_traj")

    t_steps = max(src_traj_n.shape[0], mic_traj_n.shape[0])
    if src_traj_n.shape[0] == 1 and t_steps > 1:
        src_traj_n = src_traj_n.expand(t_steps, -1, -1)
    if mic_traj_n.shape[0] == 1 and t_steps > 1:
        mic_traj_n = mic_traj_n.expand(t_steps, -1, -1)
    if src_traj_n.shape[0] != mic_traj_n.shape[0]:
        raise ValueError("src_traj and mic_traj must have matching time steps")
    if is_dynamic:
        scene: StaticScene | DynamicScene = DynamicScene(
            room=room,
            sources=sources,
            mics=mics,
            src_traj=src_traj_n,
            mic_traj=mic_traj_n,
            schedule=schedule,
        )
        schedule = scene.schedule
    else:
        if schedule is not None:
            raise ValueError("schedule is only valid with dynamic trajectories")
        scene = StaticScene(room=room, sources=sources, mics=mics)

    _validate_rirs_for_scene(rirs, scene)
    return _build_metadata_validated(
        scene=scene,
        rirs=rirs,
        schedule=schedule,
        time_reference=time_reference,
        signal_len=signal_len,
        source_info=source_info,
        extra=extra,
    )


def _build_metadata_validated(
    *,
    scene: StaticScene | DynamicScene,
    rirs: Tensor,
    schedule: FrameSchedule | None,
    time_reference: TimeReference | None,
    signal_len: int | None,
    source_info: Any,
    extra: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Serialize already-validated scene and RIR state without rescanning it."""

    room = scene.room
    sources = scene.sources
    mics = scene.mics
    fs = float(room.fs)
    nsample = int(rirs.shape[-1])
    if isinstance(scene, DynamicScene):
        is_dynamic = True
        src_traj = scene.src_traj
        mic_traj = scene.mic_traj
    else:
        is_dynamic = False
        src_traj = sources.positions.unsqueeze(0)
        mic_traj = mics.positions.unsqueeze(0)
    if signal_len is not None:
        signal_len = normalize_integer(
            signal_len,
            name="signal_len",
            minimum=1,
            maximum=_INT64_MAX,
        )
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping or None")
    if schedule is not None and not isinstance(schedule, FrameSchedule):
        raise TypeError("schedule must be a FrameSchedule or None")
    if schedule is not None:
        if not is_dynamic:
            raise ValueError("schedule is only valid with dynamic trajectories")
        if (
            schedule.conversion_sample_rate is not None
            and schedule.conversion_sample_rate != fs
        ):
            raise ValueError(
                "schedule seconds-conversion sample rate "
                f"{schedule.conversion_sample_rate:g} conflicts with "
                f"room sampling rate {fs:g}"
            )
        if len(schedule) != int(src_traj.shape[0]):
            raise ValueError(
                f"schedule has {len(schedule)} frames, but trajectories have "
                f"{int(src_traj.shape[0])}"
            )

    schedule_metadata: dict[str, Any] | None = None
    if schedule is not None:
        schedule_metadata = {
            "starts_samples": _validated_tensor_to_list(schedule.starts),
            "sample_rate": fs,
        }

    signal_metadata = (
        None if signal_len is None else {"sample_rate": fs, "sample_count": signal_len}
    )
    convolution_metadata: dict[str, Any] | None = None
    if time_reference is not None:
        if time_reference not in ("emission", "observation"):
            raise ValueError("time_reference must be 'emission' or 'observation'")
        if not is_dynamic or schedule is None:
            raise ValueError(
                "time_reference requires dynamic trajectories and a FrameSchedule"
            )
        if signal_len is None:
            raise ValueError("signal_len is required when time_reference is provided")
        assert isinstance(scene, DynamicScene)
        _validate_dynamic_time_reference(scene, time_reference=time_reference)
        if signal_len > _INT64_MAX - nsample + 1:
            raise ValueError(
                "signal_len and RIR length must produce an int64 output length"
            )
        endpoint = (
            signal_len if time_reference == "emission" else signal_len + nsample - 1
        )
        if schedule.starts[-1].item() >= endpoint:
            raise ValueError(
                f"last frame start must be before the {time_reference}-time endpoint"
            )
        convolution_metadata = {
            "time_reference": time_reference,
            "output_sample_count": signal_len + nsample - 1,
        }

    azimuth, elevation = _compute_doa(src_traj, mic_traj)
    center, minimum_pair_distance = _microphone_layout(mics)
    metadata: dict[str, Any] = {
        "schema": {"name": _SCHEMA_NAME, "version": _SCHEMA_VERSION},
        "generator": {
            "name": "torchrir",
            "version": _distribution_version(),
            "torch_version": str(torch.__version__),
        },
        "room": {
            "size": _validated_tensor_to_list(room.size),
            "c": float(room.c),
            "beta": (
                _validated_tensor_to_list(room.beta) if room.beta is not None else None
            ),
            "t60": float(room.t60) if room.t60 is not None else None,
            "fs": fs,
        },
        "sources": {
            "positions": _validated_tensor_to_list(sources.positions),
            "orientation": (
                _validated_tensor_to_list(sources.orientation)
                if sources.orientation is not None
                else None
            ),
            "directivity": sources.directivity,
        },
        "mics": {
            "positions": _validated_tensor_to_list(mics.positions),
            "orientation": (
                _validated_tensor_to_list(mics.orientation)
                if mics.orientation is not None
                else None
            ),
            "directivity": mics.directivity,
            "layout": {
                "kind": "single" if mics.positions.shape[0] == 1 else "custom",
                "center": _validated_tensor_to_list(center),
                "minimum_pair_distance": minimum_pair_distance,
            },
        },
        "trajectories": {
            "sources": _validated_tensor_to_list(src_traj) if is_dynamic else None,
            "mics": _validated_tensor_to_list(mic_traj) if is_dynamic else None,
        },
        "rir": {
            "shape": list(rirs.shape),
            "sample_rate": fs,
            "sample_count": nsample,
            "origin_sample": 0,
        },
        "doa": {
            "frame": "world",
            "unit": "radians",
            "azimuth": _validated_tensor_to_list(azimuth),
            "elevation": _validated_tensor_to_list(elevation),
        },
        "frame_schedule": schedule_metadata,
        "signal": signal_metadata,
        "convolution": convolution_metadata,
        "dynamic": is_dynamic,
    }

    if source_info is not None:
        metadata["source_info"] = _to_serializable(source_info)
    if extra is not None:
        metadata["extra"] = _to_serializable(extra)
    return metadata


def build_result_metadata(
    result: RIRResult,
    *,
    schedule: FrameSchedule | None = None,
    time_reference: TimeReference | None = None,
    signal_len: int | None = None,
    source_info: Any = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata from a scene-oriented simulation result."""

    result.validate()
    scene = result.scene
    scene_schedule = scene.schedule if isinstance(scene, DynamicScene) else None
    if scene_schedule is not None and schedule is not None:
        raise ValueError(
            "schedule must be omitted when DynamicScene contains a schedule"
        )
    effective_schedule = scene_schedule if scene_schedule is not None else schedule
    metadata = _build_metadata_validated(
        scene=scene,
        rirs=result.rirs,
        schedule=effective_schedule,
        time_reference=time_reference,
        signal_len=signal_len,
        source_info=source_info,
        extra=extra,
    )
    metadata["simulation"] = {
        "method": "ism",
        "config": _to_serializable(result.config),
    }
    return metadata


def save_metadata_json(path: Path, metadata: dict[str, Any]) -> None:
    """Save metadata as JSON to the given path.

    Examples:
        ```python
        save_metadata_json(Path(\"outputs/scene_metadata.json\"), metadata)
        ```
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    with tempfile.TemporaryDirectory(
        dir=path.parent,
        prefix=f".{path.name}.",
    ) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        with temporary_path.open(
            "x",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2, allow_nan=False)
            file.write("\n")
        if existing_mode is not None:
            temporary_path.chmod(existing_mode)
        temporary_path.replace(path)


def _normalize_traj(traj: Tensor | None, pos: Tensor, dim: int, name: str) -> Tensor:
    if traj is None:
        if pos.ndim != 2 or pos.shape[1] != dim:
            raise ValueError(f"{name} default positions must have shape (N, {dim})")
        return pos.unsqueeze(0)
    if not torch.is_tensor(traj):
        raise TypeError(f"{name} must be a Tensor")
    validate_supported_float_tensor(traj, name=name)
    if traj.device != pos.device or traj.dtype != pos.dtype:
        raise ValueError(f"{name} device and dtype must match entity positions")
    if not torch.all(torch.isfinite(traj)):
        raise ValueError(f"{name} must contain finite values")
    if traj.ndim == 2:
        if pos.shape[0] != 1:
            raise ValueError(f"{name} must include the {pos.shape[0]} entity axis")
        if traj.shape[1] != dim:
            raise ValueError(f"{name} must have shape (T, {dim})")
        if traj.shape[0] == 0:
            raise ValueError(f"{name} must contain at least one frame")
        return traj.unsqueeze(1)
    if traj.ndim == 3:
        if traj.shape[1:] != (pos.shape[0], dim) or traj.shape[0] == 0:
            raise ValueError(
                f"{name} must have shape (T, {pos.shape[0]}, {dim}) with T > 0"
            )
        return traj
    raise ValueError(f"{name} must have shape (T, N, {dim})")


def _compute_doa(src_traj: Tensor, mic_traj: Tensor) -> tuple[Tensor, Tensor]:
    # Metadata geometry is serialized on CPU. Float64 avoids half/bfloat16
    # angle quantization and backend-specific atan2 limitations.
    sources = src_traj.detach().cpu().to(torch.float64)
    microphones = mic_traj.detach().cpu().to(torch.float64)
    vec = sources[:, :, None, :] - microphones[:, None, :, :]
    x = vec[..., 0]
    y = vec[..., 1]
    azimuth = torch.atan2(y, x)
    if vec.shape[-1] < 3:
        elevation = torch.zeros_like(azimuth)
    else:
        z = vec[..., 2]
        r_xy = torch.hypot(x, y)
        elevation = torch.atan2(z, r_xy)
    return azimuth, elevation


def _validate_rirs_for_scene(
    rirs: Tensor,
    scene: StaticScene | DynamicScene,
) -> None:
    if not torch.is_tensor(rirs):
        raise TypeError("rirs must be a Tensor")
    validate_supported_float_tensor(rirs, name="rirs")
    if isinstance(scene, DynamicScene):
        is_dynamic = True
        expected_leading_shape = (
            int(scene.src_traj.shape[0]),
            int(scene.sources.positions.shape[0]),
            int(scene.mics.positions.shape[0]),
        )
    else:
        is_dynamic = False
        expected_leading_shape = (
            int(scene.sources.positions.shape[0]),
            int(scene.mics.positions.shape[0]),
        )
    expected_ndim = 4 if is_dynamic else 3
    if rirs.ndim != expected_ndim or tuple(rirs.shape[:-1]) != expected_leading_shape:
        kind = "dynamic" if is_dynamic else "static"
        raise ValueError(
            f"{kind} rirs must have leading shape {expected_leading_shape}, got "
            f"{tuple(rirs.shape[:-1])}"
        )
    if rirs.shape[-1] == 0:
        raise ValueError("rirs must contain at least one sample")
    if not torch.all(torch.isfinite(rirs)):
        raise ValueError("rirs must contain finite values")


def _microphone_layout(mics: MicrophoneArray) -> tuple[Tensor, float | None]:
    # Metadata geometry is small and device-independent. CPU float64 avoids
    # unsupported cdist kernels for half/bfloat16 and MPS geometry tensors.
    pos = mics.positions.detach().cpu().to(torch.float64)
    scale = pos.abs().amax(dim=0)
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    center = (pos / safe_scale).mean(dim=0) * scale
    minimum_pair_distance = None
    if pos.shape[0] >= 2:
        minimum_pair_distance = min(
            float(
                stable_vector_norm(pos[index + 1 :] - pos[index], dim=-1).min().item()
            )
            for index in range(pos.shape[0] - 1)
        )
    return center, minimum_pair_distance


def _to_serializable(value: Any, _active_ids: set[int] | None = None) -> Any:
    active_ids = set() if _active_ids is None else _active_ids
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.layout != torch.strided or value.is_nested:
            raise TypeError("metadata tensors must use a dense strided layout")
        if value.is_quantized:
            raise TypeError("metadata tensors must not be quantized")
        if value.device.type not in ("cpu", "cuda", "mps"):
            raise TypeError("metadata tensors must be on CPU, CUDA, or MPS")
        if value.is_complex():
            raise TypeError("metadata tensors must not use complex dtypes")
        if value.is_floating_point():
            validate_supported_float_tensor(value, name="metadata tensor")
            if not torch.all(torch.isfinite(value)):
                raise ValueError("metadata tensors must contain finite values")
        return _to_serializable(value.detach().cpu().tolist(), active_ids)
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            raise TypeError("metadata arrays must not use complex dtypes")
        return _serialize_recursive_container(value, value.tolist(), active_ids)
    if isinstance(value, np.complexfloating):
        raise TypeError("metadata numbers must be real")
    if isinstance(value, np.floating):
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(
                "metadata numbers must be finite and representable as JSON numbers"
            )
        return normalized
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.generic):
        raise TypeError(
            f"metadata NumPy scalar of type {type(value).__name__} is not JSON "
            "serializable"
        )
    if isinstance(value, (torch.device, torch.dtype)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    if isinstance(value, complex):
        raise TypeError("metadata numbers must be real")
    if isinstance(value, (str, int, bool)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        identity = id(value)
        _enter_serialization_container(identity, active_ids)
        try:
            return {
                item.name: _to_serializable(getattr(value, item.name), active_ids)
                for item in fields(value)
            }
        finally:
            active_ids.remove(identity)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata mapping keys must be strings")
        identity = id(value)
        _enter_serialization_container(identity, active_ids)
        try:
            return {
                key: _to_serializable(item, active_ids) for key, item in value.items()
            }
        finally:
            active_ids.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        _enter_serialization_container(identity, active_ids)
        try:
            return [_to_serializable(item, active_ids) for item in value]
        finally:
            active_ids.remove(identity)
    raise TypeError(
        f"metadata value of type {type(value).__name__} is not JSON serializable"
    )


def _serialize_recursive_container(
    owner: object,
    normalized: object,
    active_ids: set[int],
) -> Any:
    identity = id(owner)
    _enter_serialization_container(identity, active_ids)
    try:
        return _to_serializable(normalized, active_ids)
    finally:
        active_ids.remove(identity)


def _enter_serialization_container(identity: int, active_ids: set[int]) -> None:
    if identity in active_ids:
        raise ValueError("metadata values must not contain cycles")
    active_ids.add(identity)


def _validated_tensor_to_list(value: Tensor) -> Any:
    """Serialize a tensor whose finite/layout invariants were checked upstream."""

    return value.detach().cpu().tolist()


def _distribution_version() -> str:
    try:
        return version("torchrir")
    except PackageNotFoundError:  # pragma: no cover - source tree without install
        return "unknown"
