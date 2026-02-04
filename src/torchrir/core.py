from __future__ import annotations

"""Core RIR simulation functions (static and dynamic)."""

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

from .config import get_config
from .directivity import directivity_gain, split_directivity
from .room import MicrophoneArray, Room, Source
from .utils import (
    as_tensor,
    ensure_dim,
    estimate_beta_from_t60,
    estimate_t60_from_beta,
    infer_device_dtype,
    normalize_orientation,
    orientation_to_unit,
    resolve_device,
)


def simulate_rir(
    *,
    room: Room,
    sources: Source | Tensor,
    mics: MicrophoneArray | Tensor,
    max_order: int,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate a static RIR using the image source method.

    Args:
        room: Room configuration (geometry, fs, reflection coefficients).
        sources: Source positions or a Source object.
        mics: Microphone positions or a MicrophoneArray object.
        max_order: Maximum reflection order.
        nb_img: Optional per-dimension image counts (overrides max_order).
        nsample: Output length in samples.
        tmax: Output length in seconds (used if nsample is None).
        tdiff: Optional time to start diffuse tail modeling.
        directivity: Directivity pattern(s) for source and mic.
        orientation: Orientation vectors or angles.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tensor of shape (n_src, n_mic, nsample).
    """
    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None and tmax is None:
        raise ValueError("nsample or tmax must be provided")
    if nsample is None:
        nsample = int(math.ceil(tmax * room.fs))
    if nsample <= 0:
        raise ValueError("nsample must be positive")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")

    if isinstance(device, str):
        device = resolve_device(device)

    src_pos, src_ori = _prepare_entities(
        sources, orientation, which="source", device=device, dtype=dtype
    )
    mic_pos, mic_ori = _prepare_entities(
        mics, orientation, which="mic", device=device, dtype=dtype
    )

    device, dtype = infer_device_dtype(
        src_pos, mic_pos, room.size, device=device, dtype=dtype
    )
    src_pos = as_tensor(src_pos, device=device, dtype=dtype)
    mic_pos = as_tensor(mic_pos, device=device, dtype=dtype)

    if src_ori is not None:
        src_ori = as_tensor(src_ori, device=device, dtype=dtype)
    if mic_ori is not None:
        mic_ori = as_tensor(mic_ori, device=device, dtype=dtype)

    room_size = as_tensor(room.size, device=device, dtype=dtype)
    room_size = ensure_dim(room_size)
    dim = room_size.numel()

    if src_pos.ndim == 1:
        src_pos = src_pos.unsqueeze(0)
    if mic_pos.ndim == 1:
        mic_pos = mic_pos.unsqueeze(0)
    if src_pos.ndim != 2 or src_pos.shape[1] != dim:
        raise ValueError("sources must be of shape (n_src, dim)")
    if mic_pos.ndim != 2 or mic_pos.shape[1] != dim:
        raise ValueError("mics must be of shape (n_mic, dim)")

    beta = _resolve_beta(room, room_size, device=device, dtype=dtype)
    beta = _validate_beta(beta, dim)

    n_vec = _image_source_indices(max_order, dim, device=device, nb_img=nb_img)
    refl = _reflection_coefficients(n_vec, beta)

    src_pattern, mic_pattern = split_directivity(directivity)

    n_src = src_pos.shape[0]
    n_mic = mic_pos.shape[0]
    rir = torch.zeros((n_src, n_mic, nsample), device=device, dtype=dtype)

    for s_idx in range(n_src):
        img = _image_positions(src_pos[s_idx], room_size, n_vec)
        vec = mic_pos[:, None, :] - img[None, :, :]
        dist = torch.linalg.norm(vec, dim=-1)
        dist = torch.clamp(dist, min=1e-6)
        time = dist / room.c
        cfg = get_config()
        fdl = cfg["frac_delay_length"]
        fdl2 = (fdl - 1) // 2
        time = time + (fdl2 / room.fs)
        sample = time * room.fs

        gain = refl[None, :]
        if src_pattern != "omni":
            if src_ori is None:
                raise ValueError("source orientation required for non-omni directivity")
            ori = _select_orientation(src_ori, s_idx, n_src, dim)
            cos_theta = _cos_between(vec, ori)
            gain = gain * directivity_gain(src_pattern, cos_theta)
        if mic_pattern != "omni":
            if mic_ori is None:
                raise ValueError("mic orientation required for non-omni directivity")
            mic_dir = orientation_to_unit(mic_ori, dim)
            cos_theta = _cos_between(-vec, mic_dir)
            gain = gain * directivity_gain(mic_pattern, cos_theta)

        attenuation = gain / dist
        _accumulate_rir(rir[s_idx], sample, attenuation)

    if tdiff is not None and tmax is not None and tdiff < tmax:
        rir = _apply_diffuse_tail(rir, room, beta, tdiff, tmax)
    return rir


def simulate_dynamic_rir(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    max_order: int,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    directivity: str | tuple[str, str] = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate time-varying RIRs for source/mic trajectories.

    Args:
        room: Room configuration.
        src_traj: Source trajectory (T, n_src, dim).
        mic_traj: Microphone trajectory (T, n_mic, dim).
        max_order: Maximum reflection order.
        nsample: Output length in samples.
        tmax: Output length in seconds (used if nsample is None).
        directivity: Directivity pattern(s) for source and mic.
        orientation: Orientation vectors or angles.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tensor of shape (T, n_src, n_mic, nsample).
    """
    if isinstance(device, str):
        device = resolve_device(device)

    src_traj = as_tensor(src_traj, device=device, dtype=dtype)
    mic_traj = as_tensor(mic_traj, device=device, dtype=dtype)

    if src_traj.ndim == 2:
        src_traj = src_traj.unsqueeze(1)
    if mic_traj.ndim == 2:
        mic_traj = mic_traj.unsqueeze(1)
    if src_traj.ndim != 3:
        raise ValueError("src_traj must be of shape (T, n_src, dim)")
    if mic_traj.ndim != 3:
        raise ValueError("mic_traj must be of shape (T, n_mic, dim)")
    if src_traj.shape[0] != mic_traj.shape[0]:
        raise ValueError("src_traj and mic_traj must have the same time length")

    t_steps = src_traj.shape[0]
    rirs = []
    for t_idx in range(t_steps):
        rir = simulate_rir(
            room=room,
            sources=src_traj[t_idx],
            mics=mic_traj[t_idx],
            max_order=max_order,
            nsample=nsample,
            tmax=tmax,
            directivity=directivity,
            orientation=orientation,
            device=device,
            dtype=dtype,
        )
        rirs.append(rir)
    return torch.stack(rirs, dim=0)


def _prepare_entities(
    entities: Source | MicrophoneArray | Tensor,
    orientation: Optional[Tensor | tuple[Tensor, Tensor]],
    *,
    which: str,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> Tuple[Tensor, Optional[Tensor]]:
    """Extract positions and orientations from entities or raw tensors.

    Returns:
        Tuple of (positions, orientation).
    """
    if isinstance(entities, (Source, MicrophoneArray)):
        pos = entities.positions
        ori = entities.orientation
    else:
        pos = entities
        ori = None
    if orientation is not None:
        if isinstance(orientation, (list, tuple)):
            if len(orientation) != 2:
                raise ValueError("orientation tuple must have length 2")
            ori = orientation[0] if which == "source" else orientation[1]
        else:
            ori = orientation
    pos = as_tensor(pos, device=device, dtype=dtype)
    if ori is not None:
        ori = as_tensor(ori, device=device, dtype=dtype)
    return pos, ori


def _resolve_beta(
    room: Room, room_size: Tensor, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Resolve reflection coefficients from beta/t60/defaults.

    Returns:
        Tensor of reflection coefficients per wall.
    """
    if room.beta is not None:
        return as_tensor(room.beta, device=device, dtype=dtype)
    if room.t60 is not None:
        return estimate_beta_from_t60(room_size, room.t60, device=device, dtype=dtype)
    dim = room_size.numel()
    default_faces = 4 if dim == 2 else 6
    return torch.ones((default_faces,), device=device, dtype=dtype)


def _validate_beta(beta: Tensor, dim: int) -> Tensor:
    """Validate beta size against room dimension.

    Returns:
        The validated beta tensor.
    """
    expected = 4 if dim == 2 else 6
    if beta.numel() != expected:
        raise ValueError(f"beta must have {expected} elements for {dim}D")
    return beta


def _image_source_indices(
    max_order: int,
    dim: int,
    *,
    device: torch.device,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
) -> Tensor:
    """Generate image source index vectors up to the given order.

    Returns:
        Tensor of shape (n_images, dim).
    """
    if nb_img is not None:
        nb = as_tensor(nb_img, device=device, dtype=torch.int64)
        if nb.numel() != dim:
            raise ValueError("nb_img must match room dimension")
        ranges = [torch.arange(-n, n + 1, device=device, dtype=torch.int64) for n in nb]
        grids = torch.meshgrid(*ranges, indexing="ij")
        return torch.stack([g.reshape(-1) for g in grids], dim=-1)
    rng = torch.arange(-max_order, max_order + 1, device=device, dtype=torch.int64)
    grids = torch.meshgrid(*([rng] * dim), indexing="ij")
    n_vec = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    order = torch.sum(torch.abs(n_vec), dim=-1)
    return n_vec[order <= max_order]


def _image_positions(src: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image source positions for a given source.

    Returns:
        Tensor of image positions (n_images, dim).
    """
    n_vec_f = n_vec.to(dtype=src.dtype)
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(dtype=src.dtype)
    n = torch.floor_divide(n_vec + 1, 2).to(dtype=src.dtype)
    return 2.0 * room_size * n + sign * src


def _reflection_coefficients(n_vec: Tensor, beta: Tensor) -> Tensor:
    """Compute reflection coefficients for each image source.

    Returns:
        Tensor of shape (n_images,) with per-image gains.
    """
    dim = n_vec.shape[1]
    beta = beta.view(dim, 2)
    beta_lo = beta[:, 0]
    beta_hi = beta[:, 1]

    n = n_vec
    k = torch.abs(n)
    n_hi = torch.where(n >= 0, (n + 1) // 2, k // 2)
    n_lo = torch.where(n >= 0, n // 2, (k + 1) // 2)

    n_hi = n_hi.to(dtype=beta.dtype)
    n_lo = n_lo.to(dtype=beta.dtype)

    coeff = (beta_hi**n_hi) * (beta_lo**n_lo)
    return torch.prod(coeff, dim=1)


def _select_orientation(orientation: Tensor, idx: int, count: int, dim: int) -> Tensor:
    """Pick the correct orientation vector for a given entity index."""
    if orientation.ndim == 0:
        return orientation_to_unit(orientation, dim)
    if orientation.ndim == 1:
        return orientation_to_unit(orientation, dim)
    if orientation.ndim == 2 and orientation.shape[0] == count:
        return orientation_to_unit(orientation[idx], dim)
    raise ValueError("orientation must be shape (dim,), (count, dim), or angles")


def _cos_between(vec: Tensor, orientation: Tensor) -> Tensor:
    """Compute cosine between direction vectors and orientation."""
    orientation = normalize_orientation(orientation)
    unit = vec / torch.linalg.norm(vec, dim=-1, keepdim=True)
    return torch.sum(unit * orientation, dim=-1)


def _accumulate_rir(rir: Tensor, sample: Tensor, amplitude: Tensor) -> None:
    """Accumulate fractional-delay contributions into the RIR tensor."""
    idx0 = torch.floor(sample).to(torch.int64)
    frac = sample - idx0.to(sample.dtype)

    n_mic, nsample = rir.shape
    cfg = get_config()
    fdl = cfg["frac_delay_length"]
    lut_gran = cfg["sinc_lut_granularity"]
    use_lut = cfg["use_lut"]
    fdl2 = (fdl - 1) // 2

    n = torch.arange(fdl, device=rir.device, dtype=sample.dtype)
    offsets = n - fdl2
    window = torch.hann_window(fdl, periodic=False, device=rir.device, dtype=sample.dtype)

    if use_lut:
        sinc_lut = _get_sinc_lut(fdl, lut_gran, device=rir.device, dtype=sample.dtype)

    for m in range(n_mic):
        idx = idx0[m]
        amp = amplitude[m]
        frac_m = frac[m]

        if use_lut:
            x_off_frac = (1.0 - frac_m) * lut_gran
            lut_gran_off = torch.floor(x_off_frac).to(torch.int64)
            x_off = x_off_frac - lut_gran_off.to(sample.dtype)
            lut_pos = lut_gran_off[:, None] + (n[None, :].to(torch.int64) * lut_gran)

            s0 = torch.take(sinc_lut, lut_pos)
            s1 = torch.take(sinc_lut, lut_pos + 1)
            interp = s0 + x_off[:, None] * (s1 - s0)
            filt = interp * window[None, :]
        else:
            t = n[None, :] - fdl2 - frac_m[:, None]
            filt = torch.sinc(t) * window[None, :]

        contrib = amp[:, None] * filt
        target = idx[:, None] + offsets[None, :]
        valid = (target >= 0) & (target < nsample)
        if not valid.any():
            continue

        target = target[valid].to(torch.int64)
        values = contrib[valid]
        rir[m].scatter_add_(0, target, values)


def _get_sinc_lut(fdl: int, lut_gran: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create a sinc lookup table for fractional delays."""
    fdl2 = (fdl - 1) // 2
    lut_size = (fdl + 1) * lut_gran + 1
    n = torch.linspace(-fdl2 - 1, fdl2 + 1, lut_size, device=device, dtype=dtype)
    return torch.sinc(n)


def _apply_diffuse_tail(rir: Tensor, room: Room, beta: Tensor, tdiff: float, tmax: float) -> Tensor:
    """Apply a diffuse reverberation tail after tdiff.

    Returns:
        RIR tensor with diffuse tail applied.
    """
    nsample = rir.shape[-1]
    tdiff_idx = min(nsample - 1, int(math.floor(tdiff * room.fs)))
    if tdiff_idx <= 0:
        return rir
    tail_len = nsample - tdiff_idx
    t = torch.arange(tail_len, device=rir.device, dtype=rir.dtype) / room.fs

    t60 = estimate_t60_from_beta(room.size, beta)
    if math.isinf(t60) or t60 <= 0:
        decay = torch.exp(-t * 3.0)
    else:
        tau = t60 / 6.9078
        decay = torch.exp(-t / tau)

    gen = torch.Generator(device=rir.device)
    gen.manual_seed(0)
    noise = torch.randn(rir[..., tdiff_idx:].shape, device=rir.device, dtype=rir.dtype, generator=gen)
    scale = torch.linalg.norm(rir[..., tdiff_idx - 1 : tdiff_idx], dim=-1, keepdim=True) + 1e-8
    rir[..., tdiff_idx:] = noise * decay * scale
    return rir
