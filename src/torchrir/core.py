from __future__ import annotations

"""Core RIR simulation functions (static and dynamic)."""

import math
from collections.abc import Callable
from typing import Optional, Tuple

import torch
from torch import Tensor

from .config import SimulationConfig, default_config
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
    max_order: int | None,
    nb_img: Optional[Tensor | Tuple[int, ...]] = None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    tdiff: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate a static RIR using the image source method.

    Args:
        room: Room configuration (geometry, fs, reflection coefficients).
        sources: Source positions or a Source object.
        mics: Microphone positions or a MicrophoneArray object.
        max_order: Maximum reflection order (uses config if None).
        nb_img: Optional per-dimension image counts (overrides max_order).
        nsample: Output length in samples.
        tmax: Output length in seconds (used if nsample is None).
        tdiff: Optional time to start diffuse tail modeling.
        directivity: Directivity pattern(s) for source and mic (uses config if None).
        orientation: Orientation vectors or angles.
        config: Optional simulation configuration overrides.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tensor of shape (n_src, n_mic, nsample).

    Example:
        >>> room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
        >>> sources = Source.from_positions([[1.0, 2.0, 1.5]])
        >>> mics = MicrophoneArray.from_positions([[2.0, 2.0, 1.5]])
        >>> rir = simulate_rir(
        ...     room=room,
        ...     sources=sources,
        ...     mics=mics,
        ...     max_order=6,
        ...     tmax=0.3,
        ... )
    """
    cfg = config or default_config()
    cfg.validate()

    if device is None and cfg.device is not None:
        device = cfg.device

    if max_order is None:
        if cfg.max_order is None:
            raise ValueError("max_order must be provided if not set in config")
        max_order = cfg.max_order

    if tmax is None and nsample is None and cfg.tmax is not None:
        tmax = cfg.tmax

    if directivity is None:
        directivity = cfg.directivity or "omni"

    if not isinstance(room, Room):
        raise TypeError("room must be a Room instance")
    if nsample is None:
        if tmax is None:
            raise ValueError("nsample or tmax must be provided")
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
    mic_dir = None
    if mic_pattern != "omni":
        if mic_ori is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = orientation_to_unit(mic_ori, dim)

    n_src = src_pos.shape[0]
    n_mic = mic_pos.shape[0]
    rir = torch.zeros((n_src, n_mic, nsample), device=device, dtype=dtype)
    fdl = cfg.frac_delay_length
    fdl2 = (fdl - 1) // 2
    img_chunk = cfg.image_chunk_size
    if img_chunk <= 0:
        img_chunk = n_vec.shape[0]

    src_dirs = None
    if src_pattern != "omni":
        if src_ori is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = orientation_to_unit(src_ori, dim)
        if src_dirs.ndim == 1:
            src_dirs = src_dirs.unsqueeze(0).repeat(n_src, 1)
        if src_dirs.ndim != 2 or src_dirs.shape[0] != n_src:
            raise ValueError("source orientation must match number of sources")

    for start in range(0, n_vec.shape[0], img_chunk):
        end = min(start + img_chunk, n_vec.shape[0])
        n_vec_chunk = n_vec[start:end]
        refl_chunk = refl[start:end]
        sample_chunk, attenuation_chunk = _compute_image_contributions_batch(
            src_pos,
            mic_pos,
            room_size,
            n_vec_chunk,
            refl_chunk,
            room,
            fdl2,
            src_pattern=src_pattern,
            mic_pattern=mic_pattern,
            src_dirs=src_dirs,
            mic_dir=mic_dir,
        )
        _accumulate_rir_batch(rir, sample_chunk, attenuation_chunk, cfg)

    if tdiff is not None and tmax is not None and tdiff < tmax:
        rir = _apply_diffuse_tail(rir, room, beta, tdiff, tmax, seed=cfg.seed)
    return rir


def simulate_dynamic_rir(
    *,
    room: Room,
    src_traj: Tensor,
    mic_traj: Tensor,
    max_order: int | None,
    nsample: Optional[int] = None,
    tmax: Optional[float] = None,
    directivity: str | tuple[str, str] | None = "omni",
    orientation: Optional[Tensor | tuple[Tensor, Tensor]] = None,
    config: Optional[SimulationConfig] = None,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Simulate time-varying RIRs for source/mic trajectories.

    Args:
        room: Room configuration.
        src_traj: Source trajectory (T, n_src, dim).
        mic_traj: Microphone trajectory (T, n_mic, dim).
        max_order: Maximum reflection order (uses config if None).
        nsample: Output length in samples.
        tmax: Output length in seconds (used if nsample is None).
        directivity: Directivity pattern(s) for source and mic (uses config if None).
        orientation: Orientation vectors or angles.
        config: Optional simulation configuration overrides.
        device: Output device.
        dtype: Output dtype.

    Returns:
        Tensor of shape (T, n_src, n_mic, nsample).

    Example:
        >>> room = Room.shoebox(size=[6.0, 4.0, 3.0], fs=16000, beta=[0.9] * 6)
        >>> from torchrir import linear_trajectory
        >>> src_traj = torch.stack(
        ...     [linear_trajectory(torch.tensor([1.0, 2.0, 1.5]),
        ...                        torch.tensor([4.0, 2.0, 1.5]), 8)],
        ...     dim=1,
        ... )
        >>> mic_pos = torch.tensor([[2.0, 2.0, 1.5]])
        >>> mic_traj = mic_pos.unsqueeze(0).repeat(8, 1, 1)
        >>> rirs = simulate_dynamic_rir(
        ...     room=room,
        ...     src_traj=src_traj,
        ...     mic_traj=mic_traj,
        ...     max_order=4,
        ...     tmax=0.3,
        ... )
    """
    cfg = config or default_config()
    cfg.validate()

    if device is None and cfg.device is not None:
        device = cfg.device

    if max_order is None:
        if cfg.max_order is None:
            raise ValueError("max_order must be provided if not set in config")
        max_order = cfg.max_order

    if tmax is None and nsample is None and cfg.tmax is not None:
        tmax = cfg.tmax

    if directivity is None:
        directivity = cfg.directivity or "omni"

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
            config=config,
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


def _image_positions_batch(src_pos: Tensor, room_size: Tensor, n_vec: Tensor) -> Tensor:
    """Compute image source positions for multiple sources."""
    sign = torch.where((n_vec % 2) == 0, 1.0, -1.0).to(dtype=src_pos.dtype)
    n = torch.floor_divide(n_vec + 1, 2).to(dtype=src_pos.dtype)
    base = 2.0 * room_size * n
    return base[None, :, :] + sign[None, :, :] * src_pos[:, None, :]


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


def _compute_image_contributions(
    src: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    fdl2: int,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dir: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute sample positions and attenuation for a source and all mics."""
    img = _image_positions(src, room_size, n_vec)
    vec = mic_pos[:, None, :] - img[None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    dist = torch.clamp(dist, min=1e-6)
    time = dist / room.c
    time = time + (fdl2 / room.fs)
    sample = time * room.fs

    gain = refl[None, :]
    if src_pattern != "omni":
        if src_dir is None:
            raise ValueError("source orientation required for non-omni directivity")
        cos_theta = _cos_between(vec, src_dir)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        cos_theta = _cos_between(-vec, mic_dir)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = gain / dist
    return sample, attenuation


def _compute_image_contributions_batch(
    src_pos: Tensor,
    mic_pos: Tensor,
    room_size: Tensor,
    n_vec: Tensor,
    refl: Tensor,
    room: Room,
    fdl2: int,
    *,
    src_pattern: str,
    mic_pattern: str,
    src_dirs: Optional[Tensor],
    mic_dir: Optional[Tensor],
) -> Tuple[Tensor, Tensor]:
    """Compute samples/attenuation for all sources/mics/images in batch."""
    img = _image_positions_batch(src_pos, room_size, n_vec)
    vec = mic_pos[None, :, None, :] - img[:, None, :, :]
    dist = torch.linalg.norm(vec, dim=-1)
    dist = torch.clamp(dist, min=1e-6)
    time = dist / room.c
    time = time + (fdl2 / room.fs)
    sample = time * room.fs

    gain = refl.view(1, 1, -1)
    if src_pattern != "omni":
        if src_dirs is None:
            raise ValueError("source orientation required for non-omni directivity")
        src_dirs = src_dirs[:, None, None, :]
        cos_theta = _cos_between(vec, src_dirs)
        gain = gain * directivity_gain(src_pattern, cos_theta)
    if mic_pattern != "omni":
        if mic_dir is None:
            raise ValueError("mic orientation required for non-omni directivity")
        mic_dir = (
            mic_dir[None, :, None, :]
            if mic_dir.ndim == 2
            else mic_dir.view(1, 1, 1, -1)
        )
        cos_theta = _cos_between(-vec, mic_dir)
        gain = gain * directivity_gain(mic_pattern, cos_theta)

    attenuation = gain / dist
    return sample, attenuation


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


def _accumulate_rir(
    rir: Tensor, sample: Tensor, amplitude: Tensor, cfg: SimulationConfig
) -> None:
    """Accumulate fractional-delay contributions into the RIR tensor."""
    idx0 = torch.floor(sample).to(torch.int64)
    frac = sample - idx0.to(sample.dtype)

    n_mic, nsample = rir.shape
    fdl = cfg.frac_delay_length
    lut_gran = cfg.sinc_lut_granularity
    use_lut = cfg.use_lut and rir.device.type != "mps"
    fdl2 = (fdl - 1) // 2

    dtype = amplitude.dtype
    n = _get_fdl_grid(fdl, device=rir.device, dtype=dtype)
    offsets = _get_fdl_offsets(fdl, device=rir.device)
    window = _get_fdl_window(fdl, device=rir.device, dtype=dtype)

    if use_lut:
        sinc_lut = _get_sinc_lut(fdl, lut_gran, device=rir.device, dtype=dtype)

    mic_offsets = (
        torch.arange(n_mic, device=rir.device, dtype=torch.int64) * nsample
    ).view(n_mic, 1, 1)
    rir_flat = rir.view(-1)

    chunk_size = cfg.accumulate_chunk_size
    n_img = idx0.shape[1]
    for start in range(0, n_img, chunk_size):
        end = min(start + chunk_size, n_img)
        idx = idx0[:, start:end]
        amp = amplitude[:, start:end]
        frac_m = frac[:, start:end]

        if use_lut:
            x_off_frac = (1.0 - frac_m) * lut_gran
            lut_gran_off = torch.floor(x_off_frac).to(torch.int64)
            x_off = x_off_frac - lut_gran_off.to(dtype)
            lut_pos = lut_gran_off[..., None] + (
                n[None, None, :].to(torch.int64) * lut_gran
            )

            s0 = torch.take(sinc_lut, lut_pos)
            s1 = torch.take(sinc_lut, lut_pos + 1)
            interp = s0 + x_off[..., None] * (s1 - s0)
            filt = interp * window[None, None, :]
        else:
            t = n[None, None, :] - fdl2 - frac_m[..., None]
            filt = torch.sinc(t) * window[None, None, :]

        contrib = amp[..., None] * filt
        target = idx[..., None] + offsets[None, None, :]
        valid = (target >= 0) & (target < nsample)
        if not valid.any():
            continue

        target = target + mic_offsets
        target_flat = target[valid].to(torch.int64)
        values_flat = contrib[valid]
        rir_flat.scatter_add_(0, target_flat, values_flat)


def _accumulate_rir_batch(
    rir: Tensor, sample: Tensor, amplitude: Tensor, cfg: SimulationConfig
) -> None:
    """Accumulate fractional-delay contributions for all sources/mics."""
    fn = _get_accumulate_fn(cfg, rir.device, amplitude.dtype)
    return fn(rir, sample, amplitude)


def _accumulate_rir_batch_impl(
    rir: Tensor,
    sample: Tensor,
    amplitude: Tensor,
    *,
    fdl: int,
    lut_gran: int,
    use_lut: bool,
    chunk_size: int,
) -> None:
    """Implementation for batch accumulation (optionally compiled)."""
    idx0 = torch.floor(sample).to(torch.int64)
    frac = sample - idx0.to(sample.dtype)

    n_src, n_mic, nsample = rir.shape
    n_sm = n_src * n_mic
    idx0 = idx0.view(n_sm, -1)
    frac = frac.view(n_sm, -1)
    amplitude = amplitude.view(n_sm, -1)

    fdl2 = (fdl - 1) // 2

    n = _get_fdl_grid(fdl, device=rir.device, dtype=sample.dtype)
    offsets = _get_fdl_offsets(fdl, device=rir.device)
    window = _get_fdl_window(fdl, device=rir.device, dtype=sample.dtype)

    if use_lut:
        sinc_lut = _get_sinc_lut(fdl, lut_gran, device=rir.device, dtype=sample.dtype)

    sm_offsets = (
        torch.arange(n_sm, device=rir.device, dtype=torch.int64) * nsample
    ).view(n_sm, 1, 1)
    rir_flat = rir.view(-1)

    n_img = idx0.shape[1]
    for start in range(0, n_img, chunk_size):
        end = min(start + chunk_size, n_img)
        idx = idx0[:, start:end]
        amp = amplitude[:, start:end]
        frac_m = frac[:, start:end]

        if use_lut:
            x_off_frac = (1.0 - frac_m) * lut_gran
            lut_gran_off = torch.floor(x_off_frac).to(torch.int64)
            x_off = x_off_frac - lut_gran_off.to(sample.dtype)
            lut_pos = lut_gran_off[..., None] + (
                n[None, None, :].to(torch.int64) * lut_gran
            )

            s0 = torch.take(sinc_lut, lut_pos)
            s1 = torch.take(sinc_lut, lut_pos + 1)
            interp = s0 + x_off[..., None] * (s1 - s0)
            filt = interp * window[None, None, :]
        else:
            t = n[None, None, :] - fdl2 - frac_m[..., None]
            filt = torch.sinc(t) * window[None, None, :]

        contrib = amp[..., None] * filt
        target = idx[..., None] + offsets[None, None, :]
        valid = (target >= 0) & (target < nsample)
        if not valid.any():
            continue

        target = target + sm_offsets
        target_flat = target[valid].to(torch.int64)
        values_flat = contrib[valid]
        rir_flat.scatter_add_(0, target_flat, values_flat)


_SINC_LUT_CACHE: dict[tuple[int, int, str, torch.dtype], Tensor] = {}
_FDL_GRID_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}
_FDL_OFFSETS_CACHE: dict[tuple[int, str], Tensor] = {}
_FDL_WINDOW_CACHE: dict[tuple[int, str, torch.dtype], Tensor] = {}
_AccumFn = Callable[[Tensor, Tensor, Tensor], None]
_ACCUM_BATCH_COMPILED: dict[tuple[str, torch.dtype, int, int, bool, int], _AccumFn] = {}


def _get_accumulate_fn(
    cfg: SimulationConfig, device: torch.device, dtype: torch.dtype
) -> _AccumFn:
    """Return an accumulation function with config-bound constants."""
    use_lut = cfg.use_lut and device.type != "mps"
    fdl = cfg.frac_delay_length
    lut_gran = cfg.sinc_lut_granularity
    chunk_size = cfg.accumulate_chunk_size

    def _fn(rir: Tensor, sample: Tensor, amplitude: Tensor) -> None:
        _accumulate_rir_batch_impl(
            rir,
            sample,
            amplitude,
            fdl=fdl,
            lut_gran=lut_gran,
            use_lut=use_lut,
            chunk_size=chunk_size,
        )

    if device.type not in ("cuda", "mps") or not cfg.use_compile:
        return _fn
    key = (str(device), dtype, fdl, lut_gran, use_lut, chunk_size)
    compiled = _ACCUM_BATCH_COMPILED.get(key)
    if compiled is None:
        compiled = torch.compile(_fn, dynamic=True)
        _ACCUM_BATCH_COMPILED[key] = compiled
    return compiled


def _get_fdl_grid(fdl: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (fdl, str(device), dtype)
    cached = _FDL_GRID_CACHE.get(key)
    if cached is None:
        cached = torch.arange(fdl, device=device, dtype=dtype)
        _FDL_GRID_CACHE[key] = cached
    return cached


def _get_fdl_offsets(fdl: int, *, device: torch.device) -> Tensor:
    key = (fdl, str(device))
    cached = _FDL_OFFSETS_CACHE.get(key)
    if cached is None:
        fdl2 = (fdl - 1) // 2
        cached = torch.arange(fdl, device=device, dtype=torch.int64) - fdl2
        _FDL_OFFSETS_CACHE[key] = cached
    return cached


def _get_fdl_window(fdl: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    key = (fdl, str(device), dtype)
    cached = _FDL_WINDOW_CACHE.get(key)
    if cached is None:
        cached = torch.hann_window(fdl, periodic=False, device=device, dtype=dtype)
        _FDL_WINDOW_CACHE[key] = cached
    return cached


def _get_sinc_lut(
    fdl: int, lut_gran: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """Create a sinc lookup table for fractional delays."""
    key = (fdl, lut_gran, str(device), dtype)
    cached = _SINC_LUT_CACHE.get(key)
    if cached is not None:
        return cached
    fdl2 = (fdl - 1) // 2
    lut_size = (fdl + 1) * lut_gran + 1
    n = torch.linspace(-fdl2 - 1, fdl2 + 1, lut_size, device=device, dtype=dtype)
    cached = torch.sinc(n)
    _SINC_LUT_CACHE[key] = cached
    return cached


def _apply_diffuse_tail(
    rir: Tensor,
    room: Room,
    beta: Tensor,
    tdiff: float,
    tmax: float,
    *,
    seed: Optional[int] = None,
) -> Tensor:
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
    gen.manual_seed(0 if seed is None else seed)
    noise = torch.randn(
        rir[..., tdiff_idx:].shape, device=rir.device, dtype=rir.dtype, generator=gen
    )
    scale = (
        torch.linalg.norm(rir[..., tdiff_idx - 1 : tdiff_idx], dim=-1, keepdim=True)
        + 1e-8
    )
    rir[..., tdiff_idx:] = noise * decay * scale
    return rir
