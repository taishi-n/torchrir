"""Validated configuration objects for TorchRIR simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Literal

import torch
from torch import Tensor

from .util.device import DeviceSpec
from .util._scalars import normalize_finite_real, normalize_integer


_HPF_FAMILIES = frozenset({"bessel", "butter", "cheby1", "cheby2", "ellip"})
_INT64_MAX = torch.iinfo(torch.int64).max
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _finite_real(
    value: object,
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    return normalize_finite_real(
        value,
        name=name,
        positive=positive,
        non_negative=non_negative,
    )


def _validate_positive_int64(value: object, *, name: str) -> int:
    normalized = normalize_integer(value, name=name, minimum=1)
    if normalized > _INT64_MAX:
        raise ValueError(f"{name} must be in the positive int64 range")
    return normalized


def _ceil_sample_count(duration: float, sample_rate: float) -> int:
    product = duration * sample_rate
    if not math.isfinite(product) or product <= 0.0:
        raise ValueError(
            "tmax and fs must produce a sample count in the positive int64 range"
        )
    count = math.ceil(product)
    if count > _INT64_MAX:
        raise ValueError(
            "tmax and fs must produce a sample count in the positive int64 range"
        )
    return count


@dataclass(frozen=True, slots=True, kw_only=True)
class RIRHighPassConfig:
    """Optional IIR high-pass filter applied after RIR generation.

    High-pass filtering is opt-in. Use ``SimulationConfig(high_pass=...)`` to
    enable it. The default ``phase="zero_phase"`` matches pyroomacoustics'
    forward-backward filtering. ``phase="causal"`` is an explicit alternative
    that preserves the physical time origin and prefix invariance.

    ``filter_family`` accepts ``"bessel"``, ``"butter"``, ``"cheby1"``,
    ``"cheby2"``, or ``"ellip"``. Install ``torchrir[hpf]`` to enable this
    SciPy CPU post-process, which detaches the RIR from PyTorch autograd.
    """

    cutoff_hz: float = 10.0
    order: int = 2
    passband_ripple_db: float = 5.0
    stopband_attenuation_db: float = 60.0
    filter_family: str = "butter"
    phase: Literal["causal", "zero_phase"] = "zero_phase"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cutoff_hz",
            _finite_real(self.cutoff_hz, name="high-pass cutoff_hz", positive=True),
        )
        object.__setattr__(
            self,
            "order",
            normalize_integer(self.order, name="high-pass order", minimum=1),
        )
        object.__setattr__(
            self,
            "passband_ripple_db",
            _finite_real(
                self.passband_ripple_db,
                name="high-pass passband_ripple_db",
                non_negative=True,
            ),
        )
        object.__setattr__(
            self,
            "stopband_attenuation_db",
            _finite_real(
                self.stopband_attenuation_db,
                name="high-pass stopband_attenuation_db",
                non_negative=True,
            ),
        )
        if not isinstance(self.filter_family, str):
            raise TypeError("high-pass filter_family must be a string")
        family = self.filter_family.strip().lower()
        if family not in _HPF_FAMILIES:
            supported = ", ".join(sorted(_HPF_FAMILIES))
            raise ValueError(f"high-pass filter_family must be one of: {supported}")
        object.__setattr__(self, "filter_family", family)
        if family in ("cheby1", "ellip") and self.passband_ripple_db <= 0:
            raise ValueError(
                f"passband_ripple_db must be positive for {family} filters"
            )
        if family in ("cheby2", "ellip") and self.stopband_attenuation_db <= 0:
            raise ValueError(
                f"stopband_attenuation_db must be positive for {family} filters"
            )
        if (
            family == "ellip"
            and self.stopband_attenuation_db <= self.passband_ripple_db
        ):
            raise ValueError(
                "stopband_attenuation_db must exceed passband_ripple_db for "
                "ellip filters"
            )
        if self.phase not in ("causal", "zero_phase"):
            raise ValueError("high-pass phase must be 'causal' or 'zero_phase'")


@dataclass(frozen=True, slots=True, kw_only=True)
class SimulationConfig:
    """Complete request for one image-source simulation.

    Exactly one image limit (``max_order`` or ``nb_img``) and exactly one
    output limit (``tmax`` or ``nsample``) are required at construction. Room
    sampling rate, endpoint directivity, positions, and orientations belong to
    the scene rather than this algorithm configuration.

    Attributes:
        max_order: Maximum L1 reflection order, mutually exclusive with
            ``nb_img``.
        nb_img: Non-negative image-grid half-width per room dimension,
            mutually exclusive with ``max_order``.
        tmax: Requested duration in seconds, mutually exclusive with
            ``nsample``. Resolution uses ``ceil(tmax * fs)``.
        nsample: Exact output sample count, mutually exclusive with ``tmax``.
        tdiff: Optional strictly positive diffuse-tail handoff time.
        device: Execution device. ``None`` inherits the scene; ``"auto"``
            selects CUDA, then compatible MPS, then CPU.
        dtype: Execution dtype. ``None`` inherits the scene dtype.
        min_source_mic_distance: Minimum permitted endpoint separation in
            metres.
        seed: Optional diffuse-tail random seed.
        use_lut: Use sinc lookup-table interpolation where supported.
        frac_delay_length: Positive odd fractional-delay tap count.
        sinc_lut_granularity: Lookup-table subdivisions per sample.
        image_chunk_size: Images processed per geometry chunk.
        accumulate_chunk_size: Image contributions accumulated per chunk.
        use_compile: Compile accelerator accumulation with ``torch.compile``.
        high_pass: Optional explicit post-simulation high-pass configuration.
    """

    max_order: int | None = None
    nb_img: Tensor | tuple[int, ...] | None = None
    tmax: float | None = None
    nsample: int | None = None
    tdiff: float | None = None
    device: torch.device | str | None = None
    dtype: torch.dtype | None = None
    min_source_mic_distance: float = 1.0e-6
    seed: int | None = None
    use_lut: bool = True
    frac_delay_length: int = 81
    sinc_lut_granularity: int = 20
    image_chunk_size: int = 2048
    accumulate_chunk_size: int = 4096
    use_compile: bool = False
    high_pass: RIRHighPassConfig | None = None

    def __post_init__(self) -> None:
        self._normalize_nb_img()
        self._validate()

    def _normalize_nb_img(self) -> None:
        if self.nb_img is None:
            return
        if torch.is_tensor(self.nb_img):
            values = self.nb_img
            if values.layout != torch.strided or values.is_nested:
                raise TypeError("nb_img Tensor must use strided layout")
            if values.device.type == "meta":
                raise ValueError("nb_img Tensor must be materialized")
            if values.ndim != 1 or values.numel() == 0:
                raise ValueError("nb_img must be a non-empty one-dimensional sequence")
            if values.dtype == torch.bool:
                raise TypeError("nb_img must contain integers, not booleans")
            if values.dtype not in _INTEGER_DTYPES:
                raise TypeError("nb_img Tensor must have an integer dtype")
            normalized = tuple(int(value) for value in values.detach().cpu().tolist())
        elif isinstance(self.nb_img, tuple):
            if not self.nb_img:
                raise ValueError("nb_img must be a non-empty one-dimensional sequence")
            if any(isinstance(value, bool) for value in self.nb_img):
                raise TypeError("nb_img must contain integers, not booleans")
            try:
                normalized = tuple(
                    normalize_integer(value, name="nb_img value", minimum=0)
                    for value in self.nb_img
                )
            except TypeError as exc:
                raise TypeError("nb_img must contain integers") from exc
            except ValueError as exc:
                raise ValueError("nb_img must contain non-negative integers") from exc
        else:
            raise TypeError("nb_img must be a Tensor or tuple of integers")
        if any(value < 0 or value > _INT64_MAX for value in normalized):
            raise ValueError(
                "nb_img must contain values in the non-negative int64 range"
            )
        object.__setattr__(self, "nb_img", normalized)

    def _validate(self) -> None:
        if (self.max_order is None) == (self.nb_img is None):
            raise ValueError("exactly one of max_order or nb_img must be provided")
        if self.max_order is not None:
            object.__setattr__(
                self,
                "max_order",
                normalize_integer(self.max_order, name="max_order", minimum=0),
            )

        if (self.tmax is None) == (self.nsample is None):
            raise ValueError("exactly one of tmax or nsample must be provided")
        if self.tmax is not None:
            object.__setattr__(
                self,
                "tmax",
                _finite_real(self.tmax, name="tmax", positive=True),
            )
        if self.nsample is not None:
            object.__setattr__(
                self,
                "nsample",
                _validate_positive_int64(self.nsample, name="nsample"),
            )

        if self.tdiff is not None:
            object.__setattr__(
                self,
                "tdiff",
                _finite_real(self.tdiff, name="tdiff", positive=True),
            )
        if self.tdiff is not None and self.tmax is not None and self.tdiff >= self.tmax:
            raise ValueError("tdiff must be smaller than tmax")
        object.__setattr__(
            self,
            "min_source_mic_distance",
            _finite_real(
                self.min_source_mic_distance,
                name="min_source_mic_distance",
                positive=True,
            ),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                normalize_integer(
                    self.seed,
                    name="seed",
                    minimum=0,
                    maximum=_INT64_MAX,
                ),
            )
        if self.dtype is not None:
            _validate_supported_dtype(self.dtype)
        if self.device is not None and not isinstance(
            self.device,
            (str, torch.device),
        ):
            raise TypeError("device must be a string, torch.device, or None")
        if isinstance(self.device, str):
            normalized_device = self.device.strip().lower()
            if not normalized_device:
                raise ValueError("device must be non-empty")
            if normalized_device != "auto":
                try:
                    torch.device(normalized_device)
                except (RuntimeError, ValueError) as exc:
                    raise ValueError(f"invalid device: {self.device}") from exc
            if normalized_device != "auto":
                _validate_supported_device(torch.device(normalized_device))
            object.__setattr__(self, "device", normalized_device)
        elif isinstance(self.device, torch.device):
            _validate_supported_device(self.device)
        frac_delay_length = normalize_integer(
            self.frac_delay_length,
            name="frac_delay_length",
        )
        if frac_delay_length <= 0 or frac_delay_length % 2 == 0:
            raise ValueError("frac_delay_length must be a positive odd integer")
        object.__setattr__(self, "frac_delay_length", frac_delay_length)
        for name in (
            "sinc_lut_granularity",
            "image_chunk_size",
            "accumulate_chunk_size",
        ):
            object.__setattr__(
                self,
                name,
                normalize_integer(getattr(self, name), name=name, minimum=1),
            )
        for name in ("use_lut", "use_compile"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.high_pass is not None and not isinstance(
            self.high_pass, RIRHighPassConfig
        ):
            raise TypeError("high_pass must be RIRHighPassConfig or None")

    def replace(self, **kwargs: object) -> "SimulationConfig":
        """Return a validated configuration with selected fields replaced."""

        return replace(self, **kwargs)


@dataclass(frozen=True, slots=True, kw_only=True)
class ResolvedSimulationConfig:
    """Validated effective settings shared by a kernel and result metadata.

    This record is produced internally by `torchrir.sim.simulate`.
    ``tmax`` is exactly ``nsample / fs`` after resolution.
    """

    fs: float
    max_order: int | None
    nb_img: tuple[int, ...] | None
    nsample: int
    tmax: float
    tdiff: float | None
    device: torch.device
    dtype: torch.dtype
    min_source_mic_distance: float
    seed: int | None
    use_lut: bool
    frac_delay_length: int
    sinc_lut_granularity: int
    image_chunk_size: int
    accumulate_chunk_size: int
    use_compile: bool
    high_pass: RIRHighPassConfig | None

    def __post_init__(self) -> None:
        fs = _finite_real(self.fs, name="fs", positive=True)
        object.__setattr__(self, "fs", fs)

        if (self.max_order is None) == (self.nb_img is None):
            raise ValueError("exactly one of max_order or nb_img must be resolved")
        if self.max_order is not None:
            object.__setattr__(
                self,
                "max_order",
                normalize_integer(self.max_order, name="max_order", minimum=0),
            )
        if self.nb_img is not None:
            if not isinstance(self.nb_img, tuple) or len(self.nb_img) not in (2, 3):
                raise ValueError("nb_img must be a 2D or 3D tuple")
            try:
                normalized_nb_img = tuple(
                    normalize_integer(value, name="nb_img value", minimum=0)
                    for value in self.nb_img
                )
            except TypeError as exc:
                raise TypeError("nb_img must contain integers") from exc
            except ValueError as exc:
                raise ValueError("nb_img must contain non-negative integers") from exc
            if any(value > _INT64_MAX for value in normalized_nb_img):
                raise ValueError(
                    "nb_img must contain values in the non-negative int64 range"
                )
            object.__setattr__(self, "nb_img", normalized_nb_img)

        object.__setattr__(
            self,
            "nsample",
            _validate_positive_int64(self.nsample, name="nsample"),
        )
        tmax = _finite_real(self.tmax, name="tmax", positive=True)
        expected_tmax = self.nsample / fs
        if not math.isfinite(expected_tmax) or expected_tmax <= 0.0:
            raise ValueError("nsample / fs must be positive and finite")
        if not math.isclose(
            tmax,
            expected_tmax,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ):
            raise ValueError("tmax must equal nsample / fs in a resolved config")
        object.__setattr__(self, "tmax", tmax)

        if self.tdiff is not None:
            tdiff = _finite_real(self.tdiff, name="tdiff", positive=True)
            if tdiff >= tmax:
                raise ValueError("tdiff must be smaller than tmax")
            object.__setattr__(self, "tdiff", tdiff)

        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a resolved torch.device")
        _validate_supported_device(self.device)
        _validate_supported_dtype(self.dtype)
        if self.device.type == "mps" and self.dtype == torch.float64:
            raise ValueError("MPS does not support float64")

        object.__setattr__(
            self,
            "min_source_mic_distance",
            _finite_real(
                self.min_source_mic_distance,
                name="min_source_mic_distance",
                positive=True,
            ),
        )
        if self.seed is not None:
            object.__setattr__(
                self,
                "seed",
                normalize_integer(
                    self.seed,
                    name="seed",
                    minimum=0,
                    maximum=_INT64_MAX,
                ),
            )
        for name in ("use_lut", "use_compile"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        frac_delay_length = normalize_integer(
            self.frac_delay_length,
            name="frac_delay_length",
        )
        if frac_delay_length <= 0 or frac_delay_length % 2 == 0:
            raise ValueError("frac_delay_length must be a positive odd integer")
        object.__setattr__(self, "frac_delay_length", frac_delay_length)
        for name in (
            "sinc_lut_granularity",
            "image_chunk_size",
            "accumulate_chunk_size",
        ):
            object.__setattr__(
                self,
                name,
                normalize_integer(getattr(self, name), name=name, minimum=1),
            )
        if self.high_pass is not None and not isinstance(
            self.high_pass,
            RIRHighPassConfig,
        ):
            raise TypeError("high_pass must be RIRHighPassConfig or None")
        if self.high_pass is not None and self.high_pass.cutoff_hz >= fs / 2:
            raise ValueError("high-pass cutoff_hz must be smaller than fs/2")


def _resolve_simulation_config(
    config: SimulationConfig,
    *,
    fs: float,
    room_dimension: int,
    tensor_values: tuple[Tensor, ...],
) -> ResolvedSimulationConfig:
    """Resolve one request exactly once before entering an ISM kernel."""

    resolved_fs = _finite_real(fs, name="fs", positive=True)
    if room_dimension not in (2, 3):
        raise ValueError("room_dimension must be 2 or 3")
    if config.nb_img is not None and len(config.nb_img) != room_dimension:
        raise ValueError("nb_img must match room dimension")
    for tensor in tensor_values:
        if not torch.is_tensor(tensor):  # pragma: no cover - internal contract
            raise TypeError("simulation tensor_values must contain only Tensors")
        try:
            _validate_supported_dtype(tensor.dtype)
        except TypeError as exc:
            raise TypeError(
                "simulation scene tensors must use torch.float32 or torch.float64"
            ) from exc

    if config.nsample is None:
        assert config.tmax is not None
        nsample = _ceil_sample_count(config.tmax, resolved_fs)
    else:
        nsample = config.nsample
    duration = nsample / resolved_fs
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("nsample and fs must produce a positive finite duration")
    if config.tdiff is not None and config.tdiff >= duration:
        raise ValueError("tdiff must be smaller than the RIR duration")

    device, dtype = DeviceSpec(
        device=config.device,
        dtype=config.dtype,
    ).resolve(*tensor_values)
    if device.type == "cpu":
        device = torch.device("cpu")
    elif device.type == "cuda" and device.index is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    elif (
        device.type == "mps"
        and device.index is None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps", 0)

    use_lut = config.use_lut and device.type != "mps"
    use_compile = config.use_compile and device.type in ("cuda", "mps")

    normalized_nb_img = config.nb_img
    if torch.is_tensor(normalized_nb_img):  # pragma: no cover - normalized in init
        raise RuntimeError("SimulationConfig.nb_img was not normalized")

    return ResolvedSimulationConfig(
        fs=resolved_fs,
        max_order=config.max_order,
        nb_img=normalized_nb_img,
        nsample=nsample,
        tmax=duration,
        tdiff=config.tdiff,
        device=device,
        dtype=dtype,
        min_source_mic_distance=config.min_source_mic_distance,
        seed=config.seed,
        use_lut=use_lut,
        frac_delay_length=config.frac_delay_length,
        sinc_lut_granularity=config.sinc_lut_granularity,
        image_chunk_size=config.image_chunk_size,
        accumulate_chunk_size=config.accumulate_chunk_size,
        use_compile=use_compile,
        high_pass=config.high_pass,
    )


def _validate_supported_dtype(dtype: object) -> None:
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("simulation dtype must be torch.float32 or torch.float64")


def _validate_supported_device(device: torch.device) -> None:
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device type must be cpu, cuda, or mps")
    if device.type == "cpu" and device.index is not None:
        raise ValueError("CPU device must not include an index")
    if device.type == "mps" and device.index not in (None, 0):
        raise ValueError("MPS device index must be 0 when specified")
    if device.type == "cuda" and device.index is not None and device.index < 0:
        raise ValueError("CUDA device index must be non-negative")


__all__ = [
    "RIRHighPassConfig",
    "ResolvedSimulationConfig",
    "SimulationConfig",
]
