from __future__ import annotations

"""Global configuration toggles for RIR simulation."""

_USE_LUT = True
_MIXED_PRECISION = False
_FRAC_DELAY_LENGTH = 81
_SINC_LUT_GRANULARITY = 20
_IMAGE_CHUNK_SIZE = 2048
_ACCUMULATE_CHUNK_SIZE = 4096
_USE_COMPILE = False


def activate_lut(activate: bool = True) -> None:
    """Enable or disable the fractional-delay LUT."""
    global _USE_LUT
    _USE_LUT = bool(activate)


def activate_mixed_precision(activate: bool = True) -> None:
    """Enable or disable mixed-precision accumulation."""
    global _MIXED_PRECISION
    _MIXED_PRECISION = bool(activate)


def set_frac_delay_length(length: int) -> None:
    """Set the fractional-delay filter length (odd positive integer)."""
    if length <= 0 or length % 2 == 0:
        raise ValueError("frac_delay_length must be a positive odd integer")
    global _FRAC_DELAY_LENGTH
    _FRAC_DELAY_LENGTH = int(length)


def set_sinc_lut_granularity(granularity: int) -> None:
    """Set the LUT granularity for sinc interpolation."""
    if granularity <= 0:
        raise ValueError("sinc_lut_granularity must be positive")
    global _SINC_LUT_GRANULARITY
    _SINC_LUT_GRANULARITY = int(granularity)


def set_image_chunk_size(size: int) -> None:
    """Set image source batch size for memory control."""
    if size <= 0:
        raise ValueError("image_chunk_size must be positive")
    global _IMAGE_CHUNK_SIZE
    _IMAGE_CHUNK_SIZE = int(size)


def set_accumulate_chunk_size(size: int) -> None:
    """Set accumulation chunk size for memory control."""
    if size <= 0:
        raise ValueError("accumulate_chunk_size must be positive")
    global _ACCUMULATE_CHUNK_SIZE
    _ACCUMULATE_CHUNK_SIZE = int(size)


def activate_compile(activate: bool = True) -> None:
    """Enable or disable torch.compile optimization."""
    global _USE_COMPILE
    _USE_COMPILE = bool(activate)


def get_config() -> dict[str, int | bool]:
    """Return the current configuration dictionary."""
    return {
        "use_lut": _USE_LUT,
        "mixed_precision": _MIXED_PRECISION,
        "frac_delay_length": _FRAC_DELAY_LENGTH,
        "sinc_lut_granularity": _SINC_LUT_GRANULARITY,
        "image_chunk_size": _IMAGE_CHUNK_SIZE,
        "accumulate_chunk_size": _ACCUMULATE_CHUNK_SIZE,
        "use_compile": _USE_COMPILE,
    }
