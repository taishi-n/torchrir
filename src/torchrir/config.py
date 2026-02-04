from __future__ import annotations

_USE_LUT = True
_MIXED_PRECISION = False
_FRAC_DELAY_LENGTH = 81
_SINC_LUT_GRANULARITY = 20


def activate_lut(activate: bool = True) -> None:
    global _USE_LUT
    _USE_LUT = bool(activate)


def activate_mixed_precision(activate: bool = True) -> None:
    global _MIXED_PRECISION
    _MIXED_PRECISION = bool(activate)


def set_frac_delay_length(length: int) -> None:
    if length <= 0 or length % 2 == 0:
        raise ValueError("frac_delay_length must be a positive odd integer")
    global _FRAC_DELAY_LENGTH
    _FRAC_DELAY_LENGTH = int(length)


def set_sinc_lut_granularity(granularity: int) -> None:
    if granularity <= 0:
        raise ValueError("sinc_lut_granularity must be positive")
    global _SINC_LUT_GRANULARITY
    _SINC_LUT_GRANULARITY = int(granularity)


def get_config() -> dict[str, int | bool]:
    return {
        "use_lut": _USE_LUT,
        "mixed_precision": _MIXED_PRECISION,
        "frac_delay_length": _FRAC_DELAY_LENGTH,
        "sinc_lut_granularity": _SINC_LUT_GRANULARITY,
    }
