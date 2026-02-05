"""Simulation engines and configuration for RIR generation.

Includes the ISM implementation (in ``torchrir.sim.ism``), directivity helpers,
and simulator interfaces for ISM plus placeholder ray-tracing/FDTD backends.
"""

from .ism import simulate_dynamic_rir, simulate_rir
from .directivity import directivity_gain, split_directivity
from .simulators import ISMSimulator, RIRSimulator

__all__ = [
    "ISMSimulator",
    "RIRSimulator",
    "directivity_gain",
    "simulate_dynamic_rir",
    "simulate_rir",
    "split_directivity",
]
