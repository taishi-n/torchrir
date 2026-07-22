# Numerical test methodology

The test suite separates independent analytic checks from cross-implementation
comparisons. Shape-only checks are not accepted as evidence of numerical
correctness for simulation or convolution kernels.

## Test layers

- `test_numerical_ism.py` checks image enumeration, mirror geometry, asymmetric
  reflection gains, directivity reflection parity, and fractional-delay
  accumulation against small direct formulas.
- `test_simulation_invariants.py` checks reciprocity, permutation equivariance,
  translation invariance, static/dynamic equivalence, chunk invariance, and
  diffuse-tail reproducibility.
- `test_signal.py` compares FFT and dynamic convolution with direct NumPy
  convolution using fixed random seeds.
- `test_compare_pyroomacoustics.py` compares native, unaligned RIR and signal
  samples. The expected lag is zero and relative L2 error must be below `2e-3`.
- `test_compare_gpurir.py` compares CUDA RIRs without alignment and checks
  trajectory convolution through gpuRIR's own `simulateTrajectory` API.

## Reference provenance

The test design was reviewed against these source snapshots:

- pyroomacoustics `ff7d61f219e4eb41489963c4bb5f57bea5bc2c69`
  (released as `0.9.0` and locked in `uv.lock`)
- gpuRIR `fd8af43a4a113d3c2c05f0085a0119ecb1f1a484`
- SciPy `3df7bcd2cecfa875ff09f5fb02bf69b7fbcfc6c9`
- k-Wave Python `011c8ed3a9f8c44c47cbb4df3edd2ea63330673c`

SciPy informed the independent direct-convolution cases. k-Wave Python
informed the staged oracle design and metric-specific tolerances. No generated
reference arrays are committed: deterministic analytic cases are preferred,
and external comparisons execute the pinned reference implementation directly.

## Commands

```bash
uv run pytest -q
uv run pytest -q -m numerical
uv run --group comparison pytest -q -m comparison
uv run pytest -q --cov=torchrir --cov-branch --cov-report=term-missing
```

Dedicated comparison jobs set `TORCHRIR_REQUIRE_COMPARISON=1`; therefore a
missing reference dependency cannot turn the entire comparison into a passing
skip.
