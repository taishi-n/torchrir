# Test policy and numerical methodology

## Current-contract test policy

Follow the documentation-first, test-first development cycle in
[AGENTS.md](../AGENTS.md) and the current [specification](../README.md#specification).
For behavior changes, first update the specification, then write or update tests
and confirm the expected failure before implementing the minimum change.

TorchRIR does not preserve backward compatibility or provide data migration.
Remove tests whose only purpose is preserving historical behavior or asserting
that particular retired APIs or fields remain absent. Test the current API and
schema directly instead of maintaining lists of historical names. Do not add
replacement tests when existing current-contract tests already cover the behavior.

Retain numerical regression tests, current-format serialization round trips,
input validation, supported corpus/format interoperability, external numerical
comparisons, and filesystem failure/crash recovery. These test present behavior,
not migration from earlier TorchRIR versions.

## Numerical methodology

The test suite separates independent analytic checks from cross-implementation
comparisons. Shape-only checks are not accepted as evidence of numerical
correctness for simulation or convolution kernels.

## Test layers

- `test_numerical_ism.py` checks image enumeration, mirror geometry, asymmetric
  reflection gains, directivity reflection parity, and fractional-delay
  accumulation against small direct formulas. It also exercises finite
  `float64` geometry near `1e308`, stable directivity normalization,
  exponent-scaled delay conversion, and masking before `int64` accumulation.
- `test_simulation_invariants.py` checks reciprocity, permutation equivariance,
  translation invariance, static/dynamic equivalence, chunk invariance, and
  diffuse-tail reproducibility.
- `test_diffuse_hpf.py` checks extreme/subnormal handoff RMS, sample-index
  overflow handling, representability failures, crossfade behavior, and both
  explicit high-pass phases.
- `test_signal.py` compares FFT and dynamic convolution with direct NumPy
  convolution using fixed random seeds.
- `test_util_contracts.py` exercises shared scalar/tensor boundaries, including
  max-scaled vector norms for extreme and subnormal `float64` vectors.
- `test_outputs_logging.py` checks schema and output contracts, including
  overflow-resistant microphone centers, pair distances, and DOA for extreme
  finite coordinates.
- `test_dataset_security.py` checks descriptor-relative no-symlink reads,
  transfer deadlines and byte limits, exact `Content-Length`, atomic
  no-replace/exchange races, lock deadlines, manifest identity recovery, and
  preservation of third-party entries. These filesystem cases require Linux or
  macOS primitives and explicitly test the unsupported-platform failure path.
- `test_compare_pyroomacoustics.py` removes pyroomacoustics 0.9.0's documented
  40-sample fractional-delay offset, then compares RIR and signal samples
  without estimated alignment. It includes analytic source-only and
  microphone-only directivity; the expected remaining lag is zero and relative
  L2 error must be below `2e-3`.
- `test_compare_rir_generator.py` compares static 3D RIRs after only the known
  `4*pi` amplitude conversion. Both outputs already use a physical time axis;
  the test matches the fractional-delay support instead of shifting either
  waveform. It covers asymmetric walls, orders 0/1/3, and analytic microphone
  directivity.
- `test_compare_gpurir.py` compares CUDA RIRs without alignment and checks
  trajectory convolution through gpuRIR's own `simulateTrajectory` API.

## Reference provenance

Runtime reference implementations are distinct from source snapshots used to
review test design:

- The pyroomacoustics runtime oracle is the PyPI `0.9.0` artifact resolved and
  hashed in `uv.lock`.
- The CUDA workflow installs gpuRIR commit
  `fd8af43a4a113d3c2c05f0085a0119ecb1f1a484`.
- The optional rir-generator oracle must report version `0.3.0`; it is not a
  project dependency and is skipped unless installed explicitly.

The test methodology was also reviewed against these source snapshots:

- pyroomacoustics `ff7d61f219e4eb41489963c4bb5f57bea5bc2c69`
  (a 2026-07-17 `master` snapshot, not the PyPI `0.9.0` source revision)
- gpuRIR `fd8af43a4a113d3c2c05f0085a0119ecb1f1a484`
- rir-generator `v0.3.0`
- SciPy `3df7bcd2cecfa875ff09f5fb02bf69b7fbcfc6c9`
- k-Wave Python `011c8ed3a9f8c44c47cbb4df3edd2ea63330673c`

SciPy informed the independent direct-convolution cases. k-Wave Python
informed the staged oracle design and metric-specific tolerances. No generated
reference arrays are committed: deterministic analytic cases are preferred,
and external comparisons execute the pinned reference implementation directly.

## Reference-oracle policy

An external implementation is evidence, not an unconditional ground truth.
Independent analytic and metamorphic tests remain authoritative when a
reference implementation has a different documented convention or a relevant
known issue.

- Scale, wall-coefficient, image-count, time-origin, fractional-delay, and HPF
  conventions are matched explicitly. Tests never estimate a scale or shift by
  optimizing against the output.
- gpuRIR has a project-maintained list of
  [known issues](https://github.com/DavidDiazGuerra/gpuRIR/issues) and uses an
  internal signed reflection-coefficient convention. Raw signed reflected RIR
  samples are therefore not used as a TorchRIR correctness oracle. Static and
  dynamic RIR parity is restricted to direct paths, while
  `simulateTrajectory` is checked with identical synthetic RIR tensors.
- pyroomacoustics 0.9.0 analytic directivity tests construct orientations with
  `DirectionVector`; its ndarray constructor is broken in that release. Its
  process-global HPF setting is restored in a `finally` path after every call.
- rir-generator comparisons disable its HPF, leave TorchRIR's optional HPF
  unset, apply the specified `4*pi` scale, and compare the shared physical time
  axis directly. The residual fractional-delay window difference is covered by
  a fixed numerical tolerance.

## Commands

```bash
uv run --group test pytest -q
uv run --group test pytest -q -m numerical
uv run --group test --group comparison pytest -q -m comparison
TORCHRIR_REQUIRE_RIR_GENERATOR_COMPARISON=1 uv run --group test --with 'rir-generator==0.3.0' pytest -q tests/test_compare_rir_generator.py
uv run --group test pytest -q --cov=torchrir --cov-branch --cov-report=term-missing
```

Dedicated comparison jobs set `TORCHRIR_REQUIRE_COMPARISON=1`; therefore a
missing reference dependency cannot turn the entire comparison into a passing
skip. A dedicated rir-generator job instead sets
`TORCHRIR_REQUIRE_RIR_GENERATOR_COMPARISON=1` because rir-generator is not part
of the normal comparison dependency group.
