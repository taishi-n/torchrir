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
- Optional manual gpuRIR comparisons use commit
  `fd8af43a4a113d3c2c05f0085a0119ecb1f1a484`; no GPU runner is required by CI.
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

## Test and CI plan

### Current state

The automated validation entry point is
[ci.yml](../.github/workflows/ci.yml). It has a documentation job, a Linux test
matrix for Python 3.11.4/3.12/3.13, and one job combining lint, types, CPU
reference comparisons, and package checks. The
[release workflow](../.github/workflows/release.yml) calls this validation at the
same commit and publishes its checked artifact only after validation succeeds.
There is no dedicated CUDA or MPS workflow. Keep the accelerator tests for
manual runs; a successful CPU run says nothing about their execution status.

The existing numerical suite already tests analytic path delays/gains,
fractional-delay interpolation, directivity, diffuse tails, filtering,
reciprocity, static/dynamic equivalence, and both convolution time references.
Extend those tests for concrete gaps rather than adding duplicate shape checks
or introducing compatibility/migration tests.

The jobs and additions below are a plan, not implemented CI behavior.

### Required automated jobs

Use GitHub-hosted CPU runners for pull requests and pushes to `main`, with a
manual dispatch entry point. Keep one validation workflow with these jobs;
accelerators and real corpus downloads are not prerequisites.

| Job | Environment | Scope and acceptance criteria | Change from current CI |
| --- | --- | --- | --- |
| Quality | Linux, Python 3.11.4 | Ruff format/lint and ty must pass; validate workflow syntax with actionlint. | Separate from optional integrations and package builds; add workflow validation. |
| CPU tests | Linux, Python 3.11.4/3.12/3.13 | Run all non-comparison, non-accelerator tests, including numerical tests, audio I/O, synthetic dataset builds, and filesystem recovery. Measure branch-inclusive coverage on one designated Linux job and keep the existing 75% threshold. | Explicitly select CPU tests and report skip reasons; avoid repeating coverage collection across the entire matrix. |
| macOS CPU tests | GitHub-hosted macOS, Python 3.13 | Run the same CPU suite, especially descriptor-relative reads, atomic publication, locks, and recovery. | Add coverage for the separate Darwin filesystem implementation; do not require MPS availability. |
| CPU comparisons | Linux, Python 3.11.4 | Run the pinned pyroomacoustics and rir-generator test files with their required-dependency flags; no skipped comparison is acceptable. | Separate external-reference installation/execution from fast quality checks. |
| Documentation | Linux, Python 3.11.4, docs-only dependencies | Strict Zensical build; verify the index/API pages and exclusion of private API helpers. | Preserve the existing minimal-dependency job. |
| Distribution | Linux, Python 3.11.4 | Build sdist/wheel, check license/changelog metadata, and run installed-package smoke tests described below. | Extend the existing import-only wheel check to exercise the installed library. |

Use `uv sync --locked` with only the groups/extras each job needs, then preserve
that environment for the job. Do not install `oobss` or all extras solely to run
lint or CPU reference comparisons. Keep SciPy, SoundFile, and visualization
dependencies present in the full CPU test jobs so their absence cannot hide
required coverage. Keep test inputs small, synthetic, and local; mock network
responses instead of downloading speech corpora.

Run the validation workflow for every pull request so its result is always
reported. Include all workflow files in any push path filter. Cancel superseded
validation runs for the same pull request/ref using workflow-specific
concurrency groups, but do not cancel an in-progress publication. Record test
counts, skip reasons, and coverage in the job output; retain JUnit and coverage
reports for failed-run diagnosis. A skipped optional codec case needs an explicit
reason; required dependencies, empty test selections, and entirely skipped jobs
must fail validation.

### Test additions and concrete gaps

1. **Make CPU/device selection accurate.**
   `test_collate_dataset_items_rejects_mixed_devices_when_available` currently
   selects an accelerator at runtime without a device marker. Parameterize it
   with explicit `cuda`/`mps` marks before using marker selection as the CPU CI
   boundary. Ignore the `test_compare_*.py` files in CPU-only jobs to avoid
   optional-reference imports during collection. Keep ordinary unmarked tests;
   selecting only `unit` or `numerical` would omit current API contracts.
2. **Exercise the installed distribution.**
   The current wheel check only imports the package, while `tests/conftest.py`
   inserts the checkout's `src` directory into `sys.path`. Add a small smoke
   script and execute it outside the checkout with no source-path injection.
   With only base dependencies, simulate a tiny direct path and check its
   physical arrival/gain, then check static and both dynamic convolution
   conventions against direct sums. Confirm imports resolve inside the clean
   environment. In a separate environment with the required extras, check a
   multichannel floating-WAV round trip and the installed builder CLI's `--help`.
3. **Exercise actual media output.**
   `test_viz_animation.py` exercises actual GIF/MP4 rendering for 2D/3D scenes
   with source annotations enabled and disabled. Extend encoded-output checks
   to duration and audio-stream presence when muxing is requested, and run them
   in one Linux job with
   Pillow and system ffmpeg/ffprobe; use synthetic audio and avoid pixel hashes
   or wall-clock performance thresholds.
4. **Extend accelerator coverage when hardware is available.**
   Existing device tests cover basic static/dynamic RIR parity and emission-time
   convolution gradients. Add static and observation-time output/gradient parity,
   multi-source/microphone and chunk-boundary cases, and CUDA eager/compiled
   accumulation parity with LUT enabled/disabled. Use float32/float64 on CUDA
   and float32 on MPS. Verify the actual output/config device; silent CPU
   fallback must not count as a device pass. Compilation validation needs an
   actual supported backend, not only a mocked flag. These remain manual until
   reliable hardware is explicitly provided.

### Release validation

The release workflow calls `ci.yml` from the event commit and requires all its
jobs to succeed. `scripts/check_distribution.py` verifies the project and lock
versions, wheel/sdist metadata, license, changelog, and optional expected tag.
The artifact is named with the event SHA and uploaded only after its checks;
the publication job downloads it without checking out or rebuilding source.
`workflow_dispatch` takes an expected `release-tag` and exercises the same gate
without granting publication permission or running `uv publish`.
Version-rejection tests use synthetic archives; a manual dispatch validates the
complete gate without uploading a release.

### Manual accelerator checks

Prepare the desired PyTorch/CUDA or PyTorch/MPS environment before running these
commands. They use `--no-sync` to preserve that prepared environment. Record the
commit, Python/PyTorch version, actual device, driver/runtime where applicable,
and the pytest summary. An unavailable accelerator fails the preflight; it must
not be recorded as a successful GPU check.

```bash
# CUDA: RIR parity, sample schedules, and emission-time convolution/autograd.
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
uv run --no-sync pytest -q -rs tests/test_device_parity.py tests/test_signal.py -m cuda

# MPS: run separately on an environment with an available MPS device.
uv run --no-sync python -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'
uv run --no-sync pytest -q -rs tests/test_device_parity.py tests/test_signal.py -m mps
```

Run gpuRIR separately only when the pinned reference revision is installed and
the CUDA preflight has passed:

```bash
TORCHRIR_REQUIRE_COMPARISON=1 uv run --no-sync pytest -q -rs tests/test_compare_gpurir.py
```

The required-reference flag makes a missing gpuRIR import fail, but the tests
still skip without CUDA, so the preflight and skip-summary review are necessary.
Preserve the existing explicit amplitude/time conventions and direct-path-only
RIR comparison scope. Do not make this external CUDA build a dependency of CPU
validation or restore a scheduled self-hosted GPU workflow without a maintained
runner and successful manual verification.

### Implementation order

1. **P1: CPU selection and CI separation.** Update the specification/test markers,
   then split quality, comparisons, and distribution jobs; add macOS CPU coverage
   and workflow syntax checks. Done when all required jobs run on hosted CPU
   runners and skip reports contain no unexplained omissions.
2. **P1: Distribution smoke and release gate.** Write behavior-based smoke cases
   before wiring them into CI; then reuse validation on tag builds. Done when
   the installed artifact is exercised and failed validation prevents publishing.
3. **P2: Real media integration.** Add the tiny encoded-output test, demonstrate
   failure for missing/invalid output, and enable its prepared Linux environment.
4. **P2, hardware-dependent: Accelerator extensions.** Add the missing parity
   cases and record manual results on actual devices. CPU success does not close
   this item, and there is no requirement to add a replacement GPU workflow now.

Each step follows documentation first, failing tests for new behavior, minimal
implementation, and final removal of unused code or contradictory documentation.
Deleting a workflow or revising this plan does not warrant a permanent test that
asserts a retired filename is absent.

GitHub Actions references for implementation:
[hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
and [concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
