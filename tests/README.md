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
- `test_cli_integration.py` executes the unified CLI, all three standalone
  scenarios, and the dataset builder in fresh Python processes against tiny
  local synthetic CMU-layout recordings. JSON/YAML configuration round trips,
  explicit option precedence, time-reference metadata, sample counts, and
  mixture/reference sums are checked. No corpus downloads are needed.
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
  Child-process interruption cases stop real publication after the initial
  manifest write and initialization rename,
  backup rename, backup manifest, final rename, and published manifest (plus
  create-only rename). The parent confirms the lock is held, kills the writer,
  and starts a fresh recovery process. Complete file contents, transaction cleanup,
  and lock reacquisition are required; synthetic interrupted-state tests remain.
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

The automated validation entry point is [ci.yml](../.github/workflows/ci.yml).
It separates quality, CPU tests, CPU comparisons, documentation, and distribution.
CPU tests run on Linux Python 3.11.4/3.12/3.13 and macOS Python 3.13; only Linux
3.11.4 collects branch-inclusive coverage with the 75% threshold. Each test job
retains JUnit reports and explicit skip reasons; empty/all-skipped reports fail.
Comparison reports additionally reject any skips. ffmpeg/ffprobe are installed
in CPU jobs so real-media tests cannot disappear behind a missing-codec skip.
Actionlint 1.7.12 validates workflow syntax in the quality job.
The [release workflow](../.github/workflows/release.yml) reuses all validation
jobs at the same commit before publishing the checked artifact. Pull requests
always trigger validation; pushes watch all workflows and relevant project files.
Workflow-specific concurrency cancels obsolete CI runs, while release validation
and publication are never cancelled by this policy. There is no GPU workflow;
a successful CPU run says nothing about manual accelerator execution.

The existing numerical suite already tests analytic path delays/gains,
fractional-delay interpolation, directivity, diffuse tails, filtering,
reciprocity, static/dynamic equivalence, and both convolution time references.
Extend those tests for concrete gaps rather than adding duplicate shape checks
or introducing compatibility/migration tests.

The table records the implemented CPU validation contract. Accelerator extensions
remain deferred as noted in the README TODO.

### Required automated jobs

Use GitHub-hosted CPU runners for pull requests and pushes to `main`, with a
manual dispatch entry point. Keep one validation workflow with these jobs;
accelerators and real corpus downloads are not prerequisites.

| Job | Environment | Scope and acceptance criteria |
| --- | --- | --- |
| Quality | Linux, Python 3.11.4 | Ruff format/lint, ty, and actionlint 1.7.12. Uses the quality/test groups and visualization/CLI extras needed for static analysis. |
| CPU tests | Linux, Python 3.11.4/3.12/3.13 | All non-comparison/non-accelerator tests, required codecs, JUnit reports; branch-inclusive coverage only on 3.11.4 (75% minimum). |
| macOS CPU tests | macOS, Python 3.13 | Same CPU suite, including Darwin filesystem publication and recovery. |
| CPU comparisons | Linux, Python 3.11.4 | Pinned pyroomacoustics/rir-generator; required-dependency flags and zero skips. |
| Documentation | Linux, Python 3.11.4 | Docs-only strict build and generated index/API checks. |
| Distribution | Linux, Python 3.11.4 | Version/metadata validation, isolated installed-wheel checks, and upload of the tested sdist/wheel. |

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

1. **Keep CPU/device selection accurate.**
   Mixed-device collate cases have explicit `cuda`/`mps` marks. CPU jobs select
   `not comparison and not cuda and not mps` and ignore `test_compare_*.py`
   during collection. Ordinary unmarked tests remain part of the CPU suite.
2. **Exercise the installed distribution.**
   `scripts/smoke_wheel.py` runs under isolated Python (`-I`) from a temporary
   directory, without pytest/conftest or source-path injection. It requires the
   imported package to reside inside the fresh environment. The base case checks
   physical direct-path arrival/gain and static/emission/observation convolution
   against direct sums. A separate audio/datasets/cli environment checks a FLOAT
   multichannel WAV round trip (including amplitudes above one) and the installed
   builder entry point's `--help`. Both environments install the same checked wheel.
3. **Exercise actual media output.**
   `test_viz_animation.py` exercises actual GIF/MP4 rendering for 2D/3D scenes
   with source annotations enabled and disabled, exact timing, decoding, HD
   dimensions, and requested audio streams. Failure-injection tests cover missing
   tools/inputs, encoder failures, atomic destination preservation, temporary-name
   collisions, and Figure cleanup. CPU jobs provide Pillow and ffmpeg/ffprobe;
   inputs are synthetic, with no pixel hashes or wall-clock performance thresholds.
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

Follow the checklist order in [README TODO](../README.md#todo): visualization
correctness, release gate, installed distributions, CPU CI, CLI/examples, media
failure handling, and process-interruption recovery. Commit each completed item.
The accelerator extension remains unchecked until hardware is available.

Each step follows documentation first, failing tests for new behavior, minimal
implementation, and final removal of unused code or contradictory documentation.
Deleting a workflow or revising this plan does not warrant a permanent test that
asserts a retired filename is absent.

GitHub Actions references for implementation:
[hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
and [concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
