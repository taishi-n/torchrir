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
  convolution using fixed random seeds. Gradient checks cover uneven
  observation-time chunks, FFT-size changes, and tail-only frames. Manual
  CUDA/MPS tests compare outputs and gradients for both dynamic conventions.
- `test_signal_work_budget.py` bounds the total transformed samples for
  static/emission/observation convolution. The budgets require source reduction
  before inverse FFT and observation-time overlap-save, including nonuniform
  intervals and frames in the reverberation tail. Independent direct sums and
  gradients verify numerical behavior. These are deterministic operation-volume
  checks, not machine-dependent latency thresholds or bitwise comparisons of
  different FFT algorithms.
- `test_cli_integration.py` executes the unified CLI, all three standalone
  scenarios, and the dataset builder in fresh Python processes against tiny
  local synthetic CMU-layout recordings. JSON/YAML configuration round trips,
  explicit option precedence, time-reference metadata, sample counts, and
  mixture/reference sums are checked. The general dataset example also runs
  against synthetic CMU ARCTIC and LibriSpeech trees at 8 kHz, checking that
  the room, RIR length, metadata, and WAV output use the input audio's sample
  rate. No corpus downloads are needed.
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
3.11.4 collects branch-inclusive coverage with the 75% threshold. The workflow
sets `UV_PYTHON` to 3.11.4 by default and overrides it from each CPU matrix entry
so the development `.python-version` cannot select a different interpreter.
Before pytest, the environment's actual Python version must match the requested
major/minor, and the patch when specified.
Each test job retains JUnit reports and explicit skip reasons;
empty/all-skipped reports fail.
Comparison reports additionally reject any skips. ffmpeg/ffprobe are installed
in CPU jobs so real-media tests cannot disappear behind a missing-codec skip.
Actionlint 1.7.12 validates workflow syntax and invokes the runner's ShellCheck
for embedded shell scripts. The quality job requires ShellCheck and prints its
version before actionlint, so a missing executable fails rather than silently
omitting shell analysis. Local workflow checks must also provide ShellCheck.
Optional reference libraries are imported at runtime; the quality environment
does not need them to type-check the repository. When checking a separate local
environment, pass its interpreter explicitly with `ty check --python PATH`.
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

The table records the implemented CPU validation contract. Manual accelerator
results are recorded below; they remain separate from automated CPU validation.

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
   dimensions, and requested audio streams. Low-FPS MP4 cases verify complete
   decoded audio sample coverage and channel correlation through the tail.
   Failure-injection tests cover missing tools/inputs, encoder failures, atomic
   destination preservation, temporary-name collisions, and Figure cleanup.
   CPU jobs provide Pillow and ffmpeg/ffprobe;
   inputs are synthetic, with no pixel hashes or wall-clock performance thresholds.
4. **Retain manual accelerator validation.**
   Existing device tests cover basic static/dynamic RIR parity and both dynamic
   convolution conventions' gradients. Extended checks cover static and
   observation-time output/gradient parity, multi-source/microphone and
   chunk-boundary cases, and CUDA eager/compiled accumulation parity with LUT
   enabled/disabled.
   Use float32/float64 on CUDA. Verify the actual output/config device; silent CPU
   fallback must not count as a device pass. Compilation validation needs an
   actual supported backend, not only a mocked flag. The
   [CUDA validation record](#cuda-validation-record) closes the current P2 item;
   repeat CUDA validation and float32 MPS checks when accelerator paths change.
   These checks remain manual.

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

### Local workflow validation with act

Use `act` with Docker to exercise the Linux jobs from the actual workflow,
including dependency installation and artifact uploads. Use the default
`actions/checkout` event ref/SHA selection: GitHub validates the event commit,
while act 0.2.89 copies the local working tree, including uncommitted source
changes. Do not override checkout's `ref` or use act's `--no-skip-checkout` for
local validation. Keep act's default container copy mode rather than `--bind`.
The medium runner image does not include every tool installed on GitHub-hosted
runners. In particular,
install ShellCheck in the local image before running quality checks; actionlint
can otherwise omit shell analysis when ShellCheck is absent.

From the repository root, prepare the local runner:

```bash
docker build --platform linux/amd64 -t torchrir-act:local - <<'DOCKERFILE'
FROM catthehacker/ubuntu:act-24.04
RUN apt-get update && apt-get install -y --no-install-recommends shellcheck \
    && rm -rf /var/lib/apt/lists/*
DOCKERFILE

act_args=(
  workflow_dispatch -W .github/workflows/ci.yml
  --container-architecture linux/amd64
  -P ubuntu-latest=torchrir-act:local --pull=false
  --container-options '-v torchrir-act-uv-cache:/tmp/setup-uv-cache'
  --container-daemon-socket -
  --artifact-server-path /tmp/torchrir-act-artifacts
)
for job in quality docs comparison distribution; do
  act "${act_args[@]}" -j "$job" || exit 1
done
for python_version in 3.11.4 3.12 3.13; do
  act "${act_args[@]}" -j test \
    --matrix os:ubuntu-latest --matrix python:"$python_version" || exit 1
done
```

The architecture matches the Linux GitHub runners, including on Apple Silicon.
Invoke jobs and matrix entries separately to limit Docker memory and disk usage;
act's `--concurrent-jobs 1` does not serialize entries within a matrix job.
The artifact server keeps report/distribution uploads local; no GitHub or PyPI
token is required.
The named Docker volume reuses downloaded packages between jobs while every
job still creates its own environment with `uv sync --locked`.
Rerun only the affected job or matrix entry after diagnosing a failure. Preserve
logs and inspect JUnit counts and coverage rather than relying only on the
workflow exit code.

Docker-based Linux runs do not validate the macOS job, accelerator execution,
GitHub permissions/concurrency/timeouts, or PyPI publication. See act's
[unsupported functionality](https://nektosact.com/not_supported.html).
Check the macOS job on GitHub or in a separate native macOS environment.
For release-gate validation, use the release workflow's `workflow_dispatch`
event with `release-tag`; never use a tag-push event for a local validation run.

### Manual accelerator checks

Prepare the desired PyTorch/CUDA or PyTorch/MPS environment before running these
commands. They use `--no-sync` to preserve that prepared environment. Record the
commit, Python/PyTorch version, actual device, driver/runtime where applicable,
and the pytest summary. An unavailable accelerator fails the preflight; it must
not be recorded as a successful GPU check.

```bash
# CUDA: RIR parity, schedules, both convolution gradients, and mixed-device collate.
uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
uv run --no-sync pytest -q -rs tests/test_device_parity.py tests/test_signal.py tests/test_datasets.py -m cuda

# MPS: run separately on an environment with an available MPS device.
uv run --no-sync python -c 'import torch; assert torch.backends.mps.is_available(), "MPS unavailable"'
PYTORCH_ENABLE_MPS_FALLBACK=0 uv run --no-sync pytest -q -rs tests/test_device_parity.py tests/test_signal.py tests/test_datasets.py -m mps
```

Run MPS checks with access to Metal; a sandboxed process can report an unavailable
device even when the host supports MPS. A preflight or suite that skips all MPS
cases is not a successful hardware check. Keep CPU fallback disabled and assert
that outputs and gradients remain on MPS during extended validation.

The marked suite covers basic static/dynamic RIR parity, frame schedules,
emission/observation convolution gradients, and mixed-device collate validation.
Extended validation also requires static convolution output and
gradient parity, multiple sources/microphones, and image/accumulation and
convolution frame-chunk boundaries. Compare against CPU results and use
independent direct sums for convolution outputs and gradients. CUDA validation
also checks float32/float64 eager/compiled accumulation with LUT enabled/disabled,
including execution through a real supported compiler backend.

#### MPS validation record

MPS was validated on 2026-10-06 at commit `0ccdbf2` on an Apple M3 Max with
macOS 15.7.7, Python 3.11.11, and PyTorch 2.10.0. With CPU fallback disabled,
the marked suite passed all six tests with zero skips. A separate manual harness
passed 24 additional float32 cases: 11 static/emission/observation convolution
cases compared with CPU and independent float64 direct sums and analytic
gradients; 12 static/moving-source/moving-microphone RIR cases with multiple
sources/microphones, different image/accumulation chunks, and requested LUT
enabled/disabled; and one static RIR position-gradient case. Actual outputs,
resolved configurations, and gradients were checked on MPS. LUT resolved to
disabled on MPS as specified.

The maximum absolute MPS/CPU differences were `1.20e-6` across convolution
outputs/gradients, `2.64e-6` for RIRs, and `2.27e-5` for position gradients.
All passed the predefined tolerances: `rtol=3e-4, atol=3e-5` for convolution,
and `rtol=1e-3, atol=1e-4` for RIRs/position gradients. This records numerical
validation on the tested fixtures; it does not establish bitwise equality,
CUDA coverage, or a performance improvement.

#### CUDA validation record

CUDA was validated on 2026-10-07 at commit
`b8bcc5dc21d5606db634c47b1d7ee164d4ca0da5` (TorchRIR 3.0.3), with a clean
working tree during the accelerator runs. The environment was Ubuntu 22.04.5
LTS, x86_64, kernel `5.15.0-198-generic`, Python 3.11.11, NumPy 2.4.2, and
PyTorch `2.10.0+cu128` (CUDA runtime 12.8). The NVIDIA open kernel driver was
`580.178.04`. Both GPUs were RTX A4000s with 16 GiB memory, at PCI bus IDs
`00000000:C1:00.0` and `00000000:C2:00.0`.

Each physical GPU ran in a separate process with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `CUDA_VISIBLE_DEVICES=0` or `1`. In either process the selected GPU appeared
as logical `cuda:0`. A preliminary computation/autograd check passed float32
and float64 on both devices.

| Validation | GPU 0 | GPU 1 |
| --- | --- | --- |
| Marked CUDA suite: RIR parity, schedules, both dynamic convolution gradients, mixed-device collate | 6 passed, 0 skipped | 6 passed, 0 skipped |
| Extended eager harness | 58 passed, 0 skipped | 58 passed, 0 skipped |
| Extended compiled harness | 16 passed, 0 skipped | 16 passed, 0 skipped |

The extended harness had SHA-256
`3b2491b9b1e7832d500a5a2fcef4d95ce8a12c0467beff76b06a4b5c6be85b20`.
Its eager phase covered both dtypes with 18 FFT/static/emission/observation
convolution cases, 36 static/moving-source/moving-microphone RIR cases, and
4 static source/microphone position-gradient cases. Convolution outputs and
signal/RIR gradients were compared with CPU results and independent float64
direct sums and analytic gradients. Cases included source broadcasting,
nonuniform schedules, the eight-frame batch boundary, FFT-size changes, and
frames active only in the convolution tail.

RIR fixtures used two sources, three microphones, four dynamic frames, order 3,
512 output samples, and LUT enabled/disabled. Eager image/accumulation chunk
pairs were `(2048, 4096)`, `(11, 4)`, and `(1, 1)`. The compiled phase used
`(11, 4)` for 12 RIR and 4 position-gradient cases and compared with CPU and
CUDA eager results. Outputs, resolved configs, and gradients were asserted to
remain on the selected CUDA device with the requested dtype and effective
LUT/compile settings. Finite, nonzero outputs/gradients were required.

Compilation used the actual Inductor backend with compiler error suppression
disabled and `torch._dynamo.config.recompile_limit=64` for the fixture matrix.
Each compiled case required execution of an Inductor graph containing scatter
accumulation. Both GPUs recorded 20 such compiled graphs and 96 graph executions;
a requested compile flag alone did not count as a pass.

All comparisons passed the predefined `rtol, atol` pairs:

| Comparison | float32 | float64 |
| --- | --- | --- |
| Convolution outputs/gradients | `3e-4, 3e-5` | `1e-10, 1e-11` |
| RIR outputs | `1e-4, 1e-5` | `1e-9, 1e-11` |
| Position-gradient cases, including RIR outputs | `1e-3, 1e-4` | `1e-8, 1e-9` |

Both GPUs reported maximum absolute errors of `3.40e-6` (float32) and `7.11e-15`
(float64) for eager convolution outputs/gradients, and `4.90e-6` and `9.05e-15`
for eager/compiled RIR outputs. Position-gradient case summaries include their
RIR comparisons and are not isolated gradient-error measurements.
These results establish numerical agreement on the listed fixtures, with each
GPU exercised individually.

The server reports were `/tmp/torchrir_cuda_eager_gpu0.json`,
`/tmp/torchrir_cuda_eager_gpu1.json`, `/tmp/torchrir_cuda_compile_gpu0.json`,
and `/tmp/torchrir_cuda_compile_gpu1.json`.

A subsequent full-suite run on the server, after applying the MP4 audio-tail
fix to the same base revision, passed 893 tests with 20 skips. The skips were
14 unavailable external-reference cases (pyroomacoustics, gpuRIR, rir-generator)
and 6 unavailable MPS cases; none were CUDA skips. This full-suite result is
distinct from the zero-skip accelerator runs and does not complete the skipped
external comparisons.

#### External CUDA reference checks

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

The P2 CUDA validation item in [README TODO](../README.md#todo) was completed
on 2026-10-07 for the environment and fixtures in the
[CUDA validation record](#cuda-validation-record). Repeat manual validation
when accelerator paths change.
Commit each completed implementation item.

For subsequent P3 acoustic-model work, follow the
[acoustic model roadmap](../docs/acoustic-roadmap.md): orientation trajectories
and spatial response plots, joint motion, and staged non-shoebox geometry.
Energy-threshold path selection remains deferred outside the current
implementation sequence. Ray tracing requires a separate evaluation gate; FDTD
remains deferred. The roadmap defines the CPU analytic, limiting-case, and
convergence checks required before each feature is complete.

Each step follows documentation first, failing tests for new behavior, minimal
implementation, and final removal of unused code or contradictory documentation.
Deleting a workflow or revising this plan does not warrant a permanent test that
asserts a retired filename is absent.

GitHub Actions references for implementation:
[hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners),
[reusable workflows](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows),
and [concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
