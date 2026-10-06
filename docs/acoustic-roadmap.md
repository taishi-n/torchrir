# Acoustic Model Roadmap

This roadmap defines the order and scope of future acoustic-model work.
It does not add public APIs or mark the proposed features as implemented.
The [README specification](https://github.com/taishi-n/torchrir/blob/main/README.md#specification)
defines current behavior, and its
[TODO checklist](https://github.com/taishi-n/torchrir/blob/main/README.md#p3-acoustic-models-and-spatial-visualization)
tracks completion.

## Current model and scope

`simulate(scene, config)` implements ISM for 2D/3D shoebox rooms. The room has
one scalar reflection coefficient per wall, or coefficients derived from T60.
Sources and microphones support analytic directivity with fixed per-entity
orientations. Dynamic RIRs sample positions on an explicit frame schedule.
Signal synthesis supports moving sources with fixed microphones, or fixed
sources with moving microphones, under their respective time references.

The optional diffuse tail is a statistical continuation of the ISM response.
The optional high-pass filter is an explicit RIR post-process. Neither option
adds ray tracing, irregular boundaries, or a wave-equation solver.

| Area | Status | Scope decision |
|---|---|---|
| Shoebox ISM | Implemented | Keep the current model as the numerical baseline for the extensions below. |
| Path selection, orientation trajectories, and joint motion | Planned | Develop incrementally within shoebox scenes, with separate acceptance gates. |
| Non-shoebox geometry | Candidate | Begin with static 2D convex polygons and direct/first-order specular paths. |
| Ray tracing | Later candidate | Evaluate after geometry and boundary contracts are defined; no backend is scheduled yet. |
| FDTD | Deferred | Outside the active implementation sequence; revisit only for a specified wave-equation use case. |

The comparison tables label the future backend explicitly as **Ray Tracing**.
This separates it from the existing ISM. For terminology, the
[pyroomacoustics room-simulation documentation](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html)
also distinguishes ISM from its hybrid ISM/ray-tracing simulator. That
documentation is a conceptual reference, not a replacement for the pinned
runtime oracles in the test plan.

## Implementation order and acceptance gates

### 1. Path selection within shoebox ISM

Start with first-K arrival selection for each source/microphone pair and each
dynamic frame. Before implementation, define the candidate image set, whether
the direct path counts toward K, equal-arrival tie ordering, and treatment of
paths outside the requested output horizon. The selection must be independent
of image chunk size and enumeration order.

The first implementation must match direct sums of independently enumerated
paths in small rooms. Selecting every eligible path must reproduce the current
unfiltered ISM result within stated numerical tolerances. Cover asymmetric
walls, non-omnidirectional endpoints, static/dynamic equivalence, and selections
spanning chunk boundaries. Define how selection interacts with diffuse-tail
level estimation before allowing those options together.

Strongest-K and energy-threshold selection follow separately. Their specification
must define the ranking quantity, normalization, and whether the threshold
refers to individual path contributions or the accumulated waveform. Do not
infer a waveform-energy guarantee from a path-gain ranking.

### 2. Orientation trajectories and spatial response plots

Add orientation changes for sources and microphones on the same scene schedule
as their positions. Specify world/local coordinates, rotation representation,
normalization, and interpolation before choosing an API. State which time
reference selects an endpoint's orientation and reject combinations the initial
single-moving-side model cannot represent.

Acceptance requires constant-orientation equivalence, analytically checked
rotation of cardioid/bidirectional responses, reflected-source orientation
parity, and rejection of ambiguous shapes or mismatched schedules. Plotting
must use the same coordinate and directivity conventions. Validate static 3D
response plots first, then scheduled orientation changes; plotted maxima/nulls
must agree with the numerical gains.

### 3. Simultaneous source and microphone motion

Define source geometry at emission time and receiver geometry at observation
time, including delay solving, interpolation, gain, time support, and endpoint
orientation. A position snapshot at one shared frame time is insufficient as
the specification for this extension. Use the
[two-time-kernel comparison](comparisons.md#processing-and-time-coordinates)
as design input; upstream implementations are not unconditional correctness
oracles.

Begin with direct sound and a small CPU reference calculation. Require analytic
arrival checks, static and one-moving-side limits, source superposition,
startup/tail coverage, and convergence as the temporal sampling is refined.
State the conditions under which the limits match existing frame-based
convolution; do not require sample equality between different approximations.
Add reflected paths only after the direct-sound contract passes.

### 4. Non-shoebox geometry

The first geometric extension is a static 2D convex polygon with planar edges,
per-edge scalar reflection coefficients, and direct/first-order specular paths.
Define containment, boundary-point rejection, edge ordering, reflection-point
validity, visibility, and duplicate-path handling before adding a room model.
This stage does not include arbitrary 3D meshes, moving geometry, diffraction,
or a material database.

Acceptance requires hand-computed single-reflection cases, a rectangle that
matches the existing shoebox model at the same reflection order, invalid or
degenerate polygon rejection, and coordinate-transform invariance. Extend to
higher orders, non-convex geometry, and 3D meshes in separate stages with
explicit visibility and boundary tests. A geometry extension does not by
itself provide ray tracing or full dataset-reproduction fidelity.

### 5. Ray-tracing evaluation

Reconsider ray tracing after the non-shoebox geometry contract is usable and
a concrete scene requires a model beyond the implemented ISM. An evaluation
must specify source emission and receiver capture rules, energy/amplitude
conversion, boundary reflection/scattering, random seeds, termination, and
whether the result is standalone or combined with ISM.

Set quantitative CPU memory/runtime budgets and convergence tolerances before
implementation. Validate a simple room against independent analytic quantities
and the established ISM baseline where their assumptions match. A hybrid must
define how it avoids counting the same contribution twice. Only then decide
whether a backend belongs in the public API.

## Features that require their own physical specification

Microphone hardware response and near-field speech remain separate planned
items. They need identified target behavior before model parameters are added:

- **Microphone response:** specify sensitivity units, response/filter latency,
  self-noise spectrum and level, channel correlations, and random seeding.
  Require calibrated gain/filter cases and statistical noise checks with
  explicit tolerances; establish the units before claiming an absolute SPL.
- **Near-field speech:** specify the source model, valid distance/frequency
  range, and an analytic or measured reference. Tests must establish those
  limits and the intended transition to the point-source model. A smaller
  source/microphone separation alone does not define the new model.

These items can proceed independently once their specifications and references
are available; they are not prerequisites for the first path-selection task.

## FDTD reconsideration criteria

FDTD has no implementation milestone in the current sequence. Reopening it
requires a concrete use case, target frequency range, boundary/source/receiver
definitions, CPU memory/runtime budget, and an independent reference solution.
Its acceptance plan must include stability, spatial/temporal convergence, and
boundary-error checks. No solver selector or placeholder class is added merely
to reserve this possibility.

## Development and validation

For each implementation item, update the README specification first, resolve
the choices listed above, then add failing tests and the minimum implementation.
Keep analytic and metamorphic CPU checks authoritative; optional external
comparisons supplement them. GPU parity remains a separate manual gate and
does not block documenting or validating a CPU reference model.

The roadmap records scope and acceptance criteria independently of the
outstanding TODO checklist. A feature is complete only after its documented
acceptance criteria pass. A documentation-only roadmap change needs link and
strict-build checks, not a test asserting the absence of future APIs. See the
[test and CI plan](https://github.com/taishi-n/torchrir/blob/main/tests/README.md#test-and-ci-plan).
