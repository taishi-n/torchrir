# Repository Guidelines

## Project Structure & Module Organization
- Core library code lives under `src/torchrir/`:
  - Models and configuration: `models/`, `config.py`
  - Simulation: `sim/` (the image-source kernel lives in `sim/ism/`)
  - Static and dynamic convolution: `signal/`
  - Datasets and utilities: `datasets/`, `geometry/`, `util/`
  - Plotting and output helpers: `viz/`, `io/`
- Tests live under `tests/` and use `pytest`.
- Examples live under `examples/` and should import from `torchrir` (no duplicated utilities).
- Keep large assets out of the repo; use `assets/` only for small static files.

## Development Policy and Required Cycle
- This project is under active development. Do not preserve backward
  compatibility or implement data migration. Rewrite affected code toward the
  simplest coherent design for the current requirements (KISS), and regenerate
  generated data when its format or numerical meaning changes.
- Do not retain compatibility shims, deprecated APIs, migration utilities, or
  tests whose only purpose is preserving historical behavior or remembering
  removed APIs or fields. Update repository callers to the canonical API.
- Follow this cycle for every change:
  1. **Documentation first:** update the relevant specifications and documents,
     starting with `README.md` when requirements change. Check related documents
     and examples for contradictions and stale information before changing tests
     or implementation.
  2. **Tests first (TDD):** write or update tests for the documented behavior and
     verify the expected failure before implementing a behavior change. For
     documentation-only changes or removal of obsolete tests with no behavior
     change, review the retained coverage without inventing an artificial failure.
  3. **Minimal implementation:** implement only what satisfies the current
     specification and tests, without speculative abstractions or compatibility
     branches.
  4. **Final cleanup:** run the relevant checks and review the diff for unused
     code, obsolete tests, and contradictory or outdated documentation.
- Keep tests for current numerical correctness, input validation, supported
  external integrations, and filesystem failure/crash recovery. These protect
  current behavior and are not backward-compatibility or data-migration tests.

## Build, Test, and Development Commands
- This repository uses `uv` for local development and publishing.
- Common commands:
  - `uv sync --group test` to create/update the test environment
  - `uv run --group test pytest` to run tests
  - `uv build` / `uv publish` for releases (must still support `pip install torchrir`)
  - `uv run --group docs zensical build --strict` to build docs locally
  - `uv run ruff format .` for formatting
  - `uv run ty check` for type checking
- `uv.lock` is committed. Update it **only** when `pyproject.toml` changes.

## Coding Style & Naming Conventions
- Prefer Python for core implementation, with PyTorch used for computation.
- Use 4-space indentation and follow PEP 8 naming (snake_case for functions/variables, PascalCase for classes).
- Suggested naming patterns:
  - RIR generation: `simulate(scene, config)`
  - Scene models: `StaticScene`, `DynamicScene`
  - Modules: `simulators.py`, `dynamic.py`, `room.py`, `directivity.py`
- Use `SimulationConfig` for simulation parameters; avoid global config state.
- If you add formatters/linters (e.g., `black`, `ruff`), document exact versions and run commands here.

## Testing Guidelines
- Use `pytest` with filenames like `test_*.py` under `tests/`.
- Add unit tests for geometry, ISM correctness, and dynamic trajectory handling.
  - Prefer parity tests across `cpu`, `cuda`, and `mps` where available.

## Commit & Pull Request Guidelines
- Codex proposes commit message drafts; the user reviews/approves before committing.
- All commits must follow Conventional Commits via Commitizen (use `uv run cz commit`).
- Pull requests should include:
  - A concise summary of changes
  - Any linked issues or design notes
  - Example outputs or benchmarks when touching performance-critical code

## Versioning & Releases
- Use Commitizen for version bumps, tagging, and changelog updates.
  - Run: `uv run cz bump --changelog --yes`
  - Version comes from `pyproject.toml` (PEP 621) and tag format is `vX.Y.Z`.
- Warning: on every version bump, ensure `uv.lock` is updated to the same project version and committed before pushing.
  - Before `git push`, verify there is no leftover `uv.lock` diff and include it in the release-related commits when changed.
- After bumping, push commits and tags (`git push origin main --tags`).
- Signed tags are created manually by the user. Example:
  - `git tag -s vX.Y.Z -m "vX.Y.Z"`

## Security & Configuration Tips
- Avoid committing large audio files or datasets; prefer documented download steps.
- Keep device selection explicit in APIs (e.g., `device="cpu"` or `"cuda"`).
  - MPS support is expected on Apple Silicon; fallback to CPU where needed.

## Agent-Specific Instructions
- Follow the specification section in `README.md` as the source of truth for required features and APIs.
- Update the specification section in `README.md` when user requirements change.
- Keep README links in Markdown format.
