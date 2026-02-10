# Repository Guidelines

## Project Structure & Module Organization
- Core library code lives under `src/torchrir/`:
  - Simulation: `core.py`, `room.py`, `directivity.py`, `config.py`
  - Dynamic convolution: `dynamic.py`, `signal.py`
  - Datasets + utilities: `datasets/` (CMU ARCTIC support + template stub)
  - Plotting + scene helpers: `plotting.py`, `plotting_utils.py`, `scene_utils.py`
- Tests live under `tests/` and use `pytest`.
- Examples live under `examples/` and should import from `torchrir` (no duplicated utilities).
- Keep large assets out of the repo; use `assets/` only for small static files.

## Build, Test, and Development Commands
- This repository uses `uv` for local development and publishing.
- Common commands:
  - `uv sync` to create/update the virtual environment
  - `uv run pytest` to run tests
  - `uv build` / `uv publish` for releases (must still support `pip install torchrir`)
  - `uv run --group docs sphinx-build -b html docs docs/_build/html` to build docs locally
  - `uv run ruff format .` for formatting
  - `uv run ty check` for type checking
- `uv.lock` is committed. Update it **only** when `pyproject.toml` changes.

## Coding Style & Naming Conventions
- Prefer Python for core implementation, with PyTorch used for computation.
- Use 4-space indentation and follow PEP 8 naming (snake_case for functions/variables, PascalCase for classes).
- Suggested naming patterns:
  - RIR generation: `simulate_rir`, `simulate_dynamic_rir`
  - Modules: `core.py`, `room.py`, `directivity.py`, `utils.py`
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
