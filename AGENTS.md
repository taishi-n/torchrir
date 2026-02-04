# Repository Guidelines

## Project Structure & Module Organization
- Core code lives under `src/torchrir/` (main APIs in `core.py`, data models in `room.py`, patterns in `directivity.py`, helpers in `utils.py`).
- Tests should go under `tests/` when added.
- If you add data or example audio, store it under `assets/` or `examples/` to keep core logic clean.

## Build, Test, and Development Commands
- This repository uses `uv` for local development and publishing.
- Suggested commands once packaging is added:
  - `uv sync` to create/update the local environment
  - `uv run pytest` to run tests
  - `uv run ruff check .` or `uv run black .` for lint/format
- Publishing should be done with `uv` so the package is available to all users via `pip install torchrir`.

## Coding Style & Naming Conventions
- Prefer Python for core implementation, with PyTorch used for computation.
- Use 4-space indentation and follow PEP 8 naming (snake_case for functions/variables, PascalCase for classes).
- Suggested naming patterns:
  - RIR generation: `simulate_rir`, `simulate_dynamic_rir`
  - Modules: `core.py`, `room.py`, `directivity.py`, `utils.py`
- If you add formatters/linters (e.g., `black`, `ruff`), document exact versions and run commands here.

## Testing Guidelines
- No testing framework is configured yet.
- When tests are added, use `pytest` with filenames like `test_*.py` under `tests/`.
- Add unit tests for geometry, ISM correctness, and dynamic trajectory handling.

## Commit & Pull Request Guidelines
- No Git history is present, so commit conventions are not established.
- Recommended commit style: short, imperative subject (e.g., `Add ISM core implementation`).
- Pull requests should include:
  - A concise summary of changes
  - Any linked issues or design notes
  - Example outputs or benchmarks when touching performance-critical code

## Security & Configuration Tips
- Avoid committing large audio files or datasets; prefer documented download steps.
- Keep device selection explicit in APIs (e.g., `device="cpu"` or `"cuda"`).

## Agent-Specific Instructions
- Follow the specification section in `README.md` as the source of truth for required features and APIs.
- Update the specification section in `README.md` when user requirements change.
