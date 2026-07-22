"""Regression tests for time conventions used by dynamic examples."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _dynamic_convolver_time_references(
    path: Path,
    function_name: str | None,
) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scope: ast.AST = tree
    if function_name is not None:
        scope = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        )

    time_references = []
    for node in ast.walk(scope):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "DynamicConvolver":
            continue
        time_reference = next(
            (
                keyword.value.value
                for keyword in node.keywords
                if keyword.arg == "time_reference"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ),
            None,
        )
        if time_reference is not None:
            time_references.append(time_reference)
    return time_references


def _call_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return None


@pytest.mark.parametrize(
    ("relative_path", "function_name", "expected_time_references"),
    [
        ("examples/cli.py", "_run_dynamic_src", ["emission"]),
        ("examples/cli.py", "_run_dynamic_mic", ["observation"]),
        (
            "examples/benchmark_device.py",
            "_bench_dynamic",
            ["observation", "observation"],
        ),
        ("examples/dynamic_src.py", "main", ["emission"]),
        ("examples/dynamic_mic.py", "main", ["observation"]),
        ("examples/build_dynamic_dataset.py", "main", ["emission"]),
        ("examples/getting_started.py", None, ["emission"]),
    ],
)
def test_dynamic_examples_use_motion_appropriate_time_convention(
    relative_path: str,
    function_name: str | None,
    expected_time_references: list[str],
) -> None:
    time_references = _dynamic_convolver_time_references(
        REPOSITORY_ROOT / relative_path, function_name
    )

    assert time_references == expected_time_references


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/cli.py",
        "examples/benchmark_device.py",
        "examples/dynamic_src.py",
        "examples/dynamic_mic.py",
        "examples/build_dynamic_dataset.py",
        "examples/getting_started.py",
    ],
)
def test_dynamic_examples_derive_trajectory_grid_from_exact_schedule(
    relative_path: str,
) -> None:
    path = REPOSITORY_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    normalized_progress_calls = [
        call for call in calls if _call_name(call) == "normalized_progress"
    ]
    linear_calls = [call for call in calls if _call_name(call) == "linear_trajectory"]

    assert normalized_progress_calls, (
        f"{relative_path} does not derive schedule progress"
    )
    assert linear_calls, (
        f"{relative_path} does not use explicit trajectory interpolation"
    )
    for call in linear_calls:
        assert any(keyword.arg == "progress" for keyword in call.keywords), (
            f"{relative_path}:{call.lineno} omits explicit progress"
        )
