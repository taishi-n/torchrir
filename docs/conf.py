from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _read_project_version() -> str:
    pyproject = PROJECT_ROOT / "pyproject.toml"
    try:
        with pyproject.open("rb") as f:
            data = tomllib.load(f)
        return str(data["project"]["version"])
    except Exception:
        return "0.0.0"


project = "torchrir"
author = "torchrir contributors"
release = _read_project_version()
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.linkcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

autosummary_generate = True

myst_enable_extensions = [
    "colon_fence",
]

html_theme = "sphinx_rtd_theme"

exclude_patterns = ["_build"]

REPO_URL = "https://github.com/taishi-n/torchrir"


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode()
            .strip()
        )
    except Exception:
        return "main"


_REV = _git_rev()


def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    if domain != "py":
        return None
    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname or not fullname:
        return None
    try:
        mod = importlib.import_module(modname)
    except Exception:
        return None

    obj = mod
    for part in fullname.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None

    try:
        filename = inspect.getsourcefile(obj)
        if not filename:
            return None
        relpath = os.path.relpath(filename, PROJECT_ROOT)
        source, start = inspect.getsourcelines(obj)
    except Exception:
        return None

    end = start + len(source) - 1
    return f"{REPO_URL}/blob/{_REV}/{relpath}#L{start}-L{end}"
