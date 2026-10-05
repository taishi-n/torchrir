"""Reject inconsistent release inputs before publishing any artifact."""

import io
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

CHECK = Path(__file__).resolve().parents[1] / "scripts/check_distribution.py"


@pytest.mark.parametrize(
    "mismatch", [None, "tag", "lock", "wheel", "sdist", "license", "changelog"]
)
def test_distribution_gate(tmp_path, mismatch):
    version = "3.0.0"
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname="torchrir"\nversion="{version}"\n'
    )
    lock_version = "2.0.0" if mismatch == "lock" else version
    (tmp_path / "uv.lock").write_text(
        f'[[package]]\nname="torchrir"\nversion="{lock_version}"\nsource={{editable="."}}\n'
    )
    dist = tmp_path / "dist"
    dist.mkdir()
    with zipfile.ZipFile(dist / "torchrir-3.0.0-py3-none-any.whl", "w") as wheel:
        wheel_version = "2.0.0" if mismatch == "wheel" else version
        license_ = "MIT" if mismatch == "license" else "Apache-2.0"
        wheel.writestr(
            "torchrir-3.0.0.dist-info/METADATA",
            f"Name: torchrir\nVersion: {wheel_version}\nLicense-Expression: {license_}\n",
        )
    with tarfile.open(dist / "torchrir-3.0.0.tar.gz", "w:gz") as sdist:
        sdist_version = "2.0.0" if mismatch == "sdist" else version
        entries = {"PKG-INFO": f"Name: torchrir\nVersion: {sdist_version}\n"}
        if mismatch != "changelog":
            entries["CHANGELOG.md"] = "Release notes"
        for name, content in entries.items():
            data = content.encode()
            info = tarfile.TarInfo(f"torchrir-3.0.0/{name}")
            info.size = len(data)
            sdist.addfile(info, io.BytesIO(data))
    tag = "v2.0.0" if mismatch == "tag" else "v3.0.0"
    result = subprocess.run(
        [sys.executable, str(CHECK), "--tag", tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if mismatch is None:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
        assert mismatch in result.stderr.lower(), result.stderr
