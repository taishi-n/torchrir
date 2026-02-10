"""Safe archive extraction helpers for dataset downloads."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path


def safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract a tar archive while rejecting unsafe paths and links."""
    root = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f"archive contains links, which are not allowed: {member.name}")

        target = (root / member.name).resolve()
        if os.path.commonpath([str(root), str(target)]) != str(root):
            raise ValueError(f"unsafe archive member path: {member.name}")

    tar.extractall(root)
