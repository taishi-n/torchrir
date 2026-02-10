from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from torchrir.datasets._archive import safe_extractall


def _write_tar_with_member(path: Path, member_name: str, data: bytes = b"x") -> None:
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def test_safe_extractall_allows_normal_members(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tar.gz"
    out_dir = tmp_path / "out"
    _write_tar_with_member(archive, "dir/file.txt", b"ok")
    out_dir.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        safe_extractall(tar, out_dir)

    assert (out_dir / "dir" / "file.txt").exists()


def test_safe_extractall_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    out_dir = tmp_path / "out"
    _write_tar_with_member(archive, "../evil.txt", b"bad")
    out_dir.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="unsafe archive member path"):
            safe_extractall(tar, out_dir)


def test_safe_extractall_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "link.tar.gz"
    out_dir = tmp_path / "out"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="sym")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        tar.addfile(info)
    out_dir.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="archive contains links"):
            safe_extractall(tar, out_dir)
