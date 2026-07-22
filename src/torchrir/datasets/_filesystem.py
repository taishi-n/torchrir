"""Platform contracts for secure mutable dataset filesystem operations."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import sys


_ATOMIC_NOREPLACE_PLATFORMS = frozenset({"darwin", "linux"})

_SECURE_POSIX_DATASET_IO_AVAILABLE = (
    os.name == "posix"
    and sys.platform in _ATOMIC_NOREPLACE_PLATFORMS
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
)

_RENAME_NOREPLACE = 1
_RENAME_EXCHANGE = 2
_RENAME_EXCL = 0x00000004
_RENAME_SWAP = 0x00000002


def require_posix_dataset_filesystem() -> None:
    """Reject platforms that cannot provide the dataset safety contract."""

    if not _SECURE_POSIX_DATASET_IO_AVAILABLE:
        raise NotImplementedError(
            "dataset filesystem operations require Linux or macOS with POSIX "
            "dir_fd, O_DIRECTORY, and O_NOFOLLOW support"
        )


def rename_entry_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename an entry while refusing an existing destination."""

    require_posix_dataset_filesystem()
    if sys.platform not in _ATOMIC_NOREPLACE_PLATFORMS:
        raise NotImplementedError(
            "atomic no-replace dataset publication is supported only on Linux and macOS"
        )
    source = Path(source)
    destination = Path(destination)
    source_parent_descriptor = os.open(source.parent, _directory_open_flags())
    try:
        destination_parent_descriptor = os.open(
            destination.parent,
            _directory_open_flags(),
        )
        try:
            if sys.platform == "linux":
                _linux_rename(
                    source_parent_descriptor,
                    source.name,
                    destination_parent_descriptor,
                    destination.name,
                    flags=_RENAME_NOREPLACE,
                    operation="renameat2(RENAME_NOREPLACE)",
                )
            else:
                _darwin_rename(
                    source_parent_descriptor,
                    source.name,
                    destination_parent_descriptor,
                    destination.name,
                    flags=_RENAME_EXCL,
                    operation="renameatx_np(RENAME_EXCL)",
                )
        finally:
            os.close(destination_parent_descriptor)
    finally:
        os.close(source_parent_descriptor)


def exchange_entries(first: Path, second: Path) -> None:
    """Atomically exchange two existing filesystem entries."""

    require_posix_dataset_filesystem()
    if sys.platform not in _ATOMIC_NOREPLACE_PLATFORMS:
        raise NotImplementedError(
            "atomic dataset entry exchange is supported only on Linux and macOS"
        )
    first = Path(first)
    second = Path(second)
    first_parent_descriptor = os.open(first.parent, _directory_open_flags())
    try:
        second_parent_descriptor = os.open(second.parent, _directory_open_flags())
        try:
            if sys.platform == "linux":
                _linux_rename(
                    first_parent_descriptor,
                    first.name,
                    second_parent_descriptor,
                    second.name,
                    flags=_RENAME_EXCHANGE,
                    operation="renameat2(RENAME_EXCHANGE)",
                )
            else:
                _darwin_rename(
                    first_parent_descriptor,
                    first.name,
                    second_parent_descriptor,
                    second.name,
                    flags=_RENAME_SWAP,
                    operation="renameatx_np(RENAME_SWAP)",
                )
        finally:
            os.close(second_parent_descriptor)
    finally:
        os.close(first_parent_descriptor)


def _linux_rename(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    flags: int,
    operation: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise NotImplementedError(
            "this Linux runtime does not provide renameat2(RENAME_NOREPLACE)"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_parent_descriptor,
        os.fsencode(source_name),
        destination_parent_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    _raise_rename_error_if_needed(
        result,
        source_name=source_name,
        destination_name=destination_name,
        operation=operation,
    )


def _darwin_rename(
    source_parent_descriptor: int,
    source_name: str,
    destination_parent_descriptor: int,
    destination_name: str,
    *,
    flags: int,
    operation: str,
) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameatx_np = libc.renameatx_np
    except AttributeError as exc:
        raise NotImplementedError(
            "this macOS runtime does not provide renameatx_np(RENAME_EXCL)"
        ) from exc
    renameatx_np.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        source_parent_descriptor,
        os.fsencode(source_name),
        destination_parent_descriptor,
        os.fsencode(destination_name),
        flags,
    )
    _raise_rename_error_if_needed(
        result,
        source_name=source_name,
        destination_name=destination_name,
        operation=operation,
    )


def _raise_rename_error_if_needed(
    result: int,
    *,
    source_name: str,
    destination_name: str,
    operation: str,
) -> None:
    if result == 0:
        return
    error_number = ctypes.get_errno()
    unsupported_errors = {
        errno.ENOSYS,
        errno.EINVAL,
        getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
        errno.EOPNOTSUPP,
    }
    if error_number in unsupported_errors:
        raise NotImplementedError(
            f"the dataset filesystem does not support {operation}"
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        source_name,
        destination_name,
    )


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    return flags


__all__: list[str] = []
