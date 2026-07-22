"""Process-safe writer serialization for mutable dataset trees."""

from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import stat
import time
from typing import Iterator

from ._filesystem import require_posix_dataset_filesystem
from ..util._scalars import normalize_finite_real


DATASET_LOCK_TIMEOUT_SECONDS = 1800.0
_POLL_INTERVAL_SECONDS = 0.05


@contextmanager
def dataset_write_lock(
    path: Path,
    *,
    timeout: float = DATASET_LOCK_TIMEOUT_SECONDS,
) -> Iterator[None]:
    """Serialize dataset writers with an OS-released advisory file lock."""

    require_posix_dataset_filesystem()
    normalized_timeout = normalize_finite_real(
        timeout,
        name="timeout",
        positive=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _open_regular_lock_file(path)
    lock_file = os.fdopen(descriptor, "r+b", buffering=0)
    acquired = False
    try:
        deadline = time.monotonic() + normalized_timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for dataset lock: {path}")
            try:
                _try_lock(lock_file.fileno())
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"timed out waiting for dataset lock: {path}"
                    ) from exc
                time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))
                continue
            if time.monotonic() >= deadline:
                _unlock(lock_file.fileno())
                raise TimeoutError(f"timed out waiting for dataset lock: {path}")
            if not _lock_path_matches_descriptor(path, lock_file.fileno()):
                _unlock(lock_file.fileno())
                raise RuntimeError(
                    f"dataset lock path changed during acquisition: {path}"
                )
            acquired = True
            break
        yield
    finally:
        if acquired:
            _unlock(lock_file.fileno())
        lock_file.close()


def _open_regular_lock_file(path: Path) -> int:
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"dataset lock path is unsafe: {path}") from exc
    try:
        link_status = path.lstat()
        file_status = os.fstat(descriptor)
        same_file = (
            link_status.st_dev == file_status.st_dev
            and link_status.st_ino == file_status.st_ino
        )
        if not stat.S_ISREG(link_status.st_mode) or not same_file:
            raise ValueError(f"dataset lock path is unsafe: {path}")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _lock_path_matches_descriptor(path: Path, descriptor: int) -> bool:
    """Revalidate the acquired inode to prevent a split-brain lock path."""

    try:
        link_status = path.lstat()
        file_status = os.fstat(descriptor)
    except OSError:
        return False
    return (
        stat.S_ISREG(link_status.st_mode)
        and stat.S_ISREG(file_status.st_mode)
        and link_status.st_dev == file_status.st_dev
        and link_status.st_ino == file_status.st_ino
    )


if os.name == "posix":
    import fcntl

    def _try_lock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)

else:

    def _try_lock(descriptor: int) -> None:
        del descriptor
        raise NotImplementedError("dataset locking requires POSIX flock support")

    def _unlock(descriptor: int) -> None:
        del descriptor
        raise NotImplementedError("dataset locking requires POSIX flock support")


__all__: list[str] = []
