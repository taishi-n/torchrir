"""Safe archive, temporary-workspace, and dataset publication helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
from typing import BinaryIO, Literal
import uuid

from ._locking import dataset_write_lock
from ._filesystem import (
    rename_entry_noreplace,
    require_posix_dataset_filesystem,
)


MAX_ARCHIVE_MEMBERS = 1_000_000
MAX_ARCHIVE_FILE_BYTES = 8 * 1024**3
MAX_ARCHIVE_TOTAL_BYTES = 512 * 1024**3

_TRANSACTION_MARKER = "torchrir dataset publication transaction\n"
_TEMPORARY_MARKER = "torchrir temporary workspace\n"
_OWNER_FILE_NAME = "owner"
_TRANSACTION_MANIFEST = "manifest.json"
_TRANSACTION_MANIFEST_TEMP_PREFIX = f".{_TRANSACTION_MANIFEST}."
_TRANSACTION_MANIFEST_TEMP_SUFFIX = ".tmp"
_TRANSACTION_PHASES = frozenset({"initialized", "backed_up", "published"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Stable device/inode identity of an already-open filesystem object."""

    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class TransactionManifest:
    staged: FileIdentity
    original: FileIdentity
    phase: str


class ArchiveDigestMismatch(OSError):
    """An opened archive did not match its pinned digest."""

    def __init__(
        self,
        path: Path,
        *,
        identity: FileIdentity,
        algorithm: str,
        expected: str,
        actual: str,
    ) -> None:
        super().__init__(
            f"{algorithm.upper()} mismatch for {path}: expected {expected}, got {actual}"
        )
        self.path = path
        self.identity = identity


def is_regular_file_below(path: Path, root: Path) -> bool:
    """Return whether ``path`` is a non-symlink file contained below ``root``."""

    try:
        with open_regular_file_below(path, root=root):
            return True
    except (OSError, ValueError):
        return False


@contextmanager
def open_regular_file_below(
    path: Path,
    *,
    root: Path,
) -> Iterator[tuple[BinaryIO, FileIdentity]]:
    """Open one contained regular file without following its final symlink.

    The returned identity and stream refer to the same descriptor. Consumers
    must continue using that stream instead of reopening ``path``.
    """

    require_posix_dataset_filesystem()
    stack = ExitStack()
    descriptor = -1
    try:
        parent_descriptor, name = stack.enter_context(
            _open_parent_directory_below(path, root=root)
        )
        descriptor = os.open(
            name,
            _regular_file_open_flags(),
            dir_fd=parent_descriptor,
        )
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(opened_status.st_mode):
            raise ValueError(
                f"file must be a contained non-symlink regular file: {path}"
            )
        identity = _identity_from_status(opened_status)
        file = os.fdopen(descriptor, "rb")
        descriptor = -1
    except ValueError:
        if descriptor >= 0:
            os.close(descriptor)
        stack.close()
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        stack.close()
        raise ValueError(
            f"file must be a contained non-symlink regular file: {path}"
        ) from exc
    try:
        with file:
            yield file, identity
    finally:
        stack.close()


@contextmanager
def open_verified_tar(
    path: Path,
    *,
    root: Path,
    algorithm: str,
    expected_digest: str,
    mode: Literal["r:gz", "r:bz2"],
) -> Iterator[tarfile.TarFile]:
    """Verify and open a tar archive through one unchanged descriptor."""

    with open_regular_file_below(path, root=root) as (file, identity):
        digest = hashlib.new(algorithm, usedforsecurity=False)
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise ArchiveDigestMismatch(
                path,
                identity=identity,
                algorithm=algorithm,
                expected=expected_digest,
                actual=actual_digest,
            )
        file.seek(0)
        with tarfile.open(fileobj=file, mode=mode) as archive:
            yield archive


def safe_extractall(tar: tarfile.TarFile, dest: Path) -> None:
    """Prevalidate and extract a bounded regular-file/directory archive."""

    require_posix_dataset_filesystem()
    root = dest.resolve()
    members: list[tarfile.TarInfo] = []
    total_size = 0
    for member in tar:
        members.append(member)
        if len(members) > MAX_ARCHIVE_MEMBERS:
            raise ValueError(
                f"archive member count exceeds limit {MAX_ARCHIVE_MEMBERS}"
            )
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"archive contains a special member: {member.name}")
        if member.isfile():
            if member.size < 0:
                raise ValueError(f"archive member has a negative size: {member.name}")
            if member.size > MAX_ARCHIVE_FILE_BYTES:
                raise ValueError(
                    "archive member exceeds single-file size limit "
                    f"{MAX_ARCHIVE_FILE_BYTES}: {member.name}"
                )
            total_size += member.size
            if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError(
                    "archive total extracted size exceeds limit "
                    f"{MAX_ARCHIVE_TOTAL_BYTES}"
                )

        target = (root / member.name).resolve()
        if os.path.commonpath([str(root), str(target)]) != str(root):
            raise ValueError(f"unsafe archive member path: {member.name}")

    tar.extractall(root, members=members, filter="data")


def extract_or_download_archive(
    archive_path: Path,
    *,
    root: Path,
    download: Callable[[], None],
    extract: Callable[[], None],
    operation_logger: logging.Logger,
) -> None:
    """Extract one archive, repairing only a pinned-digest mismatch."""

    require_posix_dataset_filesystem()
    del root
    if archive_path.is_symlink():
        operation_logger.warning("Replacing unsafe archive symlink: %s", archive_path)
        download()
    elif not _path_exists(archive_path):
        download()
    try:
        extract()
    except ArchiveDigestMismatch:
        operation_logger.warning(
            "Replacing archive with an invalid digest: %s", archive_path
        )
        download()
        extract()


@contextmanager
def managed_temporary_directory(
    *,
    parent: Path,
    prefix: str,
    marker_content: str = _TEMPORARY_MARKER,
    preserve_on_exit: Callable[[], bool] | None = None,
) -> Iterator[Path]:
    """Create an owned temporary directory and warn if cleanup fails."""

    require_posix_dataset_filesystem()
    directory = _create_temporary_directory(
        parent=parent,
        prefix=prefix,
        marker_content=marker_content,
    )
    try:
        yield directory
    finally:
        if preserve_on_exit is not None and preserve_on_exit():
            logger.warning("Retaining torchrir temporary directory: %s", directory)
        else:
            _cleanup_directory(directory)


def extraction_temporary_prefix(target: Path) -> str:
    """Return the reserved extraction-workspace prefix for one dataset target."""

    return f".{target.name}.torchrir-extract."


def cleanup_stale_extraction_directories(*, parent: Path, target: Path) -> None:
    """Remove owned extraction workspaces under a dataset writer lock."""

    cleanup_stale_temporary_directories(
        parent=parent,
        prefix=extraction_temporary_prefix(target),
    )


def cleanup_stale_temporary_directories(
    *,
    parent: Path,
    prefix: str,
    marker_content: str = _TEMPORARY_MARKER,
) -> None:
    """Remove only owned or empty stale workspaces matching ``prefix``."""

    require_posix_dataset_filesystem()
    if not parent.is_dir():
        return
    try:
        candidates = [path for path in parent.iterdir() if path.name.startswith(prefix)]
    except OSError as exc:
        logger.warning(
            "Could not inspect stale torchrir workspaces in %s: %s", parent, exc
        )
        return
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_dir():
            logger.warning(
                "Leaving unsafe stale torchrir workspace untouched: %s", candidate
            )
            continue
        marker = candidate / _OWNER_FILE_NAME
        if not _path_exists(marker):
            try:
                if next(candidate.iterdir(), None) is None:
                    candidate.rmdir()
                else:
                    logger.warning(
                        "Leaving unowned stale torchrir workspace untouched: %s",
                        candidate,
                    )
            except OSError as exc:
                logger.warning(
                    "Could not inspect stale torchrir workspace %s: %s", candidate, exc
                )
            continue
        try:
            owned = (
                not marker.is_symlink()
                and marker.is_file()
                and marker.read_text(encoding="utf-8") == marker_content
            )
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "Could not verify stale torchrir workspace %s: %s", candidate, exc
            )
            continue
        if not owned:
            logger.warning(
                "Leaving unowned stale torchrir workspace untouched: %s", candidate
            )
            continue
        _cleanup_directory(candidate)


def publish_staged_directory(
    staged: Path,
    target: Path,
    *,
    replace_existing: bool = True,
) -> None:
    """Publish a validated directory with writer locking and crash recovery."""

    require_posix_dataset_filesystem()
    if not isinstance(replace_existing, bool):
        raise TypeError("replace_existing must be a bool")
    staged_identity = _directory_identity(
        staged,
        error_message="staged dataset must be a non-symlink directory",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.parent / f".{target.name}.torchrir-publish.lock"
    with dataset_write_lock(lock_path):
        _require_directory_identity(
            staged,
            expected=staged_identity,
            error_message="staged dataset changed while waiting for publication lock",
        )
        _publish_staged_directory_locked(
            staged,
            target,
            replace_existing=replace_existing,
            staged_identity=staged_identity,
        )


def recover_staged_publication(target: Path) -> None:
    """Recover or clean an interrupted publication for ``target`` if present."""

    require_posix_dataset_filesystem()
    if not target.parent.is_dir():
        return
    transaction = target.parent / f".{target.name}.torchrir-transaction"
    initialization_prefixes = _transaction_initialization_prefixes(transaction)
    try:
        entries = tuple(target.parent.iterdir())
    except OSError as exc:
        raise RuntimeError(
            f"could not inspect dataset publication state: {target.parent}"
        ) from exc
    has_initialization = any(
        entry.name.startswith(prefix)
        for entry in entries
        for prefix in initialization_prefixes
    )
    if not _path_exists(transaction) and not has_initialization:
        return
    lock_path = target.parent / f".{target.name}.torchrir-publish.lock"
    with dataset_write_lock(lock_path):
        _cleanup_transaction_initializations(transaction)
        _recover_publication(target, transaction)


def _publish_staged_directory_locked(
    staged: Path,
    target: Path,
    *,
    replace_existing: bool,
    staged_identity: FileIdentity,
) -> None:
    transaction = target.parent / f".{target.name}.torchrir-transaction"
    _cleanup_transaction_initializations(transaction)
    _recover_publication(target, transaction)
    _require_directory_identity(
        staged,
        expected=staged_identity,
        error_message="staged dataset changed before publication",
    )
    target_identity = _optional_directory_identity(
        target,
        error_message="existing dataset target must be a non-symlink directory",
    )
    if target_identity is not None and not replace_existing:
        raise FileExistsError(f"dataset target already exists: {target}")
    if target_identity is None:
        try:
            rename_entry_noreplace(staged, target)
        except BaseException as publish_error:
            published_identity = _optional_directory_identity(
                target,
                error_message="unexpected object appeared at dataset target",
                error_type=RuntimeError,
            )
            if published_identity == staged_identity:
                return
            if published_identity is not None or _path_exists(target):
                raise RuntimeError(
                    "dataset publication failed with a third-party target present"
                ) from publish_error
            raise
        _require_directory_identity(
            target,
            expected=staged_identity,
            error_message="published dataset target has an unexpected identity",
            error_type=RuntimeError,
        )
        return

    _initialize_transaction(
        transaction,
        staged_identity=staged_identity,
        original_identity=target_identity,
    )
    backup = transaction / "previous"
    _require_directory_identity(
        target,
        expected=target_identity,
        error_message="dataset target changed before backup",
        error_type=RuntimeError,
    )
    try:
        rename_entry_noreplace(target, backup)
    except BaseException as backup_error:
        try:
            current_target = _optional_directory_identity(
                target,
                error_message="dataset target became unsafe during backup",
                error_type=RuntimeError,
            )
            current_backup = _optional_directory_identity(
                backup,
                error_message="dataset backup became unsafe during rename",
                error_type=RuntimeError,
            )
        except RuntimeError as identity_error:
            raise RuntimeError(
                "dataset backup outcome is uncertain; transaction retained"
            ) from identity_error
        if current_target == target_identity and current_backup is None:
            _cleanup_transaction(transaction)
            raise backup_error
        if current_backup == target_identity and current_target is None:
            try:
                rename_entry_noreplace(backup, target)
                _require_directory_identity(
                    target,
                    expected=target_identity,
                    error_message="dataset backup rollback identity mismatch",
                    error_type=RuntimeError,
                )
            except BaseException as rollback_error:
                raise RuntimeError(
                    "dataset backup completed but rollback failed; transaction "
                    f"retained at {transaction}"
                ) from rollback_error
            _cleanup_transaction(transaction)
            raise backup_error
        raise RuntimeError(
            "dataset backup outcome is uncertain; transaction retained at "
            f"{transaction}"
        ) from backup_error
    _require_directory_identity(
        backup,
        expected=target_identity,
        error_message="dataset backup has an unexpected identity",
        error_type=RuntimeError,
    )
    _write_transaction_manifest(
        transaction,
        TransactionManifest(
            staged=staged_identity,
            original=target_identity,
            phase="backed_up",
        ),
    )
    _require_directory_identity(
        staged,
        expected=staged_identity,
        error_message="staged dataset changed before final publication",
        error_type=RuntimeError,
    )
    try:
        rename_entry_noreplace(staged, target)
    except BaseException as publish_error:
        try:
            published_identity = _optional_directory_identity(
                target,
                error_message="unexpected object appeared at dataset target",
                error_type=RuntimeError,
            )
        except RuntimeError as identity_error:
            raise RuntimeError(
                "dataset publication failed with an unsafe third-party target; "
                f"the previous tree is retained at {backup}"
            ) from identity_error
        if published_identity == staged_identity:
            _cleanup_transaction(transaction)
            return
        if published_identity is not None or _path_exists(target):
            raise RuntimeError(
                "dataset publication failed with a third-party target; the "
                f"previous tree is retained at {backup}"
            ) from publish_error
        try:
            _require_directory_identity(
                backup,
                expected=target_identity,
                error_message="dataset backup changed before rollback",
                error_type=RuntimeError,
            )
            rename_entry_noreplace(backup, target)
            _require_directory_identity(
                target,
                expected=target_identity,
                error_message="rolled-back dataset has an unexpected identity",
                error_type=RuntimeError,
            )
        except BaseException as rollback_error:
            raise RuntimeError(
                "dataset publication and rollback failed; the previous tree "
                f"is retained at {backup}"
            ) from rollback_error
        _cleanup_transaction(transaction)
        raise publish_error
    _require_directory_identity(
        target,
        expected=staged_identity,
        error_message=(
            "published dataset target has an unexpected identity; the previous "
            f"tree is retained at {backup}"
        ),
        error_type=RuntimeError,
    )
    _write_transaction_manifest(
        transaction,
        TransactionManifest(
            staged=staged_identity,
            original=target_identity,
            phase="published",
        ),
    )
    _cleanup_transaction(transaction)


def _initialize_transaction(
    transaction: Path,
    *,
    staged_identity: FileIdentity,
    original_identity: FileIdentity,
) -> None:
    initialization = _create_temporary_directory(
        parent=transaction.parent,
        prefix=f"{transaction.name}.init.",
        marker_content=_TRANSACTION_MARKER,
    )
    try:
        _write_transaction_manifest(
            initialization,
            TransactionManifest(
                staged=staged_identity,
                original=original_identity,
                phase="initialized",
            ),
        )
        rename_entry_noreplace(initialization, transaction)
        _fsync_directory(transaction.parent)
    except BaseException:
        _cleanup_directory(initialization)
        raise


def _recover_publication(target: Path, transaction: Path) -> None:
    if not _path_exists(transaction):
        return
    if transaction.is_symlink() or not transaction.is_dir():
        raise RuntimeError(f"unsafe dataset transaction state: {transaction}")
    if _path_exists(target) and (target.is_symlink() or not target.is_dir()):
        raise RuntimeError(
            "unsafe dataset publication target while recoverable state exists: "
            f"{target}"
        )
    marker = transaction / _OWNER_FILE_NAME
    if not _path_exists(marker):
        try:
            is_empty = next(transaction.iterdir(), None) is None
        except OSError as exc:
            raise RuntimeError(
                f"unreadable dataset transaction state: {transaction}"
            ) from exc
        if _path_exists(target) and is_empty:
            transaction.rmdir()
            return
        raise RuntimeError(f"unrecognized dataset transaction state: {transaction}")
    try:
        recognized = (
            not marker.is_symlink()
            and marker.is_file()
            and marker.read_text(encoding="utf-8") == _TRANSACTION_MARKER
        )
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            f"unreadable dataset transaction state: {transaction}"
        ) from exc
    if not recognized:
        raise RuntimeError(f"unrecognized dataset transaction state: {transaction}")
    _prepare_transaction_directory(transaction)
    manifest = _read_transaction_manifest(transaction)
    backup = transaction / "previous"
    backup_identity: FileIdentity | None = None
    if _path_exists(backup):
        backup_identity = _optional_directory_identity(
            backup,
            error_message=f"unsafe dataset transaction backup: {backup}",
            error_type=RuntimeError,
        )
    target_identity = _optional_directory_identity(
        target,
        error_message=(
            "unsafe dataset publication target while recoverable state exists: "
            f"{target}"
        ),
        error_type=RuntimeError,
    )
    if backup_identity is None:
        if target_identity in (manifest.original, manifest.staged):
            _cleanup_transaction(transaction)
            if _path_exists(transaction):
                raise RuntimeError(
                    f"could not clean dataset transaction: {transaction}"
                )
            return
        raise RuntimeError(
            f"dataset transaction has no recoverable tree: {transaction}"
        )
    if backup_identity != manifest.original:
        raise RuntimeError(f"dataset transaction backup identity changed: {backup}")
    if target_identity == manifest.staged:
        _cleanup_transaction(transaction)
        if _path_exists(transaction):
            raise RuntimeError(f"could not clean dataset transaction: {transaction}")
        return
    if target_identity is not None:
        raise RuntimeError(
            "third-party dataset target found while recoverable backup is retained: "
            f"{target}"
        )
    rename_entry_noreplace(backup, target)
    _require_directory_identity(
        target,
        expected=backup_identity,
        error_message="recovered dataset target has an unexpected identity",
        error_type=RuntimeError,
    )
    _cleanup_transaction(transaction)
    if _path_exists(transaction):
        raise RuntimeError(f"could not clean dataset transaction: {transaction}")


def _cleanup_transaction(transaction: Path) -> None:
    """Remove recoverable state while keeping its marker until backup removal."""

    try:
        _prepare_transaction_directory(transaction)
        backup = transaction / "previous"
        if _path_exists(backup):
            if not backup.is_symlink() and backup.is_dir():
                shutil.rmtree(backup)
            else:
                backup.unlink()
        marker = transaction / _OWNER_FILE_NAME
        (transaction / _TRANSACTION_MANIFEST).unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        transaction.rmdir()
    except (OSError, RuntimeError) as exc:
        logger.warning("Could not clean dataset transaction %s: %s", transaction, exc)


def _prepare_transaction_directory(transaction: Path) -> None:
    """Remove owned manifest temporaries after rejecting unknown entries."""

    try:
        entries = tuple(transaction.iterdir())
    except OSError as exc:
        raise RuntimeError(
            f"could not inspect dataset transaction: {transaction}"
        ) from exc
    allowed_names = {_OWNER_FILE_NAME, _TRANSACTION_MANIFEST, "previous"}
    temporary_entries: list[tuple[Path, FileIdentity]] = []
    unknown_entries: list[Path] = []
    for entry in entries:
        if entry.name in allowed_names:
            continue
        if not _is_transaction_manifest_temporary_name(entry.name):
            unknown_entries.append(entry)
            continue
        try:
            status = entry.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"could not inspect dataset transaction entry: {entry}"
            ) from exc
        if not stat.S_ISREG(status.st_mode):
            unknown_entries.append(entry)
            continue
        temporary_entries.append((entry, _identity_from_status(status)))
    if unknown_entries:
        names = ", ".join(sorted(entry.name for entry in unknown_entries))
        raise RuntimeError(
            f"unrecognized dataset transaction entries in {transaction}: {names}"
        )
    for entry, expected_identity in temporary_entries:
        try:
            current_status = entry.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"dataset transaction entry changed before cleanup: {entry}"
            ) from exc
        if (
            not stat.S_ISREG(current_status.st_mode)
            or _identity_from_status(current_status) != expected_identity
        ):
            raise RuntimeError(
                f"dataset transaction entry changed before cleanup: {entry}"
            )
        entry.unlink()
    if temporary_entries:
        _fsync_directory(transaction)


def _is_transaction_manifest_temporary_name(name: str) -> bool:
    if not (
        name.startswith(_TRANSACTION_MANIFEST_TEMP_PREFIX)
        and name.endswith(_TRANSACTION_MANIFEST_TEMP_SUFFIX)
    ):
        return False
    token = name[
        len(_TRANSACTION_MANIFEST_TEMP_PREFIX) : -len(_TRANSACTION_MANIFEST_TEMP_SUFFIX)
    ]
    return len(token) == 32 and all(
        character in "0123456789abcdef" for character in token
    )


def _cleanup_transaction_initializations(transaction: Path) -> None:
    for prefix in _transaction_initialization_prefixes(transaction):
        cleanup_stale_temporary_directories(
            parent=transaction.parent,
            prefix=prefix,
            marker_content=_TRANSACTION_MARKER,
        )


def _transaction_initialization_prefixes(transaction: Path) -> tuple[str, ...]:
    # The second spelling cleans workspaces left by the pre-hardening code.
    return (f"{transaction.name}.init.", f".{transaction.name}.init.")


def _write_transaction_manifest(
    transaction: Path,
    manifest: TransactionManifest,
) -> None:
    if manifest.phase not in _TRANSACTION_PHASES:
        raise ValueError(f"invalid dataset transaction phase: {manifest.phase}")
    payload = {
        "schema": 1,
        "phase": manifest.phase,
        "staged": {
            "device": manifest.staged.device,
            "inode": manifest.staged.inode,
        },
        "original": {
            "device": manifest.original.device,
            "inode": manifest.original.inode,
        },
    }
    temporary = transaction / f".{_TRANSACTION_MANIFEST}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as file:
            json.dump(payload, file, sort_keys=True, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(transaction / _TRANSACTION_MANIFEST)
        _fsync_directory(transaction)
    finally:
        temporary.unlink(missing_ok=True)


def _read_transaction_manifest(transaction: Path) -> TransactionManifest:
    path = transaction / _TRANSACTION_MANIFEST
    try:
        with open_regular_file_below(path, root=transaction) as (file, _):
            payload = json.loads(file.read().decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != 1:
            raise ValueError
        phase = payload["phase"]
        staged = payload["staged"]
        original = payload["original"]
        if phase not in _TRANSACTION_PHASES:
            raise ValueError
        if not isinstance(staged, dict) or not isinstance(original, dict):
            raise ValueError
        return TransactionManifest(
            staged=FileIdentity(
                device=int(staged["device"]),
                inode=int(staged["inode"]),
            ),
            original=FileIdentity(
                device=int(original["device"]),
                inode=int(original["inode"]),
            ),
            phase=phase,
        )
    except (KeyError, OSError, UnicodeError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"unrecognized dataset transaction manifest: {path}"
        ) from exc


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, _directory_open_flags())
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_temporary_directory(
    *,
    parent: Path,
    prefix: str,
    marker_content: str,
) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(dir=parent, prefix=prefix))
    marker = directory / _OWNER_FILE_NAME
    try:
        with marker.open("x", encoding="utf-8") as file:
            file.write(marker_content)
            file.flush()
            os.fsync(file.fileno())
    except BaseException:
        _cleanup_directory(directory)
        raise
    return directory


def _cleanup_directory(path: Path) -> None:
    """Remove temporary state without changing the primary operation result."""

    try:
        shutil.rmtree(path)
    except OSError as exc:
        logger.warning("Could not clean torchrir temporary directory %s: %s", path, exc)


def _directory_identity(
    path: Path,
    *,
    error_message: str,
    error_type: type[Exception] = ValueError,
) -> FileIdentity:
    """Open and identify a directory while rejecting final-component swaps."""

    try:
        with _open_parent_directory_below(path, root=path.parent) as (
            parent_descriptor,
            name,
        ):
            descriptor = os.open(
                name,
                _directory_open_flags(),
                dir_fd=parent_descriptor,
            )
            try:
                opened_status = os.fstat(descriptor)
                current_status = os.stat(
                    name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISDIR(opened_status.st_mode)
                    or not stat.S_ISDIR(current_status.st_mode)
                    or not _same_status_identity(opened_status, current_status)
                ):
                    raise error_type(error_message)
                return _identity_from_status(opened_status)
            finally:
                os.close(descriptor)
    except error_type:
        raise
    except (OSError, ValueError) as exc:
        raise error_type(error_message) from exc


def _optional_directory_identity(
    path: Path,
    *,
    error_message: str,
    error_type: type[Exception] = ValueError,
) -> FileIdentity | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise error_type(error_message) from exc
    return _directory_identity(
        path,
        error_message=error_message,
        error_type=error_type,
    )


def _require_directory_identity(
    path: Path,
    *,
    expected: FileIdentity,
    error_message: str,
    error_type: type[Exception] = ValueError,
) -> None:
    current = _directory_identity(
        path,
        error_message=error_message,
        error_type=error_type,
    )
    if current != expected:
        raise error_type(error_message)


@contextmanager
def _open_parent_directory_below(
    path: Path,
    *,
    root: Path,
) -> Iterator[tuple[int, str]]:
    """Walk from an opened root without following directory symlinks."""

    root_path = Path(os.path.abspath(os.fspath(root)))
    candidate = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError(f"file must remain below archive root: {path}") from exc
    parts = relative.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"file must remain below archive root: {path}")

    descriptors: list[int] = []
    try:
        root_descriptor = os.open(root_path, _directory_open_flags())
        descriptors.append(root_descriptor)
        root_status = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_status.st_mode):
            raise ValueError(f"archive root must be a non-symlink directory: {root}")
        current = root_descriptor
        for component in parts[:-1]:
            current = os.open(
                component,
                _directory_open_flags(),
                dir_fd=current,
            )
            descriptors.append(current)
            if not stat.S_ISDIR(os.fstat(current).st_mode):
                raise ValueError(f"file parent must be a directory: {path}")
        yield current, parts[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _regular_file_open_flags() -> int:
    flags = os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK", "O_BINARY"):
        flags |= int(getattr(os, flag_name, 0))
    return flags


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_DIRECTORY", "O_BINARY"):
        flags |= int(getattr(os, flag_name, 0))
    return flags


def _identity_from_status(status: os.stat_result) -> FileIdentity:
    return FileIdentity(device=int(status.st_dev), inode=int(status.st_ino))


def _same_status_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


__all__: list[str] = []
