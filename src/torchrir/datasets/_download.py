"""Verified, concurrency-safe streaming downloads for dataset archives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import http.client
import logging
from pathlib import Path
import stat
import time
import urllib.error
import urllib.request

from ._archive import (
    cleanup_stale_temporary_directories,
    managed_temporary_directory,
    open_regular_file_below,
)
from ._filesystem import (
    exchange_entries,
    rename_entry_noreplace,
    require_posix_dataset_filesystem,
)
from ..util._scalars import normalize_finite_real, normalize_integer


DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS = 6 * 60 * 60.0
MAX_DOWNLOAD_BYTES = 64 * 1024**3


class DownloadIntegrityError(OSError):
    """A completed transfer failed its size or digest contract."""


class DownloadTransportError(OSError):
    """A connection or response body failed before verification."""


@dataclass(frozen=True, slots=True)
class _EntrySnapshot:
    device: int
    inode: int
    file_type: int


def retry_download(
    operation: Callable[[], None],
    *,
    retries: int,
    logger: logging.Logger,
) -> None:
    """Retry only transient transport/integrity failures."""

    retries = normalize_integer(retries, name="retries", minimum=0)
    for attempt in range(retries + 1):
        try:
            operation()
            return
        except Exception as exc:
            if attempt >= retries or not _is_retryable_download_error(exc):
                raise
            logger.warning("Download failed (%s); retrying...", exc)


def stream_verified_download(
    url: str,
    destination: Path,
    *,
    algorithm: str,
    expected_digest: str,
    timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    total_timeout: float = DEFAULT_DOWNLOAD_TOTAL_TIMEOUT_SECONDS,
) -> None:
    """Download within I/O and total deadlines, then atomically publish."""

    require_posix_dataset_filesystem()
    timeout = normalize_finite_real(timeout, name="timeout", positive=True)
    total_timeout = normalize_finite_real(
        total_timeout,
        name="total_timeout",
        positive=True,
    )
    started_at = time.monotonic()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_snapshot = _entry_snapshot(destination)
    if destination_snapshot is not None and destination_snapshot.file_type not in (
        stat.S_IFREG,
        stat.S_IFLNK,
    ):
        raise ValueError(
            "download destination must be absent, a regular file, or a symlink"
        )
    preserve_workspace = False
    with managed_temporary_directory(
        parent=destination.parent,
        prefix=download_temporary_prefix(destination),
        preserve_on_exit=lambda: preserve_workspace,
    ) as workspace:
        temporary_path = workspace / "archive.part"
        response = _open_url(
            url,
            timeout=_remaining_io_timeout(
                started_at,
                total_timeout,
                timeout,
            ),
        )
        with response:
            _remaining_io_timeout(started_at, total_timeout, timeout)
            expected_size = _response_length(response)
            if expected_size is not None and expected_size > MAX_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Content-Length exceeds download byte limit {MAX_DOWNLOAD_BYTES}"
                )
            with temporary_path.open("xb") as temporary_file:
                digest = hashlib.new(algorithm, usedforsecurity=False)
                downloaded = 0
                while True:
                    read_timeout = _remaining_io_timeout(
                        started_at,
                        total_timeout,
                        timeout,
                    )
                    _set_response_read_timeout(response, read_timeout)
                    try:
                        chunk = response.read(1024 * 1024)
                    except (OSError, http.client.HTTPException) as exc:
                        raise DownloadTransportError(
                            f"response body failed for {url}"
                        ) from exc
                    _remaining_io_timeout(started_at, total_timeout, timeout)
                    if not chunk:
                        break
                    next_downloaded = downloaded + len(chunk)
                    if next_downloaded > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"download exceeds byte limit {MAX_DOWNLOAD_BYTES}"
                        )
                    if expected_size is not None and next_downloaded > expected_size:
                        raise DownloadIntegrityError(
                            "download exceeded declared Content-Length: "
                            f"{next_downloaded} > {expected_size} bytes"
                        )
                    temporary_file.write(chunk)
                    digest.update(chunk)
                    downloaded = next_downloaded
        if expected_size is not None and downloaded != expected_size:
            raise DownloadIntegrityError(
                f"incomplete download: {downloaded} of {expected_size} bytes"
            )
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            raise DownloadIntegrityError(
                f"{algorithm.upper()} mismatch for {url}: expected "
                f"{expected_digest}, got {actual_digest}"
            )
        if destination_snapshot is None:
            try:
                rename_entry_noreplace(temporary_path, destination)
            except FileExistsError as exc:
                raise RuntimeError(
                    "download destination appeared during verified publication"
                ) from exc
        else:
            if _entry_snapshot(destination) != destination_snapshot:
                raise RuntimeError(
                    "download destination changed during verified transfer"
                )
            exchange_entries(temporary_path, destination)
            displaced_snapshot = _entry_snapshot(temporary_path)
            if displaced_snapshot != destination_snapshot:
                try:
                    exchange_entries(temporary_path, destination)
                except BaseException as rollback_error:
                    preserve_workspace = True
                    raise RuntimeError(
                        "download destination changed during atomic publication; "
                        f"conflicting entry retained in {workspace}"
                    ) from rollback_error
                if _entry_snapshot(destination) != displaced_snapshot:
                    preserve_workspace = True
                    raise RuntimeError(
                        "download destination rollback identity mismatch; "
                        f"conflicting entry retained in {workspace}"
                    )
                raise RuntimeError(
                    "download destination changed during atomic publication"
                )


def _remaining_io_timeout(
    started_at: float,
    total_timeout: float,
    io_timeout: float,
) -> float:
    remaining = total_timeout - (time.monotonic() - started_at)
    if remaining <= 0:
        raise DownloadTransportError("download exceeded total_timeout")
    return min(io_timeout, remaining)


def _set_response_read_timeout(response: object, timeout: float) -> None:
    """Apply a per-read timeout to urllib's connected socket."""

    test_hook = getattr(response, "set_read_timeout", None)
    if callable(test_hook):
        test_hook(timeout)
        return
    socket_object: object | None = None
    for attributes in (
        ("fp", "raw", "_sock"),
        ("fp", "_sock"),
        ("raw", "_sock"),
        ("_sock",),
    ):
        candidate: object = response
        for attribute in attributes:
            candidate = getattr(candidate, attribute, None)
            if candidate is None:
                break
        else:
            socket_object = candidate
            break
    setter = getattr(socket_object, "settimeout", None)
    if callable(setter):
        try:
            setter(timeout)
        except OSError as exc:
            raise DownloadTransportError(
                "could not apply response body timeout"
            ) from exc
        return
    if isinstance(response, http.client.HTTPResponse) and response.isclosed():
        # Once http.client has consumed the declared body, a final read returns
        # EOF without touching the socket.
        return
    if isinstance(response, http.client.HTTPResponse):
        raise DownloadTransportError(
            "could not access the HTTP response socket for deadline enforcement"
        )


def cleanup_stale_download_directories(destination: Path) -> None:
    """Remove owned download workspaces while the caller holds its writer lock."""

    cleanup_stale_temporary_directories(
        parent=destination.parent,
        prefix=download_temporary_prefix(destination),
    )


def download_temporary_prefix(destination: Path) -> str:
    """Return the reserved workspace prefix for one archive destination."""

    return f".{destination.name}.torchrir-download."


def file_matches_digest(path: Path, *, algorithm: str, expected_digest: str) -> bool:
    """Return whether a local file has the expected digest."""

    digest = hashlib.new(algorithm, usedforsecurity=False)
    with open_regular_file_below(path, root=path.parent) as (file, _):
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_digest


def _open_url(url: str, *, timeout: float):
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except urllib.error.HTTPError:
        raise
    except (OSError, http.client.HTTPException) as exc:
        raise DownloadTransportError(f"connection failed for {url}") from exc


def _response_length(response: object) -> int | None:
    length = getattr(response, "length", None)
    if length is None:
        headers = getattr(response, "headers", None)
        value = headers.get("Content-Length") if headers is not None else None
        length = value if value else None
    if length is None:
        return None
    try:
        normalized = int(length)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DownloadTransportError("invalid Content-Length response header") from exc
    if normalized < 0:
        raise DownloadTransportError("invalid Content-Length response header")
    return normalized


def _entry_snapshot(path: Path) -> _EntrySnapshot | None:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"could not inspect download destination: {path}") from exc
    return _EntrySnapshot(
        device=int(status.st_dev),
        inode=int(status.st_ino),
        file_type=stat.S_IFMT(status.st_mode),
    )


def _is_retryable_download_error(error: Exception) -> bool:
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code < 600
    return isinstance(error, (DownloadIntegrityError, DownloadTransportError))


__all__: list[str] = []
