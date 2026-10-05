import hashlib
import io
import errno
import multiprocessing
import os
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

import torchrir.datasets._archive as archive_utils
import torchrir.datasets.cmu_arctic as cmu_arctic
import torchrir.datasets._download as download_utils
import torchrir.datasets._filesystem as filesystem_utils
import torchrir.datasets._locking as locking_utils
import torchrir.datasets.librispeech as librispeech
from torchrir.datasets._archive import (
    _TRANSACTION_MARKER,
    publish_staged_directory,
    safe_extractall,
)
from torchrir.datasets._download import DownloadIntegrityError
from torchrir.datasets._locking import dataset_write_lock
from torchrir.datasets.cmu_arctic import (
    ARCHIVE_SHA256,
    VALID_SPEAKERS,
    _has_sha256,
)
from torchrir.datasets.librispeech import ARCHIVE_MD5, VALID_SUBSETS, _has_md5


EXPECTED_CMU_SHA256 = {
    "aew": "645cb33c0f0b2ce41384fdd8d3db2c3f5fc15c1e688baeb74d2e08cab18ab406",
    "ahw": "024664adeb892809d646a3efd043625b46b5bfa3e6189b3500b2d0d59dfab06c",
    "aup": "2c55bc3050caa996758869126ad10cf42e1441212111db034b3a45189c18b6fc",
    "awb": "d74a950c9739a65f7bfc4dfa6187f2730fa03de5b8eb3f2da97a51b74df64d3c",
    "axb": "dd65c3d2907d1ee52f86e44f578319159e60f4bf722a9142be01161d84e330ff",
    "bdl": "26b91aaf48b2799b2956792b4632c2f926cd0542f402b5452d5adecb60942904",
    "clb": "3f16dc3f3b97955ea22623efb33b444341013fc660677b2e170efdcc959fa7c6",
    "eey": "8a0ee4e5acbd4b2f61a4fb947c1730ab3adcc9dc50b195981d99391d29928e8a",
    "fem": "3fcff629412b57233589cdb058f730594a62c4f3a75c20de14afe06621ef45e2",
    "gka": "dc82e7967cbd5eddbed33074b0699128dbd4482b41711916d58103707e38c67f",
    "jmk": "3a37c0e1dfc91e734fdbc88b562d9e2ebca621772402cdc693bbc9b09b211d73",
    "ksp": "8029cafce8296f9bed3022c44ef1e7953332b6bf6943c14b929f468122532717",
    "ljm": "b23993765cbf2b9e7bbc3c85b6c56eaf292ac81ee4bb887b638a24d104f921a0",
    "lnh": "4faf34d71aa7112813252fb20c5433e2fdd9a9de55a00701ffcbf05f24a5991a",
    "rms": "c6dc11235629c58441c071a7ba8a2d067903dfefbaabc4056d87da35b72ecda4",
    "rxr": "1fa4271c393e5998d200e56c102ff46fcfea169aaa2148ad9e9469616fbfdd9b",
    "slp": "54345ed55e45c23d419e9a823eef427f1cc93c83a710735ec667d068c916abf1",
    "slt": "7c173297916acf3cc7fcab2713be4c60b27312316765a90934651d367226b4ea",
}

EXPECTED_LIBRISPEECH_MD5 = {
    "dev-clean": "42e2234ba48799c1f50f24a7926300a1",
    "dev-other": "c8d0bcc9cca99d4f8b62fcc847357931",
    "test-clean": "32fa31d27d2e1cad72775fee3f4849a9",
    "test-other": "fb5a50374b501bb3bac4815ee91d3135",
    "train-clean-100": "2a93770f6d5c6c964bc36631d331a522",
    "train-clean-360": "c0e676e450a7ff2f54aeade5171606fa",
    "train-other-500": "d1a0fd59409feb2c614ce4d30c387708",
}


def _hold_dataset_lock_in_child(
    lock_path: str,
    acquired: Any,
    release: Any,
) -> None:
    """Hold a dataset lock until the parent process asks for its release."""

    with dataset_write_lock(Path(lock_path), timeout=5.0):
        acquired.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("parent did not release the child lock")


def _write_tar_with_member(path: Path, member_name: str, data: bytes = b"x") -> None:
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))


def _add_tar_file(tar: tarfile.TarFile, name: str, data: bytes = b"x") -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _write_test_transaction_manifest(
    transaction: Path,
    *,
    staged: Path,
    original: Path,
    phase: str,
) -> None:
    staged_status = staged.stat()
    original_status = original.stat()
    archive_utils._write_transaction_manifest(
        transaction,
        archive_utils.TransactionManifest(
            staged=archive_utils.FileIdentity(
                int(staged_status.st_dev),
                int(staged_status.st_ino),
            ),
            original=archive_utils.FileIdentity(
                int(original_status.st_dev),
                int(original_status.st_ino),
            ),
            phase=phase,
        ),
    )


def _seed_stale_temporary_workspaces(
    parent: Path,
    *,
    prefix: str,
    marker_content: str,
    label: str,
) -> dict[str, Path]:
    parent.mkdir(parents=True, exist_ok=True)
    owned = parent / f"{prefix}owned"
    owned.mkdir()
    (owned / "owner").write_text(marker_content, encoding="utf-8")
    (owned / "partial").write_bytes(b"stale")

    empty = parent / f"{prefix}empty"
    empty.mkdir()

    unowned = parent / f"{prefix}unowned"
    unowned.mkdir()
    (unowned / "user-data").write_bytes(b"preserve")

    outside = parent / f"outside-{label}"
    outside.mkdir()
    (outside / "external-data").write_bytes(b"preserve")
    symlink = parent / f"{prefix}symlink"
    symlink.symlink_to(outside, target_is_directory=True)
    return {
        "owned": owned,
        "empty": empty,
        "unowned": unowned,
        "outside": outside,
        "symlink": symlink,
    }


def _assert_only_owned_stale_workspaces_removed(paths: dict[str, Path]) -> None:
    assert not paths["owned"].exists()
    assert not paths["empty"].exists()
    assert paths["unowned"].is_dir()
    assert (paths["unowned"] / "user-data").read_bytes() == b"preserve"
    assert paths["symlink"].is_symlink()
    assert (paths["outside"] / "external-data").read_bytes() == b"preserve"


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


def test_safe_extractall_rejects_absolute_path(tmp_path: Path) -> None:
    archive = tmp_path / "absolute.tar.gz"
    out_dir = tmp_path / "out"
    _write_tar_with_member(archive, "/tmp/torchrir-absolute-escape", b"bad")
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
        with pytest.raises(ValueError, match="special member"):
            safe_extractall(tar, out_dir)


@pytest.mark.parametrize(
    "member_type",
    [tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE],
)
def test_safe_extractall_rejects_special_files(
    tmp_path: Path,
    member_type: bytes,
) -> None:
    archive = tmp_path / "special.tar.gz"
    out_dir = tmp_path / "out"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo(name="special")
        info.type = member_type
        tar.addfile(info)
    out_dir.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="special member"):
            safe_extractall(tar, out_dir)


def test_safe_extractall_validates_every_member_before_writing(tmp_path: Path) -> None:
    archive = tmp_path / "mixed.tar.gz"
    out_dir = tmp_path / "out"
    with tarfile.open(archive, "w:gz") as tar:
        safe_info = tarfile.TarInfo(name="safe.txt")
        safe_info.size = 2
        tar.addfile(safe_info, io.BytesIO(b"ok"))
        unsafe_info = tarfile.TarInfo(name="../unsafe.txt")
        unsafe_info.size = 3
        tar.addfile(unsafe_info, io.BytesIO(b"bad"))
    out_dir.mkdir()

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="unsafe archive member path"):
            safe_extractall(tar, out_dir)

    assert not (out_dir / "safe.txt").exists()


def test_safe_extractall_rejects_excessive_member_count_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "too-many-members.tar.gz"
    out_dir = tmp_path / "out"
    with tarfile.open(archive, "w:gz") as tar:
        _add_tar_file(tar, "one.txt")
        _add_tar_file(tar, "two.txt")
        _add_tar_file(tar, "three.txt")
    out_dir.mkdir()
    monkeypatch.setattr(archive_utils, "MAX_ARCHIVE_MEMBERS", 2)

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="member count"):
            safe_extractall(tar, out_dir)

    assert not list(out_dir.iterdir())


def test_safe_extractall_rejects_excessive_single_file_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "oversized-member.tar.gz"
    out_dir = tmp_path / "out"
    _write_tar_with_member(archive, "large.txt", b"abc")
    out_dir.mkdir()
    monkeypatch.setattr(archive_utils, "MAX_ARCHIVE_FILE_BYTES", 2)

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="single-file size"):
            safe_extractall(tar, out_dir)

    assert not list(out_dir.iterdir())


def test_safe_extractall_rejects_excessive_total_size_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "oversized-total.tar.gz"
    out_dir = tmp_path / "out"
    with tarfile.open(archive, "w:gz") as tar:
        _add_tar_file(tar, "one.txt", b"ab")
        _add_tar_file(tar, "two.txt", b"cd")
    out_dir.mkdir()
    monkeypatch.setattr(archive_utils, "MAX_ARCHIVE_FILE_BYTES", 4)
    monkeypatch.setattr(archive_utils, "MAX_ARCHIVE_TOTAL_BYTES", 3)

    with tarfile.open(archive, "r:gz") as tar:
        with pytest.raises(ValueError, match="total extracted size"):
            safe_extractall(tar, out_dir)

    assert not list(out_dir.iterdir())


def test_staged_directory_publication_restores_target_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def fail_staged_publication(source: Path, destination: Path) -> None:
        if source == staged:
            raise OSError("simulated publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        fail_staged_publication,
    )

    with pytest.raises(OSError, match="publication failure"):
        publish_staged_directory(staged, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()
    assert not list(tmp_path.glob(".dataset.torchrir-backup.*"))


def test_staged_directory_publication_rejects_symlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    staged = tmp_path / "staged"
    target = tmp_path / "dataset"
    external.mkdir()
    (external / "outside.txt").write_text("outside", encoding="utf-8")
    staged.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="non-symlink"):
        publish_staged_directory(staged, target)

    assert not target.exists()
    assert (external / "outside.txt").read_text(encoding="utf-8") == "outside"


def test_staged_directory_is_revalidated_after_publication_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staged = tmp_path / "staged"
    displaced = tmp_path / "displaced"
    target = tmp_path / "dataset"
    staged.mkdir()
    (staged / "safe.txt").write_text("safe", encoding="utf-8")

    @contextmanager
    def replace_while_waiting(*args: Any, **kwargs: Any) -> Iterator[None]:
        del args, kwargs
        staged.replace(displaced)
        staged.mkdir()
        (staged / "replacement.txt").write_text("replacement", encoding="utf-8")
        yield

    monkeypatch.setattr(archive_utils, "dataset_write_lock", replace_while_waiting)

    with pytest.raises(ValueError, match="changed while waiting"):
        publish_staged_directory(staged, target)

    assert not target.exists()
    assert (displaced / "safe.txt").read_text(encoding="utf-8") == "safe"


def test_staged_directory_publication_rejects_special_existing_target(
    tmp_path: Path,
) -> None:
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    staged.mkdir()
    os.mkfifo(target)

    with pytest.raises(ValueError, match="non-symlink directory"):
        publish_staged_directory(staged, target)

    assert target.exists()


def test_dataset_write_lock_rejects_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external-lock-target"
    external.write_text("preserved", encoding="utf-8")
    lock_path = tmp_path / "dataset.lock"
    lock_path.symlink_to(external)

    with pytest.raises(ValueError, match="unsafe"):
        with dataset_write_lock(lock_path):
            pytest.fail("unsafe lock must not be acquired")

    assert external.read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize(
    "timeout",
    [0.0, float("nan"), float("inf"), 10**400, True],
)
def test_dataset_write_lock_requires_positive_finite_timeout(
    tmp_path: Path,
    timeout: Any,
) -> None:
    error = TypeError if isinstance(timeout, bool) else ValueError
    with pytest.raises(error, match="timeout"):
        with dataset_write_lock(tmp_path / "dataset.lock", timeout=timeout):
            pytest.fail("invalid timeout must not acquire a lock")


def test_dataset_write_lock_times_out_and_releases_after_exception(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "dataset.lock"
    with dataset_write_lock(lock_path):
        with pytest.raises(TimeoutError, match="dataset lock"):
            with dataset_write_lock(lock_path, timeout=0.05):
                pytest.fail("second writer must not enter")

    with pytest.raises(RuntimeError, match="injected"):
        with dataset_write_lock(lock_path):
            raise RuntimeError("injected")

    with dataset_write_lock(lock_path):
        pass


def test_dataset_write_lock_serializes_independent_processes(
    tmp_path: Path,
) -> None:
    context = multiprocessing.get_context("spawn")
    lock_path = tmp_path / "dataset.lock"
    acquired = context.Event()
    release = context.Event()
    child = context.Process(
        target=_hold_dataset_lock_in_child,
        args=(str(lock_path), acquired, release),
    )
    child.start()
    try:
        assert acquired.wait(timeout=5.0), "child did not acquire the dataset lock"
        with pytest.raises(TimeoutError, match="dataset lock"):
            with dataset_write_lock(lock_path, timeout=0.1):
                pytest.fail("parent entered while the child held the lock")
    finally:
        release.set()
        child.join(timeout=5.0)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5.0)

    assert child.exitcode == 0
    with dataset_write_lock(lock_path, timeout=1.0):
        pass


def test_dataset_write_lock_rejects_path_replaced_after_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "dataset.lock"
    displaced = tmp_path / "displaced.lock"
    original_try_lock = locking_utils._try_lock

    def lock_then_replace(descriptor: int) -> None:
        original_try_lock(descriptor)
        lock_path.replace(displaced)
        lock_path.write_bytes(b"replacement")

    monkeypatch.setattr(locking_utils, "_try_lock", lock_then_replace)

    with pytest.raises(RuntimeError, match="changed during acquisition"):
        with dataset_write_lock(lock_path):
            pytest.fail("split-brain lock must not be entered")

    assert displaced.is_file()
    assert lock_path.read_bytes() == b"replacement"


def test_dataset_write_lock_sleeps_only_until_its_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    attempts = 0
    sleeps: list[float] = []

    def would_block(descriptor: int) -> None:
        nonlocal attempts
        del descriptor
        attempts += 1
        raise BlockingIOError(errno.EAGAIN, "busy")

    def advance(duration: float) -> None:
        nonlocal now
        sleeps.append(duration)
        now += duration

    monkeypatch.setattr(locking_utils, "_try_lock", would_block)
    monkeypatch.setattr(locking_utils.time, "monotonic", lambda: now)
    monkeypatch.setattr(locking_utils.time, "sleep", advance)

    with pytest.raises(TimeoutError, match="dataset lock"):
        with dataset_write_lock(tmp_path / "dataset.lock", timeout=0.005):
            pytest.fail("expired lock must not be entered")

    assert attempts == 1
    assert sleeps == pytest.approx([0.005])


def test_dataset_write_lock_releases_an_acquisition_completed_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    unlocks: list[int] = []

    def acquire_after_deadline(descriptor: int) -> None:
        nonlocal now
        del descriptor
        now = 0.006

    monkeypatch.setattr(locking_utils, "_try_lock", acquire_after_deadline)
    monkeypatch.setattr(locking_utils, "_unlock", unlocks.append)
    monkeypatch.setattr(locking_utils.time, "monotonic", lambda: now)

    with pytest.raises(TimeoutError, match="dataset lock"):
        with dataset_write_lock(tmp_path / "dataset.lock", timeout=0.005):
            pytest.fail("late acquisition must not be entered")

    assert len(unlocks) == 1


def test_regular_file_open_rejects_intermediate_directory_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    nested = root / "nested"
    displaced = root / "displaced"
    outside = tmp_path / "outside"
    nested.mkdir(parents=True)
    outside.mkdir()
    (nested / "file").write_bytes(b"safe")
    (outside / "file").write_bytes(b"outside")
    original_open = archive_utils.os.open
    swapped = False

    def swap_component_before_open(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            nested.replace(displaced)
            nested.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(archive_utils.os, "open", swap_component_before_open)

    with pytest.raises(ValueError, match="non-symlink regular file"):
        with archive_utils.open_regular_file_below(
            nested / "file",
            root=root,
        ):
            pytest.fail("swapped component must not be followed")

    assert swapped
    assert (outside / "file").read_bytes() == b"outside"


def test_open_regular_file_below_does_not_translate_consumer_oserror(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"invalid")
    mismatch = archive_utils.ArchiveDigestMismatch(
        archive,
        identity=archive_utils.FileIdentity(1, 2),
        algorithm="sha256",
        expected="0" * 64,
        actual="1" * 64,
    )

    with pytest.raises(archive_utils.ArchiveDigestMismatch) as caught:
        with archive_utils.open_regular_file_below(archive, root=tmp_path):
            raise mismatch

    assert caught.value is mismatch


def test_open_verified_tar_preserves_digest_mismatch_type(tmp_path: Path) -> None:
    archive = tmp_path / "archive.tar.gz"
    archive.write_bytes(b"invalid")

    with pytest.raises(archive_utils.ArchiveDigestMismatch):
        with archive_utils.open_verified_tar(
            archive,
            root=tmp_path,
            algorithm="sha256",
            expected_digest="0" * 64,
            mode="r:gz",
        ):
            pytest.fail("an invalid digest must fail before tar parsing")


def test_dataset_filesystem_contract_is_explicitly_posix_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_file = tmp_path / "file"
    local_file.write_bytes(b"data")
    monkeypatch.setattr(
        filesystem_utils,
        "_SECURE_POSIX_DATASET_IO_AVAILABLE",
        False,
    )

    with pytest.raises(NotImplementedError, match="POSIX"):
        with archive_utils.open_regular_file_below(local_file, root=tmp_path):
            pytest.fail("unsupported local dataset read must not start")
    with pytest.raises(NotImplementedError, match="POSIX"):
        with dataset_write_lock(tmp_path / "dataset.lock"):
            pytest.fail("unsupported dataset mutation must not start")
    with pytest.raises(NotImplementedError, match="POSIX"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            tmp_path / "archive",
            algorithm="sha256",
            expected_digest="0" * 64,
        )


def test_staged_publication_cleans_backup_shell_when_initial_rename_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    original_rename = archive_utils.rename_entry_noreplace

    def fail_initial_rename(source: Path, destination: Path) -> None:
        if source == target:
            raise OSError("simulated initial rename failure")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        fail_initial_rename,
    )

    with pytest.raises(OSError, match="initial rename failure"):
        publish_staged_directory(staged, target)

    assert target.is_dir()
    assert staged.is_dir()
    assert not list(tmp_path.glob(".dataset.torchrir-backup.*"))


def test_staged_publication_rolls_back_when_backup_rename_raises_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def rename_then_raise(source: Path, destination: Path) -> None:
        original_rename(source, destination)
        if source == target:
            raise OSError("simulated error after completed backup rename")

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        rename_then_raise,
    )

    with pytest.raises(OSError, match="completed backup rename"):
        publish_staged_directory(staged, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"
    assert not transaction.exists()


def test_staged_publication_does_not_report_cleanup_as_publish_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")

    def fail_cleanup(path: Path) -> None:
        del path
        raise OSError("cleanup failed")

    monkeypatch.setattr(
        "torchrir.datasets._archive.shutil.rmtree",
        fail_cleanup,
    )

    with caplog.at_level("WARNING", logger="torchrir.datasets._archive"):
        publish_staged_directory(staged, target)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (target / "old.txt").exists()
    assert "cleanup failed" in caplog.text
    assert str(tmp_path / ".dataset.torchrir-transaction") in caplog.text


def test_staged_publication_recovers_interrupted_previous_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    backup = transaction / "previous"
    backup.mkdir(parents=True)
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    (backup / "old.txt").write_text("old", encoding="utf-8")
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    _write_test_transaction_manifest(
        transaction,
        staged=staged,
        original=backup,
        phase="backed_up",
    )
    original_rename = archive_utils.rename_entry_noreplace

    def fail_new_publication(source: Path, destination: Path) -> None:
        if source == staged:
            raise OSError("simulated new publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        fail_new_publication,
    )

    with pytest.raises(OSError, match="new publication failure"):
        publish_staged_directory(staged, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not transaction.exists()


@pytest.mark.parametrize("target_kind", ["symlink", "fifo"])
def test_staged_publication_preserves_transaction_for_unsafe_existing_target(
    tmp_path: Path,
    target_kind: str,
) -> None:
    if target_kind == "fifo" and not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    backup = transaction / "previous"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    backup.mkdir(parents=True)
    (backup / "old.txt").write_text("old", encoding="utf-8")
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    _write_test_transaction_manifest(
        transaction,
        staged=staged,
        original=backup,
        phase="backed_up",
    )
    external = tmp_path / "external"
    if target_kind == "symlink":
        external.mkdir()
        (external / "external.txt").write_text("preserved", encoding="utf-8")
        target.symlink_to(external, target_is_directory=True)
    else:
        os.mkfifo(target)
    target_status = target.lstat()

    with pytest.raises(RuntimeError, match="unsafe dataset publication target"):
        publish_staged_directory(staged, target)

    current_status = target.lstat()
    assert (current_status.st_dev, current_status.st_ino, current_status.st_mode) == (
        target_status.st_dev,
        target_status.st_ino,
        target_status.st_mode,
    )
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert (transaction / "owner").read_text(encoding="utf-8") == (_TRANSACTION_MARKER)
    if target_kind == "symlink":
        assert target.is_symlink()
        assert (external / "external.txt").read_text(encoding="utf-8") == "preserved"


def test_staged_publication_preserves_backup_for_third_party_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def inject_target_then_fail(source: Path, destination: Path) -> None:
        if source == staged:
            target.mkdir()
            (target / "third-party.txt").write_text("third", encoding="utf-8")
            raise OSError("injected publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        inject_target_then_fail,
    )

    with pytest.raises(RuntimeError, match="third-party target"):
        publish_staged_directory(staged, target)

    assert (target / "third-party.txt").read_text(encoding="utf-8") == "third"
    assert (transaction / "previous" / "old.txt").read_text(encoding="utf-8") == ("old")
    assert (transaction / "owner").read_text(encoding="utf-8") == (_TRANSACTION_MARKER)
    with pytest.raises(RuntimeError, match="third-party dataset target"):
        archive_utils.recover_staged_publication(target)
    assert (target / "third-party.txt").read_text(encoding="utf-8") == "third"
    assert (transaction / "previous" / "old.txt").read_text(encoding="utf-8") == ("old")


def test_create_only_publication_does_not_replace_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def inject_before_rename(source: Path, destination: Path) -> None:
        if source == staged and destination == target:
            target.mkdir()
            (target / "third-party.txt").write_text("third", encoding="utf-8")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        inject_before_rename,
    )

    with pytest.raises(RuntimeError, match="third-party target"):
        publish_staged_directory(staged, target, replace_existing=False)

    assert (target / "third-party.txt").read_text(encoding="utf-8") == "third"
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"


def test_replacement_publication_does_not_replace_racing_final_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def inject_before_final_rename(source: Path, destination: Path) -> None:
        if source == staged and destination == target:
            target.mkdir()
            (target / "third-party.txt").write_text("third", encoding="utf-8")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        inject_before_final_rename,
    )

    with pytest.raises(RuntimeError, match="third-party target"):
        publish_staged_directory(staged, target)

    assert (target / "third-party.txt").read_text(encoding="utf-8") == "third"
    assert (transaction / "previous" / "old.txt").read_text(encoding="utf-8") == ("old")
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"


def test_publication_rollback_does_not_replace_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    original_rename = archive_utils.rename_entry_noreplace

    def fail_then_race_rollback(source: Path, destination: Path) -> None:
        if source == staged:
            raise OSError("simulated publication failure")
        if source == transaction / "previous" and destination == target:
            target.mkdir()
            (target / "third-party.txt").write_text("third", encoding="utf-8")
        original_rename(source, destination)

    monkeypatch.setattr(
        archive_utils,
        "rename_entry_noreplace",
        fail_then_race_rollback,
    )

    with pytest.raises(RuntimeError, match="publication and rollback failed"):
        publish_staged_directory(staged, target)

    assert (target / "third-party.txt").read_text(encoding="utf-8") == "third"
    assert (transaction / "previous" / "old.txt").read_text(encoding="utf-8") == ("old")
    assert (staged / "new.txt").read_text(encoding="utf-8") == "new"


def test_publication_recovery_rejects_symlink_backup(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    external = tmp_path / "external"
    transaction.mkdir()
    external.mkdir()
    (external / "preserved.txt").write_text("preserved", encoding="utf-8")
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    (transaction / "previous").symlink_to(external, target_is_directory=True)
    archive_utils._write_transaction_manifest(
        transaction,
        archive_utils.TransactionManifest(
            staged=archive_utils.FileIdentity(1, 1),
            original=archive_utils.FileIdentity(1, 2),
            phase="backed_up",
        ),
    )

    with pytest.raises(RuntimeError, match="unsafe dataset transaction backup"):
        archive_utils.recover_staged_publication(target)

    assert not target.exists()
    assert (external / "preserved.txt").read_text(encoding="utf-8") == "preserved"
    assert (transaction / "previous").is_symlink()


def test_publication_recovery_removes_owned_manifest_temporary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    backup = transaction / "previous"
    target.mkdir()
    backup.mkdir(parents=True)
    (target / "new.txt").write_text("new", encoding="utf-8")
    (backup / "old.txt").write_text("old", encoding="utf-8")
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    _write_test_transaction_manifest(
        transaction,
        staged=target,
        original=backup,
        phase="published",
    )
    (transaction / f".manifest.json.{'a' * 32}.tmp").write_text(
        "partial",
        encoding="utf-8",
    )

    archive_utils.recover_staged_publication(target)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not transaction.exists()


@pytest.mark.parametrize("entry_kind", ["unknown", "temporary-symlink"])
def test_publication_recovery_preserves_backup_for_unknown_transaction_entry(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    target = tmp_path / "dataset"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    backup = transaction / "previous"
    external = tmp_path / "external"
    target.mkdir()
    backup.mkdir(parents=True)
    external.write_text("external", encoding="utf-8")
    (target / "new.txt").write_text("new", encoding="utf-8")
    (backup / "old.txt").write_text("old", encoding="utf-8")
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    _write_test_transaction_manifest(
        transaction,
        staged=target,
        original=backup,
        phase="published",
    )
    if entry_kind == "unknown":
        unexpected = transaction / "intruder"
        unexpected.write_text("preserve", encoding="utf-8")
    else:
        unexpected = transaction / f".manifest.json.{'b' * 32}.tmp"
        unexpected.symlink_to(external)

    with pytest.raises(RuntimeError, match="unrecognized.*transaction entries"):
        archive_utils.recover_staged_publication(target)

    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert (transaction / "manifest.json").is_file()
    assert (transaction / "owner").is_file()
    assert unexpected.exists() or unexpected.is_symlink()
    assert external.read_text(encoding="utf-8") == "external"


def test_staged_publication_serializes_concurrent_writers(tmp_path: Path) -> None:
    target = tmp_path / "dataset"
    staged_directories = [tmp_path / "staged-a", tmp_path / "staged-b"]
    for index, staged in enumerate(staged_directories):
        staged.mkdir()
        (staged / "value.txt").write_text(str(index), encoding="utf-8")

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda staged: publish_staged_directory(staged, target),
                staged_directories,
            )
        )

    assert (target / "value.txt").read_text(encoding="utf-8") in {"0", "1"}
    assert not (tmp_path / ".dataset.torchrir-transaction").exists()


def test_staged_publication_preserves_create_only_under_concurrency(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    staged_directories = [tmp_path / "staged-a", tmp_path / "staged-b"]
    for index, staged in enumerate(staged_directories):
        staged.mkdir()
        (staged / "value.txt").write_text(str(index), encoding="utf-8")

    def publish_once(staged: Path) -> str:
        try:
            publish_staged_directory(
                staged,
                target,
                replace_existing=False,
            )
        except FileExistsError:
            return "exists"
        return "published"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish_once, staged_directories))

    assert sorted(outcomes) == ["exists", "published"]
    assert (target / "value.txt").read_text(encoding="utf-8") in {"0", "1"}


def test_staged_publication_recovers_empty_cleanup_checkpoint(
    tmp_path: Path,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    transaction = tmp_path / ".dataset.torchrir-transaction"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    staged.mkdir()
    (staged / "new.txt").write_text("new", encoding="utf-8")
    transaction.mkdir()

    publish_staged_directory(staged, target)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    assert not transaction.exists()


def test_staged_publication_recovers_only_owned_stale_initialization_workspaces(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "dataset"
    staged = tmp_path / "staged"
    target.mkdir()
    staged.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    (staged / "new.txt").write_text("new", encoding="utf-8")
    stale = _seed_stale_temporary_workspaces(
        tmp_path,
        prefix=".dataset.torchrir-transaction.init.",
        marker_content=_TRANSACTION_MARKER,
        label="transaction-init",
    )

    with caplog.at_level("WARNING", logger="torchrir.datasets._archive"):
        publish_staged_directory(staged, target)

    assert (target / "new.txt").read_text(encoding="utf-8") == "new"
    _assert_only_owned_stale_workspaces_removed(stale)
    assert str(stale["unowned"]) in caplog.text
    assert str(stale["symlink"]) in caplog.text


def test_cmu_arctic_has_pinned_sha256_for_every_speaker(tmp_path: Path) -> None:
    assert ARCHIVE_SHA256 == EXPECTED_CMU_SHA256
    assert ARCHIVE_SHA256.keys() == VALID_SPEAKERS
    assert all(
        len(value) == 64 and int(value, 16) >= 0 for value in ARCHIVE_SHA256.values()
    )

    archive = tmp_path / "archive"
    archive.write_bytes(b"verified content")
    expected = hashlib.sha256(b"verified content").hexdigest()
    assert _has_sha256(archive, expected)
    assert not _has_sha256(archive, "0" * 64)


def test_librispeech_has_pinned_md5_for_every_subset(tmp_path: Path) -> None:
    assert ARCHIVE_MD5 == EXPECTED_LIBRISPEECH_MD5
    assert ARCHIVE_MD5.keys() == VALID_SUBSETS
    assert all(
        len(value) == 32 and int(value, 16) >= 0 for value in ARCHIVE_MD5.values()
    )

    archive = tmp_path / "archive"
    archive.write_bytes(b"verified content")
    expected = hashlib.md5(b"verified content", usedforsecurity=False).hexdigest()
    assert _has_md5(archive, expected)
    assert not _has_md5(archive, "0" * 32)


@pytest.mark.parametrize(
    ("download", "checksum_keyword"),
    [
        (cmu_arctic._download, "sha256"),
        (librispeech._download, "md5"),
    ],
)
def test_dataset_download_retries_integrity_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: Callable[..., None],
    checksum_keyword: str,
) -> None:
    destination = tmp_path / "archive.tar.gz"
    attempts = 0

    def _fail_once_then_succeed(
        url: str,
        dest: Path,
        **checksum: Any,
    ) -> None:
        nonlocal attempts
        del url
        assert checksum_keyword in checksum
        attempts += 1
        if attempts == 1:
            raise DownloadIntegrityError("checksum mismatch")
        assert not dest.exists()
        dest.write_bytes(b"verified")

    module = cmu_arctic if download is cmu_arctic._download else librispeech
    monkeypatch.setattr(module, "_stream_download", _fail_once_then_succeed)
    download(
        "https://example.invalid/archive",
        destination,
        retries=1,
        **{checksum_keyword: "0"},
    )

    assert attempts == 2
    assert destination.read_bytes() == b"verified"


@pytest.mark.parametrize(
    ("download", "checksum_keyword"),
    [
        (cmu_arctic._download, "sha256"),
        (librispeech._download, "md5"),
    ],
)
def test_dataset_download_does_not_retry_local_filesystem_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    download: Callable[..., None],
    checksum_keyword: str,
) -> None:
    attempts = 0

    def fail_local_write(*args: Any, **kwargs: Any) -> None:
        nonlocal attempts
        del args, kwargs
        attempts += 1
        raise OSError("disk full")

    module = cmu_arctic if download is cmu_arctic._download else librispeech
    monkeypatch.setattr(module, "_stream_download", fail_local_write)

    with pytest.raises(OSError, match="disk full"):
        download(
            "https://example.invalid/archive",
            tmp_path / "archive.tar.gz",
            retries=1,
            **{checksum_keyword: "0"},
        )

    assert attempts == 1


@pytest.mark.parametrize(
    ("stream_download", "checksum_keyword", "algorithm"),
    [
        (cmu_arctic._stream_download, "sha256", "sha256"),
        (librispeech._stream_download, "md5", "md5"),
    ],
)
def test_stream_download_uses_unique_non_following_temp_and_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream_download: Callable[..., None],
    checksum_keyword: str,
    algorithm: str,
) -> None:
    payload = b"verified archive"
    destination = tmp_path / "archive.tar.gz"
    unowned_partial = destination.with_suffix(destination.suffix + ".part")
    outside = tmp_path / "outside"
    unowned_partial.symlink_to(outside)
    timeouts: list[float] = []

    class Response:
        length = len(payload)

        def __init__(self) -> None:
            self._read = False

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            assert size > 0
            if self._read:
                return b""
            self._read = True
            return payload

    def open_url(url: str, *, timeout: float) -> Response:
        assert url == "https://example.invalid/archive"
        timeouts.append(timeout)
        return Response()

    monkeypatch.setattr(download_utils.urllib.request, "urlopen", open_url)
    digest = hashlib.new(algorithm, payload, usedforsecurity=False).hexdigest()

    stream_download(
        "https://example.invalid/archive",
        destination,
        **{checksum_keyword: digest},
    )

    assert destination.read_bytes() == payload
    assert not destination.is_symlink()
    assert unowned_partial.is_symlink()
    assert not outside.exists()
    assert timeouts == [download_utils.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS]
    assert not list(tmp_path.glob(f".{destination.name}.*.part"))


@pytest.mark.parametrize("content_length", ["not-an-integer", "-1"])
def test_malformed_content_length_is_a_retryable_transport_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_length: str,
) -> None:
    attempts = 0

    class Response:
        length = None
        headers = {"Content-Length": content_length}

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            del size
            pytest.fail("body must not be read after an invalid Content-Length")

    def open_url(url: str, *, timeout: float) -> Response:
        nonlocal attempts
        del url, timeout
        attempts += 1
        return Response()

    monkeypatch.setattr(download_utils.urllib.request, "urlopen", open_url)

    with pytest.raises(download_utils.DownloadTransportError, match="Content-Length"):
        cmu_arctic._download(
            "https://example.invalid/archive",
            tmp_path / "archive.tar.bz2",
            sha256="0" * 64,
            retries=1,
        )

    assert attempts == 2


def test_download_response_is_closed_when_temporary_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = False
    exited = False

    class Response:
        length = 0

        def __enter__(self) -> "Response":
            nonlocal entered
            entered = True
            return self

        def __exit__(self, *args: object) -> None:
            nonlocal exited
            del args
            exited = True

    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())
    original_open = Path.open

    def fail_archive_part(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "archive.part":
            raise OSError("temporary open failed")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_archive_part)

    with pytest.raises(OSError, match="temporary open failed"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            tmp_path / "archive",
            algorithm="sha256",
            expected_digest=hashlib.sha256(b"").hexdigest(),
        )

    assert entered and exited


def test_download_rejects_declared_size_limit_before_reading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class Response:
        length = 3

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            nonlocal closed
            del args
            closed = True

        def read(self, size: int) -> bytes:
            del size
            pytest.fail("oversized declared response must not be read")

    monkeypatch.setattr(download_utils, "MAX_DOWNLOAD_BYTES", 2)
    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())

    with pytest.raises(ValueError, match="download byte limit"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            tmp_path / "archive",
            algorithm="sha256",
            expected_digest="0" * 64,
        )

    assert closed


def test_download_stops_when_body_exceeds_declared_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        length = 2

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            del size
            return b"abc"

    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())
    destination = tmp_path / "archive"

    with pytest.raises(DownloadIntegrityError, match="declared Content-Length"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            destination,
            algorithm="sha256",
            expected_digest="0" * 64,
        )

    assert not destination.exists()


def test_download_total_deadline_stops_slow_drip_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    read_timeouts: list[float] = []

    class Response:
        length = 1

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def set_read_timeout(self, timeout: float) -> None:
            read_timeouts.append(timeout)

        def read(self, size: int) -> bytes:
            nonlocal now
            del size
            now = 2.0
            return b"x"

    monkeypatch.setattr(download_utils.time, "monotonic", lambda: now)
    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())

    with pytest.raises(
        download_utils.DownloadTransportError,
        match="total_timeout",
    ):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            tmp_path / "archive",
            algorithm="sha256",
            expected_digest=hashlib.sha256(b"x").hexdigest(),
            total_timeout=1.0,
        )

    assert read_timeouts == pytest.approx([1.0])


def test_download_applies_remaining_total_deadline_to_connect_and_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monotonic_calls = 0
    connect_timeouts: list[float] = []
    read_timeouts: list[float] = []
    payload = b"x"

    def monotonic() -> float:
        nonlocal monotonic_calls
        monotonic_calls += 1
        return 0.0 if monotonic_calls == 1 else 4.0

    class Response:
        length = len(payload)

        def __init__(self) -> None:
            self.reads = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def set_read_timeout(self, timeout: float) -> None:
            read_timeouts.append(timeout)

        def read(self, size: int) -> bytes:
            del size
            self.reads += 1
            return payload if self.reads == 1 else b""

    def open_url(url: str, *, timeout: float) -> Response:
        del url
        connect_timeouts.append(timeout)
        return Response()

    monkeypatch.setattr(download_utils.time, "monotonic", monotonic)
    monkeypatch.setattr(download_utils, "_open_url", open_url)

    download_utils.stream_verified_download(
        "https://example.invalid/archive",
        tmp_path / "archive",
        algorithm="sha256",
        expected_digest=hashlib.sha256(payload).hexdigest(),
        timeout=10.0,
        total_timeout=5.0,
    )

    assert connect_timeouts == pytest.approx([1.0])
    assert read_timeouts == pytest.approx([1.0, 1.0])


def test_verified_download_does_not_replace_racing_new_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "archive"
    payload = b"verified"
    original_rename = download_utils.rename_entry_noreplace

    class Response:
        length = len(payload)

        def __init__(self) -> None:
            self.reads = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            del size
            self.reads += 1
            return payload if self.reads == 1 else b""

    def inject_destination(source: Path, target: Path) -> None:
        if target == destination:
            destination.write_bytes(b"third-party")
        original_rename(source, target)

    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())
    monkeypatch.setattr(
        download_utils,
        "rename_entry_noreplace",
        inject_destination,
    )

    with pytest.raises(RuntimeError, match="destination appeared"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            destination,
            algorithm="sha256",
            expected_digest=hashlib.sha256(payload).hexdigest(),
        )

    assert destination.read_bytes() == b"third-party"


def test_verified_download_restores_racing_replacement_during_atomic_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "archive"
    displaced = tmp_path / "old-cache"
    replacement = tmp_path / "replacement"
    destination.write_bytes(b"invalid-cache")
    replacement.write_bytes(b"third-party")
    payload = b"verified"
    original_exchange = download_utils.exchange_entries
    injected = False

    class Response:
        length = len(payload)

        def __init__(self) -> None:
            self.reads = 0

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self, size: int) -> bytes:
            del size
            self.reads += 1
            return payload if self.reads == 1 else b""

    def inject_before_exchange(first: Path, second: Path) -> None:
        nonlocal injected
        if second == destination and not injected:
            destination.replace(displaced)
            replacement.replace(destination)
            injected = True
        original_exchange(first, second)

    monkeypatch.setattr(download_utils, "_open_url", lambda *args, **kwargs: Response())
    monkeypatch.setattr(download_utils, "exchange_entries", inject_before_exchange)

    with pytest.raises(RuntimeError, match="atomic publication"):
        download_utils.stream_verified_download(
            "https://example.invalid/archive",
            destination,
            algorithm="sha256",
            expected_digest=hashlib.sha256(payload).hexdigest(),
        )

    assert injected
    assert destination.read_bytes() == b"third-party"
    assert displaced.read_bytes() == b"invalid-cache"


def _archive_case(
    tmp_path: Path,
    kind: str,
) -> tuple[Any, Path, Callable[[], object], str, str]:
    if kind == "cmu":
        archive = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic.tar.bz2"

        def cmu_constructor() -> object:
            return cmu_arctic.CmuArcticDataset(
                tmp_path,
                speaker="bdl",
                download=True,
            )

        return (
            cmu_arctic,
            archive,
            cmu_constructor,
            "_has_sha256",
            "_extract_cmu_archive_staged",
        )
    archive = tmp_path / "dev-clean.tar.gz"

    def libri_constructor() -> object:
        return librispeech.LibriSpeechDataset(
            tmp_path,
            subset="dev-clean",
            download=True,
        )

    return (
        librispeech,
        archive,
        libri_constructor,
        "_has_md5",
        "_extract_librispeech_archive_staged",
    )


def _pin_synthetic_archive_digest(
    monkeypatch: pytest.MonkeyPatch,
    *,
    kind: str,
    archive: Path,
) -> None:
    if kind == "cmu":
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        monkeypatch.setitem(cmu_arctic.ARCHIVE_SHA256, "bdl", digest)
    else:
        digest = hashlib.md5(
            archive.read_bytes(),
            usedforsecurity=False,
        ).hexdigest()
        monkeypatch.setitem(librispeech.ARCHIVE_MD5, "dev-clean", digest)


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_dataset_constructor_recovers_only_owned_stale_temporary_workspaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    module, archive, constructor, _, extract_name = _archive_case(tmp_path, kind)
    if kind == "cmu":
        target = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic"
        (target / "etc").mkdir(parents=True)
        (target / "wav").mkdir()
        (target / "etc" / "txt.done.data").write_text(
            '( arctic_a0001 "text" )\n',
            encoding="utf-8",
        )
        (target / "wav" / "arctic_a0001.wav").write_bytes(b"audio")
        extraction_parent = target.parent
    else:
        target = tmp_path / "LibriSpeech" / "dev-clean"
        chapter = target / "103" / "1240"
        chapter.mkdir(parents=True)
        (chapter / "103-1240.trans.txt").write_text(
            "103-1240-0000 TEXT\n",
            encoding="utf-8",
        )
        (chapter / "103-1240-0000.flac").write_bytes(b"audio")
        extraction_parent = tmp_path
    marker_content = archive_utils._TEMPORARY_MARKER
    download_stale = _seed_stale_temporary_workspaces(
        archive.parent,
        prefix=f".{archive.name}.torchrir-download.",
        marker_content=marker_content,
        label=f"{kind}-download",
    )
    extraction_stale = _seed_stale_temporary_workspaces(
        extraction_parent,
        prefix=f".{target.name}.torchrir-extract.",
        marker_content=marker_content,
        label=f"{kind}-extraction",
    )
    monkeypatch.setattr(
        module,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready dataset must not download"),
    )
    monkeypatch.setattr(
        module,
        extract_name,
        lambda *args, **kwargs: pytest.fail("ready dataset must not extract"),
    )

    with caplog.at_level("WARNING", logger="torchrir.datasets._archive"):
        constructor()

    _assert_only_owned_stale_workspaces_removed(download_stale)
    _assert_only_owned_stale_workspaces_removed(extraction_stale)
    for stale in (download_stale, extraction_stale):
        assert str(stale["unowned"]) in caplog.text
        assert str(stale["symlink"]) in caplog.text


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_dataset_download_serializes_concurrent_writers_and_rechecks_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module, archive, constructor, _, extract_name = _archive_case(
        tmp_path,
        kind,
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    if kind == "cmu":
        with tarfile.open(archive, "w:bz2") as tar:
            _add_tar_file(
                tar,
                "cmu_us_bdl_arctic/etc/txt.done.data",
                b'( arctic_a0001 "text" )\n',
            )
            _add_tar_file(tar, "cmu_us_bdl_arctic/wav/arctic_a0001.wav")
    else:
        with tarfile.open(archive, "w:gz") as tar:
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240.trans.txt",
                b"103-1240-0000 TEXT\n",
            )
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240-0000.flac",
            )
    _pin_synthetic_archive_digest(monkeypatch, kind=kind, archive=archive)
    monkeypatch.setattr(
        module,
        "_download",
        lambda *args, **kwargs: pytest.fail("verified local archive must be reused"),
    )
    counter_lock = threading.Lock()
    original_extract = getattr(module, extract_name)
    extraction_count = 0
    first_extraction_started = threading.Event()
    allow_first_extraction = threading.Event()
    second_lock_attempted = threading.Event()
    lock_attempts = 0

    def counted_extract(*args: Any, **kwargs: Any) -> None:
        nonlocal extraction_count
        with counter_lock:
            extraction_count += 1
            current_count = extraction_count
        if current_count == 1:
            first_extraction_started.set()
            assert allow_first_extraction.wait(timeout=5.0)
        original_extract(*args, **kwargs)

    monkeypatch.setattr(module, extract_name, counted_extract)
    original_writer_lock = module.dataset_write_lock

    @contextmanager
    def observed_writer_lock(*args: Any, **kwargs: Any) -> Iterator[None]:
        nonlocal lock_attempts
        with counter_lock:
            lock_attempts += 1
            if lock_attempts == 2:
                second_lock_attempted.set()
        with original_writer_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(module, "dataset_write_lock", observed_writer_lock)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(constructor)
        assert first_extraction_started.wait(timeout=5.0)
        second = executor.submit(constructor)
        try:
            assert second_lock_attempted.wait(timeout=5.0)
            assert not second.done()
        finally:
            allow_first_extraction.set()
        datasets = [first.result(timeout=5.0), second.result(timeout=5.0)]

    assert len(datasets) == 2
    assert extraction_count == 1


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_dataset_constructor_cleans_completed_interrupted_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module, _, constructor, _, _ = _archive_case(tmp_path, kind)
    if kind == "cmu":
        target = tmp_path / "ARCTIC" / "cmu_us_bdl_arctic"
        (target / "etc").mkdir(parents=True)
        (target / "wav").mkdir()
        (target / "etc" / "txt.done.data").write_text(
            '( arctic_a0001 "text" )\n',
            encoding="utf-8",
        )
        (target / "wav" / "arctic_a0001.wav").write_bytes(b"audio")
    else:
        target = tmp_path / "LibriSpeech" / "dev-clean"
        chapter = target / "103" / "1240"
        chapter.mkdir(parents=True)
        (chapter / "103-1240.trans.txt").write_text(
            "103-1240-0000 TEXT\n",
            encoding="utf-8",
        )
        (chapter / "103-1240-0000.flac").write_bytes(b"audio")
    transaction = target.parent / f".{target.name}.torchrir-transaction"
    backup = transaction / "previous"
    backup.mkdir(parents=True)
    (transaction / "owner").write_text(_TRANSACTION_MARKER, encoding="utf-8")
    (backup / "old.txt").write_text("old", encoding="utf-8")
    _write_test_transaction_manifest(
        transaction,
        staged=target,
        original=backup,
        phase="published",
    )
    monkeypatch.setattr(
        module,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready target must not download"),
    )

    constructor()

    assert not transaction.exists()


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_verified_archive_extraction_failure_is_not_redownloaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module, archive, constructor, _, extract = _archive_case(
        tmp_path,
        kind,
    )
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"already verified")
    monkeypatch.setattr(
        module,
        "_download",
        lambda *args, **kwargs: pytest.fail(
            "a verified archive must not be re-downloaded after local failure"
        ),
    )

    def fail_extraction(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(module, extract, fail_extraction)

    with pytest.raises(OSError, match="disk full"):
        constructor()

    assert archive.read_bytes() == b"already verified"


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_cached_archive_must_be_a_regular_file(
    tmp_path: Path,
    kind: str,
) -> None:
    _, archive, constructor, _, _ = _archive_case(tmp_path, kind)
    archive.mkdir(parents=True)

    with pytest.raises(ValueError, match="regular file"):
        constructor()


@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_cached_archive_symlink_is_preserved_until_verified_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module, archive, constructor, _, _ = _archive_case(tmp_path, kind)
    archive.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-archive"
    outside.write_bytes(b"external")
    archive.symlink_to(outside)

    def stop_before_network(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("safe redownload requested")

    monkeypatch.setattr(module, "_download", stop_before_network)

    with pytest.raises(RuntimeError, match="safe redownload"):
        constructor()

    assert archive.is_symlink()
    assert outside.read_bytes() == b"external"


def test_invalid_digest_repair_never_predeletes_a_replacement(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.tar.gz"
    replacement = tmp_path / "replacement.tar.gz"
    displaced = tmp_path / "inspected.tar.gz"
    archive.write_bytes(b"inspected")
    replacement.write_bytes(b"replacement")

    with archive_utils.open_regular_file_below(
        archive,
        root=tmp_path,
    ) as (_, identity):
        pass
    mismatch = archive_utils.ArchiveDigestMismatch(
        archive,
        identity=identity,
        algorithm="sha256",
        expected="0" * 64,
        actual="1" * 64,
    )

    def extract() -> None:
        raise mismatch

    def fail_replacement_download() -> None:
        archive.replace(displaced)
        replacement.replace(archive)
        raise RuntimeError("replacement download failed")

    with pytest.raises(RuntimeError, match="replacement download failed"):
        archive_utils.extract_or_download_archive(
            archive,
            root=tmp_path,
            download=fail_replacement_download,
            extract=extract,
            operation_logger=archive_utils.logger,
        )

    assert archive.read_bytes() == b"replacement"
    assert displaced.read_bytes() == b"inspected"


@pytest.mark.skipif(os.name == "nt", reason="requires replacing an open file path")
@pytest.mark.parametrize("kind", ["cmu", "libri"])
def test_verified_archive_is_extracted_from_same_file_after_path_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    module, archive, constructor, _, _ = _archive_case(tmp_path, kind)
    archive.parent.mkdir(parents=True, exist_ok=True)
    replacement = archive.parent / f"malicious-{archive.name}"
    if kind == "cmu":
        with tarfile.open(archive, "w:bz2") as tar:
            _add_tar_file(
                tar,
                "cmu_us_bdl_arctic/etc/txt.done.data",
                b'( arctic_a0001 "SAFE" )\n',
            )
            _add_tar_file(tar, "cmu_us_bdl_arctic/wav/arctic_a0001.wav")
        with tarfile.open(replacement, "w:bz2") as tar:
            _add_tar_file(
                tar,
                "cmu_us_bdl_arctic/etc/txt.done.data",
                b'( arctic_a0001 "MALICIOUS" )\n',
            )
            _add_tar_file(tar, "cmu_us_bdl_arctic/wav/arctic_a0001.wav")
        expected_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        monkeypatch.setitem(cmu_arctic.ARCHIVE_SHA256, "bdl", expected_digest)
    else:
        with tarfile.open(archive, "w:gz") as tar:
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240.trans.txt",
                b"103-1240-0000 SAFE\n",
            )
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240-0000.flac",
            )
        with tarfile.open(replacement, "w:gz") as tar:
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240.trans.txt",
                b"103-1240-0000 MALICIOUS\n",
            )
            _add_tar_file(
                tar,
                "LibriSpeech/dev-clean/103/1240/103-1240-0000.flac",
            )
        expected_digest = hashlib.md5(
            archive.read_bytes(),
            usedforsecurity=False,
        ).hexdigest()
        monkeypatch.setitem(librispeech.ARCHIVE_MD5, "dev-clean", expected_digest)
    malicious_bytes = replacement.read_bytes()
    monkeypatch.setattr(
        module,
        "_download",
        lambda *args, **kwargs: pytest.fail("verified archive must not download"),
    )
    original_tar_open = tarfile.open
    swapped = False

    def swap_path_before_extraction(*args: Any, **kwargs: Any) -> tarfile.TarFile:
        nonlocal swapped
        if not swapped:
            displaced = archive.parent / f"verified-{archive.name}"
            archive.replace(displaced)
            replacement.replace(archive)
            swapped = True
        return original_tar_open(*args, **kwargs)

    monkeypatch.setattr(archive_utils.tarfile, "open", swap_path_before_extraction)

    dataset = constructor()

    assert swapped
    assert archive.read_bytes() == malicious_bytes
    if isinstance(dataset, cmu_arctic.CmuArcticDataset):
        texts = [sentence.text for sentence in dataset.sentences()]
    else:
        assert isinstance(dataset, librispeech.LibriSpeechDataset)
        texts = [sentence.text for sentence in dataset.available_sentences()]
    assert texts == ["SAFE"]


def test_cmu_archive_replaces_incomplete_tree_only_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "ARCTIC"
    partial = base / "cmu_us_bdl_arctic"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("partial", encoding="utf-8")
    archive = base / "cmu_us_bdl_arctic.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        _add_tar_file(
            tar,
            "cmu_us_bdl_arctic/etc/txt.done.data",
            b'( arctic_a0001 "text" )\n',
        )
        _add_tar_file(tar, "cmu_us_bdl_arctic/wav/arctic_a0001.wav")
    _pin_synthetic_archive_digest(monkeypatch, kind="cmu", archive=archive)
    monkeypatch.setattr(
        cmu_arctic,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready local archive must not download"),
    )

    dataset = cmu_arctic.CmuArcticDataset(tmp_path, speaker="bdl", download=True)

    assert dataset.text_path.is_file()
    assert not (dataset._dataset_dir / "partial.txt").exists()


def test_librispeech_archive_replaces_incomplete_tree_only_after_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "LibriSpeech" / "dev-clean"
    partial.mkdir(parents=True)
    (partial / "partial.txt").write_text("partial", encoding="utf-8")
    archive = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_tar_file(
            tar,
            "LibriSpeech/dev-clean/103/1240/103-1240.trans.txt",
            b"103-1240-0000 TEXT\n",
        )
        _add_tar_file(
            tar,
            "LibriSpeech/dev-clean/103/1240/103-1240-0000.flac",
        )
    _pin_synthetic_archive_digest(monkeypatch, kind="libri", archive=archive)
    monkeypatch.setattr(
        librispeech,
        "_download",
        lambda *args, **kwargs: pytest.fail("ready local archive must not download"),
    )

    dataset = librispeech.LibriSpeechDataset(
        tmp_path,
        subset="dev-clean",
        download=True,
    )

    assert dataset.list_speakers() == ["103"]
    assert not (dataset._subset_dir / "partial.txt").exists()


def test_librispeech_download_repairs_missing_requested_speaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other_chapter = tmp_path / "LibriSpeech" / "dev-clean" / "103" / "1240"
    other_chapter.mkdir(parents=True)
    (other_chapter / "103-1240.trans.txt").write_text(
        "103-1240-0000 OTHER\n",
        encoding="utf-8",
    )
    (other_chapter / "103-1240-0000.flac").write_bytes(b"other")
    archive = tmp_path / "dev-clean.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add_tar_file(
            tar,
            "LibriSpeech/dev-clean/104/1250/104-1250.trans.txt",
            b"104-1250-0000 REQUESTED\n",
        )
        _add_tar_file(
            tar,
            "LibriSpeech/dev-clean/104/1250/104-1250-0000.flac",
        )
    _pin_synthetic_archive_digest(monkeypatch, kind="libri", archive=archive)
    monkeypatch.setattr(
        librispeech,
        "_download",
        lambda *args, **kwargs: pytest.fail("valid local archive must not download"),
    )

    dataset = librispeech.LibriSpeechDataset(
        tmp_path,
        subset="dev-clean",
        speaker="104",
        download=True,
    )

    assert dataset.list_speakers() == ["104"]
    assert [item.utterance_id for item in dataset.available_sentences()] == [
        "104-1250-0000"
    ]
