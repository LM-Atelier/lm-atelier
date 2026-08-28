from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import pytest

from local_lm import filesystem_links as links
from local_lm import shared_asset_contract_v1 as contract
from local_lm.shared_asset_contract_v1 import (
    INVALID_STORE,
    SCHEMA_ID,
    SharedAssetContractError,
    StoreIdentity,
    initialize_store_identity,
    negotiate_store_access,
    probe_store_root,
    read_store_identity,
    require_usable_root,
    store_access_mode,
)


def _identity_record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": SCHEMA_ID,
        "format_version": 1,
        "library_uuid": str(uuid.uuid4()),
        "min_reader_version": 1,
        "min_writer_version": 1,
    }
    record.update(overrides)
    return record


def test_initialize_creates_a_durable_identity_and_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    first = initialize_store_identity(root=root)
    assert first.format_version == 1
    assert first.min_reader_version == 1
    assert first.min_writer_version == 1
    assert str(uuid.UUID(first.library_uuid)) == first.library_uuid
    again = initialize_store_identity(root=root)
    assert again == first
    read_back = read_store_identity(root=root)
    assert read_back == first
    # The staged temp record never survives publication.
    assert [p.name for p in root.iterdir() if p.name.startswith("identity-")] == []


def _make_link_dir(base: Path, target: Path) -> Path | None:
    """Create a directory-shaped filesystem redirection, or None if this
    host cannot make one without privileges."""
    link = base / "redirect"
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
        )
        return link if completed.returncode == 0 else None
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return None
    return link


def test_concurrent_initializers_converge_on_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"
    lock = threading.Lock()
    first_published = threading.Event()
    observed_by_main = threading.Event()

    def gate() -> bool:
        """True when this caller is the first publisher.

        Both initializers pass the existing-identity check before either
        publishes. The first proceeds; the second waits until the main thread
        has OBSERVED the winner, so a publication that replaces rather than
        creates provably changes observed state.
        """

        with lock:
            if not first_published.is_set():
                return True
        assert observed_by_main.wait(timeout=30)
        return False

    # The two platforms publish through different calls, so the gate has to go
    # on the one this platform actually uses. Patching the other seam would
    # pass while testing nothing.
    if os.name == "nt":
        real_link_entry = contract.link_entry

        def gated_nt_link(anchor: object, source: str, destination: str) -> bool:
            first = gate()
            published = real_link_entry(anchor, source, destination)  # type: ignore[arg-type]
            if first:
                first_published.set()
            return published

        monkeypatch.setattr(contract, "link_entry", gated_nt_link)
    else:
        real_link = os.link

        def gated_link(source: str, destination: str, **kwargs: object) -> None:
            first = gate()
            real_link(source, destination, **kwargs)
            if first:
                first_published.set()

        monkeypatch.setattr(contract.os, "link", gated_link)
    results: list[StoreIdentity] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(initialize_store_identity(root=root))
        except BaseException as caught:  # noqa: BLE001 - surfaced below
            errors.append(caught)

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert first_published.wait(timeout=30)
    observed = (root / "store.json").read_bytes()
    observed_by_main.set()
    for thread in threads:
        thread.join(timeout=60)
    assert not errors
    assert len(results) == 2
    assert results[0] == results[1]
    # The identity never changes after observation, and later initializers
    # keep converging on it.
    assert (root / "store.json").read_bytes() == observed
    # Drop the gate whichever seam it went on, then prove a later
    # initializer still converges on the identity already observed.
    monkeypatch.undo()
    assert initialize_store_identity(root=root) == results[0]


def test_a_junction_or_symlink_root_is_refused_before_any_write(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create an unprivileged directory link")
    with pytest.raises(SharedAssetContractError):
        initialize_store_identity(root=link)
    assert list(target.iterdir()) == []
    report = probe_store_root(root=link, minimum_free_bytes=0)
    assert not report.no_reparse_points
    assert not report.usable


def test_an_existing_root_reached_through_a_junction_is_refused_unchanged(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    genuine = initialize_store_identity(root=target)
    snapshot = (target / "store.json").read_bytes()
    listing = sorted(p.name for p in target.iterdir())
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create an unprivileged directory link")
    with pytest.raises(SharedAssetContractError):
        initialize_store_identity(root=link)
    with pytest.raises(SharedAssetContractError):
        read_store_identity(root=link)
    # Zero target mutation and no identity adoption through the junction.
    assert (target / "store.json").read_bytes() == snapshot
    assert sorted(p.name for p in target.iterdir()) == listing
    assert initialize_store_identity(root=target) == genuine


def test_a_missing_descendant_below_a_junction_never_mutates_the_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create an unprivileged directory link")
    with pytest.raises(SharedAssetContractError):
        initialize_store_identity(root=link / "new-store")
    assert list(target.iterdir()) == []


def test_success_path_cleanup_failure_surfaces_only_the_fixed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"
    secret = r"secret C:\private\staging"

    def refuse_unlink(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(secret)

    # Removal goes through the shared primitive on both platforms now, and
    # only POSIX reaches os.unlink; patching that alone would leave this
    # passing on Windows while injecting nothing.
    monkeypatch.setattr(contract, "remove_entry", refuse_unlink)
    with pytest.raises(SharedAssetContractError) as caught:
        initialize_store_identity(root=root)
    assert str(caught.value) == "shared asset store is invalid"
    assert "secret" not in str(caught.value)


def test_a_staging_failure_with_failing_cleanup_still_surfaces_the_fixed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"

    def refuse_link(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(r"secret link C:\private\store")

    def refuse_unlink(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(r"secret unlink C:\private\staging")

    # Both operations go through the shared primitive now; the os functions
    # are only reached on POSIX.
    monkeypatch.setattr(contract, "link_entry", refuse_link)
    monkeypatch.setattr(contract, "remove_entry", refuse_unlink)
    with pytest.raises(SharedAssetContractError) as caught:
        initialize_store_identity(root=root)
    assert str(caught.value) == "shared asset store is invalid"
    assert "secret" not in str(caught.value)


def test_write_probes_never_run_after_containment_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create an unprivileged directory link")
    writes: list[str] = []

    def record_replace(*args: object, **kwargs: object) -> None:
        writes.append("replace")
        raise AssertionError("write probe ran after containment failure")

    real_open = os.open

    def record_open(path: object, flags: int, *args: object) -> int:
        if flags & os.O_CREAT:
            writes.append("create")
            raise AssertionError("write probe ran after containment failure")
        return real_open(path, flags, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(contract.os, "replace", record_replace)
    monkeypatch.setattr(contract.os, "open", record_open)
    report = probe_store_root(root=link, minimum_free_bytes=0)
    assert not report.usable
    assert writes == []


@pytest.mark.parametrize("target", ["stage", "link", "fsync"])
def test_injected_filesystem_failures_never_leak_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    root = tmp_path / "packages"
    secret = r"secret C:\private\store"

    def explode(*_args: object, **_kwargs: object) -> None:
        raise PermissionError(secret)

    # Each platform stages and publishes through different calls, so the
    # failure has to be injected into the one this platform uses.
    windows = os.name == "nt"
    if target == "stage":
        if windows:
            monkeypatch.setattr(links, "_nt_open_relative", explode)
        else:
            # On POSIX this also covers the anchor's own open, which must
            # refuse just as quietly.
            monkeypatch.setattr(contract.os, "open", explode)
    elif target == "link":
        if windows:
            monkeypatch.setattr(contract, "link_entry", explode)
        else:
            monkeypatch.setattr(contract.os, "link", explode)
    else:
        monkeypatch.setattr(contract.os, "fsync", explode)
    with pytest.raises(SharedAssetContractError) as caught:
        initialize_store_identity(root=root)
    assert str(caught.value) == "shared asset store is invalid"
    assert "secret" not in str(caught.value)


def _swap_root_for_redirect(root: Path, target: Path) -> str:
    """Try to replace the verified root with a redirect to `target`.

    This is the attack the anchor exists to stop: the root passes every
    containment check and is then swapped for a junction (Windows) or a
    symlink (POSIX) before the store is written. Returns "swapped" when the
    filesystem allowed it, or the refusal text when it did not.
    """

    target.mkdir(parents=True, exist_ok=True)
    try:
        os.rmdir(root)
    except OSError as error:
        return f"rmdir refused: {type(error).__name__}"
    try:
        if os.name == "nt":
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(root), str(target)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                return f"mklink refused: {completed.stderr.strip()}"
        else:
            os.symlink(str(target), str(root), target_is_directory=True)
    except OSError as error:
        return f"link refused: {type(error).__name__}"
    return "swapped"


def test_a_root_swapped_after_verification_gains_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact check-then-use window: swap the root once it is verified.

    The anchor is already held when staging begins, so on Windows the
    directory cannot be removed at all, and on POSIX every step resolves
    against the held descriptor rather than against the path. Either way the
    foreign target must gain nothing - not the record, not a staging file,
    not an empty entry. Whether the call then succeeds or refuses is
    secondary; writing through the replacement is the defect.
    """

    root = tmp_path / "packages"
    root.mkdir()
    foreign = tmp_path / "foreign"
    outcomes: list[str] = []
    original = contract._stage_and_publish

    def swap_then_stage(anchor: object, record: dict[str, object]) -> None:
        # Runs after the anchor is held and before anything is written.
        outcomes.append(_swap_root_for_redirect(root, foreign))
        original(anchor, record)  # type: ignore[arg-type]

    monkeypatch.setattr(contract, "_stage_and_publish", swap_then_stage)
    with contextlib.suppress(SharedAssetContractError):
        initialize_store_identity(root=root)

    assert outcomes, "the injected swap never ran"
    gained = sorted(entry.name for entry in foreign.iterdir())
    assert gained == [], f"the swap target gained {gained}; swap outcome was {outcomes[0]}"


def test_a_root_swapped_before_the_anchor_is_never_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The earlier window: swap between the chain check and the anchor.

    The chain check clears the root, and the swap lands before the anchor is
    even opened - so the anchor is asked to hold a junction. It must refuse
    rather than adopt it, because everything downstream trusts the anchor to
    be the verified directory. The mutation battery found this window
    untested: disabling the anchor's own reparse check left every other test
    passing.
    """

    root = tmp_path / "packages"
    root.mkdir()
    foreign = tmp_path / "foreign"
    outcomes: list[str] = []
    real_anchor = contract.AnchoredDirectory

    def swap_then_anchor(store: Path, **kwargs: object) -> object:
        outcomes.append(_swap_root_for_redirect(root, foreign))
        return real_anchor(store, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(contract, "AnchoredDirectory", swap_then_anchor)
    with contextlib.suppress(SharedAssetContractError):
        initialize_store_identity(root=root)

    assert outcomes, "the injected swap never ran"
    gained = sorted(entry.name for entry in foreign.iterdir())
    assert gained == [], f"the swap target gained {gained}; swap outcome was {outcomes[0]}"


@pytest.mark.skipif(os.name != "nt", reason="junction swaps are Windows-specific")
def test_the_held_root_cannot_be_removed_while_anchored(tmp_path: Path) -> None:
    """The Windows guarantee stated directly: no rename, no delete, no swap.

    The anchor opens the directory without FILE_SHARE_DELETE, so the removal
    a junction swap requires fails while it is held - and succeeds once it is
    released, which is what proves the refusal comes from the anchor rather
    than from something incidental about the directory.
    """

    root = tmp_path / "packages"
    root.mkdir()
    with contract.AnchoredDirectory(root), pytest.raises(OSError):
        os.rmdir(root)
    os.rmdir(root)
    assert not root.exists()


@pytest.mark.skipif(os.name == "nt", reason="directory descriptors are POSIX-only")
def test_publication_persists_the_directory_entry_on_posix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"
    synced: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        synced.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(contract.os, "fsync", record_fsync)
    initialize_store_identity(root=root)
    # One fsync for the staged record, one for the directory entry.
    assert len(synced) == 2


def test_read_returns_none_only_when_no_store_exists(tmp_path: Path) -> None:
    """None means genuine absence, and nothing else does."""

    assert read_store_identity(root=tmp_path / "absent") is None
    assert read_store_identity(root=tmp_path / "absent" / "deeper") is None
    root = tmp_path / "packages"
    root.mkdir()
    assert read_store_identity(root=root) is None


def test_a_dangling_linked_root_is_refused_rather_than_read_as_absent(
    tmp_path: Path,
) -> None:
    """A redirection whose target is gone is still a redirection.

    An existence query FOLLOWS the name, so a dangling redirect answers false.
    Classifying that as "no library here" is the one answer that invites a
    caller to go on and establish one THROUGH the redirect.
    """

    target = tmp_path / "target"
    target.mkdir()
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create a directory redirection unprivileged")
    shutil.rmtree(target)
    # The query this used to trust really does report the root as absent.
    assert not link.exists()
    with pytest.raises(SharedAssetContractError) as caught:
        read_store_identity(root=link)
    assert str(caught.value) == INVALID_STORE


def test_an_acquisition_failure_that_is_not_absence_is_refused(tmp_path: Path) -> None:
    """Absence is the only acquisition outcome that may return None."""

    occupied = tmp_path / "regular"
    occupied.write_text("content", encoding="utf-8")
    with pytest.raises(SharedAssetContractError) as caught:
        read_store_identity(root=occupied)
    assert str(caught.value) == INVALID_STORE


@pytest.mark.parametrize(
    ("name", "accepted"),
    [
        ("a" * 259, True),
        ("a" * 260, False),
        # 129 supplementary characters are 258 UTF-16 units, so the bound must
        # not reject them; 130 are 260 units while still only 130 code points,
        # which the code-point bound alone waved through.
        ("\U0001f600" * 129, True),
        ("\U0001f600" * 130, False),
    ],
)
def test_entry_names_are_bounded_in_utf16_units_not_code_points(name: str, accepted: bool) -> None:
    """The native FileName field is 260 WIDE CHARACTERS, not 260 characters.

    A supplementary character costs two UTF-16 units, so a name comfortably
    under the code-point bound could still overflow the structure it is
    assigned into and surface a raw conversion error instead of this layer's
    fixed neutral refusal.
    """

    if accepted:
        links._require_entry_name(name)
        return
    with pytest.raises(links.AnchoredDirectoryError):
        links._require_entry_name(name)


def test_an_oversized_supplementary_name_refuses_before_any_native_call(
    tmp_path: Path,
) -> None:
    """The refusal happens in validation, not as a conversion failure."""

    anchor = links.AnchoredDirectory(tmp_path, create=False)
    try:
        with pytest.raises(links.AnchoredDirectoryError):
            links.create_entry(anchor, "\U0001f600" * 130)
    finally:
        anchor.close()


@pytest.mark.parametrize(
    "raw",
    [
        b"not json at all",
        b"\xff\xfe\x00\x00",
        json.dumps({"schema": "some-other-store"}).encode(),
        json.dumps(_identity_record(schema="some-other-store")).encode(),
        json.dumps(_identity_record(format_version=0)).encode(),
        json.dumps(_identity_record(min_reader_version=True)).encode(),
        json.dumps(_identity_record(library_uuid="not-a-uuid")).encode(),
        json.dumps(_identity_record(library_uuid=uuid.uuid4().hex.upper())).encode(),
        json.dumps([1, 2, 3]).encode(),
    ],
)
def test_present_but_malformed_identity_refuses_never_reads_as_absent(
    tmp_path: Path, raw: bytes
) -> None:
    root = tmp_path / "packages"
    root.mkdir()
    (root / "store.json").write_bytes(raw)
    with pytest.raises(SharedAssetContractError) as caught:
        read_store_identity(root=root)
    # The literal, not the imported constant: the refusal message is a fixed
    # public contract and must not drift with the module.
    assert str(caught.value) == "shared asset store is invalid"
    assert INVALID_STORE == "shared asset store is invalid"


def test_an_unreadable_identity_refuses_rather_than_reading_as_absent(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    (root / "store.json").mkdir(parents=True)
    with pytest.raises(SharedAssetContractError) as caught:
        read_store_identity(root=root)
    assert str(caught.value) == INVALID_STORE


def test_rename_probe_is_wired_to_a_real_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"
    initialize_store_identity(root=root)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("rename unsupported")

    # The probe renames through the held directory now, which is a different
    # call on each platform. Patching the other one would leave the probe
    # succeeding and this test passing while proving nothing.
    if os.name == "nt":
        monkeypatch.setattr(contract, "rename_entry", refuse)
    else:
        monkeypatch.setattr(contract, "rename_entry", refuse)
    report = probe_store_root(root=root, minimum_free_bytes=0)
    assert not report.atomic_rename
    assert not report.usable
    assert store_access_mode(root=root) == "read_only"


def test_unknown_extra_keys_stay_readable_under_a_compatible_format(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    root.mkdir()
    record = _identity_record(future_optional_feature={"enabled": True})
    (root / "store.json").write_text(json.dumps(record), encoding="utf-8")
    identity = read_store_identity(root=root)
    assert identity is not None
    assert identity.library_uuid == record["library_uuid"]


@pytest.mark.parametrize(
    "root",
    [Path("relative/packages"), Path("//server/share/packages")],
)
def test_hostile_roots_are_refused_with_the_fixed_message(root: Path) -> None:
    with pytest.raises(SharedAssetContractError) as caught:
        read_store_identity(root=root)
    assert str(caught.value) == INVALID_STORE


def test_nul_bytes_in_a_root_are_refused(tmp_path: Path) -> None:
    with pytest.raises(SharedAssetContractError):
        read_store_identity(root=Path(str(tmp_path) + "\x00packages"))


def test_negotiation_orders_reader_before_writer() -> None:
    compatible = StoreIdentity(
        library_uuid=str(uuid.uuid4()),
        format_version=1,
        min_reader_version=1,
        min_writer_version=1,
    )
    assert negotiate_store_access(compatible) == "read_write"
    newer_writer = StoreIdentity(
        library_uuid=compatible.library_uuid,
        format_version=2,
        min_reader_version=1,
        min_writer_version=2,
    )
    assert negotiate_store_access(newer_writer) == "read_only"
    newer_reader = StoreIdentity(
        library_uuid=compatible.library_uuid,
        format_version=3,
        min_reader_version=2,
        min_writer_version=2,
    )
    with pytest.raises(SharedAssetContractError):
        negotiate_store_access(newer_reader)
    with pytest.raises(SharedAssetContractError):
        negotiate_store_access(compatible, reader_version=0)
    with pytest.raises(SharedAssetContractError):
        negotiate_store_access(compatible, writer_version=True)  # type: ignore[arg-type]


def test_probe_battery_accepts_a_plain_writable_directory(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    root.mkdir()
    report = probe_store_root(root=root, minimum_free_bytes=0)
    assert report.directory
    assert report.no_reparse_points
    assert report.atomic_rename
    assert report.exclusive_create
    assert report.free_space
    assert report.usable
    assert require_usable_root(root=root, minimum_free_bytes=0) == root
    # Probe artifacts never accumulate.
    assert list((root / "locks").iterdir()) == []


def _convert_to_junction_in_place(directory: Path, target: Path) -> bool:
    """Turn an existing EMPTY directory into a junction without moving it.

    This is the attack that beat the previous anchor. It needs no delete and
    no rename - which is exactly why holding the directory open against those
    two operations was not enough - and no elevation.
    """

    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    handle = kernel32.CreateFileW(
        str(directory),
        0x80000000 | 0x40000000,  # GENERIC_READ | GENERIC_WRITE
        0x00000001 | 0x00000002,  # FILE_SHARE_READ | FILE_SHARE_WRITE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == ctypes.c_void_p(-1).value or handle is None:
        return False
    substitute = f"\\??\\{target}".encode("utf-16-le")
    printed = str(target).encode("utf-16-le")
    path_buffer = substitute + b"\x00\x00" + printed + b"\x00\x00"
    buffer = bytearray()
    buffer += (0xA0000003).to_bytes(4, "little")  # IO_REPARSE_TAG_MOUNT_POINT
    buffer += (8 + len(path_buffer)).to_bytes(2, "little")
    buffer += (0).to_bytes(2, "little")
    buffer += (0).to_bytes(2, "little")
    buffer += len(substitute).to_bytes(2, "little")
    buffer += (len(substitute) + 2).to_bytes(2, "little")
    buffer += len(printed).to_bytes(2, "little")
    buffer += path_buffer
    raw = (ctypes.c_char * len(buffer)).from_buffer(buffer)
    returned = wintypes.DWORD(0)
    ok = kernel32.DeviceIoControl(
        ctypes.c_void_p(handle),
        0x000900A4,  # FSCTL_SET_REPARSE_POINT
        raw,
        len(buffer),
        None,
        0,
        ctypes.byref(returned),
        None,
    )
    kernel32.CloseHandle(ctypes.c_void_p(handle))
    return bool(ok)


@pytest.mark.skipif(os.name != "nt", reason="in-place reparse is Windows-specific")
def test_an_in_place_reparse_after_anchoring_writes_nothing_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attack that defeated the previous anchor, pinned.

    Holding the root open blocks rename and delete, and a junction needs
    neither: the conversion happens in place. Under the earlier construction
    the foreign directory gained store.json. Publication is now resolved
    against the held handle, so the create refuses rather than following the
    new redirection - and nothing reaches the target.
    """

    root = tmp_path / "packages"
    root.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    converted: list[bool] = []
    original = contract._stage_and_publish

    def convert_then_stage(anchor: object, record: dict[str, object]) -> None:
        converted.append(_convert_to_junction_in_place(root, foreign))
        original(anchor, record)  # type: ignore[arg-type]

    monkeypatch.setattr(contract, "_stage_and_publish", convert_then_stage)
    with contextlib.suppress(SharedAssetContractError):
        initialize_store_identity(root=root)

    assert converted, "the injected conversion never ran"
    if not converted[0]:
        pytest.skip("this filesystem refused the in-place conversion")
    assert sorted(entry.name for entry in foreign.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="in-place reparse is Windows-specific")
def test_a_swap_during_component_acquisition_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convert a component while the walk is still climbing towards the leaf.

    The window the walk exists to remove: a path-based anchor validates the
    chain, then opens the root, and anything converted in between is followed.
    Here the conversion lands mid-walk, after the parent has been opened and
    before the leaf is. Each component is opened relative to the one before
    it, so the redirection cannot be picked up, and the reparse check on the
    freshly opened component refuses outright.
    """

    parent = tmp_path / "lib"
    root = parent / "packages"
    root.mkdir(parents=True)
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    real_open = links._nt_open_relative
    converted: list[bool] = []

    def convert_before_leaf(handle: object, name: str, *, intent: str) -> int:
        # Fire once, on the way in to the leaf itself.
        if name == root.name and not converted:
            converted.append(_convert_to_junction_in_place(root, foreign))
        return real_open(handle, name, intent=intent)  # type: ignore[arg-type]

    monkeypatch.setattr(links, "_nt_open_relative", convert_before_leaf)
    with contextlib.suppress(SharedAssetContractError):
        initialize_store_identity(root=root)

    assert converted, "the injected conversion never ran"
    if not converted[0]:
        pytest.skip("this filesystem refused the in-place conversion")
    assert sorted(entry.name for entry in foreign.iterdir()) == []


@pytest.mark.skipif(os.name != "nt", reason="in-place reparse is Windows-specific")
def test_the_conversion_becomes_impossible_once_an_entry_exists(
    tmp_path: Path,
) -> None:
    """Why the first staged entry closes the window rather than dodging it.

    The in-place conversion requires an EMPTY directory. That is not a detail
    of this implementation but the property it leans on: once the store holds
    a single entry, the root can no longer be turned into a junction by
    anybody, so the attack has a window only before the first write and the
    first write is itself anchored.
    """

    empty = tmp_path / "empty"
    empty.mkdir()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "store.json").write_bytes(b"{}")
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    if not _convert_to_junction_in_place(empty, foreign):
        pytest.skip("this filesystem refused the in-place conversion")
    # The same call against a directory holding one entry must fail.
    assert not _convert_to_junction_in_place(occupied, foreign)


def test_a_junctioned_ancestor_is_refused_and_gains_nothing(tmp_path: Path) -> None:
    """A redirect ABOVE the root must not be followed either.

    Stated honestly, because this one does NOT distinguish the walk from what
    came before: the chain check already refused a pre-existing junctioned
    ancestor, and this passes against both constructions. It is here to keep
    that behaviour pinned, not as evidence for the walk.

    What the walk actually buys is the removal of an argument. O_NOFOLLOW and
    FILE_FLAG_OPEN_REPARSE_POINT describe only the final component, so an
    anchor that opens the root by path has to reason about whether an ancestor
    can be swapped between the check and the open - and that reasoning runs
    through whether an attacker can empty the parent, which depends on the
    child handle being held, which depends on the very thing being proven.
    Resolving every component relative to the one before it leaves no
    whole-path resolution to subvert, so the question stops being asked.
    """

    foreign = tmp_path / "foreign"
    (foreign / "packages").mkdir(parents=True)
    parent = _make_link_dir(tmp_path, foreign)
    if parent is None:
        pytest.skip("this filesystem refused a directory link")

    with pytest.raises(SharedAssetContractError):
        initialize_store_identity(root=parent / "packages")

    assert sorted(entry.name for entry in (foreign / "packages").iterdir()) == []


def test_the_probe_refuses_a_root_whose_locks_child_is_redirected(tmp_path: Path) -> None:
    """A child junction must fail the probe, not be written through.

    The battery writes its exclusive-create probe inside `locks`. Before this
    guard the directory was created with exist_ok and never inspected, so a
    pre-placed junction there redirected the probe into a foreign directory
    while the report still called the root usable - and it needed no race at
    all. `no_reparse_points` stays True on purpose: it describes the ancestor
    chain, which really is clean. The usability verdict is what must refuse.
    """

    root = tmp_path / "packages"
    root.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(root / "locks"), str(foreign)],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            pytest.skip("this filesystem refused a junction")
    else:
        os.symlink(str(foreign), str(root / "locks"), target_is_directory=True)

    report = probe_store_root(root=root, minimum_free_bytes=0)

    assert not report.exclusive_create
    assert not report.usable
    assert sorted(entry.name for entry in foreign.iterdir()) == []


def test_probe_battery_fails_closed_for_missing_or_file_roots(tmp_path: Path) -> None:
    missing = probe_store_root(root=tmp_path / "absent")
    assert not missing.directory
    assert not missing.usable
    file_root = tmp_path / "not-a-dir"
    file_root.write_bytes(b"payload")
    assert not probe_store_root(root=file_root).usable
    with pytest.raises(SharedAssetContractError):
        require_usable_root(root=file_root)


def test_every_probe_in_one_report_sees_the_same_held_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report is one result and must describe one directory identity.

    Capacity used to be taken by pathname while the capability fields were
    proved against the held directory, so a single report could describe two
    things. This pins the property rather than the symptom: whatever each
    probe is handed, it is the SAME held directory.
    """

    root = tmp_path / "packages"
    root.mkdir()
    seen: list[object] = []

    real_rename = contract._probe_atomic_rename
    real_create = contract._probe_exclusive_create
    real_capacity = contract.available_bytes

    def record_rename(anchor: object) -> bool:
        seen.append(anchor)
        return real_rename(anchor)  # type: ignore[arg-type]

    def record_create(anchor: object) -> bool:
        seen.append(anchor)
        return real_create(anchor)  # type: ignore[arg-type]

    def record_capacity(anchor: object) -> int:
        seen.append(anchor)
        return real_capacity(anchor)  # type: ignore[arg-type]

    monkeypatch.setattr(contract, "_probe_atomic_rename", record_rename)
    monkeypatch.setattr(contract, "_probe_exclusive_create", record_create)
    monkeypatch.setattr(contract, "available_bytes", record_capacity)

    report = probe_store_root(root=root)

    assert len(seen) == 3, "a probe stopped being exercised"
    first = seen[0]
    assert all(entry is first for entry in seen), "the report combined two identities"
    assert report.usable


def test_capacity_comes_from_the_volume_the_handle_is_on(tmp_path: Path) -> None:
    """A real measurement, not a placeholder that happens to satisfy the floor."""

    root = tmp_path / "packages"
    root.mkdir()
    with contract._anchored(root) as anchor:
        through_handle = links.available_bytes(anchor)
    by_name = shutil.disk_usage(root).free
    assert through_handle > 0
    # Same volume, so the two agree to within ordinary churn between the calls.
    assert abs(through_handle - by_name) < max(by_name * 0.05, 64 * 1024 * 1024)


def test_a_refused_capacity_query_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unanswerable capacity question is not a passing one."""

    root = tmp_path / "packages"
    root.mkdir()

    def refuse(_anchor: object) -> int:
        raise links.AnchoredDirectoryError(links.CONTAINMENT_REFUSED)

    monkeypatch.setattr(contract, "available_bytes", refuse)
    report = probe_store_root(root=root)
    assert not report.free_space
    assert not report.usable


def test_rename_refuses_an_existing_destination_by_default(tmp_path: Path) -> None:
    """The default must mean the same thing on both platforms.

    os.rename replaces silently and the NT call refuses, so an unqualified
    rename meant "publish over the old file" on POSIX and "do nothing" on
    Windows - and the Windows outcome arrived as a return value rather than a
    refusal, which is exactly what call sites drop.
    """

    with links.AnchoredDirectory(tmp_path) as anchor:
        os.close(links.create_entry(anchor, "source"))
        os.close(links.create_entry(anchor, "occupied"))
        with pytest.raises(links.AnchoredDirectoryError):
            links.rename_entry(anchor, "source", "occupied")
        # The refusal left both entries exactly as they were.
        assert (tmp_path / "source").is_file()
        assert (tmp_path / "occupied").is_file()


def test_rename_replaces_only_when_asked_and_does_so_on_both_platforms(
    tmp_path: Path,
) -> None:
    """Publish-over-existing is the pattern the primitive exists to support."""

    with links.AnchoredDirectory(tmp_path) as anchor:
        (tmp_path / "published").write_bytes(b"OLD")
        descriptor = links.create_entry(anchor, "staging")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"NEW")
        links.rename_entry(anchor, "staging", "published", replace=True)

    assert (tmp_path / "published").read_bytes() == b"NEW", "the publish did not happen"
    assert not (tmp_path / "staging").exists(), "the staging entry was left behind"


def test_rename_publishes_into_another_held_directory(tmp_path: Path) -> None:
    """Content-addressed publication cannot avoid crossing directories.

    The digest is not known until the bytes have been consumed, so staging
    happens somewhere chosen BEFORE the destination shard is known. Copying
    instead would defeat atomicity and double the IO on large media.
    """

    (tmp_path / "staging").mkdir()
    (tmp_path / "published").mkdir()
    with (
        links.AnchoredDirectory(tmp_path / "staging") as source,
        links.AnchoredDirectory(tmp_path / "published") as target,
    ):
        descriptor = links.create_entry(source, "incoming")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"payload")
        links.rename_entry(source, "incoming", "final", into=target)

    assert (tmp_path / "published" / "final").read_bytes() == b"payload"
    assert not (tmp_path / "staging" / "incoming").exists(), "the source entry remained"


def test_cross_directory_rename_refuses_an_occupied_destination_by_default(
    tmp_path: Path,
) -> None:
    """The default means the same thing across directories as within one."""

    (tmp_path / "staging").mkdir()
    (tmp_path / "published").mkdir()
    (tmp_path / "published" / "final").write_bytes(b"OLD")
    with (
        links.AnchoredDirectory(tmp_path / "staging") as source,
        links.AnchoredDirectory(tmp_path / "published") as target,
    ):
        os.close(links.create_entry(source, "incoming"))
        with pytest.raises(links.AnchoredDirectoryError):
            links.rename_entry(source, "incoming", "final", into=target)

    assert (tmp_path / "published" / "final").read_bytes() == b"OLD"
    assert (tmp_path / "staging" / "incoming").is_file(), "the source was consumed anyway"


def test_cross_directory_rename_replaces_when_asked(tmp_path: Path) -> None:
    """Publish-over-existing, across directories, on both platforms."""

    (tmp_path / "staging").mkdir()
    (tmp_path / "published").mkdir()
    (tmp_path / "published" / "final").write_bytes(b"OLD")
    with (
        links.AnchoredDirectory(tmp_path / "staging") as source,
        links.AnchoredDirectory(tmp_path / "published") as target,
    ):
        descriptor = links.create_entry(source, "incoming")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"NEW")
        links.rename_entry(source, "incoming", "final", into=target, replace=True)

    assert (tmp_path / "published" / "final").read_bytes() == b"NEW"
    assert not (tmp_path / "staging" / "incoming").exists()


def test_rename_into_a_free_name_works_in_either_mode(tmp_path: Path) -> None:
    """The modes differ only when the destination is occupied."""

    with links.AnchoredDirectory(tmp_path) as anchor:
        os.close(links.create_entry(anchor, "first"))
        links.rename_entry(anchor, "first", "first-moved")
        os.close(links.create_entry(anchor, "second"))
        links.rename_entry(anchor, "second", "second-moved", replace=True)

    assert (tmp_path / "first-moved").is_file()
    assert (tmp_path / "second-moved").is_file()
    assert not (tmp_path / "first").exists()
    assert not (tmp_path / "second").exists()


def test_free_space_floor_is_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "packages"
    root.mkdir()
    # Injected at the anchored capacity query, which is now the mechanism.
    # Patching a pathname-based one would no longer be reached, and a probe
    # that silently stopped being exercised would still go green.
    monkeypatch.setattr(contract, "available_bytes", lambda _anchor: 1024)
    report = probe_store_root(root=root)
    assert not report.free_space
    assert not report.usable
    with pytest.raises(SharedAssetContractError):
        probe_store_root(root=root, minimum_free_bytes=-1)
    with pytest.raises(SharedAssetContractError):
        probe_store_root(root=root, minimum_free_bytes=True)  # type: ignore[arg-type]


def test_access_mode_requires_an_existing_identity(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    root.mkdir()
    with pytest.raises(SharedAssetContractError):
        store_access_mode(root=root)


def test_access_mode_resolves_read_write_then_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "packages"
    initialize_store_identity(root=root)
    assert store_access_mode(root=root) == "read_write"
    # A store written by a newer format degrades this build to read-only.
    assert store_access_mode(root=root, writer_version=0 + 1, reader_version=1) == "read_write"
    record = _identity_record(min_writer_version=2)
    (root / "store.json").write_text(json.dumps(record), encoding="utf-8")
    assert store_access_mode(root=root) == "read_only"
    # A failing write probe also degrades, without refusing reads.
    (root / "store.json").write_text(json.dumps(_identity_record()), encoding="utf-8")
    # Injected at the anchored capacity query, which is now the mechanism.
    # Patching a pathname-based one would no longer be reached, and a probe
    # that silently stopped being exercised would still go green.
    monkeypatch.setattr(contract, "available_bytes", lambda _anchor: 1024)
    assert store_access_mode(root=root) == "read_only"


def test_access_mode_never_creates_a_library(tmp_path: Path) -> None:
    root = tmp_path / "packages"
    with pytest.raises(SharedAssetContractError):
        store_access_mode(root=root)
    assert not root.exists()


def test_the_exclusive_probe_refuses_a_wrong_failure_as_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a name collision proves exclusive creation works.

    The probe creates an entry, then creates it again and expects to be
    refused. An earlier version accepted ANY refusal as proof, so a
    filesystem that failed the second create for an unrelated reason - I/O
    error, containment refusal - produced a green battery. That is a probe
    that passes precisely when the thing it measures is broken.
    """

    root = tmp_path / "packages"
    initialize_store_identity(root=root)
    real_create = contract.create_entry
    calls: list[str] = []

    def fail_second_create(anchor: object, name: str) -> int:
        # Count only this probe's own entries; the rename probe creates one
        # too, and counting that made the stub fire on the wrong call.
        if name.startswith(contract._PROBE_CREATE_PREFIX):
            calls.append(name)
            if len(calls) > 1:
                # Not a collision: the wrong error, which must not read as
                # proof that exclusive creation works.
                raise links.AnchoredDirectoryError(links.CONTAINMENT_REFUSED)
        return real_create(anchor, name)  # type: ignore[arg-type]

    monkeypatch.setattr(contract, "create_entry", fail_second_create)
    report = probe_store_root(root=root, minimum_free_bytes=0)

    assert len(calls) >= 2, "the second create never ran"
    assert not report.exclusive_create
    assert not report.usable


def test_a_collision_still_proves_exclusive_creation(tmp_path: Path) -> None:
    """The other direction, so the control above cannot pass vacuously."""

    root = tmp_path / "packages"
    initialize_store_identity(root=root)
    report = probe_store_root(root=root, minimum_free_bytes=0)
    assert report.exclusive_create
    assert report.usable


def test_absent_and_present_entries_are_distinguished_through_the_handle(
    tmp_path: Path,
) -> None:
    """Absence is decided by the operation, not by testing a path first.

    Reading or deleting through anchor.path re-resolved a name, which is the
    thing this layer exists to avoid - and an adopted child holds only its own
    handle, so its ancestors genuinely can change underneath it.
    """

    root = tmp_path / "packages"
    root.mkdir()
    with links.AnchoredDirectory(root) as anchor:
        assert links.read_entry(anchor, "absent.json") is None
        # Removing something absent is not a failure.
        links.remove_entry(anchor, "absent.json")

        descriptor = links.create_entry(anchor, "present.json")
        with os.fdopen(descriptor, "wb") as entry:
            entry.write(b"{}")
        assert links.read_entry(anchor, "present.json") == b"{}"

        links.remove_entry(anchor, "present.json")
        assert links.read_entry(anchor, "present.json") is None
