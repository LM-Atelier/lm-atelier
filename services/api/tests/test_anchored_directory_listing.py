from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_lm import filesystem_links
from local_lm.filesystem_links import (
    _FILE_DIRECTORY_INFORMATION_HEADER,
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    _kind_from_dirent_type,
    _list_posix,
    _read_directory_records,
    _require_entry_name,
    _utf16_length,
    create_entry,
    list_entries,
    open_child_directory,
    remove_entry,
    rename_entry,
)


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at `target`, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _kinds(root: Path) -> dict[str, AnchoredEntryKind]:
    with AnchoredDirectory(root) as anchor:
        return {entry.name: entry.kind for entry in list_entries(anchor)}


def test_ordinary_files_and_directories_carry_their_kind(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    (root / "artifact.bin").write_bytes(b"payload")
    (root / "nested").mkdir()
    (root / "nested" / "inner.bin").write_bytes(b"deeper")

    kinds = _kinds(root)

    assert kinds == {
        "artifact.bin": AnchoredEntryKind.FILE,
        "nested": AnchoredEntryKind.DIRECTORY,
    }


def test_the_listing_never_includes_dot_or_dot_dot(tmp_path: Path) -> None:
    """Both are real entries on Windows and both are traversals upward.

    A caller that deleted or swept everything it listed would, with `..`
    present, be handed its own parent as a name to act on.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "only.bin").write_bytes(b"x")

    names = set(_kinds(root))

    assert names == {"only.bin"}


def test_a_linked_entry_is_reported_as_a_link_not_a_directory(tmp_path: Path) -> None:
    """The property every consumer depends on.

    A junction carries the directory attribute AND the reparse attribute, so a
    classifier that asks "is it a directory" first calls it a directory - and a
    sweep that then recurses or deletes follows the link out of the store. The
    kind must come from the reparse bit being checked first.
    """

    root = tmp_path / "store"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")
    (root / "ordinary").mkdir()

    kinds = _kinds(root)

    assert kinds["redirect"] is AnchoredEntryKind.LINK
    assert kinds["ordinary"] is AnchoredEntryKind.DIRECTORY
    assert (outside / "victim.bin").read_bytes() == b"not ours"


def test_a_link_is_not_safe_and_an_ordinary_entry_is(tmp_path: Path) -> None:
    """`is_safe` is what consumers branch on, so it is pinned separately.

    Classifying correctly and then exposing a predicate that says yes to a
    link would leave every caller correct in theory and wrong in practice.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "keep.bin").write_bytes(b"x")
    outside = tmp_path / "outside"
    outside.mkdir()
    if not _make_link_dir(root / "redirect", outside):
        pytest.skip("this host does not permit directory links")

    with AnchoredDirectory(root) as anchor:
        entries = {entry.name: entry for entry in list_entries(anchor)}

    assert entries["keep.bin"].is_safe
    assert not entries["redirect"].is_safe


def test_more_entries_than_the_bound_refuses(tmp_path: Path) -> None:
    """An unbounded listing is unbounded work for every caller of it."""

    root = tmp_path / "store"
    root.mkdir()
    for index in range(6):
        (root / f"entry-{index}.bin").write_bytes(b"x")

    with AnchoredDirectory(root) as anchor:
        assert len(list_entries(anchor, limit=6)) == 6
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=5)


def test_a_zero_or_negative_bound_refuses(tmp_path: Path) -> None:
    """Zero is not "no limit"; it is a caller that has lost track of one."""

    root = tmp_path / "store"
    root.mkdir()

    with AnchoredDirectory(root) as anchor:
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=0)
        with pytest.raises(AnchoredDirectoryError):
            list_entries(anchor, limit=-1)


def test_a_closed_anchor_refuses_rather_than_answering(tmp_path: Path) -> None:
    """Closing releases the guarantee, so the answer must go with it.

    A listing taken through a released anchor is a listing by path again, and
    it would look identical to a contained one.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "present.bin").write_bytes(b"x")

    anchor = AnchoredDirectory(root)
    assert list_entries(anchor)
    anchor.close()

    with pytest.raises(AnchoredDirectoryError):
        list_entries(anchor)


def test_a_redirected_directory_cannot_be_anchored_or_listed(tmp_path: Path) -> None:
    """Acquisition refuses first, so nothing outside is ever enumerated."""

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.bin").write_bytes(b"not ours")
    redirected = tmp_path / "store"
    if not _make_link_dir(redirected, outside):
        pytest.skip("this host does not permit directory links")

    with (
        pytest.raises(AnchoredDirectoryError),
        AnchoredDirectory(redirected) as anchor,
    ):
        list_entries(anchor)


def test_an_empty_directory_lists_as_empty_rather_than_refusing(tmp_path: Path) -> None:
    """Absence of entries is an answer; a refusal here would be a lie.

    On Windows an empty directory still yields `.` and `..`, so this also
    proves those two are filtered rather than merely absent from the fixtures.
    """

    root = tmp_path / "store"
    root.mkdir()

    with AnchoredDirectory(root) as anchor:
        assert list_entries(anchor) == ()


def test_an_entry_record_cannot_be_edited_after_it_is_read(tmp_path: Path) -> None:
    """The kind is evidence from the kernel, not a field a caller can set."""

    entry = AnchoredEntry("name.bin", AnchoredEntryKind.LINK)

    with pytest.raises((AttributeError, TypeError)):
        entry.kind = AnchoredEntryKind.FILE  # type: ignore[misc]
    assert entry.kind is AnchoredEntryKind.LINK
    assert not entry.is_safe


def test_every_dirent_type_maps_without_touching_the_filesystem() -> None:
    """The POSIX classification is a pure function of one integer.

    That is the point of it. `DirEntry.is_dir` and its siblings answer from the
    record when the filesystem supplied a type and otherwise stat by name, and
    the name is exactly what must not be consulted twice. A mapping that takes
    an int cannot do that, and this test is what says so.
    """

    assert _kind_from_dirent_type(8) is AnchoredEntryKind.FILE
    assert _kind_from_dirent_type(4) is AnchoredEntryKind.DIRECTORY
    assert _kind_from_dirent_type(10) is AnchoredEntryKind.LINK
    assert _kind_from_dirent_type(0) is AnchoredEntryKind.UNKNOWN
    for other in (1, 2, 6, 12, 14):
        assert _kind_from_dirent_type(other) is AnchoredEntryKind.OTHER


def test_an_unknown_dirent_type_stays_unknown_with_no_second_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case no filesystem in CI can produce, supplied directly.

    ext4, xfs and tmpfs all fill in d_type, so a real directory cannot exercise
    DT_UNKNOWN here - which is precisely why the earlier version of this
    primitive passed CI while still being able to fall back. The record is
    injected instead, and every route back to the filesystem is armed to fail:
    if anything resolves that name a second time, this test raises rather than
    quietly returning the right answer for the wrong reason.
    """

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("classification performed a second lookup by name")

    monkeypatch.setattr(filesystem_links, "_read_dirents", lambda _fd, _limit: [("entry.bin", 0)])
    monkeypatch.setattr(os, "stat", refuse)
    monkeypatch.setattr(os, "lstat", refuse)
    monkeypatch.setattr(os, "scandir", refuse)

    entries = _list_posix(-1, 8)

    assert [(entry.name, entry.kind) for entry in entries] == [
        ("entry.bin", AnchoredEntryKind.UNKNOWN)
    ]
    assert not entries[0].is_safe


def test_listing_the_same_anchor_twice_gives_the_same_answer(tmp_path: Path) -> None:
    """Found by CI, not by reasoning, and worth its own regression.

    The first version enumerated through a `dup`, which shares its file offset
    with the descriptor it copies. One listing left that shared offset at end
    of directory, so every later listing returned an empty tuple - no error, no
    warning, just nothing. A caller that lists, acts on what it found, and
    lists again would conclude the directory had been emptied by its own work.
    """

    root = tmp_path / "store"
    root.mkdir()
    for index in range(3):
        (root / f"entry-{index}.bin").write_bytes(b"x")

    with AnchoredDirectory(root) as anchor:
        first = sorted(entry.name for entry in list_entries(anchor))
        second = sorted(entry.name for entry in list_entries(anchor))
        third = sorted(entry.name for entry in list_entries(anchor))

    assert first == ["entry-0.bin", "entry-1.bin", "entry-2.bin"]
    assert second == first
    assert third == first


def test_listing_composes_with_every_other_operation(tmp_path: Path) -> None:
    """Every planned consumer lists, acts, and lists again.

    Catalogue prune, the artifact orphan sweep and install activation all have
    that shape, so a listing that works once and then reports a stale or empty
    directory would be worse than no listing at all - each of them would
    conclude its own work had emptied the store. This walks the whole surface
    through one anchor and requires each listing to show what actually
    happened.
    """

    root = tmp_path / "store"
    root.mkdir()
    (root / "kept.bin").write_bytes(b"one")

    with AnchoredDirectory(root) as anchor:
        assert sorted(e.name for e in list_entries(anchor)) == ["kept.bin"]

        descriptor = create_entry(anchor, "staged.bin")
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"two")
        assert sorted(e.name for e in list_entries(anchor)) == [
            "kept.bin",
            "staged.bin",
        ]

        child = open_child_directory(anchor, "nested", create=True)
        child.close()
        rename_entry(anchor, "staged.bin", "published.bin", replace=True)
        assert sorted(e.name for e in list_entries(anchor)) == [
            "kept.bin",
            "nested",
            "published.bin",
        ]

        remove_entry(anchor, "published.bin")
        final = {e.name: e.kind for e in list_entries(anchor)}

    assert final == {
        "kept.bin": AnchoredEntryKind.FILE,
        "nested": AnchoredEntryKind.DIRECTORY,
    }


def test_a_name_this_layer_cannot_represent_refuses_rather_than_raising(
    tmp_path: Path,
) -> None:
    """A POSIX filename may be any bytes; this module answers in one way.

    Under `surrogateescape` an invalid-UTF-8 name becomes a lone surrogate,
    which cannot be encoded as UTF-16 - so measuring it for the NT length bound
    raised a raw UnicodeEncodeError, escaping the module in place of its fixed
    refusal. The enumeration now decodes strictly and the bound answers
    "too long" for anything it cannot measure, so both routes refuse.
    """

    unrepresentable = b"caf\xe9.bin".decode("utf-8", errors="surrogateescape")

    with pytest.raises(AnchoredDirectoryError):
        _require_entry_name(unrepresentable)


def test_an_unencodable_name_refuses_before_any_native_length_is_built() -> None:
    """The measurement refuses; it does not answer a number.

    `_utf16_length` has three callers and only one is the name validator.
    `_nt_try_open_relative` forwards its result straight into
    UNICODE_STRING.Length and MaximumLength, and `_nt_set_name` into
    FileNameLength - and `_walk_windows` sends path components to that native
    open WITHOUT going through the validator first. A sentinel would therefore
    be handed to the object manager as a real buffer length for a name it
    cannot hold, which is the truncation this helper exists to prevent.
    """

    unencodable = b"caf\xe9.bin".decode("utf-8", errors="surrogateescape")

    with pytest.raises(AnchoredDirectoryError):
        _utf16_length(unencodable)
    with pytest.raises(AnchoredDirectoryError):
        _require_entry_name(unencodable)


def test_a_directory_record_that_cannot_be_decoded_refuses() -> None:
    """The Windows mirror of the POSIX strict-decode refusal.

    Built as a raw FILE_DIRECTORY_INFORMATION record rather than by planting a
    file, because a name whose bytes are an unpaired surrogate cannot be
    created through the filesystem at all - which is exactly why the parser has
    to be pinned directly.
    """

    name = b"\x00\xd8"  # a lone high surrogate in UTF-16-LE
    record = bytearray(_FILE_DIRECTORY_INFORMATION_HEADER + len(name))
    record[0:4] = (0).to_bytes(4, "little")  # NextEntryOffset: last record
    record[16:18] = (len(record)).to_bytes(2, "little")  # d_reclen equivalent
    record[56:60] = (0).to_bytes(4, "little")  # FileAttributes
    record[60:64] = (len(name)).to_bytes(4, "little")  # FileNameLength
    record[_FILE_DIRECTORY_INFORMATION_HEADER:] = name

    with pytest.raises(AnchoredDirectoryError):
        _read_directory_records(bytes(record), 8, 0)


def test_a_surrogate_path_component_never_reaches_the_native_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unencodable component never reaches the native open.

    `_walk_windows` hands each path component straight to the native open
    without going through `_require_entry_name`, so covering only the validator
    left this route open. Asserting that acquisition raises proves nothing on
    its own: a sentinel length also fails, just later and inside the object
    manager, and it raises the same exception type on the way out.

    What distinguishes a fixed refusal from a bogus buffer length is whether
    the native call is reached at all. This records every name handed to
    `_nt_try_open_relative` and requires the unencodable component to be absent
    from that list.
    """

    if os.name != "nt":
        pytest.skip("the native open exists only on Windows")

    seen: list[str] = []
    original = filesystem_links._nt_try_open_relative

    def spy(parent: int | None, name: str, *, intent: str) -> tuple[int, int]:
        seen.append(name)
        return original(parent, name, intent=intent)

    monkeypatch.setattr(filesystem_links, "_nt_try_open_relative", spy)
    component = b"caf\xe9".decode("utf-8", errors="surrogateescape")

    with pytest.raises(AnchoredDirectoryError):
        AnchoredDirectory(tmp_path / component / "child")

    assert component not in seen, "the unencodable name reached the native open"
