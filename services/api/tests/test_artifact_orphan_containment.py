"""The orphan sweep deletes only what this store could have left behind.

The pass used to name every candidate four times over - `iterdir`, `is_file`, a
link check, a `stat`, then an `unlink` - and each lookup after the first
reopened the window the pass has to be safe across. A name that was an ordinary
file when it was checked could be a link by the time it was unlinked, and the
unlink would have followed it out of the store.

These pin the properties that follow from holding the root and each shard: the
kind comes from the record that named the entry, an entry whose age could not
be established is skipped rather than defaulted, and a directory this pass did
not empty is left where it is.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    list_entries,
    remove_entry,
)

DIGEST_A = "aabb" + "0" * 60
DIGEST_B = "aabb" + "1" * 60
DIGEST_C = "ccdd" + "2" * 60


def _make_link_dir(link: Path, target: Path) -> bool:
    """Point `link` at a DIRECTORY, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(  # noqa: S603 - fixed argv, test-local paths
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return False
    return True


def _make_link_file(link: Path, target: Path) -> bool:
    """Point `link` at a FILE, or report that this host will not allow it."""

    if os.name == "nt":
        completed = subprocess.run(  # noqa: S603 - fixed argv, test-local paths
            ["cmd", "/c", "mklink", str(link), str(target)],  # noqa: S607
            capture_output=True,
            check=False,
        )
        return completed.returncode == 0
    try:
        os.symlink(target, link)
    except OSError:
        return False
    return True


def _age(path: Path, hours: float) -> None:
    when = time.time() - hours * 3600
    os.utime(path, (when, when))


def _write_aged(path: Path, payload: bytes, *, hours: float = 25) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    _age(path, hours)
    return path


@pytest.fixture
def store_session(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session, Path]]:
    root = tmp_path / "artifacts"
    root.mkdir()
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'artifacts.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield ArtifactStore(settings, root=root), session, root
    finally:
        session.close()
        engine.dispose()


def _sweep(store: ArtifactStore, session: Session) -> tuple[int, int]:
    summary = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=False,
    )
    session.commit()
    return summary.removed_count, summary.reclaimed_bytes


def test_an_ordinary_sweep_still_reclaims_what_it_should(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """The control. Without it every refusal below could be a broken sweep."""

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"orphaned")
    temporary = _write_aged(root / "ingest-abandoned", b"ingest")

    removed, reclaimed = _sweep(store, session)

    assert removed == 2
    assert reclaimed == len(b"orphaned") + len(b"ingest")
    assert not orphan.exists()
    assert not temporary.exists()
    assert not (root / "aa").exists()


def test_a_link_named_like_an_orphan_is_neither_followed_nor_deleted(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """The adversarial case: a name that passes every shape check this pass applies.

    Sharded where it claims to belong, unindexed, and old enough. The only
    thing between the sweep and another directory is the KIND, and the kind has
    to come from the enumeration record rather than a second lookup by name.
    """

    store, session, root = store_session
    outside = root.parent / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    disguised = root / "aa" / "bb" / DIGEST_A
    disguised.parent.mkdir(parents=True)
    if not _make_link_dir(disguised, outside):
        pytest.skip("this host does not permit directory links")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert (outside / "victim.bin").read_bytes() == b"not ours"
    assert disguised.exists()


def test_a_link_named_like_a_temporary_is_neither_followed_nor_deleted(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """The same case one level up, where the name shape is far weaker.

    Anything beginning `ingest-` in the root is eligible on its name alone, so
    the kind is the whole of the protection here.
    """

    store, session, root = store_session
    victim = root.parent / "victim.bin"
    victim.write_bytes(b"not ours")
    _age(victim, 25)
    disguised = root / "ingest-disguised"
    if not _make_link_file(disguised, victim):
        pytest.skip("this host does not permit file links")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert victim.read_bytes() == b"not ours"


def test_a_shard_that_is_a_link_is_never_descended_into(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """A redirected shard would put the whole walk inside somebody else's tree."""

    store, session, root = store_session
    outside = root.parent / "outside"
    _write_aged(outside / "bb" / DIGEST_A, b"not ours")
    if not _make_link_dir(root / "aa", outside):
        pytest.skip("this host does not permit directory links")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert (outside / "bb" / DIGEST_A).read_bytes() == b"not ours"


def test_a_digest_sharded_somewhere_else_is_left_alone(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """`_destination` could not have written it there, so this pass did not either."""

    store, session, root = store_session
    misplaced = _write_aged(root / "aa" / "bb" / DIGEST_C, b"misplaced")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert misplaced.read_bytes() == b"misplaced"


def test_a_directory_named_like_a_digest_is_never_deleted(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """DIRECTORY carries metadata too, so only the kind check refuses it."""

    store, session, root = store_session
    nested = root / "aa" / "bb" / DIGEST_A
    nested.mkdir(parents=True)
    (nested / "inner").write_bytes(b"kept")
    _age(nested, 25)

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert (nested / "inner").read_bytes() == b"kept"


def test_a_fresh_orphan_and_a_fresh_partial_both_survive(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """Age is the whole decision for a partial, and half of it for an orphan."""

    store, session, root = store_session
    fresh_orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"recent", hours=1)
    fresh_partial = _write_aged(
        root / "aa" / "bb" / f"{DIGEST_A}.restore-partial", b"restoring", hours=1
    )
    aged_partial = _write_aged(root / "aa" / "bb" / f"{DIGEST_B}.restore-partial", b"stale")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (1, len(b"stale"))
    assert fresh_orphan.read_bytes() == b"recent"
    assert fresh_partial.read_bytes() == b"restoring"
    assert not aged_partial.exists()


def test_a_shard_this_pass_did_not_empty_keeps_its_directory(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """Removing the parent is a consequence of emptying it, never an assumption."""

    store, session, root = store_session
    indexed = store.ingest_bytes(
        session,
        b"still referenced",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name="kept.png",
    )
    session.commit()
    kept = store.resolve(indexed)
    _age(kept, 25)
    orphan = _write_aged(kept.parent / f"{kept.name[:4]}{'e' * 60}", b"orphaned")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (1, len(b"orphaned"))
    assert not orphan.exists()
    assert kept.read_bytes() == b"still referenced"
    assert kept.parent.is_dir()


def test_a_deliberately_linked_root_is_resolved_once_and_still_swept(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """A configured redirected store root is supported and must still be swept.

    `ArtifactStore.__init__` resolves the root once at construction, which is
    what makes holding it possible at all: by the time the anchor is taken
    there is no redirection left on the way to it. Pinned here because this
    pass depends on that and would otherwise refuse a supported store.
    """

    _store, session, root = store_session
    target = root.parent / "elsewhere"
    target.mkdir()
    link = root.parent / "linked-root"
    if not _make_link_dir(link, target):
        pytest.skip("this host does not permit directory links")
    settings = Settings(data_dir=root.parent / "data")
    settings.prepare()
    redirected = ArtifactStore(settings, root=link)
    orphan = _write_aged(target / "aa" / "bb" / DIGEST_A, b"orphaned")

    removed, reclaimed = _sweep(redirected, session)

    assert (removed, reclaimed) == (1, len(b"orphaned"))
    assert not orphan.exists()


def test_a_dry_run_counts_without_deleting_anything(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """A dry run reports the same bytes it would have reclaimed."""

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"orphaned")
    temporary = _write_aged(root / "video-proxy-abandoned.mp4", b"proxy")

    summary = store.cleanup_retention(
        session,
        retention_days=30,
        temporary_hours=24,
        dry_run=True,
    )
    session.commit()

    assert summary.removed_count == 2
    assert summary.reclaimed_bytes == len(b"orphaned") + len(b"proxy")
    assert orphan.exists()
    assert temporary.exists()
    assert (root / "aa" / "bb").is_dir()


def _drop_link_dir(link: Path) -> None:
    """Remove a link without removing what it points at."""

    if os.name == "nt":
        os.rmdir(link)
        return
    link.unlink()


def test_a_shard_swapped_after_it_was_named_is_not_followed(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window this pass exists to close, driven rather than raced.

    Every static arrangement above is refused by the shipped sweep too - it
    checked the kind of everything it deleted. What it could not do is keep
    that answer true, because it named each shard again to walk it, and a
    directory that was ordinary when it was checked could be a redirection by
    the time it was opened.

    Both implementations are given the same push at the same point in their own
    sequence: the swap happens immediately before the shard is USED, once,
    after it has already been classified. The shipped sweep walks into the
    victim tree and deletes from it. Holding the shard means the open itself
    refuses, so there is nothing to walk.
    """

    store, session, root = store_session
    outside = root.parent / "outside"
    victim = _write_aged(outside / "bb" / DIGEST_A, b"not ours")
    shard = root / "aa"
    if not _make_link_dir(shard, outside):
        pytest.skip("this host does not permit directory links")
    _drop_link_dir(shard)
    (shard / "bb").mkdir(parents=True)
    shard_resolved = shard.resolve()
    swapped: list[str] = []

    def _swap() -> None:
        swapped.append("once")
        (shard / "bb").rmdir()
        shard.rmdir()
        _make_link_dir(shard, outside)

    real_iterdir = Path.iterdir

    def fake_iterdir(self: Path) -> Iterator[Path]:
        if not swapped and self == shard_resolved:
            _swap()
        return real_iterdir(self)

    def fake_list_entries(anchor: AnchoredDirectory, **kwargs: int) -> tuple[AnchoredEntry, ...]:
        listed = list_entries(anchor, **kwargs)
        if not swapped and anchor.path.resolve() == root.resolve():
            _swap()
        return listed

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    monkeypatch.setattr("local_lm.artifacts.list_entries", fake_list_entries)

    removed, reclaimed = _sweep(store, session)

    assert swapped == ["once"], "the seam never fired, so nothing was measured"
    assert (removed, reclaimed) == (0, 0)
    assert victim.read_bytes() == b"not ours"


def test_an_empty_shard_this_pass_did_not_touch_is_left_where_it_is(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """Removing a directory is a consequence of emptying it, not tidying.

    An empty shard is what an ingest that has chosen its destination and not
    yet written to it looks like. This pass removes the directories it emptied
    itself and leaves every other one alone, which is also what the sweep it
    replaces did.
    """

    store, session, root = store_session
    untouched = root / "cc" / "dd"
    untouched.mkdir(parents=True)
    swept = _write_aged(root / "aa" / "bb" / DIGEST_A, b"orphaned")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (1, len(b"orphaned"))
    assert not swept.exists()
    assert not (root / "aa").exists()
    assert untouched.is_dir()


def test_a_file_whose_age_could_not_be_measured_is_skipped(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measurement this pass does not have is not one it may assume.

    The contract puts size and time together so a consumer has one question to
    ask, and both are absent for an entry that vanished or refused reacquisition
    between the enumeration and the measurement. That entry is an ordinary FILE
    by kind, so the kind check does not cover it and nothing else in this file
    reaches it: a link is refused one step earlier, for a different reason.

    Constructed rather than raced, because the state is a POSIX reacquisition
    losing to a concurrent unlink and there is no way to schedule that.
    """

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"unmeasurable")

    def unmeasured(anchor: AnchoredDirectory, **kwargs: int) -> tuple[AnchoredEntry, ...]:
        return tuple(
            AnchoredEntry(name=entry.name, kind=entry.kind) if entry.name == DIGEST_A else entry
            for entry in list_entries(anchor, **kwargs)
        )

    monkeypatch.setattr("local_lm.artifacts.list_entries", unmeasured)

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert orphan.read_bytes() == b"unmeasurable"


def test_an_unmeasurable_EMPTY_file_is_skipped_rather_than_counted_as_zero(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size that a missing measurement is not allowed to become.

    The test above uses a file with CONTENT, which is why it survived a
    mutation that defaults an absent size to zero: the pass re-measures the
    entry before unlinking it, twelve bytes did not match the defaulted
    zero, and the deletion was refused for the wrong reason. The invariant
    looked pinned and was not.

    An EMPTY file closes it. Zero is exactly the value the default invents,
    so the re-measurement agrees with it and nothing downstream objects -
    the only thing standing between an unmeasurable entry and deletion is
    the skip itself.

    An empty file in a shard is not a curiosity either. It is what a
    publication looks like between the create and the write.
    """

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"")
    assert orphan.stat().st_size == 0, "the point of this test is the zero"

    def unmeasured(anchor: AnchoredDirectory, **kwargs: int) -> tuple[AnchoredEntry, ...]:
        return tuple(
            AnchoredEntry(name=entry.name, kind=entry.kind) if entry.name == DIGEST_A else entry
            for entry in list_entries(anchor, **kwargs)
        )

    monkeypatch.setattr("local_lm.artifacts.list_entries", unmeasured)

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (0, 0)
    assert orphan.is_file(), "an entry with no measurement was deleted anyway"


def test_an_aged_file_this_store_did_not_write_survives_in_the_root(
    store_session: tuple[ArtifactStore, Session, Path],
) -> None:
    """The root is not swept by age; it is swept by name AND age.

    The existing coverage put an unrelated file beside the temporaries but left
    it fresh, so the age test alone was enough to spare it and the name shapes
    were never actually load-bearing. Old enough to delete and not a shape this
    store writes is the case that separates them.
    """

    store, session, root = store_session
    settings_file = _write_aged(root / "settings.json", b"someone else's")
    almost = _write_aged(root / "video-proxy-abandoned.webm", b"wrong suffix")
    temporary = _write_aged(root / "ingest-abandoned", b"ingest")

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (1, len(b"ingest"))
    assert not temporary.exists()
    assert settings_file.read_bytes() == b"someone else's"
    assert almost.read_bytes() == b"wrong suffix"


def test_a_removal_that_refuses_is_not_counted_and_does_not_end_the_pass(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Losing the race safely means reporting it honestly too.

    An entry that was an ordinary file when the record named it and refuses
    removal is no longer one, which is the outcome this pass is built to
    accept. Counting it would report bytes that are still on disk, and letting
    the refusal out would abandon everything already reclaimed - so the pass
    keeps going and the summary describes what actually happened.
    """

    store, session, root = store_session
    refuses = _write_aged(root / "aa" / "bb" / DIGEST_A, b"refuses")
    yields = _write_aged(root / "aa" / "bb" / DIGEST_B, b"yields")
    refuses_temporary = _write_aged(root / "ingest-refuses", b"stays")
    yields_temporary = _write_aged(root / "ingest-yields", b"goes")
    refusing = {DIGEST_A, "ingest-refuses"}

    def selective(anchor: AnchoredDirectory, name: str) -> None:
        if name in refusing:
            raise AnchoredDirectoryError("refused")
        remove_entry(anchor, name)

    monkeypatch.setattr("local_lm.artifacts.remove_entry", selective)

    removed, reclaimed = _sweep(store, session)

    assert (removed, reclaimed) == (2, len(b"yields") + len(b"goes"))
    assert refuses.read_bytes() == b"refuses"
    assert refuses_temporary.read_bytes() == b"stays"
    assert not yields.exists()
    assert not yields_temporary.exists()


def test_a_same_size_replacement_is_still_refused_on_its_age(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolates the AGE half of the re-measurement.

    The general replacement test is satisfied by either guard alone, so a
    mutation battery reported both as unconstrained - each one covered for
    the other. Here the replacement is byte-for-byte the same LENGTH, so
    the size comparison cannot refuse it and only its fresh modification
    time can. This is the realistic shape too: a digest names its content,
    so a genuine republication of the same artifact is exactly this.
    """

    store, session, root = store_session
    original = b"the aged original"
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, original)
    replacement = b"a fresher origin"[: len(original)].ljust(len(original), b"!")
    assert len(replacement) == len(original) and replacement != original
    replaced: list[str] = []
    real_list_entries = list_entries

    def replace_after_listing(
        anchor: AnchoredDirectory, **kwargs: int
    ) -> tuple[AnchoredEntry, ...]:
        listed = real_list_entries(anchor, **kwargs)
        if not replaced and any(entry.name == DIGEST_A for entry in listed):
            replaced.append("once")
            orphan.write_bytes(replacement)
            now = time.time()
            os.utime(orphan, (now, now))
        return listed

    monkeypatch.setattr("local_lm.artifacts.list_entries", replace_after_listing)

    removed, reclaimed = _sweep(store, session)

    assert replaced == ["once"], "the seam never fired, so nothing was measured"
    assert (removed, reclaimed) == (0, 0)
    assert orphan.read_bytes() == replacement


def test_an_aged_replacement_of_another_size_is_refused_on_its_size(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolates the SIZE half of the re-measurement.

    Here the replacement is deliberately OLD, so the age comparison cannot
    refuse it and only the size mismatch can. A restore that puts different
    content under a name while preserving its timestamps has this shape,
    and the reason to refuse is the same either way: what is under the name
    now is not what was measured, so the record says nothing about it.
    """

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"the aged original")
    aged = os.stat(orphan)
    replaced: list[str] = []
    real_list_entries = list_entries

    def replace_after_listing(
        anchor: AnchoredDirectory, **kwargs: int
    ) -> tuple[AnchoredEntry, ...]:
        listed = real_list_entries(anchor, **kwargs)
        if not replaced and any(entry.name == DIGEST_A for entry in listed):
            replaced.append("once")
            orphan.write_bytes(b"a different length entirely, and just as old")
            os.utime(orphan, (aged.st_atime, aged.st_mtime))
        return listed

    monkeypatch.setattr("local_lm.artifacts.list_entries", replace_after_listing)

    removed, reclaimed = _sweep(store, session)

    assert replaced == ["once"], "the seam never fired, so nothing was measured"
    assert (removed, reclaimed) == (0, 0)
    assert orphan.read_bytes() == b"a different length entirely, and just as old"


def test_a_leaf_replaced_after_it_was_measured_is_not_deleted(
    store_session: tuple[ArtifactStore, Session, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window this pass has to survive: the name is resolved again to delete it.

    Holding the shard stabilises the DIRECTORY, not the leaf. Between the
    record that measured an aged orphan and the unlink that removes it,
    `_publish_under_its_digest` can rename a freshly written file over that
    exact digest - `replace=True`, and the Artifact row is flushed afterwards.
    Deleting on the strength of the old record removes the NEW file and leaves
    the publishing transaction pointing at nothing.

    Driven at the same seam the shard-swap test uses: the replacement happens
    once, after the listing has measured the entry and before it is removed.
    """

    store, session, root = store_session
    orphan = _write_aged(root / "aa" / "bb" / DIGEST_A, b"the aged original")
    replaced: list[str] = []
    real_list_entries = list_entries

    def replace_after_listing(
        anchor: AnchoredDirectory, **kwargs: int
    ) -> tuple[AnchoredEntry, ...]:
        listed = real_list_entries(anchor, **kwargs)
        if not replaced and any(entry.name == DIGEST_A for entry in listed):
            replaced.append("once")
            orphan.write_bytes(b"freshly published bytes")
            now = time.time()
            os.utime(orphan, (now, now))
        return listed

    monkeypatch.setattr("local_lm.artifacts.list_entries", replace_after_listing)

    removed, reclaimed = _sweep(store, session)

    assert replaced == ["once"], "the seam never fired, so nothing was measured"
    assert (removed, reclaimed) == (0, 0)
    assert orphan.read_bytes() == b"freshly published bytes"
