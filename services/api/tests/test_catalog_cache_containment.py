"""The catalog cache prune deletes only what this store could have written.

The pass used to name every entry four times over - `iterdir`, a link check, a
stat, then an unlink - and each lookup after the first reopened the window the
pass has to be safe across. It also decided eligibility on the suffix alone, so
anything ending `.json` in that directory was inside its budget, and it never
established that the root it was sweeping was the cache at all.

These pin the properties that follow from holding the root: the root is
verified once and held, a kind that is not an ordinary file is never deleted,
and an entry whose age could not be established is skipped rather than
defaulted.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from local_lm.catalog_cache import (
    CatalogCachePolicy,
    CatalogCacheStore,
    is_cache_name,
    is_partial_name,
)
from local_lm.filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    AnchoredEntry,
    AnchoredEntryKind,
    remove_entry,
)

KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_C = "c" * 64


def _make_link_dir(link: Path, target: Path) -> bool:
    """Create a directory-shaped redirection, or False without privileges."""

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


def _age(path: Path, seconds: float) -> None:
    when = time.time() - seconds
    os.utime(path, (when, when))


def _store(
    root: Path,
    *,
    fresh_seconds: float | None = None,
    stale_seconds: float | None = None,
    partial_seconds: float | None = None,
    max_entries: int | None = None,
    max_bytes: int | None = None,
) -> CatalogCacheStore:
    """A store overriding only the policy fields a test names.

    Spelled out rather than forwarded as `**kwargs`: the policy mixes float
    seconds with integer counts, one kwargs annotation cannot be both, and
    widening it to `Any` would silence a distinction that is real - a byte
    budget is not a duration.
    """

    defaults = CatalogCachePolicy()
    return CatalogCacheStore(
        root,
        CatalogCachePolicy(
            fresh_seconds=defaults.fresh_seconds if fresh_seconds is None else fresh_seconds,
            stale_seconds=defaults.stale_seconds if stale_seconds is None else stale_seconds,
            partial_seconds=(
                defaults.partial_seconds if partial_seconds is None else partial_seconds
            ),
            max_entries=defaults.max_entries if max_entries is None else max_entries,
            max_bytes=defaults.max_bytes if max_bytes is None else max_bytes,
        ),
    )


def test_a_redirected_cache_root_is_refused_and_its_target_untouched(
    tmp_path: Path,
) -> None:
    """The root itself was never verified before anything was deleted.

    Point the configured cache directory at somebody else's directory and the
    old pass swept it: it read the name, found stale files, and deleted them.
    Holding the root means a link on the way to it refuses before the first
    entry is named.
    """

    target = tmp_path / "not-the-cache"
    target.mkdir()
    victim = target / f"{KEY_A}.json"
    victim.write_text("theirs", encoding="utf-8")
    _age(victim, 10_000)
    link = tmp_path / "cache"
    if not _make_link_dir(link, target):
        pytest.skip("this host does not permit directory links")

    _store(link, stale_seconds=60).prune()

    assert victim.read_text(encoding="utf-8") == "theirs"
    assert sorted(entry.name for entry in target.iterdir()) == [f"{KEY_A}.json"]


def test_a_link_inside_the_cache_is_neither_followed_nor_deleted(
    tmp_path: Path,
) -> None:
    """A link named exactly like a cache entry is the adversarial case.

    Its name passes every shape check this store applies, so the only thing
    standing between the pass and another directory is the kind, and the kind
    has to come from the enumeration record.
    """

    root = tmp_path / "cache"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.bin").write_bytes(b"not ours")
    disguised = root / f"{KEY_A}.json"
    if not _make_link_dir(disguised, outside):
        pytest.skip("this host does not permit directory links")
    _age(disguised, 10_000)

    _store(root, stale_seconds=60).prune()

    assert (outside / "victim.bin").read_bytes() == b"not ours"
    assert disguised.exists()


def test_a_directory_is_never_deleted_even_when_it_is_stale(tmp_path: Path) -> None:
    """DIRECTORY carries metadata, so only the kind check refuses it."""

    root = tmp_path / "cache"
    root.mkdir()
    nested = root / f"{KEY_A}.json"
    nested.mkdir()
    (nested / "inner").write_text("kept", encoding="utf-8")
    _age(nested, 10_000)

    _store(root, stale_seconds=60).prune()

    assert (nested / "inner").read_text(encoding="utf-8") == "kept"


def test_only_names_this_store_could_have_written_are_pruned(tmp_path: Path) -> None:
    """Deciding on the suffix put every neighbouring `.json` in the budget."""

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60)
    ours = store.path(KEY_A)
    ours.write_text("ours", encoding="utf-8")
    theirs = root / "settings.json"
    theirs.write_text("theirs", encoding="utf-8")
    short = root / f"{'a' * 63}.json"
    short.write_text("near miss", encoding="utf-8")
    for path in (ours, theirs, short):
        _age(path, 10_000)

    store.prune()

    assert not ours.exists()
    assert theirs.read_text(encoding="utf-8") == "theirs"
    assert short.read_text(encoding="utf-8") == "near miss"


def test_an_uppercase_key_is_not_a_name_this_store_writes(tmp_path: Path) -> None:
    """Split from the sweep because the filesystem, not the code, decides.

    `path` refuses an uppercase key outright, so no such entry can be written
    through this store. On a case-insensitive filesystem the name still
    RESOLVES to a lowercase entry, which makes an on-disk assertion a fact
    about the host rather than about the predicate. The predicate is what this
    store controls, so the predicate is what is pinned.
    """

    assert not is_cache_name(f"{'A' * 64}.json")
    with pytest.raises(ValueError, match="hexadecimal"):
        CatalogCacheStore(tmp_path).path("A" * 64)


def test_a_leftover_partial_goes_and_an_unrelated_one_stays(tmp_path: Path) -> None:
    """Only the shape `_atomic_write` actually stages.

    That method uses `NamedTemporaryFile(prefix=f".{key}-",
    suffix=".partial")`, so the name carries the key. Matching on the dot and
    the suffix alone accepted `.download.partial` - a file this store could not
    have written - which is the same suffix-only reasoning this module removed
    from the cache branch, left in place one function below it.

    The dot-prefixed near miss is the case that matters. Without it a test can
    pass while the predicate is still wrong, because `download.partial` fails
    on the leading dot for a reason that has nothing to do with the key.
    """

    root = tmp_path / "cache"
    root.mkdir()
    leftover = root / f".{KEY_A}-abcd.partial"
    leftover.write_text("staged", encoding="utf-8")
    survivors = {
        "download.partial": "no leading dot",
        ".download.partial": "dotted, but carries no key",
        f".{KEY_A}.partial": "the key, but no temporary name",
        f".{'g' * 64}-abcd.partial": "the right shape, but not hexadecimal",
    }
    for name in survivors:
        (root / name).write_text(survivors[name], encoding="utf-8")
    for path in (leftover, *(root / name for name in survivors)):
        _age(path, 10_000)

    _store(root, partial_seconds=60).prune()

    assert not leftover.exists()
    for name, contents in survivors.items():
        assert (root / name).read_text(encoding="utf-8") == contents, name


def test_a_refused_removal_stays_in_the_entry_budget(tmp_path: Path) -> None:
    """The count budget must not be satisfied by a delete that did not happen.

    `_enforce_budget` used to drop an entry from its inventory and then attempt
    the delete. A refusal left the file in the directory while `len(kept)`
    fell, so a count-only overflow could end the loop reporting itself resolved
    with the cache still over budget. The existing refusal test uses a BYTE
    ceiling, where the unchanged total keeps the loop alive and hides this.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=10_000, max_entries=1, max_bytes=10**9)
    oldest, middle, newest = (store.path(key) for key in (KEY_A, KEY_B, KEY_C))
    for index, target in enumerate((oldest, middle, newest)):
        target.write_text("1234", encoding="utf-8")
        _age(target, 300 - index)

    def refuse_the_oldest(anchor: AnchoredDirectory, name: str) -> None:
        if name == oldest.name:
            raise AnchoredDirectoryError("refused")
        remove_entry(anchor, name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("local_lm.catalog_cache.remove_entry", refuse_the_oldest)
    try:
        store.prune()
    finally:
        monkeypatch.undo()

    # The refusal is honest about itself: the file is still there and still
    # counted, so the pass keeps going and clears everything it actually can.
    assert oldest.read_text(encoding="utf-8") == "1234"
    assert not middle.exists()
    assert not newest.exists()


def test_a_refused_stale_removal_still_counts_toward_the_budget(
    tmp_path: Path,
) -> None:
    """`_sweep_by_age` lost a refused entry from the inventory entirely.

    A stale entry whose removal refuses is still in the directory. Omitting it
    from `kept` meant the budget pass could not see it, so the cache could sit
    over `max_entries` with the pass believing it was under.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60, max_entries=1, max_bytes=10**9)
    stale, fresh = store.path(KEY_A), store.path(KEY_B)
    stale.write_text("1234", encoding="utf-8")
    _age(stale, 10_000)
    fresh.write_text("1234", encoding="utf-8")
    _age(fresh, 1)

    def refuse_the_stale(anchor: AnchoredDirectory, name: str) -> None:
        if name == stale.name:
            raise AnchoredDirectoryError("refused")
        remove_entry(anchor, name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("local_lm.catalog_cache.remove_entry", refuse_the_stale)
    try:
        store.prune()
    finally:
        monkeypatch.undo()

    # Two entries against a budget of one, and the stale one cannot go - so the
    # fresh one must, which only happens if the refused entry was still counted.
    assert stale.read_text(encoding="utf-8") == "1234"
    assert not fresh.exists()


def test_an_entry_whose_age_could_not_be_established_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`has_metadata` False is an answer, not a default of zero.

    A safe entry that vanished or refused reacquisition between the
    enumeration and the measurement carries no size and no time. Treating that
    as age zero would keep it forever; treating it as very old would delete
    something nothing ever measured. It is skipped, and the next pass decides.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60, max_entries=0, max_bytes=0)
    path = store.path(KEY_A)
    path.write_text("unmeasured", encoding="utf-8")
    _age(path, 10_000)
    monkeypatch.setattr(
        "local_lm.catalog_cache.list_entries",
        lambda anchor, limit: (AnchoredEntry(name=path.name, kind=AnchoredEntryKind.FILE),),
    )

    store.prune()

    assert path.read_text(encoding="utf-8") == "unmeasured"


def test_the_record_supplies_the_age_the_pass_decides_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enumeration record is the source, not a second stat by name.

    The file on disk is fresh; only the record says it is old. If the pass
    still deletes it, the age it acted on came from the record - which is what
    removes the window between deciding and deleting.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60)
    path = store.path(KEY_A)
    path.write_text("fresh on disk", encoding="utf-8")
    stale = datetime.fromtimestamp(time.time() - 10_000, tz=UTC)
    monkeypatch.setattr(
        "local_lm.catalog_cache.list_entries",
        lambda anchor, limit: (
            AnchoredEntry(
                name=path.name,
                kind=AnchoredEntryKind.FILE,
                size_bytes=13,
                modified_at=stale,
            ),
        ),
    )

    store.prune()

    assert not path.exists()


def test_the_protected_entry_survives_both_passes(tmp_path: Path) -> None:
    """The entry being written must not be pruned by its own write."""

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60, max_entries=1, max_bytes=4)
    protected = store.path(KEY_C)
    for key in (KEY_A, KEY_B, KEY_C):
        target = store.path(key)
        target.write_text("1234", encoding="utf-8")
        _age(target, 10_000)

    store.prune(protected=protected)

    assert protected.read_text(encoding="utf-8") == "1234"
    assert not store.path(KEY_A).exists()
    assert not store.path(KEY_B).exists()


def test_a_protected_path_outside_the_root_protects_nothing(tmp_path: Path) -> None:
    """Comparing bare names would let an unrelated path shield a real entry."""

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60)
    entry = store.path(KEY_A)
    entry.write_text("stale", encoding="utf-8")
    _age(entry, 10_000)

    store.prune(protected=tmp_path / "elsewhere" / f"{KEY_A}.json")

    assert not entry.exists()


def test_the_enumeration_ceiling_stays_above_the_store_budget(tmp_path: Path) -> None:
    """`list_entries` refuses rather than truncating.

    A ceiling at or below `max_entries` would make the pass do nothing exactly
    when the directory had outgrown its budget, which is when it is needed.
    """

    root = tmp_path / "cache"
    root.mkdir()

    assert _store(root, max_entries=512)._listing_limit() == 8192
    assert _store(root, max_entries=4000)._listing_limit() == 16000
    assert _store(root, max_entries=100_000)._listing_limit() == 400_000


def test_a_missing_cache_root_leaves_the_caller_alone(tmp_path: Path) -> None:
    """Pruning is best-effort; a write must not fail because of it."""

    _store(tmp_path / "never-created").prune()


def test_the_pruned_shape_is_exactly_what_path_produces(tmp_path: Path) -> None:
    """The write shape and the delete shape are one predicate, not two."""

    store = CatalogCacheStore(tmp_path)
    for suffix in (".json", ".bin"):
        assert is_cache_name(store.path(KEY_A, suffix=suffix).name)
    for rejected in (
        f"{KEY_A}.txt",
        f"{'a' * 63}.json",
        f"{'a' * 65}.json",
        f"{'g' * 64}.json",
        KEY_A,
        ".json",
        "",
    ):
        assert not is_cache_name(rejected)
    assert is_partial_name(f".{KEY_A}-1234.partial")
    assert not is_partial_name(f"{KEY_A}.partial")
    assert not is_partial_name(".notpartial")


def test_a_refused_removal_does_not_count_against_the_byte_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subtracting bytes that never left would end the pass early.

    The budget loop stops when it believes it is under the ceiling. A removal
    that refused and was counted anyway leaves the directory over budget while
    the pass reports itself finished, and nothing revisits it until the next
    write happens to come along.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=10_000, max_entries=8, max_bytes=4)
    older, newer = store.path(KEY_A), store.path(KEY_B)
    for index, target in enumerate((older, newer)):
        target.write_text("1234", encoding="utf-8")
        _age(target, 100 - index)

    def refuse_the_older(anchor: AnchoredDirectory, name: str) -> None:
        if name == older.name:
            raise AnchoredDirectoryError("refused")
        remove_entry(anchor, name)

    monkeypatch.setattr("local_lm.catalog_cache.remove_entry", refuse_the_older)

    store.prune()

    assert older.read_text(encoding="utf-8") == "1234"
    assert not newer.exists()


@pytest.mark.parametrize(
    "kind",
    [AnchoredEntryKind.LINK, AnchoredEntryKind.UNKNOWN, AnchoredEntryKind.OTHER],
)
def test_an_unsafe_kind_is_refused_even_when_it_arrives_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: AnchoredEntryKind
) -> None:
    """The kind check is driven directly, because nothing else falsifies it.

    Removing it from the sweep and running every other test here changes
    nothing, and the reason is worth stating: `list_entries` never attaches a
    size or a time to an unsafe kind, so those entries are already skipped for
    having no age - and a directory survives only because unlinking one fails.
    Two unrelated mechanisms were standing in for the guard, and a test suite
    that cannot tell the guard from its understudies is not testing the guard.

    So the enumeration is made to hand back an unsafe kind that IS measured,
    which is the one shape those other mechanisms do not cover. This module's
    contract is that it prunes ordinary files and nothing else, and that has to
    hold whatever the enumeration decides to attach in future.
    """

    root = tmp_path / "cache"
    root.mkdir()
    store = _store(root, stale_seconds=60)
    path = store.path(KEY_A)
    path.write_text("not an ordinary file", encoding="utf-8")
    measured = AnchoredEntry(
        name=path.name,
        kind=kind,
        size_bytes=4,
        modified_at=datetime.fromtimestamp(time.time() - 10_000, tz=UTC),
    )
    monkeypatch.setattr(
        "local_lm.catalog_cache.list_entries",
        lambda anchor, limit: (measured,),
    )

    store.prune()

    assert path.read_text(encoding="utf-8") == "not an ordinary file"
