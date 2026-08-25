"""A lock is only useful for leases if its ABSENCE is provable.

Two properties, and the second is the subtle one.

The lock must be released when its holder DIES, including on a hard kill -
that is the case a create-only sentinel file gets wrong, and the case a crash
actually produces.

And the proof must be about the right OBJECT. A lock lives on the file object,
not on the directory entry naming it. On POSIX the two separate: unlinking the
name does not close the object, so a later open at the same name creates a
DIFFERENT object that locks independently while the original holder is still
running. Identity is what keeps "I could take it" meaning "nobody holds it".
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from local_lm.filesystem_links import AnchoredDirectory
from local_lm.shared_asset_lock_v1 import (
    LOCK_REPLACED,
    LOCK_UNAVAILABLE,
    SharedAssetLockError,
    current_process_identity,
    entry_identity,
    hold,
    holder_is_gone,
)

LOCK = "lease.lock"

_HOLDER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, sys.argv[1])
    from local_lm.filesystem_links import AnchoredDirectory
    from local_lm.shared_asset_lock_v1 import hold

    with AnchoredDirectory(Path(sys.argv[2])) as anchor:
        with hold(anchor, sys.argv[3]) as (_descriptor, identity):
            print(f"HELD {identity[0]} {identity[1]}", flush=True)
            import time

            time.sleep(300)
    """
)


def _start_holder(tmp_path: Path) -> tuple[subprocess.Popen[str], tuple[int, int]]:
    """Another PROCESS holding the lock, not another descriptor in this one."""

    script = tmp_path / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    package_root = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(
        [sys.executable, str(script), package_root, str(tmp_path), LOCK],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    assert child.stdout is not None
    announced = child.stdout.readline().split()
    assert announced and announced[0] == "HELD", "the holder never took the lock"
    return child, (int(announced[1]), int(announced[2]))


def test_a_second_process_is_refused_while_the_first_holds_it(tmp_path: Path) -> None:
    child, identity = _start_holder(tmp_path)
    try:
        with AnchoredDirectory(tmp_path) as anchor:
            with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK):
                pytest.fail("two processes held the same lock")
            assert str(refusal.value) == LOCK_UNAVAILABLE
            assert not holder_is_gone(anchor, LOCK, expect=identity)
    finally:
        child.kill()
        child.wait(timeout=30)


def test_a_hard_killed_holder_releases_the_lock(tmp_path: Path) -> None:
    """The property the whole module exists for.

    A create-only sentinel file fails exactly here: the file outlives the
    process, so the resource stays locked until somebody deletes it by hand.
    """

    child, identity = _start_holder(tmp_path)
    child.kill()
    child.wait(timeout=30)

    with AnchoredDirectory(tmp_path) as anchor:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if holder_is_gone(anchor, LOCK, expect=identity):
                break
            time.sleep(0.1)
        else:
            pytest.fail("the lock outlived the process that held it")

        with hold(anchor, LOCK):
            pass


def test_replacing_the_entry_cannot_hand_the_lease_to_a_second_holder(
    tmp_path: Path,
) -> None:
    """codex/R1907, across two processes.

    Holder A owns the object. If the directory entry is replaced, a naive
    implementation lets B lock the NEW object and believe the lease is free
    while A is still running. Both questions must answer no.

    On Windows the replacement itself is refused - the open omits delete
    sharing - so the property holds there by a different mechanism, and this
    asserts that mechanism rather than skipping.
    """

    child, identity = _start_holder(tmp_path)
    try:
        entry = tmp_path / LOCK
        try:
            entry.unlink()
            replaced = True
        except OSError:
            replaced = False

        with AnchoredDirectory(tmp_path) as anchor:
            if not replaced:
                assert entry.is_file(), "the entry survived, so nothing was replaced"
                assert not holder_is_gone(anchor, LOCK, expect=identity)
                return

            # POSIX: the name is now free, so a fresh object can be created and
            # locked. Neither of these may report the lease as available.
            assert not holder_is_gone(anchor, LOCK, expect=identity), (
                "a replaced entry reported the original holder as gone"
            )
            with (
                pytest.raises(SharedAssetLockError) as refusal,
                hold(anchor, LOCK, expect=identity),
            ):
                pytest.fail("a second holder entered while the first still owned the lock")
            assert str(refusal.value) == LOCK_REPLACED
    finally:
        child.kill()
        child.wait(timeout=30)


def test_the_identity_a_holder_reports_is_the_object_it_locked(tmp_path: Path) -> None:
    with AnchoredDirectory(tmp_path) as anchor, hold(anchor, LOCK) as (descriptor, identity):
        assert identity == entry_identity(descriptor)
        assert identity == (os.stat(tmp_path / LOCK).st_dev, os.stat(tmp_path / LOCK).st_ino)


def test_the_lock_is_reusable_after_an_ordinary_release(tmp_path: Path) -> None:
    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            assert not holder_is_gone(anchor, LOCK, expect=identity)
        assert holder_is_gone(anchor, LOCK, expect=identity)
        with hold(anchor, LOCK, expect=identity):
            pass


def test_a_raising_block_still_releases_the_lock(tmp_path: Path) -> None:
    """A caller that crashes mid-lease must not wedge the resource."""

    with AnchoredDirectory(tmp_path) as anchor:
        # Established first so the assertion after the crash knows which object
        # it is asking about.
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass
        with pytest.raises(RuntimeError, match="caller exploded"), hold(anchor, LOCK):
            raise RuntimeError("caller exploded")
        assert holder_is_gone(anchor, LOCK, expect=identity)


def test_the_lock_file_is_created_inside_the_held_directory(tmp_path: Path) -> None:
    with AnchoredDirectory(tmp_path) as anchor, hold(anchor, LOCK):
        pass
    assert (tmp_path / LOCK).is_file()
    assert [entry.name for entry in tmp_path.iterdir()] == [LOCK]


def test_process_identity_is_diagnosis_and_says_so() -> None:
    """A pid is recorded to name a holder, never to prove one.

    Pinned because the temptation is to compare a stored pid against the
    running process and call that liveness. Pid numbers are reused, so that
    comparison can say "still held" about an unrelated program.
    """

    pid, interpreter = current_process_identity()

    assert pid == os.getpid()
    assert interpreter == sys.executable


def test_a_replaced_object_is_refused_even_when_nothing_holds_it(tmp_path: Path) -> None:
    """The identity guard itself, exercised without needing a POSIX-only unlink.

    The cross-process test above can only reach the interesting branch on
    POSIX, because Windows refuses to unlink an entry whose object is open. So
    it proves nothing about this logic on Windows, and a guard that is only
    covered on one of two supported platforms is half a guard.

    Here the object is replaced while FREE - which both platforms allow - and
    the point is that a caller naming the old object must still be refused. It
    is not asking "is this name free"; it is asking "is the object I mean
    free", and the answer for an object that no longer has a name is no.
    """

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, first):
            pass

        (tmp_path / LOCK).unlink()
        with hold(anchor, LOCK) as (_descriptor, second):
            pass
        assert first != second, "the replacement must be a different object"

        assert not holder_is_gone(anchor, LOCK, expect=first)
        assert holder_is_gone(anchor, LOCK, expect=second)

        with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK, expect=first):
            pytest.fail("a caller naming the replaced object was allowed in")
        assert str(refusal.value) == LOCK_REPLACED


def test_naming_no_object_answers_the_weaker_question(tmp_path: Path) -> None:
    """Without `expect` these answer "is the name free", which is not the same.

    Pinned so the difference stays visible: a lease MUST pass the identity it
    recorded, and this test exists to show what it gets if it does not.
    """

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, first):
            pass
        (tmp_path / LOCK).unlink()
        with hold(anchor, LOCK) as (_descriptor, second):
            pass

        assert first != second
        # No identity given, so this says yes about a different object.
        assert holder_is_gone(anchor, LOCK)


def test_the_entry_changing_between_open_and_lock_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The window between opening the entry and locking the object.

    On POSIX the name can be replaced in that window, so the object we end up
    holding is no longer the one the name refers to. Windows refuses the
    replacement, so the race is unreachable there - which is exactly why this
    drives it through the seam rather than by racing: a guard that only one of
    two supported platforms can falsify is not being tested.
    """

    import local_lm.shared_asset_lock_v1 as module

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass

        # Whatever we lock, report that the NAME now points somewhere else.
        monkeypatch.setattr(
            module, "_current_entry_identity", lambda _anchor, _name: (identity[0], identity[1] + 1)
        )

        with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK):
            pytest.fail("the lock was held on an object the name no longer refers to")
        assert str(refusal.value) == LOCK_REPLACED

    # And the refusal did not leave the lock held. The patch has to come off
    # first, or this would refuse for the reason under test rather than
    # proving anything about release.
    monkeypatch.undo()
    with AnchoredDirectory(tmp_path) as anchor, hold(anchor, LOCK):
        pass


def test_a_raise_after_acquiring_does_not_strand_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """codex/R1908.

    The post-acquire identity re-check calls os.fstat and can raise. If that
    happens outside the cleanup, the lock is already TAKEN and the descriptor
    is unreachable, so the lease stays unavailable until the process ends -
    the exact failure a released-on-death lock exists to avoid, reintroduced
    by the guard meant to make it safe.
    """

    import local_lm.shared_asset_lock_v1 as module

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass

        def explode(_anchor: object, _name: str) -> tuple[int, int]:
            raise OSError("fstat failed after the lock was taken")

        monkeypatch.setattr(module, "_current_entry_identity", explode)
        with pytest.raises(OSError, match="fstat failed"), hold(anchor, LOCK):
            pytest.fail("the block should not have been entered")

        monkeypatch.undo()
        # The decisive part: the lock must be free again, and free to a
        # DIFFERENT process, not merely to this one holding a stale descriptor.
        assert holder_is_gone(anchor, LOCK, expect=identity)
        with hold(anchor, LOCK, expect=identity):
            pass


def test_a_refusal_before_acquiring_does_not_release_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup must know whether the lock was ever taken.

    Releasing a lock this call never acquired would, on a shared descriptor,
    unlock somebody else's hold. The flag is what keeps `finally` honest.
    """

    import local_lm.shared_asset_lock_v1 as module

    released: list[int] = []
    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass

        monkeypatch.setattr(module, "_take", lambda _descriptor: False)
        monkeypatch.setattr(module, "_release", lambda descriptor: released.append(descriptor))

        with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK):
            pytest.fail("acquisition reported failure, so the block must not run")
        assert str(refusal.value) == LOCK_UNAVAILABLE
        assert released == [], "released a lock that was never acquired"
