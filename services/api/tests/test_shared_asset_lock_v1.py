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

from local_lm.filesystem_links import AnchoredDirectory, create_entry
from local_lm.shared_asset_lock_v1 import (
    _TOKEN_BYTES,
    _TOKEN_OFFSET,
    LOCK_INVALID,
    LOCK_REPLACED,
    LOCK_UNAVAILABLE,
    EntryIdentity,
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
            print(f"HELD {identity[0]} {identity[1]} {identity[2].hex()}", flush=True)
            import time

            time.sleep(300)
    """
)


def _start_holder(tmp_path: Path) -> tuple[subprocess.Popen[str], EntryIdentity]:
    """Another PROCESS holding the lock, not another descriptor in this one."""

    script = tmp_path / "holder.py"
    script.write_text(_HOLDER, encoding="utf-8")
    package_root = str(Path(__file__).resolve().parents[1])
    child = subprocess.Popen(
        [sys.executable, str(script), package_root, str(tmp_path), LOCK],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )
    try:
        assert child.stdout is not None
        announced = child.stdout.readline().split()
        assert announced and announced[0] == "HELD", "the holder never took the lock"
        # The token crosses the process boundary as hex; identity is not
        # identity if the parent reconstructs only two thirds of it.
        return child, (int(announced[1]), int(announced[2]), bytes.fromhex(announced[3]))
    except BaseException:
        # A holder that was started and never handed back is a holder nobody
        # will kill. It keeps the lock, and on Windows it keeps the entry open
        # so the temporary directory cannot be removed - which turns one
        # failure into every later test in the session erroring during
        # teardown. Found exactly that way: a mutation run left three holders
        # alive and every mutation after it read as caught.
        child.kill()
        child.wait(timeout=30)
        raise


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
    """Replacing the entry cannot hand the lease to a second holder.

    Driven across two processes. Holder A owns the object. If the directory
    entry is replaced, a naive implementation lets B lock the NEW object and
    believe the lease is free while A is still running. Both questions must
    answer no.

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
    """What hold() hands back describes the object at that name, all three parts."""

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (descriptor, identity):
            assert identity == entry_identity(descriptor)
            named = os.stat(tmp_path / LOCK)
            assert identity[:2] == (named.st_dev, named.st_ino)
        # The token is in the ENTRY rather than remembered by the holder, so
        # read it back through the name - the only route another process has.
        # After the release, because the Windows lock is mandatory and an
        # ordinary open of a held entry is refused outright there.
        assert (tmp_path / LOCK).read_bytes()[_TOKEN_OFFSET:] == identity[2]
        assert len(identity[2]) == _TOKEN_BYTES


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
        assert first[2] != second[2], "and it is the token that has to differ"

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
            module,
            "_current_entry_identity",
            lambda _anchor, _name: (identity[0], identity[1] + 1, identity[2]),
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
    """A raise after acquiring must not strand the lock.

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


def test_a_recreated_entry_is_distinguishable_when_the_inode_is_reused(
    tmp_path: Path,
) -> None:
    """The Ubuntu failure, pinned as a property rather than as a platform.

    `(st_dev, st_ino)` was the whole identity, and on ext4 the inode number is
    handed straight back to the next file created in the directory: an entry
    and its replacement both measured (2049, 9439958), so the two tests that
    build a replacement failed their own precondition. The measure was wrong,
    not the case.

    This asserts what the guard needs - the identities differ - and separately
    that the token is the part carrying it, so the assertion cannot start
    passing again for the accidental reason it used to pass on Windows. It
    deliberately does not require the device and inode to differ: whether they
    do is the filesystem's business, and the point is that it no longer
    matters.
    """

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, first):
            pass
        (tmp_path / LOCK).unlink()
        with hold(anchor, LOCK) as (_descriptor, second):
            pass

    assert first[2] != second[2]
    assert first != second


def test_the_entry_is_never_created_where_another_process_could_see_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry built in place is visible before it carries a token.

    `create_entry` is exclusive, so only one process creates - but between that
    create and the write the file exists and is empty, and anything opening it
    then reads no token and records an identity that can never match. That is a
    permanently unusable lease rather than a wrong one, which is why it is
    worth a test of its own: it would never show up as a false match.

    The entry is therefore staged under another name and published complete.
    Asserting the final name is never passed to `create_entry` is what
    distinguishes that from building it in place and hoping the gap is small.
    """

    import local_lm.shared_asset_lock_v1 as module

    created: list[str] = []

    def record(anchor: AnchoredDirectory, name: str) -> int:
        # The original comes from its own module rather than being read back
        # off the module under test. `--strict` refuses the latter: the name is
        # imported there and not exported, so it is not part of that module's
        # public surface even though patching it is exactly the point.
        created.append(name)
        return create_entry(anchor, name)

    monkeypatch.setattr(module, "create_entry", record)
    with AnchoredDirectory(tmp_path) as anchor, hold(anchor, LOCK):
        pass

    assert created, "nothing was created, so this proves nothing"
    assert LOCK not in created
    assert all(name.startswith(f"{LOCK}.") for name in created)


def test_publishing_the_entry_leaves_nothing_behind(tmp_path: Path) -> None:
    """Staging is an implementation detail and must not become litter."""

    with AnchoredDirectory(tmp_path) as anchor:
        for _ in range(3):
            with hold(anchor, LOCK):
                pass

    assert sorted(entry.name for entry in tmp_path.iterdir()) == [LOCK]


def test_an_entry_this_module_did_not_write_is_refused(tmp_path: Path) -> None:
    """A file at the lock's name that carries no token is not a lock.

    Reading a short entry and defaulting the missing bytes would put the false
    match straight back: two different foreign files would carry the same empty
    token and compare equal.
    """

    (tmp_path / LOCK).write_bytes(b"not a lock")

    with AnchoredDirectory(tmp_path) as anchor:
        with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK):
            pytest.fail("a foreign file was accepted as a lock entry")
        assert str(refusal.value) == LOCK_INVALID
        assert not holder_is_gone(anchor, LOCK)


def test_the_guard_holds_when_the_filesystem_reuses_the_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ext4's behaviour, reproduced on whichever platform is running.

    This is the case that reached CI rather than a local run. The inode number
    is handed straight back to the next file created in the directory, so an
    entry and its replacement measure the same device and inode - the exact
    pair the Ubuntu leg printed. Windows file IDs are not reused that way, so
    no arrangement of real files on this machine can produce the case, and the
    only honest options were to leave it untested on one platform or to drive
    it through a seam. The numbers are pinned to the ones actually observed.
    """

    import local_lm.shared_asset_lock_v1 as module

    monkeypatch.setattr(module, "_object_numbers", lambda _descriptor: (2049, 9439958))

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, first):
            pass
        (tmp_path / LOCK).unlink()
        with hold(anchor, LOCK) as (_descriptor, second):
            pass

        assert first[:2] == second[:2], "the seam is not reproducing the reuse"
        assert first != second, "identity collapsed to the reused numbers"

        assert not holder_is_gone(anchor, LOCK, expect=first)
        assert holder_is_gone(anchor, LOCK, expect=second)
        with pytest.raises(SharedAssetLockError) as refusal, hold(anchor, LOCK, expect=first):
            pytest.fail("a caller naming the replaced object was allowed in")
        assert str(refusal.value) == LOCK_REPLACED


def test_the_probe_refuses_when_the_name_changes_between_open_and_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe refuses when the entry name changes between open and lock.

    Between opening the entry and locking it, the name can be replaced on
    POSIX. The lock then proves that the object we OPENED is free - and that
    object is detached, unreachable, and of no interest to anybody. The object
    the NAME refers to is the one a caller about to steal the lease will act
    on, and it may be held.

    Driven through the seam rather than by racing, for the same reason the
    equivalent hold() test is: Windows refuses the replacement outright, so a
    real race can only reach this branch on one of the two supported platforms.
    """

    import local_lm.shared_asset_lock_v1 as module

    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass

        # Whatever we lock, report that the NAME now points somewhere else.
        monkeypatch.setattr(
            module,
            "_current_entry_identity",
            lambda _anchor, _name: (identity[0], identity[1] + 1, identity[2]),
        )

        assert not holder_is_gone(anchor, LOCK)
        assert not holder_is_gone(anchor, LOCK, expect=identity)

    # And refusing did not leave the lock held. The patch comes off first, or
    # this would refuse for the reason under test rather than proving release.
    monkeypatch.undo()
    with AnchoredDirectory(tmp_path) as anchor:
        assert holder_is_gone(anchor, LOCK, expect=identity)
        with hold(anchor, LOCK, expect=identity):
            pass


def test_the_probe_refusing_before_acquiring_releases_nothing_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same obligation as hold(), in the helper that answers most often.

    Every way out of `holder_is_gone` before the take - a foreign entry, an
    identity that does not match, and a live holder - reaches the same
    `finally`. Releasing there would be an unlock this call has no claim to,
    and the fact that it usually costs nothing is not a reason to allow it:
    the flag is either honest in both public entry points or in neither.

    Written after a mutation survived. Turning `if acquired` into `if True`
    here changed no test, while the identical mutation in hold() failed one -
    so the guard existed in both places and was evidence in only one.
    """

    import local_lm.shared_asset_lock_v1 as module

    released: list[int] = []
    with AnchoredDirectory(tmp_path) as anchor:
        with hold(anchor, LOCK) as (_descriptor, identity):
            pass

        monkeypatch.setattr(module, "_release", lambda descriptor: released.append(descriptor))

        # A live holder: the take fails and nothing was ever ours to release.
        monkeypatch.setattr(module, "_take", lambda _descriptor: False)
        assert not holder_is_gone(anchor, LOCK)
        assert not holder_is_gone(anchor, LOCK, expect=identity)
        assert released == [], "released a lock that was never acquired"

        # An identity that does not match refuses even earlier.
        monkeypatch.setattr(module, "_take", lambda _descriptor: True)
        assert not holder_is_gone(anchor, LOCK, expect=(identity[0], identity[1] + 1, identity[2]))
        assert released == [], "released after refusing on identity"
