from __future__ import annotations

import contextlib
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from local_lm import filesystem_links as links
from local_lm import shared_asset_consumer_v1 as consumer
from local_lm.shared_asset_consumer_v1 import (
    INVALID_CONSUMER,
    SharedAssetConsumerError,
    derive_consumer_identity,
)

_BASE = {
    "product_namespace": "lm-atelier",
    "channel": "stable",
    "build_fingerprint": "0.1.8",
}


def _derive(data_dir: Path, **overrides: str) -> str:
    # Derivation now REFUSES an absent root rather than establishing one, so
    # every test that is about identity rather than about establishment states
    # that precondition here, once. Tests that are about the refusal call
    # derive_consumer_identity directly and do not come through here.
    data_dir.mkdir(parents=True, exist_ok=True)
    facts = dict(_BASE)
    facts.update(overrides)
    return derive_consumer_identity(data_dir=data_dir, **facts)


def test_identity_is_stable_and_opaque_hex(tmp_path: Path) -> None:
    first = _derive(tmp_path / "data")
    again = _derive(tmp_path / "data")
    assert first == again
    assert re.fullmatch(r"[0-9a-f]{64}", first)
    # The registry's consumer grammar accepts it directly.
    assert re.fullmatch(r"[0-9a-f]{32,64}", first)


def test_every_input_dimension_changes_the_identity(tmp_path: Path) -> None:
    base = _derive(tmp_path / "data")
    assert _derive(tmp_path / "other-data") != base
    assert _derive(tmp_path / "data", product_namespace="other-product") != base
    assert _derive(tmp_path / "data", channel="beta") != base
    assert _derive(tmp_path / "data", build_fingerprint="0.1.9") != base


def test_composition_is_unambiguous_across_field_boundaries(tmp_path: Path) -> None:
    # Without length prefixes these two would collapse to one hash input.
    left = _derive(tmp_path / "data", product_namespace="ab", channel="c")
    right = _derive(tmp_path / "data", product_namespace="a", channel="bc")
    assert left != right


def test_distinct_installations_never_collide(tmp_path: Path) -> None:
    alpha = _derive(tmp_path / "machine-a")
    beta = _derive(tmp_path / "machine-b")
    assert alpha != beta
    # Same facts, same machine seed directory: stable across processes.
    assert _derive(tmp_path / "machine-a") == alpha


@pytest.mark.parametrize(
    "fact",
    [
        "",
        "UPPER",
        "has space",
        "-leading-dash",
        ".leading-dot",
        "a" * 81,
        "caf\u00e9",
        "nul\x00byte",
        "path/../escape",
    ],
)
def test_hostile_manifest_facts_are_refused(tmp_path: Path, fact: str) -> None:
    with pytest.raises(SharedAssetConsumerError) as caught:
        _derive(tmp_path / "data", product_namespace=fact)
    # The literal, not the imported constant: the refusal is a fixed public
    # contract and must not drift with the module.
    assert str(caught.value) == "shared asset consumer identity is invalid"
    assert INVALID_CONSUMER == "shared asset consumer identity is invalid"
    with pytest.raises(SharedAssetConsumerError):
        _derive(tmp_path / "data", channel=fact)
    with pytest.raises(SharedAssetConsumerError):
        _derive(tmp_path / "data", build_fingerprint=fact)


def test_known_answer_pins_context_order_and_prefixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Independently computed vector: sha256 over the v1 context byte string
    # followed by the two-byte-length-prefixed installation ("f" * 64),
    # namespace "lm-atelier", channel "stable", fingerprint "0.1.8". Pins the
    # domain-separation context, the field order, and the prefixing exactly;
    # recomputing it from module constants would defeat the pin.
    # Patched at the anchored delegate, which is what the consumer now calls.
    monkeypatch.setattr(consumer, "instance_identity_from_directory", lambda _a: "f" * 64)
    # The root must exist: derivation refuses an absent one rather than
    # establishing it, so tests state the precondition explicitly.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    derived = _derive(data_dir)
    assert derived == "8787c561932238b31ee249c8d0d124dfe18ba0ab0000a05ca4dfd667db516abb"


def test_instance_failures_surface_only_the_fixed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explode(_anchor: object) -> str:
        raise consumer.InstanceIdentityError(r"secret C:\private\seed path")

    monkeypatch.setattr(consumer, "instance_identity_from_directory", explode)
    with pytest.raises(SharedAssetConsumerError) as caught:
        _derive(tmp_path / "data")
    assert str(caught.value) == "shared asset consumer identity is invalid"
    assert "secret" not in str(caught.value)


def test_derivation_runs_against_the_directory_that_was_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One acquisition, carried through the delegate - not two resolutions.

    The defect this pins: validating a pathname and then handing the PATHNAME
    to the delegate meant the delegate resolved it a second time, and its own
    mkdir could establish a replacement root in the gap. Asserting the
    delegate receives the held directory pins that the gap does not exist.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    received: list[object] = []
    held_at_call: list[bool] = []
    real = consumer.instance_identity_from_directory

    def record(anchor: object) -> str:
        received.append(anchor)
        # Sampled HERE, while the call is in flight: the anchor is released
        # when the consumer's with-block exits, so inspecting it afterwards
        # would assert against a closed object and prove nothing.
        held_at_call.append(anchor.descriptor is not None or anchor.handle is not None)
        return real(anchor)  # type: ignore[arg-type]

    monkeypatch.setattr(consumer, "instance_identity_from_directory", record)
    derived = _derive(data_dir)

    assert len(received) == 1, "the delegate was not reached exactly once"
    anchor = received[0]
    assert isinstance(anchor, links.AnchoredDirectory), "the delegate got a pathname"
    assert anchor.path == data_dir.resolve()
    # It really was a HELD directory during the call, not a value object
    # carrying a path.
    assert held_at_call == [True]
    assert re.fullmatch(r"[0-9a-f]{64}", derived)
    # The seed landed inside the validated root and nowhere else.
    assert (data_dir / "state" / "desktop-instance-seed").is_file()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data"]


def test_a_held_data_root_cannot_be_swapped_underneath_the_derivation(
    tmp_path: Path,
) -> None:
    """Retention is what closes the gap, so retention is what gets asserted.

    Windows and POSIX enforce it differently and both are checked here rather
    than skipping the one that matters: Windows refuses to rename a directory
    while a handle is held, and POSIX allows the rename but the retained
    descriptor keeps referring to the same directory, so a replacement created
    at the old NAME is not what the derivation writes into.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    moved = tmp_path / "moved"

    with links.AnchoredDirectory(data_dir.resolve()) as anchor:
        try:
            data_dir.rename(moved)
        except OSError:
            # Windows: the held directory cannot be renamed at all.
            identity = consumer.instance_identity_from_directory(anchor)
            assert re.fullmatch(r"[0-9a-f]{64}", identity)
            assert (data_dir / "state" / "desktop-instance-seed").is_file()
            return
        # POSIX: the rename succeeded, so a hostile replacement can now sit at
        # the original name. The derivation must ignore it entirely.
        replacement = tmp_path / "data"
        replacement.mkdir()
        identity = consumer.instance_identity_from_directory(anchor)
        assert re.fullmatch(r"[0-9a-f]{64}", identity)
        assert (moved / "state" / "desktop-instance-seed").is_file()
        assert not (replacement / "state").exists(), "wrote through the replacement"


def _make_link_dir(base: Path, target: Path, name: str = "redirect") -> Path | None:
    """A directory-shaped redirection, or None where this host forbids one."""

    link = base / name
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True
        )
        return link if completed.returncode == 0 else None
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        return None
    return link


def test_a_linked_data_root_refuses_and_writes_nothing_to_the_target(
    tmp_path: Path,
) -> None:
    """Resolving before acquiring would have followed this and seen nothing wrong.

    The whole point of validating the spelling rather than the resolution: an
    anchor can only refuse a redirection it is allowed to see.
    """

    target = tmp_path / "elsewhere"
    target.mkdir()
    link = _make_link_dir(tmp_path, target)
    if link is None:
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(SharedAssetConsumerError) as caught:
        derive_consumer_identity(
            data_dir=link,
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert str(caught.value) == INVALID_CONSUMER
    assert sorted(p.name for p in target.iterdir()) == [], "wrote through the redirect"


def test_a_linked_ANCESTOR_refuses_and_writes_nothing_to_the_target(
    tmp_path: Path,
) -> None:
    """The redirection need not be the root itself to matter."""

    target = tmp_path / "elsewhere"
    (target / "data").mkdir(parents=True)
    link = _make_link_dir(tmp_path, target, name="parent-redirect")
    if link is None:
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(SharedAssetConsumerError) as caught:
        derive_consumer_identity(
            data_dir=link / "data",
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert str(caught.value) == INVALID_CONSUMER
    assert sorted(p.name for p in (target / "data").iterdir()) == [], (
        "wrote through a redirected ancestor"
    )


def test_reading_an_existing_seed_repairs_its_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observable on both platforms, enforced where a mode exists.

    The repair itself is POSIX-only because Windows carries no mode here, so
    asserting only the mode would make this test silently vacuous on Windows -
    which is where it would then be run most often. The call is asserted
    everywhere; the effect is asserted where there is one.
    """

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    seed = state / "desktop-instance-seed"
    seed.write_text("a" * 64, encoding="ascii")
    seed.chmod(0o666)

    opened: list[object] = []
    real = instance_identity.open_entry

    def record(anchor: object, name: str) -> object:
        opened.append(name)
        return real(anchor, name)  # type: ignore[arg-type]

    # The fix IS the single open: type, contents and mode are settled on one
    # descriptor rather than by looking the name up again. That happens on
    # both platforms, so it is asserted on both; only its mode effect is
    # POSIX-specific.
    monkeypatch.setattr(instance_identity, "open_entry", record)
    identity = _derive(data_dir)

    assert re.fullmatch(r"[0-9a-f]{64}", identity)
    assert opened == ["desktop-instance-seed"], "the seed was not opened exactly once"
    if os.name != "nt":
        assert seed.stat().st_mode & 0o777 == 0o600


def test_an_absent_data_root_refuses_and_creates_nothing(tmp_path: Path) -> None:
    """A mistyped root must fail, not quietly become a second consumer.

    The refusal is only half the property. The half that matters is that
    nothing is created: if derivation established the root it was asked
    about, a typo would mint an identity that owns no claims and diverges
    silently from the store the caller meant.
    """

    absent = tmp_path / "not-a-store"
    with pytest.raises(SharedAssetConsumerError) as caught:
        derive_consumer_identity(
            data_dir=absent,
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert str(caught.value) == INVALID_CONSUMER
    assert not absent.exists(), "derivation created the root it was asked about"
    # A nested root is the shape a typo actually takes, and parents=True would
    # have built the whole chain.
    nested = tmp_path / "typo" / "deeper" / "store"
    with pytest.raises(SharedAssetConsumerError):
        derive_consumer_identity(
            data_dir=nested,
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert not (tmp_path / "typo").exists()


def test_a_data_root_that_is_a_regular_file_refuses(tmp_path: Path) -> None:
    """Not a directory is not a store, and it is not an absence either."""

    occupied = tmp_path / "afile"
    occupied.write_text("x", encoding="utf-8")
    with pytest.raises(SharedAssetConsumerError) as caught:
        derive_consumer_identity(
            data_dir=occupied,
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert str(caught.value) == INVALID_CONSUMER


def test_a_tampered_seed_refuses_rather_than_deriving(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _derive(data_dir)
    seed = data_dir / "state" / "desktop-instance-seed"
    seed.write_text("not a seed at all", encoding="ascii")
    with pytest.raises(SharedAssetConsumerError):
        _derive(data_dir)


def test_non_path_data_dir_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SharedAssetConsumerError):
        derive_consumer_identity(
            data_dir="not-a-path",  # type: ignore[arg-type]
            **_BASE,
        )


def test_an_oversized_seed_is_refused_without_reading_all_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed restored seed must not cost memory in proportion to its size.

    The record is 64 hex characters. Reading one byte past the longest legal
    value is enough to tell valid from overlong, so nothing larger is ever
    pulled into memory.
    """

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    (state / "desktop-instance-seed").write_bytes(b"a" * (4 * 1024 * 1024))

    sizes: list[int] = []
    real_read = instance_identity.os.read

    def record(descriptor: int, count: int) -> bytes:
        sizes.append(count)
        return real_read(descriptor, count)

    monkeypatch.setattr(instance_identity.os, "read", record)
    with pytest.raises(SharedAssetConsumerError) as caught:
        _derive(data_dir)
    assert str(caught.value) == INVALID_CONSUMER
    assert sizes, "the seed was never read"
    assert max(sizes) <= 67, f"read {max(sizes)} bytes from a 4 MiB file"


def test_a_parent_segment_spelling_is_refused_rather_than_aliased(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One spelling per identity, and the same answer on both platforms.

    Two spellings that walk to the same directory would otherwise derive two
    identities, since the identity is bound to the name - and POSIX accepts
    ".." in the walk while Windows refuses it.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    acquired: list[object] = []

    def record(*args: object, **kwargs: object) -> object:
        acquired.append(args)
        raise AssertionError("acquisition should never be reached")

    # The refusal must come from the SPELLING check, before acquisition. The
    # walk happens to refuse ".." on Windows as well, so asserting only that
    # the call raises would pass here whether or not this fix exists.
    monkeypatch.setattr(consumer, "AnchoredDirectory", record)
    with pytest.raises(SharedAssetConsumerError) as caught:
        derive_consumer_identity(
            data_dir=tmp_path / "data" / ".." / "data",
            product_namespace="lm-atelier",
            channel="stable",
            build_fingerprint="0.1.8",
        )
    assert str(caught.value) == INVALID_CONSUMER
    assert acquired == [], "the spelling was accepted and acquisition was attempted"


def test_a_failure_before_the_seed_is_written_leaves_no_staging_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every pre-publication failure cleans up after itself."""

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    def explode(*_args: object, **_kwargs: object) -> object:
        raise OSError("no descriptor for you")

    monkeypatch.setattr(instance_identity.os, "fdopen", explode)
    with pytest.raises(SharedAssetConsumerError):
        _derive(data_dir)
    leftover = sorted(p.name for p in (data_dir / "state").iterdir())
    assert leftover == [], f"staging entries left behind: {leftover}"


def test_two_concurrent_first_starts_converge_on_one_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second start must not observe a half-written seed.

    Creating the final name directly made it visible before its bytes were
    complete, so a concurrent first start could read empty content and fail
    instead of converging. Publication by rename removes the window; this pins
    it by holding one start inside that interval while the other runs.
    """

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    inside = threading.Event()
    release = threading.Event()
    real_create = instance_identity.create_entry
    first = threading.current_thread().name

    def pause_after_create(*args: object, **kwargs: object) -> object:
        descriptor = real_create(*args, **kwargs)  # type: ignore[arg-type]
        if threading.current_thread().name == first:
            # Held between CREATE and WRITE - the interval where the old code
            # left the FINAL name existing and empty. Pausing at the rename
            # instead would prove nothing, because by then the bytes are
            # already written under either design.
            inside.set()
            release.wait(timeout=10)
        return descriptor

    monkeypatch.setattr(instance_identity, "create_entry", pause_after_create)

    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def second() -> None:
        try:
            inside.wait(timeout=10)
            results["second"] = _derive(data_dir)
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            errors.append(error)
        finally:
            release.set()

    worker = threading.Thread(target=second, name="second-start")
    worker.start()
    results["first"] = _derive(data_dir)
    worker.join(timeout=20)

    assert not errors, f"the concurrent start failed: {errors}"
    assert results["first"] == results["second"], "the two starts disagreed"


def test_open_entry_refuses_a_directory(tmp_path: Path) -> None:
    """The primitive promises a regular file, so it must enforce one.

    POSIX O_RDONLY happily returns a descriptor for a directory, and a caller
    that then reads, stats or chmods through it is operating on something with
    entirely different semantics.
    """

    (tmp_path / "a-directory").mkdir()
    with (
        links.AnchoredDirectory(tmp_path) as anchor,
        pytest.raises(links.AnchoredDirectoryError),
    ):
        links.open_entry(anchor, "a-directory")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="this platform has no FIFO")
def test_open_entry_refuses_a_named_pipe_without_hanging(tmp_path: Path) -> None:
    """Opening a FIFO for reading blocks until a writer appears.

    Without O_NONBLOCK the type check below would never be reached, so this
    pins the absence of a hang as much as the refusal.
    """

    os.mkfifo(tmp_path / "a-pipe")  # type: ignore[attr-defined]
    with (
        links.AnchoredDirectory(tmp_path) as anchor,
        pytest.raises(links.AnchoredDirectoryError),
    ):
        links.open_entry(anchor, "a-pipe")


def test_a_short_first_read_still_measures_the_overflow_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One os.read may return less than asked for before EOF.

    A 64-byte valid-looking prefix of a longer file would then satisfy the
    pattern without the overflow byte ever being read - the bound enforced
    against a number never measured.
    """

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    (state / "desktop-instance-seed").write_bytes(b"a" * 64 + b"trailing rubbish")

    real_read = instance_identity.os.read
    first = {"done": False}

    def short_first(descriptor: int, count: int) -> bytes:
        if not first["done"]:
            first["done"] = True
            # Exactly the 64 valid hex bytes. A single-read implementation
            # stops here and ACCEPTS; only continuing to EOF reveals the
            # overflow. Returning fewer would be rejected by both, and would
            # prove nothing.
            return real_read(descriptor, 64)
        return real_read(descriptor, count)

    monkeypatch.setattr(instance_identity.os, "read", short_first)
    with pytest.raises(SharedAssetConsumerError) as caught:
        _derive(data_dir)
    assert str(caught.value) == INVALID_CONSUMER


def test_a_pre_existing_staging_entry_does_not_block_a_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An abandoned staging entry must not stop a later start.

    The old name was derived from pid and thread id, both of which get reused,
    so an interrupted process could leave a name a later start would collide
    with - and the failure surfaced as a bad final seed rather than as what it
    was.
    """

    from local_lm import instance_identity

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)

    names = iter(["collides", "collides", "free"])
    monkeypatch.setattr(instance_identity.secrets, "token_hex", lambda _n: next(names))
    (state / "desktop-instance-seed.collides.tmp").write_text("abandoned", encoding="ascii")

    identity = _derive(data_dir)

    assert re.fullmatch(r"[0-9a-f]{64}", identity)
    assert (state / "desktop-instance-seed").is_file(), "the seed was not published"
    # The abandoned entry is left exactly as it was rather than reused.
    assert (state / "desktop-instance-seed.collides.tmp").read_text(encoding="ascii") == "abandoned"


def test_require_regular_refuses_a_character_device(tmp_path: Path) -> None:
    """Pinned on a non-regular entry THIS platform can open.

    The directory case is already refused by the NT open flags on Windows, so
    it cannot demonstrate the type check here - it demonstrates it on POSIX,
    where O_RDONLY opens a directory happily. A character device is something
    both platforms can hand back a descriptor for, so the contract is pinned
    where the test actually runs rather than only where CI runs it.
    """

    device = "NUL" if os.name == "nt" else "/dev/null"
    descriptor = os.open(device, os.O_RDONLY)
    try:
        with pytest.raises(links.AnchoredDirectoryError):
            links._require_regular(descriptor)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise
    # _require_regular closed it on refusal; closing again would be an error.
    with pytest.raises(OSError):
        os.fstat(descriptor)
