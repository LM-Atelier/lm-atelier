from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from local_lm.config import Settings
from local_lm.processes import (
    ProcessStateError,
    ProcessSupervisor,
    _held_state,
    _publish_bytes,
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


def test_a_redirected_state_folder_is_refused_and_gains_nothing(tmp_path: Path) -> None:
    """What is written here tells the runtime where to load assets from.

    Writing through a redirection is not merely a wrong destination - it is a
    way to point the runtime at someone else's directories.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    if not _make_link_dir(data_dir / "state", destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(ProcessStateError), _held_state(data_dir / "state") as anchor:
        _publish_bytes(anchor, "comfy-extra-model-paths.yaml", b"{}")
    assert sorted(p.name for p in destination.iterdir()) == [], "wrote through the redirect"


def test_a_redirected_root_with_ABSENT_state_creates_nothing(tmp_path: Path) -> None:
    """The case a pathname mkdir would walk straight into.

    With the state child absent, a mkdir(parents=True) ahead of acquisition
    creates it inside the destination before anything gets to refuse.
    """

    destination = tmp_path / "destination"
    destination.mkdir()
    redirected = tmp_path / "redirected-data"
    if not _make_link_dir(redirected, destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(ProcessStateError), _held_state(redirected / "state") as anchor:
        _publish_bytes(anchor, "comfy-extra-model-paths.yaml", b"{}")
    assert sorted(p.name for p in destination.iterdir()) == [], (
        "state was created inside the redirect destination"
    )


def test_publishing_replaces_the_previous_file(tmp_path: Path) -> None:
    """Publishing over an existing file is the normal case, not the edge case.

    Before the rename contract was explicit this replaced on POSIX and did
    nothing at all on Windows, leaving stale content and a staging file.
    """

    state = tmp_path / "data" / "state"
    with _held_state(state) as anchor:
        _publish_bytes(anchor, "record.json", b"first")
        _publish_bytes(anchor, "record.json", b"second")

    assert (state / "record.json").read_bytes() == b"second", "the publish did not happen"
    leftovers = [p.name for p in state.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == [], f"staging entries left behind: {leftovers}"


def test_a_planted_file_at_a_fixed_staging_name_is_not_used(tmp_path: Path) -> None:
    """The staging name is unpredictable, so a guessed one is inert.

    A fixed ".tmp" sibling could be pre-planted as a link, and a plain write
    follows one - so guarding the published name protected everything except
    the file the bytes actually travelled through.
    """

    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    planted = state / "record.json.tmp"
    planted.write_text("planted", encoding="utf-8")

    with _held_state(state) as anchor:
        _publish_bytes(anchor, "record.json", b"published")

    assert (state / "record.json").read_bytes() == b"published"
    assert planted.read_text(encoding="utf-8") == "planted", "wrote through the old name"


def test_the_worker_identity_record_is_not_written_through_a_redirect(
    tmp_path: Path,
) -> None:
    """End to end, and without a database: persisting is best effort but contained."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    if not _make_link_dir(data_dir / "state", destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    supervisor = ProcessSupervisor(Settings(data_dir=data_dir))
    supervisor._worker_identities = {"chat": []}
    # Persisting swallows failure by design, so the assertion is that the
    # destination gained nothing rather than that it raised.
    supervisor._persist_worker_identities()
    assert sorted(p.name for p in destination.iterdir()) == [], "wrote through the redirect"


def _reaper_spy(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record what startup cleanup is handed, which is the thing that matters.

    The loaded identities are consumed immediately and the record is then
    rewritten, so asserting on a SECOND load proves nothing about the first.
    The reaper is the actual consumer - and the one that terminates processes.
    """

    seen: list[dict[str, object]] = []
    real = ProcessSupervisor._reap_persisted_workers

    def spy(self: ProcessSupervisor) -> None:
        seen.append(dict(self._worker_identities))
        return real(self)

    monkeypatch.setattr(ProcessSupervisor, "_reap_persisted_workers", spy)
    return seen


def _identity_record() -> str:
    return json.dumps(
        {"version": 1, "workers": {"chat": [{"pid": 4321, "create_time": 1.0}]}},
        sort_keys=True,
    )


def test_a_redirected_state_cannot_supply_worker_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record read through a redirection would decide which processes DIE.

    ProcessSupervisor.__init__ hands the loaded identities straight to startup
    cleanup, which terminates the processes they name. So guarding publication
    and discard while reading by pathname left the dangerous direction open:
    an attacker SUPPLIES the record rather than reads it.

    The planted record is syntactically VALID, so only the containment refusal
    can be what rejects it.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    if not _make_link_dir(data_dir / "state", destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")
    (destination / "worker-processes.json").write_text(_identity_record(), encoding="utf-8")

    seen = _reaper_spy(monkeypatch)
    ProcessSupervisor(Settings(data_dir=data_dir))

    assert seen == [{}], f"startup cleanup was handed {seen} from a redirected folder"
    # The planted record is left exactly as it was rather than consumed.
    assert (destination / "worker-processes.json").is_file()


def test_a_valid_record_in_the_real_state_folder_is_still_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control for the test above.

    Without it, a load that always returned nothing would satisfy the refusal
    test perfectly while quietly breaking the feature.
    """

    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    (state / "worker-processes.json").write_text(_identity_record(), encoding="utf-8")

    seen = _reaper_spy(monkeypatch)
    ProcessSupervisor(Settings(data_dir=tmp_path / "data"))

    assert len(seen) == 1, "startup cleanup did not run"
    assert list(seen[0]) == ["chat"], "a valid record in the real folder was not read"


def _make_file_link(link: Path, target: Path) -> bool:
    """A FILE-shaped redirection, or None-equivalent where privilege is lacking.

    Windows needs Developer Mode or elevation for a file symlink, so this is
    the one case that legitimately cannot run everywhere.
    """

    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        return False
    return True


def test_a_linked_identity_record_inside_a_real_state_folder_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acquisition succeeding is not the whole guard.

    With a legitimate state folder, the remaining question is the ENTRY: a
    pathname read follows a link at the record's own name, while reading
    through the held directory refuses it. This is the case that distinguishes
    the two, and it is why the read goes through read_entry rather than merely
    happening inside an acquired block.
    """

    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(_identity_record(), encoding="utf-8")
    if not _make_file_link(state / "worker-processes.json", elsewhere):
        pytest.skip("this host cannot create a file symlink unprivileged")

    seen = _reaper_spy(monkeypatch)
    ProcessSupervisor(Settings(data_dir=tmp_path / "data"))

    assert seen == [{}], f"a linked record supplied identities: {seen}"
    assert elsewhere.read_text(encoding="utf-8") == _identity_record(), (
        "the link target was consumed"
    )


def test_a_link_at_the_comfy_launch_child_is_refused(tmp_path: Path) -> None:
    """The one write site that walks a NESTED component, pinned separately.

    `_held_state` has four call sites: two publishes, one read and one discard.
    Three of them hold the state directory itself, and only this one descends a
    NAMED CHILD - `state/comfy-launch`. Its containment therefore depends on
    the walk refusing a link at a component that is not the anchor, which
    nothing else in this file exercises: a change that stopped checking
    intermediate components would leave every other test here green.

    A junction at `comfy-launch` is the exact shape: the parent is a legitimate
    state directory, and only the child is redirected.
    """

    state = tmp_path / "data" / "state"
    state.mkdir(parents=True)
    destination = tmp_path / "elsewhere"
    destination.mkdir()
    if not _make_link_dir(state / "comfy-launch", destination):
        pytest.skip("this host does not permit directory links")

    with (
        pytest.raises(ProcessStateError),
        _held_state(state, "comfy-launch") as anchor,
    ):
        _publish_bytes(anchor, "launch.yaml", b"payload")

    assert list(destination.iterdir()) == [], "the link target was written through"
