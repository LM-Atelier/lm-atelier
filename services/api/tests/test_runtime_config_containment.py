from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from local_lm import desktop
from local_lm.runtime_config import (
    RuntimeConfigError,
    persist_runtime_values,
    runtime_config_path,
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
    """The configuration written here feeds process environment on next launch.

    So a redirected state folder is not merely a wrong place to write - it is a
    way to supply values to the next start.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if not _make_link_dir(data_dir / "state", foreign):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    environment: dict[str, str] = {}
    with pytest.raises(RuntimeConfigError):
        persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "mock"}, environment)
    assert sorted(p.name for p in foreign.iterdir()) == [], "wrote through the redirect"


def test_publishing_replaces_the_previous_configuration(tmp_path: Path) -> None:
    """Publishing over an existing file is the normal case, not the edge case.

    Before the rename contract was made explicit this replaced on POSIX and
    silently did nothing on Windows, leaving the previous configuration in
    place - so this test is the one that fails on the platform it matters on.
    """

    data_dir = tmp_path / "data"
    environment: dict[str, str] = {}
    persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "mock"}, environment)
    persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "llama.cpp"}, environment)

    written = runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert "llama.cpp" in written
    assert "mock" not in written
    assert environment["LOCAL_LM_CHAT_ENGINE"] == "llama.cpp"


def test_a_planted_file_at_the_old_fixed_staging_name_is_not_used(
    tmp_path: Path,
) -> None:
    """The staging name is unpredictable, so the old fixed one is inert.

    A fixed `.tmp` sibling could be pre-planted as a link, and a plain write
    follows one - which meant the guard on the published name protected
    everything except the file the bytes actually travelled through.
    """

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    planted = state / "runtime-config.tmp"
    planted.write_text("planted", encoding="utf-8")

    persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})

    assert "mock" in runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert planted.read_text(encoding="utf-8") == "planted", "wrote through the old name"


def test_a_relative_data_directory_still_works(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The documented .env spelling allows a relative data directory."""

    monkeypatch.chdir(tmp_path)
    persist_runtime_values(Path("developer-data"), {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})
    assert "mock" in (tmp_path / "developer-data" / "state" / "runtime-config.json").read_text(
        encoding="utf-8"
    )


def test_a_redirected_state_folder_refuses_startup_cleanly(
    tmp_path: Path, monkeypatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    """A refusal at startup must read like the sibling one, not like a crash.

    configure_desktop_environment is the first thing that touches the state
    folder, so without handling here the refusal would surface as a traceback
    rather than the message and exit code the instance-identity refusal
    already established.
    """

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    if not _make_link_dir(data_dir / "state", foreign):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    def refuse() -> None:
        raise RuntimeConfigError("LM Atelier's state folder may not be a filesystem link")

    monkeypatch.setattr(desktop, "configure_desktop_environment", refuse)
    assert desktop.main() == 2
    assert "could not establish ownership of its data folder" in capsys.readouterr().err


def test_a_redirected_root_with_ABSENT_state_creates_nothing(tmp_path: Path) -> None:
    """The case the old test missed entirely.

    The previous version created the state folder with a pathname
    mkdir(parents=True) BEFORE acquiring anything, so a redirected data root
    with no state child got one created inside the destination before
    acquisition ever refused. The old test only planted an already-existing
    redirected state leaf, where mkdir had nothing to create.
    """

    destination = tmp_path / "destination"
    destination.mkdir()
    redirected = tmp_path / "redirected"
    if not _make_link_dir(redirected, destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(RuntimeConfigError):
        persist_runtime_values(redirected, {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})
    assert sorted(p.name for p in destination.iterdir()) == [], (
        "state was created inside the redirect destination"
    )


def test_a_redirected_ANCESTOR_with_absent_state_creates_nothing(
    tmp_path: Path,
) -> None:
    """The redirection need not be the data root itself."""

    destination = tmp_path / "destination"
    (destination / "data").mkdir(parents=True)
    redirected = tmp_path / "redirected-parent"
    if not _make_link_dir(redirected, destination):
        pytest.skip("this host cannot create a directory redirection unprivileged")

    with pytest.raises(RuntimeConfigError):
        persist_runtime_values(redirected / "data", {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})
    assert sorted(p.name for p in (destination / "data").iterdir()) == [], (
        "state was created through a redirected ancestor"
    )


def test_a_failure_opening_the_staging_file_leaves_nothing_behind(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """fdopen owns the descriptor only once it succeeds."""

    from local_lm import runtime_config

    data_dir = tmp_path / "data"

    def explode(*_args: object, **_kwargs: object) -> object:
        raise OSError("no stream for you")

    monkeypatch.setattr(runtime_config.os, "fdopen", explode)
    with pytest.raises(OSError):
        persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})
    leftover = sorted(p.name for p in (data_dir / "state").iterdir())
    assert leftover == [], f"staging entries left behind: {leftover}"


def test_an_abandoned_staging_entry_does_not_block_a_write(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A deterministic staging name could be blocked by an interrupted run."""

    from local_lm import runtime_config

    data_dir = tmp_path / "data"
    state = data_dir / "state"
    state.mkdir(parents=True)
    names = iter(["collides", "collides", "free"])
    monkeypatch.setattr(runtime_config.secrets, "token_hex", lambda _n: next(names))
    (state / "runtime-config.json.collides.tmp").write_text("abandoned", encoding="utf-8")

    persist_runtime_values(data_dir, {"LOCAL_LM_CHAT_ENGINE": "mock"}, {})

    assert "mock" in runtime_config_path(data_dir).read_text(encoding="utf-8")
    assert (state / "runtime-config.json.collides.tmp").read_text(encoding="utf-8") == "abandoned"
