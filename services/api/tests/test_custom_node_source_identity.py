from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_lm.custom_nodes import CustomNodeManager
from local_lm.models import CustomNodeInstall
from local_lm.subprocess_env import git_subprocess_environment


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=git_subprocess_environment(),
    ).stdout.strip()


@pytest.fixture
def installed_source(settings):
    root = settings.custom_node_dir / "lm-atelier-node_fixture"
    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    _git(root, "config", "user.name", "ajccarlson")
    _git(root, "config", "user.email", "32660587+ajccarlson@users.noreply.github.com")
    _git(root, "config", "core.autocrlf", "false")
    (root / "__init__.py").write_text("NODE_CLASS_MAPPINGS = {}\n", encoding="utf-8")
    (root / "node.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / ".gitignore").write_text("ignored_node.py\n__pycache__/\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "Add constructed node fixture")
    install = CustomNodeInstall(
        id="node_fixture",
        name="Constructed node fixture",
        source_url="https://github.com/example/constructed-nodes.git",
        revision=_git(root, "rev-parse", "HEAD"),
        installed_path=root.name,
        tree_hash=_git(root, "rev-parse", "HEAD^{tree}"),
        trusted=True,
        active=True,
        security_json={},
    )
    return CustomNodeManager(settings), install, root


async def test_verification_accepts_unchanged_pinned_source(installed_source) -> None:
    manager, install, _ = installed_source
    await manager.verify(install)


@pytest.mark.parametrize("index_hint", [None, "--assume-unchanged", "--skip-worktree"])
async def test_verification_refuses_changed_source_even_with_index_hints(
    installed_source, index_hint: str | None
) -> None:
    manager, install, root = installed_source
    if index_hint:
        _git(root, "update-index", index_hint, "--", "node.py")
    (root / "node.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert _git(root, "rev-parse", "HEAD") == install.revision
    assert _git(root, "rev-parse", "HEAD^{tree}") == install.tree_hash
    with pytest.raises(ValueError):
        await manager.verify(install)


@pytest.mark.parametrize("filename", ["extra_node.py", "ignored_node.py"])
async def test_verification_refuses_untracked_loadable_source(
    installed_source, filename: str
) -> None:
    manager, install, root = installed_source
    (root / filename).write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError):
        await manager.verify(install)


async def test_verification_refuses_missing_tracked_source(installed_source) -> None:
    manager, install, root = installed_source
    (root / "node.py").unlink()
    with pytest.raises(ValueError):
        await manager.verify(install)


async def test_verification_accepts_nested_tree_and_reviewed_cache(installed_source) -> None:
    manager, install, root = installed_source
    for directory in ("nested", "nested.more", "nested module"):
        (root / directory).mkdir()
        (root / directory / "value.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "Add nested fixture source")
    install.revision = _git(root, "rev-parse", "HEAD")
    install.tree_hash = _git(root, "rev-parse", "HEAD^{tree}")
    cache = root / "nested" / "__pycache__"
    cache.mkdir()
    (cache / "value.cpython-312.pyc").write_bytes(b"constructed cache")
    await manager.verify(install)


async def test_verification_refuses_cache_without_reviewed_source(installed_source) -> None:
    manager, install, root = installed_source
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "absent.cpython-312.pyc").write_bytes(b"constructed cache")
    with pytest.raises(ValueError):
        await manager.verify(install)


async def test_verification_refuses_a_linked_source_directory(
    installed_source, tmp_path: Path
) -> None:
    from test_anchored_directory_listing import _make_link_dir

    manager, install, root = installed_source
    outside = tmp_path / "separate-source"
    outside.mkdir()
    (outside / "outside.py").write_text("VALUE = 4\n", encoding="utf-8")
    if not _make_link_dir(root / "additional", outside):
        pytest.skip("directory links are unavailable on this host")
    with pytest.raises(ValueError):
        await manager.verify(install)
    assert (outside / "outside.py").read_text(encoding="utf-8") == "VALUE = 4\n"


async def test_verification_refuses_changed_bytes_with_preserved_timestamp(
    installed_source,
) -> None:
    import os

    manager, install, root = installed_source
    node = root / "node.py"
    before = node.stat()
    node.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(node, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert node.stat().st_size == before.st_size
    with pytest.raises(ValueError):
        await manager.verify(install)


async def test_verification_does_not_trust_a_rewritten_git_manifest(
    installed_source, monkeypatch
) -> None:
    import hashlib

    manager, install, root = installed_source
    original = manager._run
    body = b"VALUE = 8\n"
    (root / "node.py").write_bytes(body)
    replacement = hashlib.sha1(b"blob " + str(len(body)).encode("ascii") + b"\0" + body).hexdigest()

    async def changed_manifest(*command, **kwargs):
        result = await original(*command, **kwargs)
        if "ls-tree" in command:
            rows = result.split("\0")
            rows = [
                "100644 blob " + replacement + "\tnode.py" if row.endswith("\tnode.py") else row
                for row in rows
            ]
            return "\0".join(rows)
        return result

    monkeypatch.setattr(manager, "_run", changed_manifest)
    with pytest.raises(ValueError):
        await manager.verify(install)


async def test_verification_keeps_selected_directory_through_metadata_reads(
    installed_source, tmp_path: Path, monkeypatch
) -> None:
    from test_anchored_directory_listing import _make_link_dir

    manager, install, root = installed_source
    original = manager._run
    outside = tmp_path / "separate-namespace"
    outside.mkdir()
    (outside / "outside.py").write_text("VALUE = 9\n", encoding="utf-8")
    moved = root.with_name(root.name + "-preserved")
    attempted = False

    async def checked_git(*command, **kwargs):
        nonlocal attempted
        result = await original(*command, **kwargs)
        if "rev-parse" in command and not attempted:
            attempted = True
            try:
                root.rename(moved)
            except PermissionError:
                pass  # Windows retained directory handles prevent the move.
            else:
                if not _make_link_dir(root, outside):
                    moved.rename(root)
                    pytest.skip("directory links are unavailable on this host")
        return result

    monkeypatch.setattr(manager, "_run", checked_git)
    await manager.verify(install)
    assert attempted
    assert (outside / "outside.py").read_text(encoding="utf-8") == "VALUE = 9\n"


async def test_cancelled_verification_joins_reader_before_releasing_directory(
    installed_source, monkeypatch
) -> None:
    import asyncio
    import threading
    from contextlib import suppress

    from local_lm import custom_nodes
    from local_lm.filesystem_links import list_entries

    manager, install, _ = installed_source
    started = threading.Event()
    finished = threading.Event()

    def reader(package, _manifest, _tree, should_stop):
        started.set()
        while not should_stop():
            threading.Event().wait(0.005)
        assert any(entry.name == "node.py" for entry in list_entries(package))
        finished.set()

    monkeypatch.setattr(custom_nodes, "verify_pinned_source", reader, raising=False)
    verification = asyncio.create_task(manager.verify(install))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        verification.cancel()
        with pytest.raises(asyncio.CancelledError):
            await verification
        assert finished.is_set()
    finally:
        if not verification.done():
            verification.cancel()
        with suppress(asyncio.CancelledError):
            await verification


async def test_repeated_cancellation_keeps_directory_until_reader_stops(
    installed_source, monkeypatch
) -> None:
    import asyncio
    import threading
    from contextlib import suppress

    from local_lm import custom_nodes
    from local_lm.filesystem_links import list_entries

    manager, install, _ = installed_source
    started = threading.Event()
    stopped = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    reader_errors = []

    def reader(package, _manifest, _tree, should_stop):
        started.set()
        try:
            while not should_stop():
                release.wait(0.005)
            stopped.set()
            assert release.wait(2)
            assert any(entry.name == "node.py" for entry in list_entries(package))
        except Exception as exc:
            reader_errors.append(type(exc).__name__)
        finally:
            finished.set()

    monkeypatch.setattr(custom_nodes, "verify_pinned_source", reader, raising=False)
    verification = asyncio.create_task(manager.verify(install))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        verification.cancel()
        assert await asyncio.to_thread(stopped.wait, 2)
        verification.cancel()
        await asyncio.sleep(0.02)
        assert not verification.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await verification
        assert finished.is_set()
        assert reader_errors == []
    finally:
        release.set()
        if started.is_set():
            await asyncio.to_thread(finished.wait, 2)
        if not verification.done():
            verification.cancel()
        with suppress(asyncio.CancelledError):
            await verification


@pytest.mark.parametrize("changed", [False, True])
async def test_managed_start_checks_pinned_node_bytes_before_spawning(
    client, settings, request: pytest.FixtureRequest, monkeypatch, tmp_path: Path, changed: bool
) -> None:
    from unittest.mock import AsyncMock

    from local_lm.db import SessionLocal
    from local_lm.processes import ProcessSupervisor

    runtime = tmp_path / "constructed-runtime"
    runtime.mkdir()
    (runtime / "main.py").touch()
    executable = runtime / "python.exe"
    executable.touch()
    model_paths = runtime / "models.yaml"
    model_paths.touch()
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    # Configure the runtime before constructing its managed custom-node root.
    _, install, source = request.getfixturevalue("installed_source")
    with SessionLocal() as session:
        session.add(install)
        session.commit()
    supervisor = ProcessSupervisor(settings)
    spawn = AsyncMock()
    monkeypatch.setattr(supervisor, "_replace", spawn)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)

    if changed:
        (source / "node.py").write_text("VALUE = 2\n", encoding="utf-8")
        with pytest.raises(ValueError):
            await supervisor.start_media()
        spawn.assert_not_awaited()
    else:
        await supervisor.start_media()
        spawn.assert_awaited_once()
        command = spawn.await_args.args[1]
        assert "--disable-all-custom-nodes" in command
        assert command[command.index("--whitelist-custom-nodes") + 1 :] == [
            "lm-atelier-node_fixture"
        ]
