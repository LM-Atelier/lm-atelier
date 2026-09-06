from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

_START = r"""
import os
import sys
import time
from pathlib import Path
from local_lm.backups import BackupManager

marker = Path(sys.argv[1])
phase = sys.argv[2]
def restore(self):
    marker.write_text('entered', encoding='ascii')
    if phase == 'hold':
        time.sleep(60)
    raise SystemExit(0)
BackupManager.apply_pending_restore = restore
if sys.argv[3] == 'desktop':
    from local_lm.desktop import main
    sys.argv = ['lm-atelier']
    raise SystemExit(main())
else:
    from local_lm import main
"""


def _environment(data_dir: Path, *, dev: bool = False) -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("LOCAL_LM_")
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        LOCAL_LM_DATA_DIR=str(data_dir),
        LOCAL_LM_DEV=str(dev).lower(),
        LOCAL_LM_CHAT_ENGINE="mock",
        LOCAL_LM_MEDIA_ENGINE="mock",
        LOCAL_LM_OPEN_BROWSER="false",
        LOCAL_LM_PORT="12349",
    )
    return environment


def _run(
    script: str, data_dir: Path, *arguments: str, dev: bool = False
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script, *arguments],
        cwd=Path(__file__).resolve().parents[3],
        env=_environment(data_dir, dev=dev),
        capture_output=True,
        text=True,
        timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


@contextmanager
def _holding_start(data_dir: Path, marker: Path) -> Iterator[subprocess.Popen[bytes]]:
    with marker.with_suffix(".log").open("wb") as log:
        environment = _environment(data_dir)
        environment["LOCAL_LM_PORT"] = "12348"
        process = subprocess.Popen(
            [sys.executable, "-c", _START, str(marker), "hold", "direct"],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            deadline = time.monotonic() + 20
            while not marker.exists() and time.monotonic() < deadline:
                assert process.poll() is None, marker.with_suffix(".log").read_text()
                time.sleep(0.01)
            assert marker.exists(), "first start did not reach the pending-restore boundary"
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)


@pytest.mark.parametrize("entrypoint", ["desktop", "direct"])
def test_second_start_refuses_before_database_work_even_on_another_port(
    tmp_path: Path, entrypoint: str
) -> None:
    data = tmp_path / "data"
    entered = tmp_path / "first-restore"
    second = tmp_path / "second-restore"
    with _holding_start(data, entered) as holder:
        result = _run(_START, data, str(second), "probe", entrypoint)
        assert result.returncode != 0, "a competing startup must refuse before database work"
        if entrypoint == "desktop":
            assert result.returncode == 2
        assert "already starting or running with this data folder" in result.stderr
        assert not second.exists(), "the second process must never enter pending restore"
        assert holder.poll() is None
        assert not (data / "state" / "local-lm.sqlite3").exists()
        holder.kill()
        holder.wait(timeout=10)
    resumed = _run(_START, data, str(second), "probe", entrypoint)
    assert resumed.returncode == 0, resumed.stderr
    assert second.exists(), "a killed owner must not leave stale exclusion behind"


def test_same_directory_refuses_and_other_directory_remains_available(tmp_path: Path) -> None:
    from local_lm.instance_lock import DataDirectoryBusy, DataDirectoryLock

    data = tmp_path / "data"
    with DataDirectoryLock(data):
        with pytest.raises(DataDirectoryBusy):
            DataDirectoryLock(data)
        with DataDirectoryLock(tmp_path / "other"), pytest.raises(DataDirectoryBusy):
            DataDirectoryLock(data)
    with DataDirectoryLock(data):
        pass


def test_relative_and_absolute_names_share_one_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm.instance_lock import DataDirectoryBusy, DataDirectoryLock

    monkeypatch.chdir(tmp_path)
    with DataDirectoryLock(Path("data")), pytest.raises(DataDirectoryBusy):
        DataDirectoryLock(tmp_path / "data")


def test_linked_ancestor_is_refused_before_creating_children(tmp_path: Path) -> None:
    from local_lm.instance_lock import DataDirectoryLock, DataDirectoryLockError

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    if sys.platform == "win32":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(linked), str(outside)],
            capture_output=True,
            check=True,
        )
    else:
        linked.symlink_to(outside, target_is_directory=True)
    try:
        with pytest.raises(DataDirectoryLockError, match="filesystem links"):
            DataDirectoryLock(linked / "must-not-be-created" / "data")
        assert list(outside.iterdir()) == []
    finally:
        if sys.platform == "win32":
            linked.rmdir()
        else:
            linked.unlink()


def test_failed_app_construction_releases_ownership(tmp_path: Path) -> None:
    script = r"""
from local_lm.backups import BackupManager
from local_lm.config import get_settings
from local_lm.instance_lock import DataDirectoryLock

def fail(self):
    raise RuntimeError('constructed restore failure')
BackupManager.apply_pending_restore = fail
try:
    from local_lm import main
except RuntimeError as exc:
    assert str(exc) == 'constructed restore failure'
else:
    raise AssertionError('startup failure did not propagate')
with DataDirectoryLock(get_settings().data_dir):
    pass
"""
    result = _run(script, tmp_path / "data")
    assert result.returncode == 0, result.stderr


def test_reload_supervisor_releases_ownership_before_starting_child(tmp_path: Path) -> None:
    script = r"""
from local_lm import main
from local_lm.config import get_settings
from local_lm.instance_lock import DataDirectoryLock

def reload(application, **kwargs):
    assert application == 'local_lm.main:app'
    assert kwargs['reload'] is True
    with DataDirectoryLock(get_settings().data_dir):
        pass
main.uvicorn.run = reload
main.run()
"""
    result = _run(script, tmp_path / "data", dev=True)
    assert result.returncode == 0, result.stderr


def test_regular_server_retains_ownership_while_serving(tmp_path: Path) -> None:
    script = r"""
from local_lm import main
from local_lm.config import get_settings
from local_lm.instance_lock import DataDirectoryBusy, DataDirectoryLock

def serve(application, **kwargs):
    assert application is main.app
    assert kwargs['reload'] is False
    try:
        DataDirectoryLock(get_settings().data_dir)
    except DataDirectoryBusy:
        return
    raise AssertionError('server lost ownership before serving')
main.uvicorn.run = serve
main.run()
"""
    result = _run(script, tmp_path / "data")
    assert result.returncode == 0, result.stderr


def test_schema_export_uses_its_own_folder_while_an_instance_is_running(tmp_path: Path) -> None:
    from local_lm.instance_lock import DataDirectoryLock

    occupied = tmp_path / "occupied-data"
    repository = Path(__file__).resolve().parents[3]
    with DataDirectoryLock(occupied):
        result = subprocess.run(
            [sys.executable, str(repository / "scripts" / "export-openapi.py")],
            cwd=repository,
            env=_environment(occupied),
            capture_output=True,
            text=True,
            timeout=25,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert result.returncode == 0, result.stderr
        schema = json.loads(result.stdout)
        assert "/api/ready" in schema["paths"]
        assert "JobOut" in schema["components"]["schemas"]
        assert list(occupied.iterdir()) == [], (
            "export must not prepare the running instance's folder"
        )
