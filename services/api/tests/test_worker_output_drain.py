from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock

import psutil
import pytest

from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport
from local_lm.config import Settings
from local_lm.processes import ProcessSupervisor, WorkerRecord, WorkerStopIncomplete


def _read_fixture_pid(path: Path) -> int | None:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


@pytest.mark.parametrize("hold_output", [False, True])
async def test_inherited_output_preserves_incomplete_shutdown_until_eof(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    unused_tcp_port: int,
    hold_output: bool,
) -> None:
    settings.comfy_url = f"http://127.0.0.1:{unused_tcp_port}"
    settings.worker_shutdown_seconds = 1
    supervisor = ProcessSupervisor(settings, liveness_interval_seconds=60)
    started = settings.data_dir / "parent-started"
    signal = settings.data_dir / "spawn-child"
    child_started = settings.data_dir / "child-started"
    release = settings.data_dir / "release-output"
    child_script = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "release = pathlib.Path(sys.argv[2]); "
        "exec('while not release.exists(): time.sleep(0.01)'); "
        "print('neutral child output', flush=True)"
    )
    parent_script = (
        "import os, pathlib, subprocess, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
        "signal = pathlib.Path(sys.argv[2]); "
        "exec('while not signal.exists(): time.sleep(0.01)'); "
        f"subprocess.Popen([sys.executable, '-c', {child_script!r}, sys.argv[3], sys.argv[4]], "
        "stdout=sys.stdout, stderr=sys.stderr, "
        "creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0); "
        "time.sleep(60)"
    )

    async def wait_for_start(record: WorkerRecord, _health_url: str) -> None:
        async with asyncio.timeout(10):
            while _read_fixture_pid(started) != record.process.pid:
                await asyncio.sleep(0.01)

    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock(side_effect=wait_for_start))
    executable = getattr(sys, "_base_executable", None)
    assert isinstance(executable, str)
    support = ComfyEditorBridgeSupport(
        True, "ready", "Native workflow editing is available.", "0.28.0", "1.45.21"
    )
    await supervisor._replace(
        "media",
        [
            executable,
            "-c",
            parent_script,
            str(started),
            str(signal),
            str(child_started),
            str(release),
        ],
        f"{settings.comfy_url}/health",
        launch_scope_sha256="a" * 64,
        editor_bridge_support=support,
    )
    record = supervisor._workers["media"]
    assert int(started.read_text()) == record.process.pid
    child: psutil.Process | None = None
    child_birth: float | None = None
    original_terminate = record.process.terminate

    def spawn_after_snapshot() -> None:
        nonlocal child, child_birth
        signal.write_text("start")
        deadline = time.monotonic() + 5
        pid = _read_fixture_pid(child_started)
        while pid is None and time.monotonic() < deadline:
            time.sleep(0.01)
            pid = _read_fixture_pid(child_started)
        assert pid is not None
        child = psutil.Process(pid)
        child_birth = child.create_time()
        assert child.ppid() == record.process.pid
        if not hold_output:
            release.write_text("finish")
        original_terminate()

    monkeypatch.setattr(record.process, "terminate", spawn_after_snapshot)
    try:
        if hold_output:
            with pytest.raises(WorkerStopIncomplete):
                await asyncio.wait_for(supervisor.stop("media"), timeout=8)
            assert child is not None and child.create_time() == child_birth and child.is_running()
            assert record.process.returncode is not None
            assert supervisor._workers.get("media") is record
            assert record.stopping and record.shutdown_incomplete
            assert record.output_task is not None
            assert not record.output_task.done() and not record.output_task.cancelled()
            assert not record.log._handle.closed
            record.log.write(b"neutral output after incomplete drain\n")
            assert supervisor.launch_scope_sha256("media") is None
            assert supervisor.workflow_editor_runtime_identity() is None
            assert supervisor.workflow_editor_bridge_support() is None
            release.write_text("finish")
            await asyncio.wait_for(asyncio.shield(record.output_task), timeout=5)
        stopped = await asyncio.wait_for(supervisor.stop("media"), timeout=8)
        assert stopped.state == "stopped" and not stopped.running and not stopped.managed
        assert record.output_task is not None and record.output_task.done()
        assert record.log._handle.closed
    finally:
        release.write_text("finish")
        monkeypatch.setattr(record.process, "terminate", original_terminate)
        if child is not None:
            try:
                if child.create_time() == child_birth:
                    await asyncio.to_thread(child.wait, timeout=5)
            except psutil.TimeoutExpired:
                if child.create_time() == child_birth:
                    child.terminate()
                    await asyncio.to_thread(child.wait, timeout=5)
            except psutil.NoSuchProcess:
                pass
        await supervisor.close()
