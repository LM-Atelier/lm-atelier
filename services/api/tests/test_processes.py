from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import psutil
import pytest
from sqlalchemy.orm import object_session

import local_lm.comfy_registry_interpreter as registry_interpreter_module
from local_lm.comfy_editor_bridge import BRIDGE_DIRECTORY_NAME
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import SessionLocal
from local_lm.events import EventBroker
from local_lm.models import CustomNodeInstall, ModelInstall, ModelProfile
from local_lm.network import shared_tls_context
from local_lm.processes import (
    WORKER_STDERR_DISPLAY_CHARS,
    WORKER_STDERR_DISPLAY_LINES,
    WORKER_STDERR_TAIL_BYTES,
    ProcessSupervisor,
    WorkerRecord,
    _RotatingWorkerLog,
    _with_comfy_registry_overlays,
)
from local_lm.worker_failures import WorkerFailureCode
from local_lm.workflow_activations import (
    WorkflowActivationLaunchScope,
    WorkflowAssetLaunchBinding,
    WorkflowModelLaunchBinding,
)

# Reproduced from CPython 3.12 by making the socket teardown inside
# `_ProactorBasePipeTransport._call_connection_lost` raise. asyncio composes the
# first two lines; the rest is the traceback it prints for the callback.
CONNECTION_TEARDOWN_BLOCK = (
    b"Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)\n"
    b"handle: <Handle _ProactorBasePipeTransport._call_connection_lost(None)>\n"
    b"Traceback (most recent call last):\n"
    b'  File "asyncio\\events.py", line 88, in _run\n'
    b"    self._context.run(self._callback, *self._args)\n"
    b'  File "asyncio\\proactor_events.py", line 165, in _call_connection_lost\n'
    b"    self._sock.shutdown(socket.SHUT_RDWR)\n"
    b"OSError: [WinError 10022] An invalid argument was supplied\n"
)

# The same failure as a real frozen media worker writes it, captured from
# `media-worker.log.3` on a monitored install. A bare interpreter is not what
# ships: the level tag comes first, the callback renders with empty parentheses,
# and the traceback carries no source lines. Matching only the reproduction above
# would have filtered nothing that actually occurs.
FROZEN_RUNTIME_TEARDOWN_BLOCK = (
    b"[ERROR] Exception in callback _ProactorBasePipeTransport._call_connection_lost()\n"
    b"handle: <Handle _ProactorBasePipeTransport._call_connection_lost()>\n"
    b"Traceback (most recent call last):\n"
    b'  File "asyncio\\events.py", line 89, in _run\n'
    b'  File "asyncio\\proactor_events.py", line 165, in _call_connection_lost\n'
    b"OSError: [WinError 10022] An invalid argument was supplied\n"
)


async def wait_for_worker_event(events: EventBroker, event_type: str) -> None:
    for _ in range(200):
        if any(event.type == event_type for event in events.since(0)):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"worker event {event_type!r} was not published")


class FakeRunningProcess:
    stdout = None
    stderr = None

    def __init__(self, pid: int, *, terminate_code: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.terminate_code = terminate_code
        self.exited = asyncio.Event()
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = self.terminate_code
        self.exited.set()

    def kill(self) -> None:
        self.returncode = -9
        self.exited.set()

    async def wait(self) -> int:
        await self.exited.wait()
        assert self.returncode is not None
        return self.returncode


def test_llama_arguments_are_explicit_and_shell_free() -> None:
    arguments = ProcessSupervisor._llama_load_arguments(
        {
            "context_length": 16_384,
            "gpu_layers": 40,
            "flash_attention": True,
            "mmap": False,
            "mlock": True,
            "tensor_split": "3,1",
        }
    )
    assert arguments == [
        "--ctx-size",
        "16384",
        "--n-gpu-layers",
        "40",
        "--tensor-split",
        "3,1",
        "--flash-attn",
        "on",
        "--no-mmap",
        "--mlock",
    ]


def test_gguf_resolution_rejects_ambiguous_installs(tmp_path: Path) -> None:
    (tmp_path / "a.gguf").touch()
    (tmp_path / "b.gguf").touch()
    with pytest.raises(ValueError, match="multiple standalone GGUF models"):
        ProcessSupervisor._gguf_path(tmp_path, {})


def test_gguf_resolution_uses_first_shard_from_complete_manifest(tmp_path: Path) -> None:
    first = tmp_path / "model-Q4_K_M-00001-of-00002.gguf"
    second = tmp_path / "model-Q4_K_M-00002-of-00002.gguf"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    manifest = {"files": [second.name, first.name]}

    assert ProcessSupervisor._gguf_paths(tmp_path, manifest) == (first, second)
    assert ProcessSupervisor._gguf_path(tmp_path, manifest) == first


def test_gguf_resolution_preserves_legacy_single_file_installs(tmp_path: Path) -> None:
    model = tmp_path / "legacy-model.gguf"
    model.write_bytes(b"GGUF")

    assert ProcessSupervisor._gguf_paths(tmp_path, {"files": [model.name]}) == (model,)
    assert ProcessSupervisor._gguf_path(tmp_path, {}) == model


def test_multimodal_projector_resolution_matches_selected_model(tmp_path: Path) -> None:
    model = tmp_path / "vision-model-4B-Q4_K_M.gguf"
    matching = tmp_path / "mmproj-vision-model-4B-f16.gguf"
    other = tmp_path / "mmproj-vision-model-8B-f16.gguf"
    for candidate in (model, matching, other):
        candidate.write_bytes(candidate.name.encode())
    manifest = {"files": [model.name, matching.name, other.name]}

    assert (
        ProcessSupervisor._llama_mmproj_path(
            tmp_path,
            manifest,
            (model,),
        )
        == matching
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows short-path behavior")
def test_long_llama_model_path_handles_shortening_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / ("descriptive-model-" * 8)
    model_dir.mkdir()
    model_path = model_dir / (("quantized-model-" * 5) + ".gguf")
    model_path.touch()
    short_alias = tmp_path / "MODEL~1.GGF"
    os.link(model_path, short_alias)
    assert len(str(model_path)) >= 240

    class FakeShortPath:
        argtypes: object = None
        restype: object = None

        def __init__(self, result: Path) -> None:
            self.result = result

        def __call__(self, _path: str, buffer: Any, _length: int) -> int:
            buffer.value = str(self.result)
            return len(str(self.result))

    get_short_path = FakeShortPath(short_alias)
    monkeypatch.setattr(
        "local_lm.processes.ctypes.WinDLL",
        lambda _name, **_kwargs: SimpleNamespace(
            GetShortPathNameW=get_short_path,
        ),
    )

    launch_path = ProcessSupervisor._llama_model_path(model_path)

    assert len(str(launch_path)) < 260
    assert os.path.samefile(launch_path, model_path)

    get_short_path.result = model_path
    with pytest.raises(OSError, match="remains too long"):
        ProcessSupervisor._llama_model_path(model_path)


def test_chat_memory_estimate_includes_model_and_context_overhead() -> None:
    gib = 1024**3
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 8192}) == 6 * gib
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 2048}) == (
        5 * gib + 512 * 1024**2
    )


def test_worker_log_rotation_enforces_file_and_retention_bounds(tmp_path: Path) -> None:
    log_path = tmp_path / "chat-worker.log"
    log_path.write_bytes(b"old-output-" * 10)
    log_path.with_name("chat-worker.log.1").write_bytes(b"older-output-" * 10)
    log_path.with_name("chat-worker.log.3").write_bytes(b"expired")

    worker_log = _RotatingWorkerLog(log_path, max_bytes=32, backup_count=2)
    worker_log.write(b"new-output-" * 12)
    worker_log.close()

    retained = [log_path, *sorted(tmp_path.glob("chat-worker.log.*"))]
    assert len(retained) == 3
    assert all(path.stat().st_size <= 32 for path in retained)
    assert not log_path.with_name("chat-worker.log.3").exists()
    assert sum(path.stat().st_size for path in retained) <= 32 * 3


async def test_private_session_suppresses_worker_logs_and_diagnostic_tail(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    marker = b"PRIVATE-WORKER-MARKER-19f25a"
    stdout.feed_data(marker)
    stdout.feed_eof()
    stderr.feed_data(marker)
    stderr.feed_eof()
    log_path = settings.log_dir / "private-worker.log"
    record = WorkerRecord(
        name="chat",
        process=SimpleNamespace(stdout=stdout, stderr=stderr),  # type: ignore[arg-type]
        command=[],
        log=_RotatingWorkerLog(log_path),
    )

    supervisor.begin_private_session()
    await supervisor._capture_process_output(record)
    supervisor.end_private_session()

    assert supervisor.private_output_suppressed is False
    assert record.stderr_tail == b""
    assert marker not in log_path.read_bytes()


async def test_startup_exit_retains_redacted_stderr_and_actionable_status(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    settings.hf_token = "hf_private_worker_token"
    settings.civitai_token = "civitai_private_worker_token"
    private_model_path = settings.model_dir / "private-model.gguf"
    stderr = (
        "Authorization: Bearer hf_private_worker_token\n"
        "CivitAI token: civitai_private_worker_token\n"
        f"failed to open {private_model_path}\n"
    )
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())

    with pytest.raises(RuntimeError) as raised:
        await supervisor._replace(
            "chat",
            [
                sys.executable,
                "-c",
                f"import sys; sys.stderr.write({stderr!r}); raise SystemExit(7)",
            ],
            "http://127.0.0.1:9/health",
        )

    status = supervisor.statuses()[0]
    surfaced = f"{raised.value}\n{status.model_dump_json()}"
    assert status.state == "exited"
    assert status.managed is True
    assert status.exit_code == 7
    assert status.failure_detail == "chat worker exited with code 7."
    assert status.stderr_tail
    assert "[redacted]" in status.stderr_tail
    assert "[data folder]" in status.stderr_tail
    assert status.log_path == "logs/chat-worker.log"
    assert settings.hf_token not in surfaced
    assert settings.civitai_token not in surfaced
    assert str(settings.data_dir.resolve()) not in surfaced
    assert all(str(settings.data_dir.resolve()) not in argument for argument in status.command)
    await supervisor.close()


async def test_startup_exit_with_empty_stderr_has_no_synthetic_tail(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())

    with pytest.raises(RuntimeError, match=r"chat worker exited with code 4\.$"):
        await supervisor._replace(
            "chat",
            [sys.executable, "-c", "raise SystemExit(4)"],
            "http://127.0.0.1:9/health",
        )

    status = supervisor.statuses()[0]
    assert status.failure_detail == "chat worker exited with code 4."
    assert status.stderr_tail is None
    await supervisor.close()


async def test_worker_subprocess_does_not_inherit_application_or_cloud_secrets(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 1
        returncode = 0

    async def create_process(*_command: str, **kwargs: object) -> FakeProcess:
        captured.update(kwargs)
        return FakeProcess()

    async def capture_output(_record: object) -> None:
        return None

    async def healthy(_record: object, _url: str) -> None:
        return None

    async def port_available(_name: str, _url: str) -> None:
        return None

    monkeypatch.setenv("LOCAL_LM_HF_TOKEN", "hf_private")
    monkeypatch.setenv("GITHUB_TOKEN", "github_private")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_private")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(supervisor, "_capture_process_output", capture_output)
    monkeypatch.setattr(supervisor, "_wait_healthy", healthy)
    monkeypatch.setattr(supervisor, "_ensure_port_available", port_available)

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "PATH" in environment
    assert "LOCAL_LM_HF_TOKEN" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    if os.name == "nt":
        assert captured["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in captured
    await supervisor.close()


async def test_worker_port_is_preflighted_before_spawn(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    create_process = AsyncMock()
    preflight = AsyncMock(side_effect=OSError("worker port is already in use"))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(supervisor, "_ensure_port_available", preflight)

    with pytest.raises(OSError, match="already in use"):
        await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")

    preflight.assert_awaited_once_with("chat", "http://127.0.0.1:12341/health")
    create_process.assert_not_awaited()
    assert "chat" not in supervisor._workers


async def test_worker_port_preflight_distinguishes_free_and_bound_ports(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as temporary:
        temporary.bind(("127.0.0.1", 0))
        free_port = temporary.getsockname()[1]

    await supervisor._ensure_port_available(
        "chat",
        f"http://127.0.0.1:{free_port}/health",
    )

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied_port = listener.getsockname()[1]
        with pytest.raises(OSError, match="already in use"):
            await supervisor._ensure_port_available(
                "chat",
                f"http://127.0.0.1:{occupied_port}/health",
            )


async def test_cancelled_worker_start_terminates_and_forgets_starting_process(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    waiting = asyncio.Event()

    class FakeProcess:
        pid = 987_654_321
        returncode: int | None = None
        stdout = None
        stderr = None
        terminated = False

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or 0

    process = FakeProcess()

    async def create_process(*_command: str, **_kwargs: object) -> FakeProcess:
        return process

    async def wait_healthy(_record: object, _url: str) -> None:
        await waiting.wait()

    async def port_available(_name: str, _url: str) -> None:
        return None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(supervisor, "_wait_healthy", wait_healthy)
    monkeypatch.setattr(supervisor, "_ensure_port_available", port_available)

    start = asyncio.create_task(
        supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")
    )
    while "chat" not in supervisor._workers:
        await asyncio.sleep(0)
    start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start

    assert process.terminated is True
    assert "chat" not in supervisor._workers


async def test_stopping_worker_terminates_descendant_process_tree(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    settings.worker_shutdown_seconds = 1
    supervisor = ProcessSupervisor(settings)
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import pathlib, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
        "time.sleep(60)"
    )

    async def healthy(_record: object, _url: str) -> None:
        return None

    async def port_available(_name: str, _url: str) -> None:
        return None

    monkeypatch.setattr(supervisor, "_wait_healthy", healthy)
    monkeypatch.setattr(supervisor, "_ensure_port_available", port_available)
    child_pid: int | None = None
    try:
        await supervisor._replace(
            "chat",
            [sys.executable, "-c", script],
            "http://127.0.0.1:12341/health",
        )
        for _attempt in range(100):
            if child_pid_file.is_file():
                child_pid = int(child_pid_file.read_text())
                break
            await asyncio.sleep(0.02)
        assert child_pid is not None
        assert psutil.pid_exists(child_pid)

        await supervisor.stop("chat")

        assert not psutil.pid_exists(child_pid)
    finally:
        if child_pid and psutil.pid_exists(child_pid):
            with contextlib.suppress(psutil.Error):
                psutil.Process(child_pid).kill()


async def test_exited_record_reaps_exact_persisted_descendants(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    persisted = SimpleNamespace(pid=987_654_399)
    terminated: list[object] = []
    record = WorkerRecord(
        name="chat",
        process=SimpleNamespace(pid=987_654_398, returncode=9),  # type: ignore[arg-type]
        command=["worker"],
        log=_RotatingWorkerLog(settings.log_dir / "exited-worker.log"),
        state="ready",
    )
    monkeypatch.setattr(
        supervisor,
        "_matching_worker_processes",
        lambda _name: [persisted],
    )
    monkeypatch.setattr(
        supervisor,
        "_terminate_processes",
        lambda processes, _timeout: terminated.extend(processes),
    )
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda _name: None)

    await supervisor._terminate_record(record)

    assert terminated == [persisted]


def test_persisted_worker_identity_reaps_only_matching_process(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=creationflags,
    )
    try:
        identity_path = settings.state_dir / "worker-processes.json"
        identity_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "workers": {
                        "chat": [
                            {
                                "pid": child.pid,
                                "create_time": psutil.Process(child.pid).create_time(),
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        ProcessSupervisor(settings)

        child.wait(timeout=5)
        assert not identity_path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_persisted_worker_identity_does_not_kill_reused_pid(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=creationflags,
    )
    try:
        identity_path = settings.state_dir / "worker-processes.json"
        identity_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "workers": {
                        "chat": [
                            {
                                "pid": child.pid,
                                "create_time": psutil.Process(child.pid).create_time() + 1,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

        ProcessSupervisor(settings)

        assert child.poll() is None
        assert not identity_path.exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


def test_malformed_worker_identity_record_fails_safe(settings) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    identity_path = settings.state_dir / "worker-processes.json"
    identity_path.write_text("not-json", encoding="utf-8")

    ProcessSupervisor(settings)

    assert not identity_path.exists()


async def test_worker_status_records_spawn_to_health_duration(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    ticks = iter((100.0, 100.456))

    class FakeProcess:
        pid = os.getpid()
        returncode: int | None = None

        def terminate(self) -> None:
            self.returncode = 0

        async def wait(self) -> int:
            return self.returncode or 0

    async def create_process(*_command: str, **_kwargs: object) -> FakeProcess:
        return FakeProcess()

    async def capture_output(_record: object) -> None:
        return None

    async def healthy(_record: object, _url: str) -> None:
        return None

    async def port_available(_name: str, _url: str) -> None:
        return None

    monkeypatch.setattr("local_lm.processes.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(supervisor, "_capture_process_output", capture_output)
    monkeypatch.setattr(supervisor, "_wait_healthy", healthy)
    monkeypatch.setattr(supervisor, "_ensure_port_available", port_available)

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")

    status = supervisor.statuses()[0]
    assert status.state == "ready"
    assert status.startup_duration_ms == 456
    await supervisor.close()


async def test_worker_health_probe_ignores_proxy_environment(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, _url: str, *, timeout: float) -> SimpleNamespace:
            assert timeout == 5.0
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr("local_lm.processes.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(supervisor, "_listener_owned_by_worker", lambda *_args: True)
    record = SimpleNamespace(
        name="chat",
        process=SimpleNamespace(pid=os.getpid(), returncode=None),
    )

    await supervisor._wait_healthy(record, "http://127.0.0.1:12341/health")

    assert captured == {
        "trust_env": False,
        "verify": shared_tls_context(trust_environment=False),
    }
    await supervisor.close()


async def test_worker_health_probe_backs_off_between_loading_responses(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    delays: list[float] = []
    record = SimpleNamespace(
        name="chat",
        process=SimpleNamespace(pid=456_789_123, returncode=None),
    )

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, _url: str, *, timeout: float) -> SimpleNamespace:
            assert timeout == 5.0
            return SimpleNamespace(is_success=False, status_code=503)

    async def sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) == 4:
            record.process.returncode = 1

    monkeypatch.setattr("local_lm.processes.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("local_lm.processes.asyncio.sleep", sleep)
    monkeypatch.setattr(supervisor, "_record_worker_process_tree", lambda *_args: None)

    with pytest.raises(RuntimeError, match="exited with code 1"):
        await supervisor._wait_healthy(record, "http://127.0.0.1:12341/health")

    assert delays == [0.25, 0.5, 1.0, 2.0]


async def test_worker_health_rejects_listener_owned_by_another_process(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def get(self, _url: str, *, timeout: float) -> SimpleNamespace:
            assert timeout == 5.0
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr("local_lm.processes.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr(supervisor, "_listener_owned_by_worker", lambda *_args: False)
    record = SimpleNamespace(
        name="chat",
        process=SimpleNamespace(pid=456_789_123, returncode=None),
    )

    with pytest.raises(RuntimeError, match="another process"):
        await supervisor._wait_healthy(record, "http://127.0.0.1:12341/health")


async def test_runtime_exit_captures_only_a_bounded_stderr_tail(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    async def healthy_immediately(*_args: object) -> None:
        return None

    monkeypatch.setattr(supervisor, "_wait_healthy", healthy_immediately)
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    await supervisor._replace(
        "chat",
        [
            sys.executable,
            "-c",
            "import sys; sys.stderr.write('x' * 20000 + '\\nlast failure\\n'); raise SystemExit(9)",
        ],
        "http://127.0.0.1:9/health",
    )
    await supervisor._workers["chat"].process.wait()
    output_task = supervisor._workers["chat"].output_task
    assert output_task is not None
    await output_task

    record = supervisor._workers["chat"]
    status = supervisor.statuses()[0]
    assert status.state == "exited"
    assert status.exit_code == 9
    assert status.failure_detail == "chat worker exited with code 9."
    assert status.stderr_tail
    assert status.stderr_tail.endswith("last failure")
    assert len(status.stderr_tail) <= WORKER_STDERR_DISPLAY_CHARS + 1
    assert len(record.stderr_tail) <= WORKER_STDERR_TAIL_BYTES
    await supervisor.close()


async def test_loading_health_503_lines_do_not_displace_stderr_tail(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_data(b"CUDA allocation failed before loading\nINFO GET /hea")
    stderr.feed_data(b"lth HTTP/1.1 503 Loading model\nfatal model error after loading\n")
    stderr.feed_eof()
    log_path = settings.log_dir / "loading-worker.log"
    record = WorkerRecord(
        name="chat",
        process=SimpleNamespace(stdout=stdout, stderr=stderr),  # type: ignore[arg-type]
        command=[],
        log=_RotatingWorkerLog(log_path),
    )

    await supervisor._capture_process_output(record)

    assert b"CUDA allocation failed" in record.stderr_tail
    assert b"fatal model error" in record.stderr_tail
    assert b"/health" not in record.stderr_tail
    assert b"GET /health HTTP/1.1 503" in log_path.read_bytes()


async def _captured_stderr(
    settings: Any,
    supervisor: ProcessSupervisor,
    payload: bytes,
    log_name: str,
) -> tuple[WorkerRecord, Path]:
    stdout = asyncio.StreamReader()
    stderr = asyncio.StreamReader()
    stdout.feed_eof()
    stderr.feed_data(payload)
    stderr.feed_eof()
    log_path = settings.log_dir / log_name
    record = WorkerRecord(
        name="media",
        process=SimpleNamespace(stdout=stdout, stderr=stderr),  # type: ignore[arg-type]
        command=[],
        log=_RotatingWorkerLog(log_path),
    )
    await supervisor._capture_process_output(record)
    return record, log_path


async def test_a_failure_is_classified_from_output_the_display_tail_cannot_show(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The line that names an out-of-memory failure is printed when it happens.

    Everything after it - unloading, shutdown, the engine's parting messages -
    pushes it out of the twelve lines a user is shown. Classifying against the
    display string would therefore miss exactly the failures worth naming, so it
    reads the whole retained buffer instead.
    """
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    record, _ = await _captured_stderr(
        settings,
        supervisor,
        b"ggml_backend_cuda_buffer_type_alloc_buffer: allocating 8192.00 MiB "
        b"on device 0 failed: out of memory\n" + (b"unloading model tensors\n" * 30),
        "buried-oom-worker.log",
    )
    record.process = SimpleNamespace(returncode=1, pid=None)  # type: ignore[assignment]
    supervisor._workers["media"] = record

    status = next(item for item in supervisor.statuses() if item.name == "media")

    assert status.stderr_tail is not None
    assert "out of memory" not in status.stderr_tail, "the display tail must have scrolled past it"
    assert status.failure_code == WorkerFailureCode.OOM_VRAM
    assert status.failure_remedy is not None
    assert "graphics card" in status.failure_remedy


async def test_connection_teardown_noise_never_displaces_the_real_failure(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The diagnostic a user sees must not be filled with routine teardown.

    asyncio logs eight lines when a proactor transport's socket refuses its
    shutdown, which on Windows is WSAEINVAL after a client disconnects. It is
    logged after `connection_lost` has already reached the protocol, so nothing
    was dropped - but two of these blocks are sixteen lines, and only the last
    twelve are shown when a worker dies. The cause would scroll away.
    """
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    record, log_path = await _captured_stderr(
        settings,
        supervisor,
        b"CUDA error: out of memory while loading the checkpoint\n"
        + CONNECTION_TEARDOWN_BLOCK
        + CONNECTION_TEARDOWN_BLOCK,
        "teardown-worker.log",
    )

    tail = supervisor._stderr_tail(record)
    assert tail is not None
    assert "CUDA error: out of memory" in tail
    assert "WinError 10022" not in tail
    assert "_call_connection_lost" not in tail
    # Whole-log fidelity is the point of keeping the file: only the twelve-line
    # diagnostic drops these lines.
    assert b"WinError 10022" in log_path.read_bytes()


async def test_the_frozen_runtime_form_of_the_teardown_block_is_filtered(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The shape that actually ships, not the one a bare interpreter prints.

    A packaged worker prefixes the level tag, renders the callback with empty
    parentheses, and omits source lines from the traceback. An anchor that
    required the line to begin with "Exception in callback" matched the
    reproduction and none of the real captures.
    """
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    record, log_path = await _captured_stderr(
        settings,
        supervisor,
        b"[ERROR] Could not allocate the requested VRAM for this workflow\n"
        + FROZEN_RUNTIME_TEARDOWN_BLOCK * 3,
        "frozen-teardown-worker.log",
    )

    tail = supervisor._stderr_tail(record)
    assert tail is not None
    assert "Could not allocate the requested VRAM" in tail
    assert "WinError 10022" not in tail
    assert "_call_connection_lost" not in tail
    assert b"WinError 10022" in log_path.read_bytes()


async def test_a_real_traceback_after_teardown_noise_is_kept_in_full(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Suppression ends at the block's own exception line, not on a line budget.

    Ending only on a budget would let the next traceback - which starts with the
    same "Traceback (most recent call last):" line - be swallowed as if it were
    more of the same block.
    """
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    real_failure = (
        b"Traceback (most recent call last):\n"
        b'  File "comfy\\execution.py", line 300, in execute\n'
        b"    raise RuntimeError(message)\n"
        b"RuntimeError: the workflow requires a node that is not installed\n"
    )
    record, _ = await _captured_stderr(
        settings,
        supervisor,
        CONNECTION_TEARDOWN_BLOCK + real_failure,
        "teardown-then-real-worker.log",
    )

    tail = supervisor._stderr_tail(record)
    assert tail is not None
    assert "WinError 10022" not in tail
    assert "the workflow requires a node that is not installed" in tail
    assert "comfy\\execution.py" in tail
    # `_stderr_tail` strips each line for display, so compare on that basis.
    assert tail.splitlines() == [
        line.decode().strip() for line in real_failure.splitlines() if line.strip()
    ]


async def test_an_unrelated_callback_exception_is_still_reported(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Only this one teardown callback is filtered, not every asyncio error."""
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    record, _ = await _captured_stderr(
        settings,
        supervisor,
        b"Exception in callback ModelLoader._finish(None)\n"
        b"OSError: [WinError 10022] An invalid argument was supplied\n",
        "unrelated-callback-worker.log",
    )

    tail = supervisor._stderr_tail(record)
    assert tail is not None
    assert "ModelLoader._finish" in tail
    assert "WinError 10022" in tail


async def test_teardown_suppression_is_bounded_when_the_block_never_ends(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """A malformed block cannot silence the stream indefinitely."""
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    unterminated = b"Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)\n"
    record, _ = await _captured_stderr(
        settings,
        supervisor,
        unterminated + (b"    frame line\n" * 40) + b"the real cause\n",
        "unterminated-teardown-worker.log",
    )

    tail = supervisor._stderr_tail(record)
    assert tail is not None
    assert "the real cause" in tail
    assert tail.count("frame line") <= WORKER_STDERR_DISPLAY_LINES


async def test_chat_first_use_provisions_missing_runtime(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF")
    executable = tmp_path / "llama-server.exe"

    async def provision(engine: str) -> None:
        assert engine == "llama.cpp"
        executable.write_bytes(b"runtime")
        settings.llama_executable = executable

    runtimes = SimpleNamespace(ensure=AsyncMock(side_effect=provision))
    supervisor = ProcessSupervisor(settings, runtimes)
    captured: dict[str, list[str]] = {}

    async def replace(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        assert name == "chat"
        assert estimated_memory_bytes is not None
        captured["command"] = command

    monkeypatch.setattr(supervisor, "_replace", replace)
    profile = ModelProfile(
        id="profile_first_use",
        model_install_id="model_first_use",
        name="First use",
        role="chat",
        engine="llama.cpp",
        load_settings_json={},
    )
    install = ModelInstall(
        id="model_first_use",
        name="First use",
        role="chat",
        engine="llama.cpp",
        local_path=str(model_path),
        active=True,
    )

    await supervisor.load_chat(profile, install)

    runtimes.ensure.assert_awaited_once_with("llama.cpp")
    assert captured["command"][0] == str(executable.resolve())


async def test_media_first_use_provisions_missing_runtime(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    runtime = tmp_path / "ComfyUI"
    executable = tmp_path / "python.exe"

    async def provision(engine: str) -> None:
        assert engine == "comfyui"
        runtime.mkdir()
        (runtime / "main.py").write_bytes(b"")
        executable.write_bytes(b"runtime")
        settings.comfy_directory = runtime
        settings.comfy_executable = executable

    runtimes = SimpleNamespace(ensure=AsyncMock(side_effect=provision))
    supervisor = ProcessSupervisor(settings, runtimes)
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}
    phases: list[str] = []

    async def record_phase(phase: str) -> None:
        phases.append(phase)

    async def trusted_nodes() -> list[str]:
        return []

    async def replace(
        name: str,
        command: list[str],
        health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        del estimated_memory_bytes
        assert name == "media"
        captured["command"] = command
        captured["health_url"] = health_url

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda *_args: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media(phase_callback=record_phase)

    runtimes.ensure.assert_awaited_once_with("comfyui")
    assert phases == [
        "Provisioning media runtime",
        "Validating media dependencies",
        "Starting media runtime",
    ]
    assert captured["command"][0] == str(executable.resolve())
    assert captured["command"][1] == str((runtime / "main.py").resolve())
    assert captured["health_url"] == settings.comfy_url + "/system_stats"


async def test_vllm_chat_launches_complete_modelopt_snapshot(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "modelopt-snapshot"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "hf_quant_config.json").write_text(
        '{"quantization": "NVFP4"}',
        encoding="utf-8",
    )
    weights = model_dir / "model-00001-of-00001.safetensors"
    weights.write_bytes(b"modelopt")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"runtime")
    settings.vllm_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    async def replace(
        name: str,
        command: list[str],
        health_url: str,
        profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        captured.update(
            name=name,
            command=command,
            health_url=health_url,
            profile_id=profile_id,
            estimated_memory_bytes=estimated_memory_bytes,
        )

    monkeypatch.setattr(supervisor, "_replace", replace)
    profile = ModelProfile(
        id="profile_nvfp4",
        model_install_id="model_nvfp4",
        name="NVFP4 vision",
        role="chat",
        engine="vllm",
        load_settings_json={
            "context_length": 4096,
            "cpu_offload_gb": 2,
        },
    )
    install = ModelInstall(
        id="model_nvfp4",
        name="NVFP4 vision",
        role="chat",
        engine="vllm",
        local_path=str(model_dir),
        manifest_json={
            "files": [
                "config.json",
                "hf_quant_config.json",
                weights.name,
            ]
        },
        active=True,
    )

    await supervisor.load_chat(profile, install)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [
        str(executable.resolve()),
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert command[command.index("--model") + 1] == str(model_dir.resolve())
    assert command[command.index("--quantization") + 1] == "modelopt"
    assert command[command.index("--max-model-len") + 1] == "4096"
    assert command[command.index("--cpu-offload-gb") + 1] == "2.0"
    assert captured["health_url"] == settings.llama_url + "/health"
    assert captured["profile_id"] == profile.id
    assert captured["estimated_memory_bytes"] == ProcessSupervisor._estimate_chat_memory(
        sum(path.stat().st_size for path in model_dir.iterdir()),
        profile.load_settings_json,
    )
    assert settings.chat_engine == "vllm"


async def test_vllm_chat_rejects_incomplete_snapshot(
    settings,
    tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "incomplete-modelopt"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"weights")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"runtime")
    settings.vllm_executable = executable
    supervisor = ProcessSupervisor(settings)
    profile = ModelProfile(
        id="profile_incomplete_nvfp4",
        model_install_id="model_incomplete_nvfp4",
        name="Incomplete NVFP4",
        role="chat",
        engine="vllm",
        load_settings_json={},
    )
    install = ModelInstall(
        id="model_incomplete_nvfp4",
        name="Incomplete NVFP4",
        role="chat",
        engine="vllm",
        local_path=str(model_dir),
        manifest_json={},
        active=True,
    )

    with pytest.raises(ValueError, match="missing ModelOpt metadata"):
        await supervisor.load_chat(profile, install)


async def test_chat_launches_split_gguf_from_first_shard(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "split-model"
    model_dir.mkdir()
    first = model_dir / "model-Q4_K_M-00001-of-00002.gguf"
    second = model_dir / "model-Q4_K_M-00002-of-00002.gguf"
    first.write_bytes(b"first-shard")
    second.write_bytes(b"second-shard")
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"runtime")
    settings.llama_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    async def replace(
        _name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        captured["command"] = command
        captured["estimated_memory_bytes"] = estimated_memory_bytes

    monkeypatch.setattr(supervisor, "_replace", replace)
    profile = ModelProfile(
        id="profile_split",
        model_install_id="model_split",
        name="Split model",
        role="chat",
        engine="llama.cpp",
        load_settings_json={},
    )
    install = ModelInstall(
        id="model_split",
        name="Split model",
        role="chat",
        engine="llama.cpp",
        local_path=str(model_dir),
        manifest_json={"files": [second.name, first.name]},
        active=True,
    )

    await supervisor.load_chat(profile, install)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--model") + 1] == str(first)
    assert captured["estimated_memory_bytes"] == ProcessSupervisor._estimate_chat_memory(
        first.stat().st_size + second.stat().st_size,
        {},
    )


async def test_chat_launches_multimodal_projector_and_includes_its_memory(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    model_dir = tmp_path / "vision-model"
    model_dir.mkdir()
    model = model_dir / "vision-model-4B-Q4_K_M.gguf"
    projector = model_dir / "mmproj-vision-model-4B-f16.gguf"
    model.write_bytes(b"model")
    projector.write_bytes(b"projector")
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"runtime")
    settings.llama_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    async def replace(
        _name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        captured["command"] = command
        captured["estimated_memory_bytes"] = estimated_memory_bytes

    monkeypatch.setattr(supervisor, "_replace", replace)
    profile = ModelProfile(
        id="profile_vision",
        model_install_id="model_vision",
        name="Vision model",
        role="chat",
        engine="llama.cpp",
        load_settings_json={},
    )
    install = ModelInstall(
        id="model_vision",
        name="Vision model",
        role="chat",
        engine="llama.cpp",
        local_path=str(model_dir),
        manifest_json={"files": [model.name, projector.name]},
        active=True,
    )

    await supervisor.load_chat(profile, install)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--mmproj") + 1] == str(projector)
    assert captured["estimated_memory_bytes"] == ProcessSupervisor._estimate_chat_memory(
        model.stat().st_size + projector.stat().st_size,
        {},
    )


async def test_media_start_disables_unapproved_custom_nodes(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    runtime = tmp_path / "comfyui"
    runtime.mkdir()
    (runtime / "main.py").touch()
    executable = tmp_path / "python.exe"
    executable.touch()
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.touch()
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, list[str]] = {}

    async def trusted_nodes() -> list[str]:
        return ["lm-atelier-node_reviewed"]

    async def replace(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        assert name == "media"
        assert estimated_memory_bytes is None
        captured["command"] = command

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    command = captured["command"]
    output_directory = settings.state_dir / "comfy-output"
    assert output_directory.is_dir()
    assert command[command.index("--output-directory") + 1] == str(output_directory.resolve())
    assert command[command.index("--preview-method") + 1] == "latent2rgb"
    assert "--disable-all-custom-nodes" in command
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == ["lm-atelier-node_reviewed"]


async def test_media_start_whitelists_only_the_verified_first_party_editor_bridge(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    portable = tmp_path / "ComfyUI_windows_portable"
    runtime = portable / "ComfyUI"
    runtime.mkdir(parents=True)
    (runtime / "main.py").touch()
    executable = portable / "python_embeded" / "python.exe"
    executable.parent.mkdir()
    executable.touch()
    dist_info = (
        executable.parent / "Lib" / "site-packages" / "comfyui_frontend_package-1.45.21.dist-info"
    )
    dist_info.mkdir(parents=True)
    (runtime / "comfyui_version.py").write_text(
        '__version__ = "0.28.0"' + chr(10),
        encoding="utf-8",
    )
    (dist_info / "METADATA").write_text(
        chr(10).join(["Name: comfyui-frontend-package", "Version: 1.45.21", ""]),
        encoding="utf-8",
    )
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.touch()
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, list[str]] = {}

    async def trusted_nodes() -> list[str]:
        return []

    async def replace(
        _name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        assert estimated_memory_bytes is None
        captured["command"] = command

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    command = captured["command"]
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == [BRIDGE_DIRECTORY_NAME]
    staged = runtime / "custom_nodes" / BRIDGE_DIRECTORY_NAME
    assert (staged / "__init__.py").is_file()
    assert (staged / "js" / "lm_atelier_workflow_editor.js").is_file()


async def test_media_start_uses_only_the_exact_activation_scope(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    runtime = tmp_path / "comfyui"
    runtime.mkdir()
    (runtime / "main.py").touch()
    executable = tmp_path / "python.exe"
    executable.touch()
    model_root = tmp_path / "selected-model"
    model_root.mkdir()
    asset_root = tmp_path / "selected-asset"
    asset_root.mkdir()
    site_packages = tmp_path / "selected-registry" / "site-packages"
    site_packages.mkdir(parents=True)
    digest = "a" * 64
    scope = WorkflowActivationLaunchScope(
        "wfact_selected",
        "wfrev_selected",
        "b" * 64,
        digest,
        ("model_selected",),
        ("asset_selected",),
        (),
        (),
        (),
        (
            WorkflowModelLaunchBinding(
                "model_selected",
                model_root.resolve(),
                (("checkpoints", "."),),
                (),
            ),
        ),
        (
            WorkflowAssetLaunchBinding(
                "asset_selected",
                asset_root.resolve(),
                "loras",
                "style.safetensors",
                "c" * 64,
            ),
        ),
        (),
        (),
        (),
    )
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}

    async def scoped_nodes(
        current: WorkflowActivationLaunchScope,
    ) -> tuple[list[str], tuple[str, ...]]:
        assert current is scope
        return ["lm-atelier-node_selected"], ("SelectedNode",)

    async def broad_nodes() -> list[str]:
        raise AssertionError("scoped launch queried broad custom-node state")

    async def replace_worker(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        _estimated_memory_bytes: int | None = None,
        *,
        environment_overrides: dict[str, str] | None = None,
        ready_check=None,  # type: ignore[no-untyped-def]
        launch_scope_sha256: str | None = None,
    ) -> None:
        captured.update(
            name=name,
            command=command,
            environment_overrides=environment_overrides,
            ready_check=ready_check,
            launch_scope_sha256=launch_scope_sha256,
        )

    monkeypatch.setattr(supervisor, "_scoped_comfy_node_folders", scoped_nodes)
    monkeypatch.setattr(
        supervisor,
        "_scoped_comfy_registry_contract",
        lambda current: (
            ComfyRegistryLaunchContract(
                ("lm-atelier-registry_selected",),
                (site_packages,),
                ("RegistryNode",),
            )
            if current is scope
            else None
        ),
    )
    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", broad_nodes)
    monkeypatch.setattr(
        supervisor,
        "_trusted_comfy_registry_contract",
        lambda: (_ for _ in ()).throw(AssertionError("scoped launch queried broad Registry state")),
    )
    monkeypatch.setattr(supervisor, "_replace", replace_worker)

    await supervisor.start_media(activation_scope=scope)

    command = captured["command"]
    assert isinstance(command, list)
    config_path = Path(command[command.index("--extra-model-paths-config") + 1])
    assert config_path.name == f"{digest}.yaml"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "local_lm_1": {
            "base_path": str(model_root.resolve()),
            "checkpoints": ".",
        },
        "local_lm_2": {
            "base_path": str(asset_root.resolve()),
            "loras": ".",
        },
    }
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == [
        "lm-atelier-node_selected",
        "lm-atelier-registry_selected",
    ]
    assert command[:4] == [str(executable.resolve()), "-c", command[2], "1"]
    assert command[4:6] == [str(site_packages.resolve()), str((runtime / "main.py").resolve())]
    assert captured["environment_overrides"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert callable(captured["ready_check"])
    assert captured["launch_scope_sha256"] == digest


async def test_activation_scoped_worker_is_reused_only_for_the_same_ready_launch(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    supervisor = ProcessSupervisor(settings)
    command = ["python", "worker.py"]
    process = FakeRunningProcess(12345, terminate_code=0)
    log = _RotatingWorkerLog(tmp_path / "reuse.log")
    record = WorkerRecord(
        "media",
        process,  # type: ignore[arg-type]
        command,
        log,
        state="ready",
        launch_scope_sha256="a" * 64,
    )
    supervisor._workers["media"] = record
    stopped = AsyncMock(side_effect=AssertionError("matching worker was restarted"))
    monkeypatch.setattr(supervisor, "_stop_unlocked", stopped)

    await supervisor._replace(
        "media",
        list(command),
        "http://127.0.0.1:8289/system_stats",
        launch_scope_sha256="a" * 64,
    )

    stopped.assert_not_awaited()
    assert supervisor._workers["media"] is record
    log.close()


async def test_activation_scope_cannot_be_broadened_with_provisional_paths(
    settings,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    supervisor = ProcessSupervisor(settings)
    scope = WorkflowActivationLaunchScope(
        "wfact_selected",
        "wfrev_selected",
        "b" * 64,
        "a" * 64,
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
    )

    with pytest.raises(ValueError, match="cannot broaden"):
        await supervisor.start_media((tmp_path, {}), activation_scope=scope)


async def test_media_start_uses_only_verified_registry_overlay_contract(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    runtime = tmp_path / "comfyui"
    runtime.mkdir()
    (runtime / "main.py").touch()
    executable = tmp_path / "python.exe"
    executable.touch()
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.touch()
    site_packages = tmp_path / "registry-environment" / "site-packages"
    site_packages.mkdir(parents=True)
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    captured: dict[str, object] = {}
    runtime_baseline = (ComfyRegistryRuntimeDistribution("torch", "2.13.0+cu130"),)

    async def trusted_nodes() -> list[str]:
        return ["lm-atelier-node_reviewed"]

    async def replace(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        _estimated_memory_bytes: int | None = None,
        *,
        environment_overrides: dict[str, str] | None = None,
        ready_check=None,  # type: ignore[no-untyped-def]
    ) -> None:
        captured.update(
            name=name,
            command=command,
            environment_overrides=environment_overrides,
            ready_check=ready_check,
        )

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(
        supervisor,
        "_trusted_comfy_registry_contract",
        lambda: ComfyRegistryLaunchContract(
            ("lm-atelier-registry_example",),
            (site_packages,),
            ("ExampleLoader",),
            runtime_baseline,
        ),
    )

    async def probe_runtime(_executable: Path):  # type: ignore[no-untyped-def]
        return {}, (), runtime_baseline

    monkeypatch.setattr(
        registry_interpreter_module,
        "probe_comfy_registry_runtime_target",
        probe_runtime,
    )
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    assert captured["name"] == "media"
    command = captured["command"]
    assert isinstance(command, list)
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == [
        "lm-atelier-node_reviewed",
        "lm-atelier-registry_example",
    ]
    assert command[:4] == [str(executable.resolve()), "-c", command[2], "1"]
    assert command[4:6] == [str(site_packages.resolve()), str((runtime / "main.py").resolve())]
    assert captured["environment_overrides"] == {"PYTHONDONTWRITEBYTECODE": "1"}
    assert callable(captured["ready_check"])


def test_registry_launch_contracts_use_the_managed_registry_root(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    supervisor = ProcessSupervisor(settings)
    expected_root = settings.registry_dir / "registry-wheel-environments"
    observed: list[Path] = []

    def trusted_contract(_session, *, custom_node_root, environment_root):  # type: ignore[no-untyped-def]
        assert custom_node_root == settings.custom_node_dir
        observed.append(environment_root)
        return ComfyRegistryLaunchContract((), (), ())

    def scoped_contract(  # type: ignore[no-untyped-def]
        _session,
        bindings,
        *,
        custom_node_root,
        environment_root,
    ):
        assert tuple(bindings) == scope.registry_packages
        assert custom_node_root == settings.custom_node_dir
        observed.append(environment_root)
        return ComfyRegistryLaunchContract((), (), ())

    monkeypatch.setattr(
        "local_lm.comfy_registry_installs.trusted_comfy_registry_launch_contract",
        trusted_contract,
    )
    monkeypatch.setattr(
        "local_lm.comfy_registry_installs.scoped_comfy_registry_launch_contract",
        scoped_contract,
    )
    scope = SimpleNamespace(
        registry_install_ids=("registry_example",),
        registry_packages=(SimpleNamespace(registry_install_id="registry_example"),),
    )

    supervisor._trusted_comfy_registry_contract()
    supervisor._scoped_comfy_registry_contract(scope)  # type: ignore[arg-type]

    assert observed == [expected_root, expected_root]


async def test_media_start_refuses_registry_overlay_after_runtime_drift(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    runtime = tmp_path / "comfyui"
    runtime.mkdir()
    (runtime / "main.py").touch()
    executable = tmp_path / "python.exe"
    executable.touch()
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    prepared = (ComfyRegistryRuntimeDistribution("torch", "2.13.0+cu130"),)

    async def trusted_nodes() -> list[str]:
        return []

    async def drifted_runtime(_executable: Path):  # type: ignore[no-untyped-def]
        return {}, (), (ComfyRegistryRuntimeDistribution("torch", "2.14.0+cu130"),)

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(
        supervisor,
        "_trusted_comfy_registry_contract",
        lambda: ComfyRegistryLaunchContract((), (tmp_path,), (), prepared),
    )
    monkeypatch.setattr(
        registry_interpreter_module,
        "probe_comfy_registry_runtime_target",
        drifted_runtime,
    )

    with pytest.raises(RuntimeError, match="changed after workflow dependencies"):
        await supervisor.start_media()


def test_registry_overlay_bootstrap_imports_without_executing_pth(tmp_path: Path) -> None:
    site_packages = tmp_path / "registry-environment" / "site-packages"
    site_packages.mkdir(parents=True)
    (site_packages / "registry_probe.py").write_text("VALUE = 7\n", encoding="utf-8")
    (site_packages / "unsafe.pth").write_text(
        "import os; os.environ['LM_ATELIER_PTH_EXECUTED'] = '1'\n",
        encoding="utf-8",
    )
    entrypoint = tmp_path / "main.py"
    entrypoint.write_text(
        "import json, os\n"
        "import registry_probe\n"
        "print(json.dumps({'value': registry_probe.VALUE, "
        "'pth': os.environ.get('LM_ATELIER_PTH_EXECUTED')}))\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("LM_ATELIER_PTH_EXECUTED", None)

    result = subprocess.run(
        _with_comfy_registry_overlays(
            [sys.executable, str(entrypoint)],
            (site_packages,),
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout) == {"value": 7, "pth": None}


@pytest.mark.parametrize(("inventory", "missing"), [({"ExampleLoader": {}}, False), ({}, True)])
async def test_registry_node_inventory_is_verified_before_ready(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    inventory: dict[str, object],
    missing: bool,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    class FakeResponse:
        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):  # type: ignore[no-untyped-def]
            yield json.dumps(inventory).encode()

    class FakeClient:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["trust_env"] is False

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        def stream(self, method: str, url: str, *, timeout: float) -> FakeResponse:
            assert method == "GET"
            assert url.endswith("/object_info")
            assert timeout == 5.0
            return FakeResponse()

    monkeypatch.setattr("local_lm.processes.httpx.AsyncClient", FakeClient)

    if missing:
        with pytest.raises(RuntimeError, match="did not load required node type"):
            await supervisor._verify_comfy_node_types(("ExampleLoader",))
    else:
        await supervisor._verify_comfy_node_types(("ExampleLoader",))


async def test_media_whitelist_contains_only_active_verified_trusted_installs(
    client,
    settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    del client
    runtime = tmp_path / "comfyui"
    custom_nodes = runtime / "custom_nodes"
    custom_nodes.mkdir(parents=True)
    settings.comfy_directory = runtime
    records = [
        CustomNodeInstall(
            id=f"node_{index}",
            name=f"Node {index}",
            source_url=f"https://github.com/example/node-{index}.git",
            revision=str(index) * 40,
            installed_path=folder,
            tree_hash=str(index) * 40,
            trusted=trusted,
            active=active,
            security_json={},
        )
        for index, folder, trusted, active in [
            (1, "lm-atelier-node_trusted", True, True),
            (2, "lm-atelier-node_untrusted", False, True),
            (3, "lm-atelier-node_inactive", True, False),
        ]
    ]
    for record in records:
        (custom_nodes / record.installed_path).mkdir()
    with SessionLocal() as session:
        session.add_all(records)
        session.commit()
    verified: list[str] = []

    async def verify(_manager: CustomNodeManager, install: CustomNodeInstall) -> None:
        assert object_session(install) is None
        verified.append(install.installed_path)

    monkeypatch.setattr(CustomNodeManager, "verify", verify)

    folders = await ProcessSupervisor(settings)._trusted_comfy_node_folders()

    assert folders == ["lm-atelier-node_trusted"]
    assert verified == folders


async def test_liveness_probe_requires_success_from_the_owned_listener(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    process = FakeRunningProcess(987_654_320, terminate_code=-15)
    record = WorkerRecord(
        name="chat",
        process=process,  # type: ignore[arg-type]
        command=["worker"],
        log=_RotatingWorkerLog(settings.log_dir / "probe-worker.log"),
        state="ready",
    )
    client = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(is_success=True)))
    monkeypatch.setattr(
        supervisor,
        "_listener_owned_by_worker",
        lambda _pid, _url: True,
    )
    recorded = []
    monkeypatch.setattr(
        supervisor,
        "_record_worker_process_tree",
        lambda name, pid: recorded.append((name, pid)),
    )

    assert await supervisor._probe_worker_health(
        client,  # type: ignore[arg-type]
        record,
        "http://127.0.0.1:12341/health",
    )
    assert recorded == [("chat", process.pid)]
    client.get.assert_awaited_once_with(
        "http://127.0.0.1:12341/health",
        timeout=5.0,
    )

    monkeypatch.setattr(
        supervisor,
        "_listener_owned_by_worker",
        lambda _pid, _url: False,
    )
    assert not await supervisor._probe_worker_health(
        client,  # type: ignore[arg-type]
        record,
        "http://127.0.0.1:12341/health",
    )
    client.get = AsyncMock(side_effect=httpx.ConnectError("offline"))
    assert not await supervisor._probe_worker_health(
        client,  # type: ignore[arg-type]
        record,
        "http://127.0.0.1:12341/health",
    )


async def test_supervisor_reports_unexpected_worker_exit_without_status_polling(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    events = EventBroker()
    supervisor = ProcessSupervisor(
        settings,
        events=events,
        liveness_interval_seconds=1,
    )
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())

    await supervisor._replace(
        "chat",
        [sys.executable, "-c", "import time; time.sleep(0.05); raise SystemExit(9)"],
        "http://127.0.0.1:12341/health",
    )
    record = supervisor._workers["chat"]
    assert record.monitor_task is not None

    await wait_for_worker_event(events, "worker.exited")

    status = supervisor.statuses()[0]
    event = next(event for event in events.since(0) if event.type == "worker.exited")
    assert status.state == "exited"
    assert status.exit_code == 9
    assert status.failure_detail == "chat worker exited with code 9."
    assert event.entity_id == "chat"
    assert event.payload == {"name": "chat", "state": "exited", "exit_code": 9}
    await asyncio.wait_for(record.monitor_task, timeout=1)
    assert record.monitor_task.done()
    await supervisor.close()


async def test_supervisor_stops_worker_after_consecutive_health_failures(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    events = EventBroker()
    supervisor = ProcessSupervisor(
        settings,
        events=events,
        liveness_interval_seconds=0.001,
        liveness_failure_threshold=2,
    )

    process = FakeRunningProcess(987_654_321, terminate_code=-15)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(supervisor, "_capture_process_output", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    probe = AsyncMock(return_value=False)
    monkeypatch.setattr(supervisor, "_probe_worker_health", probe)

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")
    await wait_for_worker_event(events, "worker.unhealthy")

    status = supervisor.statuses()[0]
    event = next(event for event in events.since(0) if event.type == "worker.unhealthy")
    assert process.terminated is True
    assert probe.await_count == 2
    assert status.state == "exited"
    assert status.failure_detail == "chat worker stopped responding to health checks."
    assert event.payload == {"name": "chat", "state": "exited", "exit_code": -15}
    await supervisor.close()


async def test_supervisor_tolerates_transient_health_failure_and_cleans_monitor(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    events = EventBroker()
    supervisor = ProcessSupervisor(
        settings,
        events=events,
        liveness_interval_seconds=0.001,
        liveness_failure_threshold=2,
    )

    process = FakeRunningProcess(987_654_322, terminate_code=0)
    probes = 0

    async def probe(*_args: object) -> bool:
        nonlocal probes
        probes += 1
        return probes > 1

    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(supervisor, "_capture_process_output", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(supervisor, "_probe_worker_health", probe)

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")
    record = supervisor._workers["chat"]
    assert record.monitor_task is not None
    for _ in range(200):
        if probes >= 3:
            break
        await asyncio.sleep(0.01)

    assert probes >= 3
    assert process.terminated is False
    assert supervisor.statuses()[0].state == "ready"
    assert all(event.type != "worker.unhealthy" for event in events.since(0))

    monitor = record.monitor_task
    await supervisor.stop("chat")
    assert monitor.done()
    assert process.terminated is True


async def test_supervisor_fails_closed_when_monitor_itself_errors(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    events = EventBroker()
    supervisor = ProcessSupervisor(
        settings,
        events=events,
        liveness_interval_seconds=0.001,
    )
    process = FakeRunningProcess(987_654_323, terminate_code=-15)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    monkeypatch.setattr(supervisor, "_capture_process_output", AsyncMock())
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    monkeypatch.setattr(supervisor, "_wait_healthy", AsyncMock())
    monkeypatch.setattr(
        supervisor,
        "_probe_worker_health",
        AsyncMock(side_effect=RuntimeError("probe exploded")),
    )

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1:12341/health")
    record = supervisor._workers["chat"]
    assert record.monitor_task is not None
    await wait_for_worker_event(events, "worker.unhealthy")

    assert process.terminated is True
    assert record.monitor_task.done()
    assert record.monitor_task.exception() is None
    assert supervisor.statuses()[0].failure_detail == (
        "chat worker supervision failed: probe exploded."
    )
    await supervisor.close()
