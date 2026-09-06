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
import local_lm.processes as processes_module
from local_lm.comfy_editor_bridge import (
    BRIDGE_COORDINATOR_CONFIG,
    ComfyEditorBridgeError,
    ComfyEditorBridgeSupport,
    bridge_directory_name,
)
from local_lm.comfy_registry_installs import ComfyRegistryLaunchContract
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import SessionLocal
from local_lm.events import EventBroker
from local_lm.models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    ModelAssetInstall,
    ModelInstall,
    ModelProfile,
)
from local_lm.network import shared_tls_context
from local_lm.processes import (
    WORKER_STDERR_DISPLAY_CHARS,
    WORKER_STDERR_DISPLAY_LINES,
    WORKER_STDERR_TAIL_BYTES,
    ProcessSupervisor,
    WorkerRecord,
    _ProcessIdentity,
    _RotatingWorkerLog,
    _with_comfy_registry_overlays,
)
from local_lm.security import trusted_browser_origins
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


async def test_trusted_registry_node_types_preserve_package_version_ownership(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:  # type: ignore[no-untyped-def]
    install = ComfyRegistryInstall(
        package_id="comfyui-kjnodes",
        package_version="1.2.3",
        registry_record_id="registry-record-kjnodes",
        repository_url="https://github.com/kijai/ComfyUI-KJNodes.git",
        download_url="https://cdn.comfy.org/kjnodes/1.2.3.zip",
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        installed_path="lm-atelier-registry_kjnodes",
        node_types_json=["ImageResizeKJ", "ColorMatch"],
        pip_dependencies_json=[],
        review_json={"review_required": True},
        trusted=True,
        active=True,
    )
    with SessionLocal() as session:
        session.add(install)
        session.commit()
        install_id = install.id

    def cleanup_install() -> None:
        with SessionLocal() as session:
            persisted = session.get(ComfyRegistryInstall, install_id)
            if persisted is not None:
                session.delete(persisted)
                session.commit()

    request.addfinalizer(cleanup_install)

    supervisor = ProcessSupervisor(settings)
    packages = {("comfyui-kjnodes", "1.2.3"): frozenset({"ImageResizeKJ", "ColorMatch"})}
    monkeypatch.setattr(
        "local_lm.comfy_registry_installs.trusted_comfy_registry_launch_contract",
        lambda *_args, **_kwargs: ComfyRegistryLaunchContract(
            ("lm-atelier-registry_kjnodes",), (), ("ImageResizeKJ", "ColorMatch")
        ),
    )

    assert await supervisor.trusted_comfy_registry_package_node_types() == packages

    monkeypatch.setattr(
        "local_lm.comfy_registry_installs.trusted_comfy_registry_launch_contract",
        lambda *_args, **_kwargs: ComfyRegistryLaunchContract(
            ("lm-atelier-registry_kjnodes",), (), ("OtherNode",)
        ),
    )
    with pytest.raises(RuntimeError, match="node ownership is inconsistent"):
        await supervisor.trusted_comfy_registry_package_node_types()


async def test_trusted_manual_node_types_preserve_reviewed_package_ownership(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:  # type: ignore[no-untyped-def]
    install = CustomNodeInstall(
        id="node_reviewed_inventory",
        name="comfyui-kjnodes",
        source_url="https://github.com/example/comfyui-kjnodes.git",
        revision="a" * 40,
        installed_path="lm-atelier-node_reviewed-inventory",
        tree_hash="b" * 40,
        trusted=True,
        active=True,
        security_json={"node_types": ["ColorMatch", "ImageResizeKJ"]},
    )
    with SessionLocal() as session:
        session.add(install)
        session.commit()

    def cleanup_install() -> None:
        with SessionLocal() as session:
            persisted = session.get(CustomNodeInstall, install.id)
            if persisted is not None:
                session.delete(persisted)
                session.commit()

    request.addfinalizer(cleanup_install)
    verified: list[str] = []

    async def verify(_manager: CustomNodeManager, current: CustomNodeInstall) -> None:
        verified.append(current.id)

    monkeypatch.setattr(CustomNodeManager, "verify", verify)
    supervisor = ProcessSupervisor(settings)

    assert await supervisor.trusted_comfy_custom_node_package_node_types() == {
        ("comfyui-kjnodes", "a" * 40): frozenset({"ImageResizeKJ", "ColorMatch"})
    }
    assert verified == [install.id]


async def test_unreviewed_manual_node_types_are_not_launchable_package_evidence(
    settings,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:  # type: ignore[no-untyped-def]
    install = CustomNodeInstall(
        id="node_unreviewed_inventory",
        name="comfyui-kjnodes",
        source_url="https://github.com/example/comfyui-kjnodes.git",
        revision="c" * 40,
        installed_path="lm-atelier-node_unreviewed-inventory",
        tree_hash="d" * 40,
        trusted=True,
        active=True,
        security_json={"review_required": True},
    )
    with SessionLocal() as session:
        session.add(install)
        session.commit()

    def cleanup_install() -> None:
        with SessionLocal() as session:
            persisted = session.get(CustomNodeInstall, install.id)
            if persisted is not None:
                session.delete(persisted)
                session.commit()

    request.addfinalizer(cleanup_install)

    async def verify(_manager: CustomNodeManager, _current: CustomNodeInstall) -> None:
        return None

    monkeypatch.setattr(CustomNodeManager, "verify", verify)

    assert await ProcessSupervisor(settings).trusted_comfy_custom_node_package_node_types() == {}


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


def test_workflow_editor_authority_requires_the_live_ready_verified_launch(
    settings,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    supervisor = ProcessSupervisor(settings)
    process = FakeRunningProcess(34567, terminate_code=0)
    record = WorkerRecord(
        "media",
        process,  # type: ignore[arg-type]
        ["comfy"],
        _RotatingWorkerLog(tmp_path / "media.log"),
        editor_bridge_launch_id="verified-launch",
        editor_bridge_support=ComfyEditorBridgeSupport(
            True,
            "ready",
            "Native workflow editing is available.",
            "0.28.0",
            "1.45.21",
        ),
    )
    supervisor._workers["media"] = record

    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None
    record.state = "ready"
    assert supervisor.workflow_editor_runtime_identity() == "verified-launch"
    assert supervisor.workflow_editor_bridge_support() == record.editor_bridge_support
    record.editor_bridge_launch_id = None
    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None
    record.editor_bridge_launch_id = "verified-launch"
    process.returncode = 1
    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None
    record.log.close()


def test_workflow_editor_support_reports_a_live_ready_refusal(
    settings,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    supervisor = ProcessSupervisor(settings)
    support = ComfyEditorBridgeSupport(
        False,
        "workflow-editor-frontend-unsupported",
        "The configured frontend is not certified for native workflow editing.",
        "0.28.0",
        "1.46.0",
    )
    record = WorkerRecord(
        "media",
        FakeRunningProcess(34568, terminate_code=0),  # type: ignore[arg-type]
        ["comfy"],
        _RotatingWorkerLog(tmp_path / "media-unsupported.log"),
        state="ready",
        editor_bridge_support=support,
    )
    supervisor._workers["media"] = record

    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() == support
    record.editor_bridge_launch_id = "invalid-authority"
    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None
    record.log.close()


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


async def test_port_refusal_names_the_process_holding_the_port(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """A user who hits this must learn what to close.

    The listener here is the test process itself, so the expected name and pid
    are known exactly rather than matched loosely.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied_port = listener.getsockname()[1]
        with pytest.raises(OSError) as refused:
            await supervisor._ensure_port_available(
                "media",
                f"http://127.0.0.1:{occupied_port}/health",
            )

    message = str(refused.value)
    assert "already in use by " in message, message
    assert f"pid {os.getpid()}" in message, message
    assert psutil.Process(os.getpid()).name() in message, message


async def test_port_refusal_keeps_its_original_wording_when_the_holder_is_unknown(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    """Enumerating sockets is privileged on some systems.

    Degradation guard rather than a regression: this passes on the parent too,
    because the parent only ever produces the unnamed message. It is here so a
    later change cannot turn an unidentifiable occupant into a wrong name or a
    lost error.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    def refuse_enumeration(*_args: object, **_kwargs: object) -> list[object]:
        raise psutil.AccessDenied(pid=None, name="net_connections")

    monkeypatch.setattr(processes_module.psutil, "net_connections", refuse_enumeration)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        occupied_port = listener.getsockname()[1]
        with pytest.raises(OSError) as refused:
            await supervisor._ensure_port_available(
                "media",
                f"http://127.0.0.1:{occupied_port}/health",
            )

    assert (
        str(refused.value)
        == f"media worker cannot start because 127.0.0.1:{occupied_port} is already in use"
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


async def test_media_replacement_rotates_verified_editor_launch_authority(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    async def healthy_immediately(*_args: object) -> None:
        return None

    monkeypatch.setattr(supervisor, "_wait_healthy", healthy_immediately)
    monkeypatch.setattr(supervisor, "_ensure_port_available", AsyncMock())
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    support = ComfyEditorBridgeSupport(
        True,
        "ready",
        "Native workflow editing is available.",
        "0.28.0",
        "1.45.21",
    )

    await supervisor._replace(
        "media",
        command,
        "http://127.0.0.1:9/health",
        editor_bridge_support=support,
    )
    first = supervisor.workflow_editor_runtime_identity()
    await supervisor._replace(
        "media",
        command,
        "http://127.0.0.1:9/health",
        editor_bridge_support=support,
    )
    second = supervisor.workflow_editor_runtime_identity()

    assert first
    assert second
    assert second != first
    assert supervisor.workflow_editor_bridge_support() == support
    await supervisor.stop("media")
    assert supervisor.workflow_editor_runtime_identity() is None
    assert supervisor.workflow_editor_bridge_support() is None


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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        del estimated_memory_bytes
        assert editor_bridge_support is not None
        assert not editor_bridge_support.supported
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


@pytest.mark.parametrize(
    ("refused_phase", "effects_before_it"),
    [
        ("Provisioning media runtime", []),
        ("Validating media dependencies", ["provisioned"]),
        ("Starting media runtime", ["provisioned", "inspected nodes", "staged model paths"]),
    ],
)
async def test_a_refused_media_phase_stops_before_the_next_process_effect(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refused_phase: str,
    effects_before_it: list[str],
) -> None:  # type: ignore[no-untyped-def]
    """A refusing callback stops the start at the phase it refused.

    The sibling above shows the ordinary run: three phases, each followed by
    the effect it announces. This is the same start with the announcement
    refused, and the sentinel is the effect list: it must end where the
    refusal was raised, whichever phase that is.

    The distinction is the point. A callback that raises because it is broken
    is still swallowed and the start still completes, because a worker must
    not be lost to a failing progress report. A callback that raises
    `WorkerStartRefused` is saying the row has moved on, and every effect
    after it - provisioning the runtime, reading the trusted node set, staging
    the model paths, launching the process - would be done on another
    attempt's behalf.
    """

    runtime = tmp_path / "ComfyUI"
    executable = tmp_path / "python.exe"
    effects: list[str] = []

    async def provision(engine: str) -> None:
        assert engine == "comfyui"
        effects.append("provisioned")
        runtime.mkdir()
        (runtime / "main.py").write_bytes(b"")
        executable.write_bytes(b"runtime")
        settings.comfy_directory = runtime
        settings.comfy_executable = executable

    runtimes = SimpleNamespace(ensure=AsyncMock(side_effect=provision))
    supervisor = ProcessSupervisor(settings, runtimes)
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.write_text("{}", encoding="utf-8")
    phases: list[str] = []

    async def refuse_at(phase: str) -> None:
        phases.append(phase)
        if phase == refused_phase:
            raise processes_module.WorkerStartRefused(phase)

    async def trusted_nodes() -> list[str]:
        effects.append("inspected nodes")
        return []

    def write_model_paths(*_args: object) -> Path:
        effects.append("staged model paths")
        return model_paths

    async def replace(*_args: object, **_kwargs: object) -> None:
        effects.append("launched")

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", write_model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    with pytest.raises(processes_module.WorkerStartRefused, match=refused_phase):
        await supervisor.start_media(phase_callback=refuse_at)

    assert phases[-1] == refused_phase, "the start announced a phase past the refusal"
    assert effects == effects_before_it


async def test_a_broken_media_phase_report_does_not_stop_the_start(
    settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    """The other half of the distinction, held in place.

    A callback that raises anything else is a reporting bug, and a reporting
    bug must not cost the caller a worker. Every phase is still attempted and
    the launch still happens.
    """

    runtime = tmp_path / "ComfyUI"
    runtime.mkdir()
    (runtime / "main.py").write_bytes(b"")
    executable = tmp_path / "python.exe"
    executable.write_bytes(b"runtime")
    settings.comfy_directory = runtime
    settings.comfy_executable = executable
    supervisor = ProcessSupervisor(settings)
    model_paths = tmp_path / "extra-model-paths.yaml"
    model_paths.write_text("{}", encoding="utf-8")
    launched: list[str] = []
    phases: list[str] = []

    async def broken_report(phase: str) -> None:
        phases.append(phase)
        raise RuntimeError("the progress channel is down")

    async def trusted_nodes() -> list[str]:
        return []

    async def replace(*_args: object, **_kwargs: object) -> None:
        launched.append("media")

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda *_args: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media(phase_callback=broken_report)

    assert phases == ["Validating media dependencies", "Starting media runtime"]
    assert launched == ["media"]


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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert name == "media"
        assert estimated_memory_bytes is None
        assert editor_bridge_support is not None
        assert not editor_bridge_support.supported
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


async def test_media_start_retains_a_bridge_staging_refusal_without_blocking_media(
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
    captured: dict[str, object] = {}

    def refuse_staging(**_kwargs: object) -> None:
        raise ComfyEditorBridgeError(
            "workflow-editor-bridge-staging-failed",
            "The verified workflow editor bridge could not be staged.",
        )

    async def replace(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert name == "media"
        assert estimated_memory_bytes is None
        captured["command"] = command
        captured["support"] = editor_bridge_support

    monkeypatch.setattr(processes_module, "prepare_comfy_editor_bridge", refuse_staging)
    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", AsyncMock(return_value=[]))
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    support = captured["support"]
    assert isinstance(support, ComfyEditorBridgeSupport)
    assert support == ComfyEditorBridgeSupport(
        False,
        "workflow-editor-bridge-staging-failed",
        "The verified workflow editor bridge could not be staged.",
    )
    assert "--whitelist-custom-nodes" not in captured["command"]


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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert estimated_memory_bytes is None
        assert editor_bridge_support is not None
        assert editor_bridge_support.supported
        assert editor_bridge_support.code == "ready"
        assert editor_bridge_support.comfyui_version == "0.28.0"
        assert editor_bridge_support.frontend_version == "1.45.21"
        captured["command"] = command

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    command = captured["command"]
    bridge_name = bridge_directory_name(trusted_browser_origins(settings))
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == [bridge_name]
    staged = runtime / "custom_nodes" / bridge_name
    assert (staged / "__init__.py").is_file()
    assert (staged / "js" / "lm_atelier_workflow_editor.js").is_file()
    config = (staged / BRIDGE_COORDINATOR_CONFIG).read_text(encoding="utf-8")
    assert f"http://127.0.0.1:{settings.port}" in config


async def test_media_start_retains_the_exact_unsupported_runtime_fact(
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
        '__version__ = "0.27.0"' + chr(10),
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
    captured: dict[str, object] = {}

    async def replace(
        _name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert estimated_memory_bytes is None
        captured["command"] = command
        captured["support"] = editor_bridge_support

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", AsyncMock(return_value=[]))
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    support = captured["support"]
    assert support == ComfyEditorBridgeSupport(
        False,
        "workflow-editor-comfyui-unsupported",
        "Native workflow editing requires ComfyUI 0.28.0; the configured runtime uses 0.27.0.",
        "0.27.0",
        "1.45.21",
    )
    command = captured["command"]
    assert isinstance(command, list)
    assert "--whitelist-custom-nodes" not in command


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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert editor_bridge_support is not None
        assert not editor_bridge_support.supported
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
    support = ComfyEditorBridgeSupport(
        True,
        "ready",
        "Native workflow editing is available.",
        "0.28.0",
        "1.45.21",
    )
    record = WorkerRecord(
        "media",
        process,  # type: ignore[arg-type]
        command,
        log,
        state="ready",
        launch_scope_sha256="a" * 64,
        editor_bridge_launch_id="same-launch",
        editor_bridge_support=support,
    )
    supervisor._workers["media"] = record
    stopped = AsyncMock(side_effect=AssertionError("matching worker was restarted"))
    monkeypatch.setattr(supervisor, "_stop_unlocked", stopped)

    await supervisor._replace(
        "media",
        list(command),
        "http://127.0.0.1:8289/system_stats",
        launch_scope_sha256="a" * 64,
        editor_bridge_support=support,
    )

    stopped.assert_not_awaited()
    assert supervisor._workers["media"] is record
    assert supervisor.workflow_editor_runtime_identity() == "same-launch"
    assert supervisor.workflow_editor_bridge_support() == support

    changed_support = ComfyEditorBridgeSupport(
        False,
        "workflow-editor-frontend-unsupported",
        "The configured frontend is not certified for native workflow editing.",
        "0.28.0",
        "1.46.0",
    )
    with pytest.raises(AssertionError, match="matching worker was restarted"):
        await supervisor._replace(
            "media",
            list(command),
            "http://127.0.0.1:8289/system_stats",
            launch_scope_sha256="a" * 64,
            editor_bridge_support=changed_support,
        )
    stopped.assert_awaited_once_with("media")
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
        editor_bridge_support: ComfyEditorBridgeSupport | None = None,
    ) -> None:
        assert editor_bridge_support is not None
        assert not editor_bridge_support.supported
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
    expected_root = registry_wheel_environment_root(settings.registry_dir)
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


async def test_every_installed_asset_kind_reaches_the_runtime(
    client,
    settings,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    """A verified asset the runtime cannot see is a download that bought nothing."""

    del client
    settings.prepare()
    kinds = {
        "diffusion_model": "diffusion_models",
        "text_encoder": "text_encoders",
        "clip_vision": "clip_vision",
        "checkpoint": "checkpoints",
        "vae": "vae",
        "lora": "loras",
    }
    with SessionLocal() as session:
        for kind in (*kinds, "configuration"):
            directory = tmp_path / f"asset-{kind}"
            directory.mkdir()
            session.add(
                ModelAssetInstall(
                    name=kind,
                    kind=kind,
                    local_path=str(directory),
                    size_bytes=1024,
                    manifest_json={},
                    active=True,
                )
            )
        inactive = tmp_path / "asset-retired"
        inactive.mkdir()
        session.add(
            ModelAssetInstall(
                name="retired",
                kind="lora",
                local_path=str(inactive),
                size_bytes=1024,
                manifest_json={},
                active=False,
            )
        )
        session.commit()

    destination = ProcessSupervisor(settings)._write_comfy_model_paths()

    published = json.loads(destination.read_text(encoding="utf-8"))
    served = {
        Path(entry["base_path"]).name: folder
        for entry in published.values()
        for folder in entry
        if folder != "base_path"
    }
    assert served == {f"asset-{kind}": folder for kind, folder in kinds.items()}


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


def test_a_python_worker_can_write_text_whatever_the_console_encoding_is() -> None:
    """The rgthree failure: a startup message with an emoji aborted the worker.

    The hidden Windows worker's standard streams are the system code page, so
    a custom node printing U+1F389 raised UnicodeEncodeError; ComfyUI then
    tried to log that traceback, hit the same error, and aborted. The API saw
    only that activation failed to start.
    """
    from local_lm.subprocess_env import python_subprocess_environment

    environment = python_subprocess_environment(source={"PATH": "/usr/bin"})

    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"


def test_an_explicit_encoding_choice_is_not_rewritten() -> None:
    """These are defaults for workers that say nothing, not a policy."""
    from local_lm.subprocess_env import python_subprocess_environment

    environment = python_subprocess_environment(
        overrides={"PYTHONIOENCODING": "utf-8:backslashreplace"},
        source={"PATH": "/usr/bin"},
    )

    assert environment["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    assert environment["PYTHONUTF8"] == "1"


def test_the_encoding_defaults_do_not_reach_other_subprocesses() -> None:
    """Git and the like keep the environment they had; this is a worker fact."""
    from local_lm.subprocess_env import git_subprocess_environment

    assert "PYTHONUTF8" not in git_subprocess_environment()


def test_a_worker_environment_still_inherits_nothing_it_should_not() -> None:
    from local_lm.subprocess_env import python_subprocess_environment

    environment = python_subprocess_environment(
        source={"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "leaked", "HF_TOKEN": "leaked"},
    )

    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "HF_TOKEN" not in environment


async def test_stopping_a_worker_with_no_live_record_still_stops_its_process(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Losing the handle must not make the worker unstoppable.

    Measured on a live install: /api/workers reported the media worker stopped
    with pid=None while the ComfyUI process the app had launched was still alive,
    still its child, and still holding 127.0.0.1:8289. `_stop_unlocked` returned
    immediately because `self._workers` had no record, so the stop was a no-op,
    and the port preflight in `_replace` then refused every start and restart
    with "already in use". Nothing in the product recovered it; the process had
    to be killed from outside.

    The identities are persisted so a worker can be recognised after the handle
    is lost. This drives the exact state that occurred - a persisted identity
    with no in-memory record - and requires that stopping actually stops it.
    """

    settings.prepare()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=creationflags,
    )
    try:
        supervisor = ProcessSupervisor(settings)
        # A persisted identity and NO live record: the state the install was in.
        supervisor._worker_identities["media"] = [
            _ProcessIdentity(pid=child.pid, create_time=psutil.Process(child.pid).create_time())
        ]
        assert "media" not in supervisor._workers

        await supervisor._stop_unlocked("media")

        child.wait(timeout=10)
        assert child.poll() is not None, (
            "the worker process survived a stop, so the port it holds stays held "
            "and every later start refuses"
        )
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5)


async def test_stopping_a_worker_with_no_record_does_not_kill_a_reused_pid(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The recovery must not become a licence to kill by pid alone.

    The control for the test above. A persisted identity whose creation time no
    longer matches is a pid the operating system has since handed to something
    else, and stopping the worker must leave it alone. Without this, recovering a
    wedged worker could terminate an unrelated process that merely inherited its
    number.
    """

    settings.prepare()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    stranger = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=creationflags,
    )
    try:
        supervisor = ProcessSupervisor(settings)
        # Same pid, wrong creation time: a reused number, not our worker.
        supervisor._worker_identities["media"] = [
            _ProcessIdentity(
                pid=stranger.pid,
                create_time=psutil.Process(stranger.pid).create_time() - 3600.0,
            )
        ]

        await supervisor._stop_unlocked("media")

        assert stranger.poll() is None, (
            "a process that merely reused the worker's pid was killed; identity "
            "is pid AND creation time, and only the pair may authorise a kill"
        )
    finally:
        stranger.kill()
        stranger.wait(timeout=5)


async def test_a_port_held_by_our_own_child_is_reclaimed_without_any_identity(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The incident had no usable identity, and this is the path that recovers it.

    The persisted identities are not sufficient on their own, and believing they
    were is what made an earlier version of this fix narrower than it claimed.
    They can be absent entirely, and they can be emptied while a process still
    holds the port, because the snapshot only covers the tree as it stood when it
    was taken. A descendant that appeared afterwards is invisible to them.

    So this drives the state that leaves the application wedged with NOTHING
    recorded: a live child of this process listening on the port, no worker
    record, and no identity at all. Recovery has to come from ownership we can
    still prove - it is our descendant, and it is on the port we are about to
    bind.
    """

    settings.prepare()
    port = _free_port()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time;"
            "s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
            f"s.bind(('127.0.0.1',{port}));s.listen(8);time.sleep(120)",
        ],
        creationflags=creationflags,
    )
    try:
        supervisor = ProcessSupervisor(settings)
        assert "media" not in supervisor._workers
        assert not supervisor._worker_identities.get("media"), (
            "this test is about the case with NO identity; if one exists it is "
            "measuring the other path"
        )

        async def bound() -> bool:
            while True:
                if supervisor._own_descendants_blocking("media", "127.0.0.1", port):
                    return True
                await asyncio.sleep(0.05)

        assert await asyncio.wait_for(bound(), timeout=30) is True

        await supervisor._reclaim_port_from_our_own_children(
            "media", f"http://127.0.0.1:{port}/health"
        )

        holder.wait(timeout=15)
        assert holder.poll() is not None, "our own child kept the port after a reclaim"
        assert supervisor._port_is_free("127.0.0.1", port), (
            "the port was not bindable after reclaiming it, so a start would still refuse"
        )
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


async def test_a_port_held_by_a_process_we_did_not_parent_is_left_alone(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Reclaiming must never become killing by port number.

    The control for the test above. A listener this application did not parent is
    not ours, whatever it is holding. Both tests are required together: descendant
    AND on the port. Without this one, a recovery would be free to terminate an
    unrelated service that happened to occupy the address.

    The stranger here is deliberately NOT a descendant of the test process: it is
    detached, so `children(recursive=True)` from our pid cannot reach it.
    """

    settings.prepare()
    port = _free_port()
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(8)
    try:
        supervisor = ProcessSupervisor(settings)
        # Held by THIS process, which is not one of our descendants.
        assert supervisor._own_descendants_blocking("media", "127.0.0.1", port) == []

        await supervisor._reclaim_port_from_our_own_children(
            "media", f"http://127.0.0.1:{port}/health"
        )

        # Still bound, and this process is still alive to prove nothing was killed.
        assert not supervisor._port_is_free("127.0.0.1", port), (
            "the reclaim freed a port it could not prove it owned"
        )
    finally:
        listener.close()


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def test_replace_reclaims_the_port_before_it_judges_the_port(
    settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """The reclaim must be wired in, and wired in BEFORE the preflight.

    The other reclaim tests call it directly, so they would all still pass with
    its call site deleted - a recovery nothing reaches is decorative. This binds
    the wiring and the order together: stopping first, then reclaiming, then
    judging whether the port is free. Reclaiming after the preflight would be
    useless, because the preflight is what refuses.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    order: list[str] = []

    async def record_stop(name: str) -> None:
        order.append("stop")

    async def record_reclaim(name: str, url: str) -> None:
        order.append("reclaim")

    async def record_preflight(name: str, url: str) -> None:
        order.append("preflight")
        # Stop _replace here: everything after this launches a real process, and
        # the ordering is the whole assertion.
        raise OSError("preflight reached")

    monkeypatch.setattr(supervisor, "_stop_unlocked", record_stop)
    monkeypatch.setattr(supervisor, "_reclaim_port_from_our_own_children", record_reclaim)
    monkeypatch.setattr(supervisor, "_ensure_port_available", record_preflight)

    with pytest.raises(OSError, match="preflight reached"):
        await supervisor._replace(
            "media",
            [sys.executable, "-c", "pass"],
            "http://127.0.0.1:65530/health",
        )

    assert order == ["stop", "reclaim", "preflight"], (
        f"expected the port to be reclaimed between stopping and judging it, got {order}"
    )


def _listener_child(address: str, port: int) -> subprocess.Popen[bytes]:
    """A child of this process that holds one listening socket and then waits."""

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import socket,time;"
            "s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1);"
            f"s.bind(('{address}',{port}));s.listen(8);time.sleep(120)",
        ],
        creationflags=creationflags,
    )


def _tree_pids(root_pid: int) -> list[int]:
    """That pid and every pid beneath it."""

    try:
        process = psutil.Process(root_pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []
    pids = [root_pid]
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        pids.extend(child.pid for child in process.children(recursive=True))
    return pids


async def _await_listener(root_pid: int, address: str, port: int) -> int:
    """Wait until something in this child's tree is listening, and say which pid.

    The pid `Popen` returns is not necessarily the pid that binds. A virtual
    environment's interpreter is a trampoline that re-execs the real one, so the
    listener is commonly a CHILD of the process we spawned. An assertion written
    against the spawned pid therefore compares the launcher against the listener
    and fails for a reason that has nothing to do with the code under test -
    which is exactly what it did before this helper existed. Everything here is
    asserted about the spawned TREE.
    """

    while True:
        for pid in _tree_pids(root_pid):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                if ProcessSupervisor._listening_match(psutil.Process(pid), address, port) == (
                    "exact"
                ):
                    return pid
        await asyncio.sleep(0.05)


async def test_a_descendant_on_another_address_at_the_same_port_survives(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The port number is not the endpoint, and terminating on it is destruction.

    A child of ours listening on 127.0.0.2 at the same number is not what stands
    between the worker and 127.0.0.1. Killing it frees nothing - the real
    blocker keeps the address - and it destroys a process that was doing its
    job. An earlier version of this fix did exactly that, because it compared
    `laddr.port` and discarded the host it had already parsed.

    The target endpoint is held here by the TEST process, which is deliberate on
    two counts. The reclaim returns immediately on a free endpoint, so something
    must hold it or the selection under test is never reached. And the test
    process is not a descendant of itself, so it stands in for the foreign
    blocker: the correct outcome is that the reclaim proves it owns nothing,
    terminates nothing, and leaves the address exactly as busy as it found it.
    """

    settings.prepare()
    port = _free_port()
    stranger = _listener_child("127.0.0.2", port)
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("127.0.0.1", port))
        blocker.listen(8)
        supervisor = ProcessSupervisor(settings)
        listener = await asyncio.wait_for(
            _await_listener(stranger.pid, "127.0.0.2", port), timeout=30
        )
        assert supervisor._own_descendants_blocking("media", "127.0.0.1", port) == [], (
            "a descendant on another address was selected as a blocker of this one"
        )

        await supervisor._reclaim_port_from_our_own_children(
            "media", f"http://127.0.0.1:{port}/health"
        )

        assert stranger.poll() is None, (
            "the reclaim terminated a descendant that was not blocking the "
            "endpoint it was asked to free"
        )
        assert psutil.pid_exists(listener), (
            "the reclaim terminated the listener beneath that descendant, which "
            "is the same harm reached one level down"
        )
    finally:
        blocker.close()
        if stranger.poll() is None:
            stranger.kill()
            stranger.wait(timeout=5)


async def test_a_wildcard_descendant_is_recognised_as_holding_the_address(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The other half of the same distinction, and it must not be lost with it.

    A listener on the unspecified address covers every address of its family.
    Narrowing the match to exact address equality would leave this one
    unrecognised, and on the platforms where a wildcard really does exclude a
    specific bind that means the workspace still refuses to start - the failure
    this row exists to fix.

    This asserts RECOGNITION, not a completed reclaim, and the distinction is
    measured rather than stylistic. On Windows a specific address may be bound
    alongside a wildcard at the same port, so `_port_is_free('127.0.0.1', port)`
    answers True with this child running and the reclaim correctly returns
    having done nothing. On POSIX the same probe answers False. An end-to-end
    assertion here would therefore be asserting the platform. What must hold
    everywhere is that this child is identified as holding the address, which is
    what the selection is for.
    """

    settings.prepare()
    port = _free_port()
    holder = _listener_child("0.0.0.0", port)
    try:
        supervisor = ProcessSupervisor(settings)
        listener = await asyncio.wait_for(_await_listener(holder.pid, "0.0.0.0", port), timeout=30)

        selected = supervisor._own_descendants_blocking("media", "127.0.0.1", port)
        chosen = [process.pid for process in selected]

        assert chosen == [listener], (
            "a wildcard listener of ours was not recognised as holding the address"
        )
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


async def test_an_exact_holder_is_preferred_and_a_wildcard_bystander_is_spared(
    settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Both tiers present, and only the exact one may be terminated.

    This is the case the two socket tests cannot reach, and the reason is the
    platform rather than the design: an exact holder and a wildcard holder can
    coexist at one port on Windows and cannot on POSIX, so a test built from
    real listeners could only run where that arrangement is possible - and the
    rule it checks has to hold on both. The descendants are therefore supplied
    directly.

    Why preference rather than taking both. Where a wildcard does not exclude a
    specific bind, the exact holder is the only thing that can be holding the
    address, so terminating the wildcard alongside it kills a working child of
    ours for no gain. Where a wildcard does exclude one, the two cannot both be
    there, so nothing is given up by preferring exact. Taking both is wrong on
    one platform and pointless on the other.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    exact = SimpleNamespace(pid=4001)
    bystander = SimpleNamespace(pid=4002)
    kinds = {4001: "exact", 4002: "wildcard"}

    monkeypatch.setattr(supervisor, "_own_descendants", lambda: [bystander, exact])
    monkeypatch.setattr(
        supervisor,
        "_listening_match",
        lambda process, host, port: kinds.get(process.pid),
    )

    chosen = supervisor._own_descendants_blocking("media", "127.0.0.1", 65532)

    assert [process.pid for process in chosen] == [exact.pid], (
        "the wildcard bystander was selected alongside the exact holder, which "
        "on a platform where a wildcard does not block is a working child of "
        "ours terminated for nothing"
    )


async def test_localhost_names_the_loopback_addresses_and_nothing_else(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """A named host has to be resolved deliberately, in both directions.

    `localhost` is a SUPPORTED value here - `validate_worker_url` accepts it
    beside literal loopback addresses - so refusing to resolve it would leave
    this recovery silently inoperative for anyone whose URLs are written that
    way. Nothing would fail; the worker would simply never be reclaimed. A
    mutation found that, not a reader.

    Resolving it must not become resolving anything. The three assertions are
    the three cases and they have to hold together: localhost reaches the
    loopback address, localhost does NOT reach the rest of 127.0.0.0/8, and a
    name this code cannot prove reaches nothing at all. Widening the first
    without the second would put the different-address bystander back in range
    of termination through the back door.
    """

    settings.prepare()
    port = _free_port()
    loopback = _listener_child("127.0.0.1", port)
    elsewhere = _listener_child("127.0.0.2", port)
    try:
        loopback_pid = await asyncio.wait_for(
            _await_listener(loopback.pid, "127.0.0.1", port), timeout=30
        )
        elsewhere_pid = await asyncio.wait_for(
            _await_listener(elsewhere.pid, "127.0.0.2", port), timeout=30
        )
        supervisor = ProcessSupervisor(settings)

        assert (
            supervisor._listening_match(psutil.Process(loopback_pid), "localhost", port) == "exact"
        ), "localhost did not reach the loopback listener, so the reclaim would never fire"
        assert (
            supervisor._listening_match(psutil.Process(elsewhere_pid), "localhost", port) is None
        ), "localhost reached 127.0.0.2, which is loopback but is not localhost"
        assert (
            supervisor._listening_match(psutil.Process(loopback_pid), "example.invalid", port)
            is None
        ), "a name this code cannot prove was treated as naming an address"
    finally:
        for child in (loopback, elsewhere):
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)


async def test_the_reclaim_retries_a_briefly_unbindable_address(
    settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """A terminated listener can leave the address unbindable for a moment.

    Judging that moment as a refusal would fail the start for a condition that
    resolves on its own, so the bind is retried. The retry existed before this
    test and nothing drove it: every other case here frees the address on the
    first probe, so the loop could have been deleted with the suite still green.

    The probe is scripted rather than raced. Busy on the first question, which
    is what admits the selection; then two transient failures; then bindable.
    Four probes means the loop kept asking past a failure and stopped at the
    first success, which is both halves of the guarantee. A version that judged
    one attempt would ask twice and give up with the address still counted busy.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    answers = iter((False, False, False, True))
    probes = 0

    def scripted_probe(host: str, port: int) -> bool:
        nonlocal probes
        probes += 1
        return next(answers, True)

    def one_holder(name: str, host: str, port: int) -> list[psutil.Process]:
        return [psutil.Process(os.getpid())]

    terminated: list[int] = []

    def record_termination(processes, timeout) -> None:  # type: ignore[no-untyped-def]
        terminated.append(len(processes))

    monkeypatch.setattr(supervisor, "_port_is_free", scripted_probe)
    monkeypatch.setattr(supervisor, "_own_descendants_blocking", one_holder)
    monkeypatch.setattr(supervisor, "_terminate_processes", record_termination)
    monkeypatch.setattr(supervisor, "_refresh_worker_identities_after_stop", lambda name: None)

    await supervisor._reclaim_port_from_our_own_children("media", "http://127.0.0.1:65531/health")

    assert terminated == [1], f"expected the one selected holder to be terminated, got {terminated}"
    assert probes == 4, (
        f"expected the bind to be retried past two transient failures, it probed {probes} time(s)"
    )


async def test_a_sibling_worker_on_the_same_endpoint_is_never_terminated(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Endpoint ownership proves a process is in the way, not that it is ours to take.

    Nothing requires two workers to be configured on DIFFERENT endpoints:
    `validate_worker_url` checks each URL is loopback and stops there. So
    `llama_url` and `comfy_url` may name the same host and port, and then the
    other worker is a descendant of ours holding exactly the address this one
    wants - indistinguishable, by endpoint alone, from the lost worker being
    reclaimed. Selecting it tears down a live sibling, and `_terminate_processes`
    takes its whole subtree, so a running service is stopped where the ordinary
    preflight would merely have refused.

    THE HOLDER IS FOUND RATHER THAN ASSUMED. On Windows this interpreter is a
    trampoline: `subprocess.Popen` returns the pid of a launcher that holds no
    socket, and the real listener is its child. Registering the returned pid as
    the sibling's identity would therefore name a process that owns nothing, and
    the test would pass while proving the opposite of what it claims.
    """

    settings.prepare()
    port = _free_port()
    sibling = _listener_child("127.0.0.1", port)
    try:
        supervisor = ProcessSupervisor(settings)

        def holders() -> list[psutil.Process]:
            return [
                process
                for process in supervisor._own_descendants()
                if supervisor._listening_match(process, "127.0.0.1", port) == "exact"
            ]

        async def bound() -> list[psutil.Process]:
            while True:
                found = holders()
                if found:
                    return found
                await asyncio.sleep(0.05)

        actual = await asyncio.wait_for(bound(), timeout=30)
        assert actual, "the sibling never took the endpoint, so nothing was measured"
        # Without the exclusion this is exactly what would be terminated.
        assert supervisor._own_descendants_blocking("media", "127.0.0.1", port) == actual

        # Now it is chat's, and we are replacing media.
        supervisor._worker_identities["chat"] = [
            _ProcessIdentity(pid=process.pid, create_time=process.create_time())
            for process in actual
        ]

        assert supervisor._own_descendants_blocking("media", "127.0.0.1", port) == [], (
            "a live sibling worker holding the same configured endpoint was "
            "selected for termination"
        )

        await supervisor._reclaim_port_from_our_own_children(
            "media", f"http://127.0.0.1:{port}/health"
        )
        assert all(process.is_running() for process in actual), (
            "the reclaim terminated another worker's process"
        )
    finally:
        if sibling.poll() is None:
            sibling.kill()
            sibling.wait(timeout=5)


def test_an_ipv6_wildcard_is_not_evidence_for_an_ipv4_target(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """Cross-family wildcard evidence is unprovable here, so it must not authorise.

    A socket bound to `::` accepts IPv4 connections wherever the platform leaves
    IPV6_V6ONLY off - and psutil cannot report that flag, so whether a given `::`
    listener holds an IPv4 endpoint is exactly what this cannot observe. An
    earlier version accepted it anyway, reasoning that the endpoint had already
    been measured busy. That reasoning is wrong: the thing making it busy can be
    a FOREIGN IPv4 holder while an IPv6-only child of ours sits innocently at the
    same number, and it would then be killed for a bind it could never block.

    Driven through _listening_match with a stand-in rather than a real dual-stack
    listener, because what is asserted is the rule, and arranging a genuine
    IPv6-only holder would make the test assert the platform instead.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    listener = SimpleNamespace(
        status=psutil.CONN_LISTEN,
        laddr=SimpleNamespace(ip="::", port=51234),
    )
    process = SimpleNamespace(net_connections=lambda kind="tcp": [listener])

    assert supervisor._listening_match(process, "127.0.0.1", 51234) is None, (
        "an IPv6 wildcard was accepted as evidence for an IPv4 target, which "
        "this cannot prove and must not act on"
    )
    assert supervisor._listening_match(process, "::1", 51234) == "wildcard", (
        "the same-family wildcard is real evidence and must still count"
    )


def test_localhost_selects_only_the_family_the_probe_uses(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    """The evidence and the decision have to be about one endpoint.

    `_port_is_free` and `_ensure_port_available` both pick the address family
    from whether the host text contains a colon, so `localhost` is probed as IPv4
    only. Selecting against both families while probing one would let an IPv4
    busy result authorise terminating an exact ::1 descendant that had nothing to
    do with it.
    """

    settings.prepare()
    supervisor = ProcessSupervisor(settings)
    on_v6 = SimpleNamespace(
        net_connections=lambda kind="tcp": [
            SimpleNamespace(status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="::1", port=51235))
        ]
    )
    on_v4 = SimpleNamespace(
        net_connections=lambda kind="tcp": [
            SimpleNamespace(
                status=psutil.CONN_LISTEN, laddr=SimpleNamespace(ip="127.0.0.1", port=51235)
            )
        ]
    )

    assert supervisor._listening_match(on_v6, "localhost", 51235) is None, (
        "an IPv6 loopback descendant was selected on an IPv4 probe of localhost"
    )
    assert supervisor._listening_match(on_v4, "localhost", 51235) == "exact", (
        "localhost stopped matching the family it is actually probed as"
    )
