from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import object_session

from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import SessionLocal
from local_lm.models import CustomNodeInstall, ModelInstall, ModelProfile
from local_lm.processes import (
    WORKER_STDERR_DISPLAY_CHARS,
    WORKER_STDERR_TAIL_BYTES,
    ProcessSupervisor,
    WorkerRecord,
    _RotatingWorkerLog,
)


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
def test_long_llama_model_path_uses_existing_short_name(tmp_path: Path) -> None:
    model_dir = tmp_path / ("descriptive-model-" * 8)
    model_dir.mkdir()
    model_path = model_dir / (("quantized-model-" * 5) + ".gguf")
    model_path.touch()
    assert len(str(model_path)) >= 240

    launch_path = ProcessSupervisor._llama_model_path(model_path)

    assert len(str(launch_path)) < 260
    assert os.path.samefile(launch_path, model_path)


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
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    settings.hf_token = "hf_private_worker_token"
    private_model_path = settings.model_dir / "private-model.gguf"
    stderr = f"Authorization: Bearer hf_private_worker_token\nfailed to open {private_model_path}\n"
    supervisor = ProcessSupervisor(settings)

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
    assert str(settings.data_dir.resolve()) not in surfaced
    assert all(str(settings.data_dir.resolve()) not in argument for argument in status.command)
    await supervisor.close()


async def test_startup_exit_with_empty_stderr_has_no_synthetic_tail(
    settings,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

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

    monkeypatch.setenv("LOCAL_LM_HF_TOKEN", "hf_private")
    monkeypatch.setenv("GITHUB_TOKEN", "github_private")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws_private")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    monkeypatch.setattr(supervisor, "_capture_process_output", capture_output)
    monkeypatch.setattr(supervisor, "_wait_healthy", healthy)

    await supervisor._replace("chat", ["worker"], "http://127.0.0.1/health")

    environment = captured["env"]
    assert isinstance(environment, dict)
    assert "PATH" in environment
    assert "LOCAL_LM_HF_TOKEN" not in environment
    assert "GITHUB_TOKEN" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment
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

        async def get(self, _url: str, *, timeout: int) -> SimpleNamespace:
            assert timeout == 1
            return SimpleNamespace(is_success=True)

    monkeypatch.setattr("local_lm.processes.httpx.AsyncClient", FakeClient)
    record = SimpleNamespace(name="chat", process=SimpleNamespace(returncode=None))

    await supervisor._wait_healthy(record, "http://127.0.0.1:12341/health")

    assert captured == {"trust_env": False}
    await supervisor.close()


async def test_runtime_exit_captures_only_a_bounded_stderr_tail(
    settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:  # type: ignore[no-untyped-def]
    settings.prepare()
    supervisor = ProcessSupervisor(settings)

    async def healthy_immediately(*_args: object) -> None:
        return None

    monkeypatch.setattr(supervisor, "_wait_healthy", healthy_immediately)
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
    captured: dict[str, list[str]] = {}

    async def trusted_nodes() -> list[str]:
        return []

    async def replace(
        name: str,
        command: list[str],
        _health_url: str,
        _profile_id: str | None = None,
        *,
        estimated_memory_bytes: int | None = None,
    ) -> None:
        del estimated_memory_bytes
        assert name == "media"
        captured["command"] = command

    monkeypatch.setattr(supervisor, "_trusted_comfy_node_folders", trusted_nodes)
    monkeypatch.setattr(supervisor, "_write_comfy_model_paths", lambda *_args: model_paths)
    monkeypatch.setattr(supervisor, "_replace", replace)

    await supervisor.start_media()

    runtimes.ensure.assert_awaited_once_with("comfyui")
    assert captured["command"][0] == str(executable.resolve())
    assert captured["command"][1] == str((runtime / "main.py").resolve())


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
