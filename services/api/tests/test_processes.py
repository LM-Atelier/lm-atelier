from __future__ import annotations

from pathlib import Path

import pytest

from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import SessionLocal
from local_lm.models import CustomNodeInstall
from local_lm.processes import ProcessSupervisor


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
    with pytest.raises(ValueError, match="exactly one"):
        ProcessSupervisor._gguf_path(tmp_path, {})


def test_chat_memory_estimate_includes_model_and_context_overhead() -> None:
    gib = 1024**3
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 8192}) == 6 * gib
    assert ProcessSupervisor._estimate_chat_memory(5 * gib, {"context_length": 2048}) == (
        5 * gib + 512 * 1024**2
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
        verified.append(install.installed_path)

    monkeypatch.setattr(CustomNodeManager, "verify", verify)

    folders = await ProcessSupervisor(settings)._trusted_comfy_node_folders()

    assert folders == ["lm-atelier-node_trusted"]
    assert verified == folders
