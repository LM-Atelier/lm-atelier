from __future__ import annotations

import asyncio
import io
import zipfile

from httpx2 import AsyncClient

from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.downloads import DownloadManager


async def wait_for_assistant(client: AsyncClient, chat_id: str, expected_type: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/chats/{chat_id}")
        assert response.status_code == 200
        chat = response.json()
        assistant = [message for message in chat["messages"] if message["role"] == "assistant"][-1]
        if assistant["status"] in {"complete", "failed", "cancelled"}:
            assert assistant["status"] == "complete", assistant
            assert any(part["type"] == expected_type for part in assistant["parts"])
            return assistant
        await asyncio.sleep(0.03)
    raise AssertionError("assistant run did not complete")


async def wait_for_run(client: AsyncClient, run_id: str) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == "complete", run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


async def test_project_and_chat_management_contract(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Research Lab"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Model notes", "project_id": project["id"]})
    ).json()

    project_search = await client.get("/api/projects", params={"query": "research"})
    chat_search = await client.get("/api/chats", params={"query": "notes"})
    assert [item["id"] for item in project_search.json()] == [project["id"]]
    assert [item["id"] for item in chat_search.json()] == [chat["id"]]

    archived = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"title": "Archived notes", "archived": True, "project_id": None},
    )
    assert archived.status_code == 200
    assert archived.json()["project_id"] is None
    assert (await client.get("/api/chats")).json() == []
    archived_list = await client.get("/api/chats", params={"include_archived": True})
    assert [item["id"] for item in archived_list.json()] == [chat["id"]]

    restored = await client.patch(f"/api/chats/{chat['id']}", json={"archived": False})
    assert restored.status_code == 200
    assert (await client.delete(f"/api/projects/{project['id']}")).status_code == 204
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["project_id"] is None
    assert (await client.delete(f"/api/chats/{chat['id']}")).status_code == 204
    assert (await client.get(f"/api/chats/{chat['id']}")).status_code == 404


async def test_project_chat_text_and_inline_image_flow(client: AsyncClient) -> None:
    project_response = await client.post("/api/projects", json={"name": "Demo"})
    assert project_response.status_code == 201
    project = project_response.json()

    chat_response = await client.post(
        "/api/chats", json={"title": "New chat", "project_id": project["id"]}
    )
    assert chat_response.status_code == 201
    chat = chat_response.json()

    text_response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Explain local inference", "mode": "auto", "idempotency_key": "text-1"},
    )
    assert text_response.status_code == 202
    assert text_response.json()["run"]["operation"] == "text"
    assistant = await wait_for_assistant(client, chat["id"], "text")
    assert "Mock local response" in assistant["parts"][0]["text"]
    text_metadata = next(
        part for part in assistant["parts"] if part["type"] == "generation_metadata"
    )["metadata_json"]
    assert text_metadata["provenance"]["output"]["kind"] == "text"
    assert len(text_metadata["provenance"]["output"]["sha256"]) == 64
    assert "resolved_settings" in text_metadata["provenance"]

    image_response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a purple observatory",
            "mode": "auto",
            "idempotency_key": "image-1",
        },
    )
    assert image_response.status_code == 202
    assert image_response.json()["run"]["operation"] == "text_to_image"
    image_message = await wait_for_assistant(client, chat["id"], "image")
    image_part = next(part for part in image_message["parts"] if part["type"] == "image")
    image_metadata = next(
        part for part in image_message["parts"] if part["type"] == "generation_metadata"
    )["metadata_json"]
    image_output = image_metadata["provenance"]["outputs"][0]
    assert image_output["artifact_id"] == image_part["artifact_id"]
    assert image_output["sha256"] == image_part["artifact"]["sha256"]
    assert image_part["artifact"]["kind"] == "image"
    assert image_part["artifact"]["metadata_json"].get("temporary_preview") is None
    content = await client.get(f"/api/artifacts/{image_part['artifact_id']}/content")
    assert content.status_code == 200
    assert content.headers["content-type"].startswith("image/svg+xml")
    partial = await client.get(
        f"/api/artifacts/{image_part['artifact_id']}/content", headers={"range": "bytes=0-15"}
    )
    assert partial.status_code == 206
    assert partial.headers["content-range"].startswith("bytes 0-15/")
    assert len(partial.content) == 16

    followup = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Animate that image gently", "mode": "auto"},
    )
    assert followup.status_code == 202
    assert followup.json()["run"]["operation"] == "image_to_video"
    assert followup.json()["run"]["provenance_json"]["input_artifact_ids"] == [
        image_part["artifact_id"]
    ]
    await wait_for_assistant(client, chat["id"], "video")


async def test_inline_video_and_project_export(client: AsyncClient) -> None:
    project = (await client.post("/api/projects", json={"name": "Film lab"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Storyboard", "project_id": project["id"]})
    ).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a short video of dawn", "mode": "video"},
    )
    assert response.status_code == 202
    message = await wait_for_assistant(client, chat["id"], "video")
    video_part = next(part for part in message["parts"] if part["type"] == "video")
    assert video_part["artifact_id"]
    metadata = next(part for part in message["parts"] if part["type"] == "generation_metadata")[
        "metadata_json"
    ]
    output = metadata["provenance"]["outputs"][0]
    assert output["artifact_id"] == video_part["artifact_id"]
    assert output["sha256"] == video_part["artifact"]["sha256"]
    assert output["kind"] == "video"

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert "manifest.json" in bundle.namelist()
        assert any(name.startswith("artifacts/") for name in bundle.namelist())


async def test_backup_create_verify_and_restore_marker(client: AsyncClient) -> None:
    created = await client.post("/api/backups")
    assert created.status_code == 201
    name = created.json()["name"]
    verified = await client.post(f"/api/backups/{name}/verify")
    assert verified.status_code == 200
    assert verified.json()["verified"] is True
    restore = await client.post(f"/api/backups/{name}/restore")
    assert restore.status_code == 200
    assert restore.json()["restore_pending"] is True


async def test_worker_management_reports_missing_local_binaries(client: AsyncClient) -> None:
    workers = await client.get("/api/workers")
    assert workers.status_code == 200
    assert {item["name"] for item in workers.json()} == {"chat", "media"}
    assert all(item["state"] == "stopped" for item in workers.json())
    assert all(item["active_jobs"] == 0 and item["queued_jobs"] == 0 for item in workers.json())
    assert all(item["current_memory_bytes"] is None for item in workers.json())
    media = await client.post("/api/workers/media/start")
    assert media.status_code == 422


async def test_chat_tool_capability_probe_executes_declared_schema(client: AsyncClient) -> None:
    response = await client.post("/api/engines/chat/tool-probe")
    assert response.status_code == 200
    assert response.json()["passed"] is True
    assert response.json()["arguments"] == {"mode": "image", "confidence": 1}


async def test_turn_idempotency_returns_original_run(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Idempotency"})).json()
    payload = {"text": "Hello", "mode": "text", "idempotency_key": "stable-key"}
    first = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    second = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["run"]["id"] == second.json()["run"]["id"]


async def test_active_chat_run_can_be_cancelled_directly(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Stop response"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Write a response long enough to stop", "mode": "text"},
    )
    assert turn.status_code == 202
    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    run = (await client.get(f"/api/runs/{turn.json()['run']['id']}")).json()
    assert run["status"] == "cancelled"


async def test_editing_user_message_creates_new_active_branch(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Edit branch"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Original question", "mode": "text"}
    )
    assert first.status_code == 202
    await wait_for_run(client, first.json()["run"]["id"])
    second = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "Old branch follow-up", "mode": "text"}
    )
    assert second.status_code == 202
    await wait_for_run(client, second.json()["run"]["id"])

    branched = await client.post(
        f"/api/messages/{first.json()['user_message']['id']}/branch",
        json={"text": "Edited question"},
    )
    assert branched.status_code == 202
    payload = branched.json()
    assert payload["run"]["operation"] == "text"
    assert payload["user_message"]["parent_id"] == first.json()["user_message"]["parent_id"]
    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert detail["active_head_message_id"] == payload["assistant_message"]["id"]


async def test_new_turn_uses_persisted_active_branch_head(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Branch head"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "First", "mode": "text"},
    )
    first_run = await wait_for_run(client, first.json()["run"]["id"])
    first_assistant_id = first_run["assistant_message_id"]

    refreshed = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert refreshed["active_head_message_id"] == first_assistant_id

    second = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Second", "mode": "text"},
    )
    assert second.status_code == 202
    assert second.json()["user_message"]["parent_id"] == first_assistant_id
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == second.json()["assistant_message"]["id"]


async def test_long_chat_records_visible_context_truncation(client: AsyncClient) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    profile = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{profile['id']}",
        json={
            "load_settings": {"context_length": 512},
            "request_settings": {"max_tokens": 64},
        },
    )
    assert updated.status_code == 200
    chat = (await client.post("/api/chats", json={"title": "Long context"})).json()

    final_run: dict = {}
    for index in range(6):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": f"Turn {index}: " + ("context detail " * 12),
                "mode": "text",
            },
        )
        assert response.status_code == 202
        final_run = await wait_for_run(client, response.json()["run"]["id"])

    context = final_run["provenance_json"]["context"]
    assert context["messages_omitted"] > 0
    assert context["input_tokens"] <= context["input_budget"]

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assistant = [item for item in detail["messages"] if item["role"] == "assistant"][-1]
    metadata = next(part for part in assistant["parts"] if part["type"] == "generation_metadata")
    assert metadata["metadata_json"]["context"]["messages_omitted"] > 0


async def test_artifact_upload_deduplicates_content(client: AsyncClient) -> None:
    first = await client.post(
        "/api/artifacts",
        files={"file": ("one.png", b"same-bytes", "image/png")},
    )
    second = await client.post(
        "/api/artifacts",
        files={"file": ("two.png", b"same-bytes", "image/png")},
    )
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]


async def test_workflow_revisions_and_validation(client: AsyncClient) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Custom",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "Mock"}},
            "input_schema": {
                "type": "object",
                "properties": {"steps": {"type": "integer", "default": 20}},
            },
            "trusted": True,
        },
    )
    assert created.status_code == 201
    workflow = created.json()
    revision = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={"api_graph": {"node": {"class_type": "MockV2"}}, "trusted": True},
    )
    assert revision.status_code == 201
    assert revision.json()["version"] == 2
    validation = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert validation.status_code == 200
    assert validation.json()["valid"] is True

    exported = await client.get(f"/api/workflows/{workflow['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-workflow"
    assert bundle["source_revision"] == 2

    cloned = await client.post(
        f"/api/workflows/{workflow['id']}/clone", json={"name": "Custom copy"}
    )
    assert cloned.status_code == 201
    assert cloned.json()["name"] == "Custom copy"

    bundle["name"] = "Imported workflow"
    imported = await client.post("/api/workflows/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["revisions"][0]["api_graph_json"] == {"node": {"class_type": "MockV2"}}

    restored = await client.post(
        f"/api/workflows/{workflow['id']}/revisions/{workflow['revisions'][0]['id']}/restore"
    )
    assert restored.status_code == 201
    assert restored.json()["version"] == 3
    assert restored.json()["api_graph_json"] == {"node": {"class_type": "Mock"}}

    missing_dependency = await client.post(
        f"/api/workflows/{workflow['id']}/revisions",
        json={
            "api_graph": {"node": {"class_type": "Mock"}},
            "dependencies": {"models": ["not-installed"]},
            "trusted": True,
        },
    )
    assert missing_dependency.status_code == 201
    validation = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert validation.json()["valid"] is False
    assert "missing model dependency" in validation.json()["errors"][0]


async def test_profile_edit_clone_reset_and_portable_bundle(client: AsyncClient) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    source = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{source['id']}",
        json={
            "name": "Focused chat",
            "load_settings": {"context_length": 16_384},
            "request_settings": {"temperature": 0.25},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["load_settings_json"] == {"context_length": 16_384}

    cloned = await client.post(
        f"/api/profiles/{source['id']}/clone", json={"name": "Focused chat copy"}
    )
    assert cloned.status_code == 201
    assert cloned.json()["request_settings_json"] == {"temperature": 0.25}
    assert cloned.json()["is_default"] is False

    exported = await client.get(f"/api/profiles/{source['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-profile"
    assert bundle["version"] == 1

    bundle["name"] = "Imported portable chat"
    bundle["model_install_id"] = "missing-on-this-machine"
    imported = await client.post("/api/profiles/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["model_install_id"] is None
    assert imported.json()["load_settings_json"] == {"context_length": 16_384}

    reset = await client.post(f"/api/profiles/{source['id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["load_settings_json"] == {}
    assert reset.json()["request_settings_json"] == {}

    invalid = await client.post(
        "/api/profiles/import",
        json={
            "format": "lm-atelier-profile",
            "version": 1,
            "name": "Invalid",
            "role": "chat",
            "engine": "llama.cpp",
            "load_settings": {"not_a_setting": True},
        },
    )
    assert invalid.status_code == 422


async def test_preset_lifecycle_and_portable_bundle(client: AsyncClient) -> None:
    created = await client.post(
        "/api/presets",
        json={"name": "Precise", "role": "chat", "settings": {"temperature": 0.1}},
    )
    assert created.status_code == 201
    preset = created.json()

    cloned = await client.post(f"/api/presets/{preset['id']}/clone", json={"name": "Precise copy"})
    assert cloned.status_code == 201
    assert cloned.json()["settings_json"] == {"temperature": 0.1}

    exported = await client.get(f"/api/presets/{preset['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-preset"

    bundle["name"] = "Imported precise"
    imported = await client.post("/api/presets/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["settings_json"] == {"temperature": 0.1}

    reset = await client.post(f"/api/presets/{preset['id']}/reset")
    assert reset.status_code == 200
    assert reset.json()["settings_json"] == {}

    deleted = await client.delete(f"/api/presets/{cloned.json()['id']}")
    assert deleted.status_code == 204


async def test_model_storage_cleanup_and_shared_path_deletion(
    client: AsyncClient, settings: Settings
) -> None:
    shared = settings.model_dir / "shared-revision"
    shared.mkdir(parents=True)
    (shared / "model.gguf").write_bytes(b"shared-model")
    payload = {
        "name": "Shared model",
        "role": "chat",
        "engine": "llama.cpp",
        "local_path": str(shared),
    }
    first = await client.post("/api/models/import", json=payload)
    second = await client.post("/api/models/import", json={**payload, "name": "Second selection"})
    assert first.status_code == 201
    assert second.status_code == 201

    profile = await client.post(
        "/api/profiles",
        json={
            "name": "Bound profile",
            "role": "chat",
            "engine": "llama.cpp",
            "model_install_id": first.json()["id"],
        },
    )
    assert profile.status_code == 201
    assert (await client.delete(f"/api/models/{first.json()['id']}")).status_code == 409
    assert (await client.delete(f"/api/profiles/{profile.json()['id']}")).status_code == 204

    assert (await client.delete(f"/api/models/{first.json()['id']}")).status_code == 204
    assert (shared / "model.gguf").is_file()
    assert (await client.delete(f"/api/models/{second.json()['id']}")).status_code == 204
    assert not shared.exists()

    partial = settings.download_dir / "orphan.partial"
    partial.mkdir(parents=True)
    (partial / "chunk").write_bytes(b"incomplete")
    storage = await client.get("/api/models/storage")
    assert storage.status_code == 200
    assert storage.json()["partial_download_count"] == 1
    assert storage.json()["partial_download_bytes"] == len(b"incomplete")
    cleaned = await client.post("/api/downloads/cleanup")
    assert cleaned.status_code == 200
    assert cleaned.json() == {"removed_count": 1, "reclaimed_bytes": len(b"incomplete")}
    assert not partial.exists()

    unsafe = settings.data_dir / "unsafe-import"
    unsafe.mkdir()
    (unsafe / "weights.ckpt").write_bytes(b"pickle-compatible")
    blocked = await client.post(
        "/api/models/import",
        json={
            "name": "Unsafe directory",
            "role": "image",
            "engine": "comfyui",
            "local_path": str(unsafe),
        },
    )
    assert blocked.status_code == 422


async def test_download_pause_resume_and_cancel(client: AsyncClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    gate = asyncio.Event()

    async def wait_for_release(_manager: DownloadManager, _job_id: str) -> None:
        await gate.wait()

    monkeypatch.setattr(DownloadManager, "_download", wait_for_release)
    created = await client.post(
        "/api/downloads",
        json={"remote_id": "owner/model", "role": "chat", "engine": "llama.cpp"},
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    paused = await client.post(f"/api/downloads/{job_id}/pause")
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"

    resumed = await client.post(f"/api/downloads/{job_id}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "queued"

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    cannot_resume = await client.post(f"/api/downloads/{job_id}/resume")
    assert cannot_resume.status_code == 409


async def test_catalog_preflight_blocks_gated_unsafe_weights(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Unsafe model",
                "gated": True,
                "compatibility": "advanced_import",
                "compatibility_reasons": ["manual review required"],
            },
            "revision": revision,
            "files": [{"filename": "weights/model.ckpt", "size": 1024, "sha256": None}],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "main",
            "role": "image",
            "engine": "comfyui",
            "selected_files": ["weights/model.ckpt"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["can_install"] is False
    blocked = {check["id"] for check in payload["checks"] if check["status"] == "block"}
    assert {"weights", "access"}.issubset(blocked)


async def test_catalog_preflight_autoselects_smallest_gguf(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def inspect(
        _catalog: HuggingFaceCatalog,
        remote_id: str,
        revision: str = "main",
        requested_role: str | None = None,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "model": {
                "remote_id": remote_id,
                "name": "Safe model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "large.gguf", "size": 2048, "sha256": None},
                {"filename": "small.gguf", "size": 1024, "sha256": None},
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/model/preflight",
        json={
            "revision": "abc123",
            "role": "chat",
            "engine": "llama.cpp",
            "selected_files": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["can_install"] is True
    assert response.json()["selected_files"] == ["small.gguf"]
