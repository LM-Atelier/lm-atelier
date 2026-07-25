from __future__ import annotations

import asyncio
import io
import zipfile
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from httpx2 import AsyncClient

from local_lm import __version__
from local_lm.adapters.base import ChatEvent, ChatRequest
from local_lm.adapters.mock import MockChatAdapter
from local_lm.catalog import HuggingFaceCatalog
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus, utcnow
from local_lm.downloads import DownloadManager
from local_lm.models import Artifact, Chat, Job, Run, WorkflowDefinition
from local_lm.orchestrator import ConversationOrchestrator


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
            assert run["started_at"] is not None
            assert run["completed_at"] is not None
            assert isinstance(run["duration_ms"], int)
            assert run["duration_ms"] >= 0
            assert run["provenance_json"]["timings"]["duration_ms"] == run["duration_ms"]
            return run
        await asyncio.sleep(0.03)
    raise AssertionError("run did not complete")


async def test_health_probes_the_database(client: AsyncClient, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    healthy = await client.get("/api/health")
    assert healthy.status_code == 200
    assert healthy.json()["database"] is True
    assert healthy.json()["version"] == __version__

    def fail_probe(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        from sqlalchemy.exc import OperationalError

        raise OperationalError("SELECT 1", {}, Exception("database unavailable"))

    monkeypatch.setattr("sqlalchemy.orm.Session.execute", fail_probe)
    degraded = await client.get("/api/health")
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["database"] is False


async def test_fast_readiness_probe_reports_version(client: AsyncClient) -> None:
    response = await client.get("/api/ready")
    assert response.status_code == 200
    assert response.json() == {"version": __version__}


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
    assert output["kind"] == video_part["artifact"]["kind"]

    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert "manifest.json" in bundle.namelist()
        assert any(name.startswith("artifacts/") for name in bundle.namelist())
        manifest = bundle.read("manifest.json")
        assert b'"version": 2' in manifest

    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "film-lab.lm-atelier.zip",
                archive.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    imported_chats = await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    assert [item["title"] for item in imported_chats.json()] == ["Storyboard"]
    imported_detail = (await client.get(f"/api/chats/{imported_chats.json()[0]['id']}")).json()
    imported_video = next(
        part
        for message in imported_detail["messages"]
        for part in message["parts"]
        if part["type"] == "video"
    )
    assert imported_video["artifact_id"] == video_part["artifact_id"]


async def test_metadata_only_project_export_import_marks_missing_media(
    client: AsyncClient,
) -> None:
    project = (await client.post("/api/projects", json={"name": "Portable"})).json()
    chat = (
        await client.post("/api/chats", json={"title": "Images", "project_id": project["id"]})
    ).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a portable image", "mode": "image"},
    )
    await wait_for_assistant(client, chat["id"], "image")
    exported = await client.post(
        f"/api/projects/{project['id']}/export", params={"include_media": False}
    )
    archive = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        assert not any(name.startswith("artifacts/") for name in bundle.namelist())

    imported = await client.post(
        "/api/projects/import",
        files={"archive": ("metadata.lm-atelier.zip", archive.content, "application/zip")},
    )
    assert imported.status_code == 201, imported.text
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    detail = (await client.get(f"/api/chats/{imported_chat['id']}")).json()
    image_part = next(
        part
        for message in detail["messages"]
        for part in message["parts"]
        if part["type"] == "image"
    )
    assert image_part["artifact_id"] is None
    assert image_part["metadata_json"]["missing_import_artifact_id"].startswith("sha256:")


async def test_project_import_rejects_unsafe_archive_paths(client: AsyncClient) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("../outside", b"unsafe")
        bundle.writestr("manifest.json", b"{}")
    response = await client.post(
        "/api/projects/import",
        files={"archive": ("unsafe.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 422
    assert "unsafe path" in response.json()["detail"]


async def test_backup_create_verify_and_restore_marker(
    client: AsyncClient, settings: Settings
) -> None:
    created = await client.post("/api/backups")
    assert created.status_code == 201
    name = created.json()["name"]
    verified = await client.post(f"/api/backups/{name}/verify")
    assert verified.status_code == 200
    assert verified.json()["verified"] is True
    restore = await client.post(f"/api/backups/{name}/restore")
    assert restore.status_code == 200
    assert restore.json()["restore_pending"] is True

    await client.post(
        "/api/artifacts",
        files={"file": ("backup-image.png", b"backup-media", "image/png")},
    )
    media_backup = await client.post("/api/backups", params={"include_media": True})
    assert media_backup.status_code == 201
    assert media_backup.json()["media_included"] is True
    assert media_backup.json()["media_size_bytes"] > 0
    assert (settings.backup_dir / f"{media_backup.json()['name']}.media.zip").is_file()
    assert (await client.post(f"/api/backups/{media_backup.json()['name']}/verify")).json()[
        "verified"
    ] is True


async def test_diagnostic_bundle_is_redacted(client: AsyncClient) -> None:
    secret_prompt = "diagnostic-secret-prompt-that-must-not-leak"
    chat = (await client.post("/api/chats", json={"title": "Private"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": secret_prompt, "mode": "text"}
    )
    await wait_for_run(client, turn.json()["run"]["id"])

    created = await client.post("/api/diagnostics")
    assert created.status_code == 201
    archive = await client.get(created.json()["url"])
    assert secret_prompt.encode() not in archive.content
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        payload = bundle.read("diagnostics.json")
        assert secret_prompt.encode() not in payload
        assert b'"prompts_included": false' in payload
        assert b'"tokens_included": false' in payload
        assert b'"comfy_inactivity_seconds": 600' in payload


async def test_worker_management_reports_missing_local_binaries(client: AsyncClient) -> None:
    workers = await client.get("/api/workers")
    assert workers.status_code == 200
    assert {item["name"] for item in workers.json()} == {"chat", "media"}
    assert all(item["state"] == "stopped" for item in workers.json())
    assert all(item["active_jobs"] == 0 and item["queued_jobs"] == 0 for item in workers.json())
    assert all(item["current_memory_bytes"] is None for item in workers.json())
    media = await client.post("/api/workers/media/start")
    assert media.status_code == 422


async def test_worker_changes_are_rejected_while_generation_is_pending(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        session.add(
            Job(
                kind="chat",
                status="queued",
                phase="queued",
                payload_json={},
            )
        )
        session.commit()

    stopped = await client.post("/api/workers/chat/stop")
    assert stopped.status_code == 409
    assert "active or queued job" in stopped.json()["detail"]

    profiles = (await client.get("/api/profiles?role=chat")).json()
    loaded = await client.post(f"/api/workers/chat/load/{profiles[0]['id']}")
    assert loaded.status_code == 409


async def test_chat_tool_capability_probe_executes_declared_schema(client: AsyncClient) -> None:
    response = await client.post("/api/engines/chat/tool-probe")
    assert response.status_code == 200
    assert response.json()["passed"] is True
    assert response.json()["arguments"] == {"mode": "image", "confidence": 1}


async def test_engine_api_isolates_media_settings_by_role(client: AsyncClient) -> None:
    response = await client.get("/api/engines")
    assert response.status_code == 200
    media = next(item for item in response.json() if {"image", "video"} <= set(item["roles"]))

    image_keys = [field["key"] for field in media["settings_by_role"]["image"]]
    video_keys = [field["key"] for field in media["settings_by_role"]["video"]]
    assert image_keys == [
        "negative_prompt",
        "seed",
        "width",
        "height",
        "steps",
        "cfg",
        "sampler",
        "scheduler",
        "denoise",
        "batch_size",
        "loras",
    ]
    assert video_keys == [
        "seed",
        "width",
        "height",
        "frames",
        "fps",
        "steps",
        "guidance",
        "motion_strength",
        "codec",
    ]
    assert "frames" not in image_keys
    assert "negative_prompt" not in video_keys
    assert len(media["settings"]) == len(image_keys) + len(video_keys)


async def test_random_media_seed_is_resolved_and_persisted(
    client: AsyncClient, monkeypatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.secrets.randbelow", lambda _upper: 8675309)
    chat = (await client.post("/api/chats", json={"title": "Seed provenance"})).json()

    random_seed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image of a seed vault", "mode": "image"},
    )
    assert random_seed.status_code == 202
    random_run = random_seed.json()["run"]
    assert random_run["settings_json"]["seed"] == 8675309
    assert random_run["provenance_json"]["resolved_settings"]["seed"] == 8675309

    explicit_seed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a second image of a seed vault",
            "mode": "image",
            "settings": {"seed": 42},
        },
    )
    assert explicit_seed.status_code == 202
    explicit_run = explicit_seed.json()["run"]
    assert explicit_run["settings_json"]["seed"] == 42
    assert explicit_run["provenance_json"]["resolved_settings"]["seed"] == 42


async def test_uncertain_auto_media_requires_confirmation(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Routing"})).json()
    payload = {
        "text": "Maybe create an image of a quiet harbor",
        "mode": "auto",
        "idempotency_key": "uncertain-media-route",
    }
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "route_confirmation_required"
    assert detail["plan"]["operation"] == "text_to_image"

    confirmed = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={**payload, "confirm_media": True},
    )
    assert confirmed.status_code == 202
    assert confirmed.json()["run"]["operation"] == "text_to_image"


async def test_expensive_auto_video_reports_estimate_before_queueing(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Video estimate"})).json()
    payload = {
        "text": "Generate a video of a lighthouse through the night",
        "mode": "auto",
        "settings": {"width": 1024, "height": 576, "frames": 121, "steps": 30, "fps": 24},
        "idempotency_key": "expensive-video-route",
    }
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert response.status_code == 409
    detail = response.json()["detail"]
    estimate = detail["plan"]["parameter_overrides"]["_generation_estimate"]
    assert detail["plan"]["operation"] == "text_to_video"
    assert estimate["duration_seconds"] == 5.04
    assert estimate["estimated_intermediate_bytes"] > estimate["estimated_output_bytes"]

    confirmed = await client.post(
        f"/api/chats/{chat['id']}/turns", json={**payload, "confirm_media": True}
    )
    assert confirmed.status_code == 202
    assert confirmed.json()["run"]["provenance_json"]["generation_estimate"] == estimate


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
    assistant_id = turn.json()["assistant_message"]["id"]
    deadline = asyncio.get_running_loop().time() + 5
    streamed_text = ""
    while asyncio.get_running_loop().time() < deadline:
        assistant = (await client.get(f"/api/messages/{assistant_id}")).json()
        streamed_text = "".join(
            part["text"] or "" for part in assistant["parts"] if part["type"] == "text"
        )
        if streamed_text:
            break
        await asyncio.sleep(0.01)
    assert streamed_text

    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    run = (await client.get(f"/api/runs/{turn.json()['run']['id']}")).json()
    assert run["status"] == "cancelled"
    assistant = (await client.get(f"/api/messages/{assistant_id}")).json()
    assert assistant["status"] == "cancelled"
    final_text = "".join(
        part["text"] or "" for part in assistant["parts"] if part["type"] == "text"
    )
    assert final_text.startswith(streamed_text.rstrip())
    assert not any(part["type"] == "error" for part in assistant["parts"])


async def test_failed_chat_run_preserves_streamed_text_and_reports_error(
    client: AsyncClient, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    async def fail_after_delta(
        _adapter: MockChatAdapter, _request: ChatRequest
    ) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="delta", text="Partial response that should remain")
        yield ChatEvent(type="error", data={"error": ""})

    monkeypatch.setattr(MockChatAdapter, "stream", fail_after_delta)
    chat = (await client.post("/api/chats", json={"title": "Failed response"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Start a response", "mode": "text"},
    )
    assert turn.status_code == 202

    deadline = asyncio.get_running_loop().time() + 5
    run = turn.json()["run"]
    while asyncio.get_running_loop().time() < deadline:
        run = (await client.get(f"/api/runs/{run['id']}")).json()
        if run["status"] == "failed":
            break
        await asyncio.sleep(0.01)
    assert run["status"] == "failed"
    assert run["error"] == "Chat engine stream failed"

    assistant = (await client.get(f"/api/messages/{turn.json()['assistant_message']['id']}")).json()
    assert assistant["status"] == "failed"
    assert [
        (part["type"], part["text"])
        for part in assistant["parts"]
        if part["type"] in {"text", "error"}
    ] == [
        ("text", "Partial response that should remain"),
        ("error", "Chat engine stream failed"),
    ]


async def test_cancelled_media_run_can_be_retried(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Retry cancelled media"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a retryable video", "mode": "video"},
    )
    assert turn.status_code == 202

    cancelled = await client.post(f"/api/chats/{chat['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    retried = await client.post(f"/api/jobs/{cancelled.json()['id']}/retry")
    assert retried.status_code == 200
    assert retried.json()["status"] in {"queued", "running"}

    run = await wait_for_run(client, turn.json()["run"]["id"])
    assert run["operation"] == "text_to_video"
    jobs = (await client.get("/api/jobs")).json()
    completed = next(job for job in jobs if job["id"] == cancelled.json()["id"])
    assert completed["status"] == "complete"
    assert completed["attempt"] == 2


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
    assert final_run["settings_json"]["context_length"] == 512
    assert final_run["provenance_json"]["resolved_settings"]["context_length"] == 512

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    assistant = [item for item in detail["messages"] if item["role"] == "assistant"][-1]
    metadata = next(part for part in assistant["parts"] if part["type"] == "generation_metadata")
    assert metadata["metadata_json"]["context"]["messages_omitted"] > 0


async def test_turn_rejects_load_only_setting_overrides(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Load settings"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Do not restart the worker",
            "mode": "text",
            "settings": {"context_length": 512},
        },
    )
    assert response.status_code == 422
    assert "unsupported settings: context_length" in response.json()["detail"]


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


async def test_media_library_reports_references_and_storage(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Gallery"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image for the gallery", "mode": "image"},
    )
    message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in message["parts"] if part["type"] == "image")

    gallery = await client.get("/api/artifacts", params={"kind": "image"})
    assert gallery.status_code == 200
    item = next(artifact for artifact in gallery.json() if artifact["id"] == image["artifact_id"])
    assert item["reference_count"] == 1
    assert item["chat_ids"] == [chat["id"]]

    storage = await client.get("/api/artifacts/storage")
    assert storage.status_code == 200
    assert storage.json()["referenced_count"] >= 1
    assert storage.json()["retention_pending_count"] == 0
    assert storage.json()["retention_days"] == 30
    assert turn.status_code == 202


async def test_text_turn_after_image_keeps_alternating_chat_context(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Mixed media context"})).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a reference image", "mode": "image"},
    )
    await wait_for_assistant(client, chat["id"], "image")

    with SessionLocal() as session:
        persisted_chat = session.get(Chat, chat["id"])
        assert persisted_chat
        routing_context = ConversationOrchestrator._routing_context(
            session,
            persisted_chat,
            persisted_chat.active_head_message_id,
        )
    assert routing_context == [
        {"role": "user", "content": "Create a reference image"},
        {
            "role": "assistant",
            "content": 'Generated image from this prompt: "Create a reference image".',
        },
    ]

    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Now answer a text question", "mode": "text"},
    )
    assert accepted.status_code == 202
    await wait_for_assistant(client, chat["id"], "text")

    with SessionLocal() as session:
        run = session.get(Run, accepted.json()["run"]["id"])
        assert run
        generation_context = ConversationOrchestrator._context_messages(session, run)
    assert generation_context == [
        {"role": "user", "content": "Create a reference image"},
        {
            "role": "assistant",
            "content": 'Generated image from this prompt: "Create a reference image".',
        },
        {"role": "user", "content": "Now answer a text question"},
    ]


async def test_image_request_uses_referenced_text_from_the_active_chat_branch(
    client: AsyncClient,
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Story illustration"})).json()
    story_request = "Write a short story about a silver fox crossing a glass city at dusk."
    text_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": story_request, "mode": "text"},
    )
    assert text_turn.status_code == 202
    story_message = await wait_for_assistant(client, chat["id"], "text")
    story_text = "\n".join(
        part["text"] for part in story_message["parts"] if part["type"] == "text"
    ).strip()

    image_turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make an image based on the previous story", "mode": "auto"},
    )
    assert image_turn.status_code == 202
    image_run = image_turn.json()["run"]
    assert image_run["operation"] == "text_to_image"
    assert "Source chat text:" in image_run["standalone_prompt"]
    assert story_request in image_run["standalone_prompt"]
    assert story_text in image_run["standalone_prompt"]

    image_message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in image_message["parts"] if part["type"] == "image")
    assert (
        image["artifact"]["metadata_json"]["semantic_description"] == image_run["standalone_prompt"]
    )


async def test_prior_image_edit_falls_back_to_accumulated_text_prompt(
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        for workflow in session.query(WorkflowDefinition).filter_by(operation="image_to_image"):
            session.delete(workflow)
        session.commit()

    chat = (await client.post("/api/chats", json={"title": "Semantic image edit"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make me an image of an apple", "mode": "auto"},
    )
    assert first.status_code == 202
    await wait_for_assistant(client, chat["id"], "image")

    edited = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Make it green", "mode": "auto"},
    )
    assert edited.status_code == 202
    edited_run = await wait_for_run(client, edited.json()["run"]["id"])

    assert edited_run["operation"] == "text_to_image"
    assert edited_run["standalone_prompt"] == (
        "Make me an image of an apple. Follow-up instruction: Make it green"
    )
    edited_message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in edited_message["parts"] if part["type"] == "image")
    assert image["artifact"]["metadata_json"]["semantic_description"] == (
        "Make me an image of an apple. Follow-up instruction: Make it green"
    )


async def test_retention_cleanup_only_removes_expired_unreferenced_artifacts(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.bin", b"expired-orphan", "application/octet-stream")},
    )
    artifact_id = uploaded.json()["id"]
    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact
        artifact.metadata_json = {
            **artifact.metadata_json,
            "unreferenced_at": (datetime.now(UTC) - timedelta(days=31)).isoformat(),
        }
        session.commit()

    preview = await client.post("/api/artifacts/cleanup", json={"dry_run": True})
    assert preview.status_code == 200
    assert preview.json()["removed_count"] == 1
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200

    cleaned = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleaned.status_code == 200
    assert cleaned.json()["removed_count"] == 1
    assert cleaned.json()["reclaimed_bytes"] == len(b"expired-orphan")
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 404


async def test_cleanup_marks_new_unreferenced_artifacts_for_recovery(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.png", b"new-orphan", "image/png")},
    )
    artifact_id = uploaded.json()["id"]

    cleaned = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleaned.status_code == 200
    assert cleaned.json()["marked_count"] == 1
    assert cleaned.json()["removed_count"] == 0

    with SessionLocal() as session:
        artifact = session.get(Artifact, artifact_id)
        assert artifact
        assert isinstance(artifact.metadata_json.get("unreferenced_at"), str)


async def test_media_library_can_delete_a_referenced_artifact(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Delete media"})).json()
    await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create an image to delete", "mode": "image"},
    )
    message = await wait_for_assistant(client, chat["id"], "image")
    image = next(part for part in message["parts"] if part["type"] == "image")
    artifact_id = image["artifact_id"]

    deleted = await client.delete(f"/api/artifacts/{artifact_id}")
    assert deleted.status_code == 200
    assert deleted.json()["artifact_id"] == artifact_id
    assert deleted.json()["reference_count"] == 1
    assert deleted.json()["removed_count"] == 1
    assert deleted.json()["reclaimed_bytes"] > 0
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 404

    detail = (await client.get(f"/api/chats/{chat['id']}")).json()
    deleted_part = next(
        part for item in detail["messages"] for part in item["parts"] if part["id"] == image["id"]
    )
    assert deleted_part["artifact_id"] is None


async def test_chat_delete_can_remove_exclusive_generated_media(client: AsyncClient) -> None:
    keep_chat = (await client.post("/api/chats", json={"title": "Keep media"})).json()
    await client.post(
        f"/api/chats/{keep_chat['id']}/turns",
        json={"text": "Create an image to keep", "mode": "image"},
    )
    keep_message = await wait_for_assistant(client, keep_chat["id"], "image")
    keep_artifact_id = next(
        part["artifact_id"] for part in keep_message["parts"] if part["type"] == "image"
    )

    assert (await client.delete(f"/api/chats/{keep_chat['id']}")).status_code == 204
    assert (await client.get(f"/api/artifacts/{keep_artifact_id}")).status_code == 200

    delete_chat = (await client.post("/api/chats", json={"title": "Delete media"})).json()
    await client.post(
        f"/api/chats/{delete_chat['id']}/turns",
        json={"text": "Create an image to remove", "mode": "image"},
    )
    delete_message = await wait_for_assistant(client, delete_chat["id"], "image")
    delete_artifact_id = next(
        part["artifact_id"] for part in delete_message["parts"] if part["type"] == "image"
    )

    deleted = await client.delete(
        f"/api/chats/{delete_chat['id']}",
        params={"delete_generated_media": True},
    )
    assert deleted.status_code == 204
    assert (await client.get(f"/api/artifacts/{delete_artifact_id}")).status_code == 404


async def test_chat_delete_keeps_generated_media_referenced_by_another_chat(
    client: AsyncClient,
) -> None:
    source_chat = (await client.post("/api/chats", json={"title": "Source"})).json()
    await client.post(
        f"/api/chats/{source_chat['id']}/turns",
        json={"text": "Create a shared image", "mode": "image"},
    )
    source_message = await wait_for_assistant(client, source_chat["id"], "image")
    artifact_id = next(
        part["artifact_id"] for part in source_message["parts"] if part["type"] == "image"
    )

    other_chat = (await client.post("/api/chats", json={"title": "Other"})).json()
    attached = await client.post(
        f"/api/chats/{other_chat['id']}/turns",
        json={
            "text": "Describe this image",
            "mode": "text",
            "input_artifact_ids": [artifact_id],
        },
    )
    assert attached.status_code == 202

    deleted = await client.delete(
        f"/api/chats/{source_chat['id']}",
        params={"delete_generated_media": True},
    )
    assert deleted.status_code == 204
    assert (await client.get(f"/api/artifacts/{artifact_id}")).status_code == 200


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


async def test_workflow_vram_requirement_uses_device_capacity(
    client: AsyncClient, monkeypatch
) -> None:
    memory = {
        "total": 16 * 1024**3,
        "available": 8 * 1024**3,
    }

    def system_info(_settings):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            devices=[
                SimpleNamespace(
                    kind="gpu",
                    total_memory_bytes=memory["total"],
                    available_memory_bytes=memory["available"],
                )
            ]
        )

    monkeypatch.setattr("local_lm.api.collect_system_info", system_info)
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "VRAM capacity contract",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": "Mock"}},
                "dependencies": {"minimum_vram_bytes": 12 * 1024**3},
                "trusted": True,
            },
        )
    ).json()

    under_pressure = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert under_pressure.status_code == 200
    assert under_pressure.json()["valid"] is True
    assert under_pressure.json()["errors"] == []
    assert under_pressure.json()["warnings"] == [
        "currently available accelerator memory is below the workflow requirement"
    ]

    memory["total"] = 8 * 1024**3
    insufficient = await client.post(f"/api/workflows/{workflow['id']}/validate")
    assert insufficient.status_code == 200
    assert insufficient.json()["valid"] is False
    assert insufficient.json()["warnings"] == []
    assert insufficient.json()["errors"] == [
        "accelerator memory capacity is below the workflow requirement"
    ]


async def test_project_pins_an_immutable_media_workflow_revision(client: AsyncClient) -> None:
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Project image recipe",
                "operation": "text_to_image",
                "api_graph": {"1": {"class_type": "SaveImage", "inputs": {}}},
                "trusted": True,
            },
        )
    ).json()
    revision_id = workflow["current_revision_id"]
    project_response = await client.post(
        "/api/projects",
        json={"name": "Pinned studio", "image_workflow_revision_id": revision_id},
    )
    assert project_response.status_code == 201
    project = project_response.json()
    assert project["image_workflow_revision_id"] == revision_id

    incompatible = await client.patch(
        f"/api/projects/{project['id']}",
        json={"video_workflow_revision_id": revision_id},
    )
    assert incompatible.status_code == 422

    chat = (
        await client.post("/api/chats", json={"title": "Pinned run", "project_id": project["id"]})
    ).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw a blue cabin", "mode": "image"},
    )
    assert turn.status_code == 202
    assert turn.json()["run"]["workflow_revision_id"] == revision_id


async def test_media_workflow_follows_the_selected_model_dependency(
    client: AsyncClient, tmp_path: Path
) -> None:
    installs = []
    for name in ("first-image-model", "second-image-model"):
        model_dir = tmp_path / name
        model_dir.mkdir()
        (model_dir / f"{name}.safetensors").write_bytes(b"safe")
        installs.append(
            (
                await client.post(
                    "/api/models/import",
                    json={
                        "name": name,
                        "role": "image",
                        "engine": "mock",
                        "local_path": str(model_dir),
                    },
                )
            ).json()
        )
    profiles = (await client.get("/api/profiles")).json()
    selected_profile = next(
        profile for profile in profiles if profile["model_install_id"] == installs[0]["id"]
    )
    revisions = []
    for index, install in enumerate(installs, start=1):
        workflow = (
            await client.post(
                "/api/workflows",
                json={
                    "name": f"Model-specific image workflow {index}",
                    "operation": "text_to_image",
                    "engine": "mock",
                    "api_graph": {"node": {"class_type": f"MockImage{index}"}},
                    "dependencies": {"model_install_ids": [install["id"]]},
                    "trusted": True,
                },
            )
        ).json()
        revisions.append(workflow["current_revision_id"])

    chat = (await client.post("/api/chats", json={"title": "Dependency routing"})).json()
    updated = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": selected_profile["id"]},
    )
    assert updated.status_code == 200
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Draw with the first model", "mode": "image"},
    )

    assert turn.status_code == 202
    assert turn.json()["run"]["workflow_revision_id"] == revisions[0]


async def test_pinned_workflow_schema_drives_generation_settings(client: AsyncClient) -> None:
    workflow = (
        await client.post(
            "/api/workflows",
            json={
                "name": "Constrained video recipe",
                "operation": "text_to_video",
                "engine": "mock",
                "api_graph": {"node": {"class_type": "Mock"}},
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "width": {"type": "integer", "const": 832},
                        "height": {"type": "integer", "const": 480},
                        "frames": {"type": "integer", "default": 81, "minimum": 1},
                        "fps": {"type": "number", "default": 16, "minimum": 1},
                        "steps": {
                            "type": "integer",
                            "default": 12,
                            "minimum": 1,
                            "maximum": 30,
                        },
                        "codec": {"type": "string", "const": "h264"},
                        "camera_strength": {
                            "type": "number",
                            "default": 0.5,
                            "minimum": 0,
                            "maximum": 1,
                        },
                    },
                },
                "trusted": True,
            },
        )
    ).json()
    project = (
        await client.post(
            "/api/projects",
            json={
                "name": "Constrained video project",
                "video_workflow_revision_id": workflow["current_revision_id"],
            },
        )
    ).json()
    profile = (
        await client.post(
            "/api/profiles",
            json={
                "name": "Reusable video profile",
                "role": "video",
                "engine": "mock",
                "request_settings": {"width": 768, "steps": 20},
            },
        )
    ).json()
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Reusable video preset",
            "role": "video",
            "settings": {"frames": 49, "codec": "vp9"},
            "is_default": True,
        },
    )
    assert preset.status_code == 201
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "Workflow settings", "project_id": project["id"]},
        )
    ).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_video_profile_id": profile["id"]},
    )
    assert selected.status_code == 200

    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a short camera move",
            "mode": "video",
            "settings": {"camera_strength": 0.75},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["workflow_revision_id"] == workflow["current_revision_id"]
    assert run["settings_json"]["width"] == 832
    assert run["settings_json"]["height"] == 480
    # Compatible stored profile/preset values still apply, while values that
    # conflict with fixed workflow controls fall back to the workflow defaults.
    assert run["settings_json"]["frames"] == 49
    assert run["settings_json"]["fps"] == 16
    assert run["settings_json"]["steps"] == 20
    assert run["settings_json"]["codec"] == "h264"
    assert run["settings_json"]["camera_strength"] == 0.75
    assert run["provenance_json"]["generation_estimate"]["width"] == 832
    assert run["provenance_json"]["generation_estimate"]["height"] == 480

    regenerated = await client.post(
        f"/api/messages/{run['assistant_message_id']}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    assert regenerated.json()["run"]["settings_json"]["camera_strength"] == 0.75

    branched = await client.post(
        f"/api/messages/{run['user_message_id']}/branch",
        json={"text": "Create a different camera move", "settings": {}},
    )
    assert branched.status_code == 202
    assert branched.json()["run"]["settings_json"]["camera_strength"] == 0.75

    incompatible = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Try an unsupported codec",
            "mode": "video",
            "settings": {"codec": "vp9"},
        },
    )
    assert incompatible.status_code == 422
    assert "codec must be one of" in incompatible.json()["detail"]


async def test_profile_edit_clone_reset_and_portable_bundle(client: AsyncClient) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    source = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{source['id']}",
        json={
            "name": "Focused chat",
            "use_case": "Programming, code review, and technical explanations",
            "load_settings": {"context_length": 16_384},
            "request_settings": {"temperature": 0.25},
        },
    )
    assert updated.status_code == 200
    assert updated.json()["use_case"] == "Programming, code review, and technical explanations"
    assert updated.json()["load_settings_json"] == {"context_length": 16_384}

    cloned = await client.post(
        f"/api/profiles/{source['id']}/clone", json={"name": "Focused chat copy"}
    )
    assert cloned.status_code == 201
    assert cloned.json()["use_case"] == updated.json()["use_case"]
    assert cloned.json()["request_settings_json"] == {"temperature": 0.25}
    assert cloned.json()["is_default"] is False

    exported = await client.get(f"/api/profiles/{source['id']}/export")
    assert exported.status_code == 200
    bundle = exported.json()
    assert bundle["format"] == "lm-atelier-profile"
    assert bundle["version"] == 1
    assert bundle["use_case"] == updated.json()["use_case"]

    bundle["name"] = "Imported portable chat"
    bundle["model_install_id"] = "missing-on-this-machine"
    imported = await client.post("/api/profiles/import", json=bundle)
    assert imported.status_code == 201
    assert imported.json()["model_install_id"] is None
    assert imported.json()["use_case"] == updated.json()["use_case"]
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


async def test_auto_model_selection_matches_profile_use_case(client: AsyncClient) -> None:
    programming = await client.post(
        "/api/profiles",
        json={
            "name": "Code specialist",
            "use_case": "Python programming, code review, debugging, and software architecture",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert programming.status_code == 201
    creative = await client.post(
        "/api/profiles",
        json={
            "name": "Creative writer",
            "use_case": "Fiction, poetry, characters, and narrative prose",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert creative.status_code == 201

    chat = (await client.post("/api/chats", json={"title": "Automatic model"})).json()
    assert chat["active_chat_profile_id"] == "__auto__"
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Review and debug this Python code", "mode": "text"},
    )
    assert response.status_code == 202
    selection = response.json()["run"]["provenance_json"]["model_selection"]
    assert selection["mode"] == "auto"
    assert selection["profile_id"] == programming.json()["id"]
    assert {"python", "code"}.issubset(selection["matched_terms"])


async def test_auto_model_selection_normalizes_common_intent_terms(client: AsyncClient) -> None:
    technical = await client.post(
        "/api/profiles",
        json={
            "name": "Technical specialist",
            "use_case": "Software development and application architecture",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert technical.status_code == 201
    visual = await client.post(
        "/api/profiles",
        json={
            "name": "Visual specialist",
            "use_case": "Illustration, photography, and artwork",
            "role": "chat",
            "engine": "mock",
        },
    )
    assert visual.status_code == 201

    chat = (await client.post("/api/chats", json={"title": "Intent aliases"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Help me code a small command-line tool", "mode": "text"},
    )

    assert response.status_code == 202
    selection = response.json()["run"]["provenance_json"]["model_selection"]
    assert selection["profile_id"] == technical.json()["id"]
    assert "code" in selection["matched_terms"]


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


async def test_default_preset_is_resolved_between_profile_and_turn_settings(
    client: AsyncClient,
) -> None:
    profiles = (await client.get("/api/profiles?role=chat")).json()
    profile = profiles[0]
    updated = await client.patch(
        f"/api/profiles/{profile['id']}",
        json={"request_settings": {"temperature": 0.25, "max_tokens": 32}},
    )
    assert updated.status_code == 200
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Default precise",
            "role": "chat",
            "settings": {"temperature": 0.1, "max_tokens": 48},
            "is_default": True,
        },
    )
    assert preset.status_code == 201
    chat = (await client.post("/api/chats", json={"title": "Preset resolution"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Apply all setting layers",
            "mode": "text",
            "settings": {"max_tokens": 16},
        },
    )
    assert turn.status_code == 202
    run = turn.json()["run"]
    assert run["settings_json"]["temperature"] == 0.1
    assert run["settings_json"]["max_tokens"] == 16
    assert run["provenance_json"]["preset"] == {
        "id": preset.json()["id"],
        "name": "Default precise",
        "role": "chat",
        "settings": {"temperature": 0.1, "max_tokens": 48},
    }


async def test_chat_preset_rejects_load_only_settings(client: AsyncClient) -> None:
    preset = await client.post(
        "/api/presets",
        json={
            "name": "Invalid load preset",
            "role": "chat",
            "settings": {"context_length": 512},
        },
    )
    assert preset.status_code == 422
    assert "unsupported settings: context_length" in preset.json()["detail"]


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
    chat = (await client.post("/api/chats", json={"title": "Cascade model"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_chat_profile_id": profile.json()["id"]},
    )
    assert selected.status_code == 200
    assert (await client.delete(f"/api/models/{first.json()['id']}")).status_code == 409
    assert (
        await client.delete(f"/api/models/{first.json()['id']}", params={"delete_profiles": True})
    ).status_code == 204
    profiles = (await client.get("/api/profiles")).json()
    assert not any(item["model_install_id"] == first.json()["id"] for item in profiles)
    updated_chat = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert updated_chat["active_chat_profile_id"] == "__auto__"
    assert (shared / "model.gguf").is_file()
    assert (
        await client.delete(f"/api/models/{second.json()['id']}", params={"delete_profiles": True})
    ).status_code == 204
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

    failed = await client.post(
        "/api/downloads",
        json={"remote_id": "owner/retry", "role": "chat", "engine": "llama.cpp"},
    )
    failed_id = failed.json()["id"]
    with SessionLocal() as session:
        failed_job = session.get(Job, failed_id)
        assert failed_job
        failed_job.status = JobStatus.FAILED.value
        failed_job.phase = "failed"
        failed_job.error = "incomplete HTTP read"
        failed_job.completed_at = utcnow()
        session.commit()

    retried = await client.post(f"/api/downloads/{failed_id}/resume")
    assert retried.status_code == 200
    assert retried.json()["status"] == "queued"
    assert retried.json()["phase"] == "retry queued"
    assert retried.json()["error"] is None
    await client.post(f"/api/jobs/{failed_id}/cancel")


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
                {"filename": "large.gguf", "size": 2048, "sha256": "b" * 64},
                {"filename": "small.gguf", "size": 1024, "sha256": "a" * 64},
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
    assert response.json()["expected_sha256"] == {"small.gguf": "a" * 64}
    checksum = next(check for check in response.json()["checks"] if check["id"] == "checksum")
    assert checksum["status"] == "pass"


async def test_catalog_preflight_prefers_balanced_gguf_quantization(
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
                "name": "Quantized model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["GGUF artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "model-Q2_K.gguf", "size": 1024, "sha256": "a" * 64},
                {"filename": "model-Q4_K_M.gguf", "size": 2048, "sha256": "b" * 64},
                {"filename": "model-Q8_0.gguf", "size": 4096, "sha256": "c" * 64},
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
    assert response.json()["selected_files"] == ["model-Q4_K_M.gguf"]


async def test_catalog_preflight_autoselects_safe_media_checkpoint(
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
                "name": "Safe image model",
                "license_id": "apache-2.0",
                "compatibility": "likely",
                "compatibility_reasons": ["safetensors artifact detected"],
            },
            "revision": revision,
            "files": [
                {"filename": "model.safetensors", "size": 2048, "sha256": "a" * 64},
                {"filename": "vae.safetensors", "size": 1024, "sha256": "b" * 64},
            ],
        }

    monkeypatch.setattr(HuggingFaceCatalog, "inspect", inspect)
    response = await client.post(
        "/api/catalog/owner/image-model/preflight",
        json={
            "revision": "abc123",
            "role": "image",
            "engine": "comfyui",
            "selected_files": [],
        },
    )
    assert response.status_code == 200
    assert response.json()["can_install"] is True
    assert response.json()["selected_files"] == ["model.safetensors"]
