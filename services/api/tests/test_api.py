from __future__ import annotations

import asyncio
import io
import zipfile

from httpx2 import AsyncClient


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
    assert any(part["artifact_id"] for part in message["parts"] if part["type"] == "video")

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
    media = await client.post("/api/workers/media/start")
    assert media.status_code == 422


async def test_turn_idempotency_returns_original_run(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Idempotency"})).json()
    payload = {"text": "Hello", "mode": "text", "idempotency_key": "stable-key"}
    first = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    second = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json()["run"]["id"] == second.json()["run"]["id"]


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
