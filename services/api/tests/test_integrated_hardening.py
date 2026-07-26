from __future__ import annotations

import asyncio
import io
import json
import zipfile
from collections.abc import AsyncIterator
from types import SimpleNamespace

from httpx2 import AsyncClient
from sqlalchemy import select, update

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.auxiliary_assets import checkpoint_lora_extension
from local_lm.db import SessionLocal
from local_lm.domain import utcnow
from local_lm.models import (
    Job,
    ModelAssetInstall,
    Run,
    WorkflowDefinition,
    WorkflowRevision,
)


async def _wait_for_run(
    client: AsyncClient,
    run_id: str,
    expected: str = "complete",
) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 8
    run: dict = {}  # type: ignore[type-arg]
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        run = response.json()
        if run["status"] in {"complete", "failed", "cancelled"}:
            assert run["status"] == expected, run
            return run
        await asyncio.sleep(0.03)
    raise AssertionError(f"run {run_id} did not become {expected}: {run}")


async def _wait_for_step_states(
    client: AsyncClient,
    plan_id: str,
    expected: list[str],
) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 8
    plan: dict = {}  # type: ignore[type-arg]
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/work-plans/{plan_id}")
        assert response.status_code == 200
        plan = response.json()
        if [step["status"] for step in plan["steps"]] == expected:
            return plan
        await asyncio.sleep(0.03)
    raise AssertionError(f"plan {plan_id} did not reach {expected}: {plan}")


async def test_ordered_story_image_video_summary_retries_only_cancelled_video(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_generate = MockMediaAdapter.generate
    video_started = asyncio.Event()

    async def hold_video(
        adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        if "video" not in request.operation:
            async for event in original_generate(adapter, request):
                yield event
            return
        video_started.set()
        while request.run_id not in adapter._cancelled:
            await asyncio.sleep(0.01)
        adapter._cancelled.discard(request.run_id)
        yield MediaEvent(type="cancelled", phase="cancelled")

    monkeypatch.setattr(MockMediaAdapter, "generate", hold_video)
    chat = (await client.post("/api/chats", json={"title": "Integrated ordered retry"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": (
                "Write a short story about a lantern, then create an image based on it, "
                "then animate the image into a video, then summarize the video"
            ),
            "mode": "auto",
            "confirm_media": True,
        },
    )
    assert accepted.status_code == 202
    plan_id = accepted.json()["run"]["work_plan_id"]
    plan = (await client.get(f"/api/work-plans/{plan_id}")).json()
    assert [step["operation"] for step in plan["steps"]] == [
        "text",
        "text_to_image",
        "image_to_video",
        "text",
    ]
    await asyncio.wait_for(video_started.wait(), timeout=8)

    video_step = plan["steps"][2]
    cancelled = await client.post(f"/api/work-steps/{video_step['id']}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    blocked = await _wait_for_step_states(
        client,
        plan_id,
        ["complete", "complete", "cancelled", "blocked"],
    )
    image_run_id = blocked["summary_json"]["run_ids"][1]
    with SessionLocal() as session:
        image_run_before = session.get(Run, image_run_id)
        assert image_run_before
        image_artifacts_before = [image_run_before.provenance_json["outputs"][0]["artifact_id"]]

    monkeypatch.setattr(MockMediaAdapter, "generate", original_generate)
    retried = await client.post(f"/api/work-steps/{video_step['id']}/retry")
    assert retried.status_code == 200
    for run_id in blocked["summary_json"]["run_ids"]:
        await _wait_for_run(client, run_id)
    completed = await _wait_for_step_states(
        client,
        plan_id,
        ["complete", "complete", "complete", "complete"],
    )
    assert completed["status"] == "complete"

    with SessionLocal() as session:
        jobs = session.scalars(
            select(Job).where(Job.work_plan_id == plan_id).order_by(Job.work_step_id)
        ).all()
        attempts = {job.work_step_id: job.attempt for job in jobs}
        assert attempts[plan["steps"][0]["id"]] == 1
        assert attempts[plan["steps"][1]["id"]] == 1
        assert attempts[plan["steps"][2]["id"]] == 2
        assert attempts[plan["steps"][3]["id"]] == 1
        image_run_after = session.get(Run, image_run_id)
        assert image_run_after
        assert [image_run_after.provenance_json["outputs"][0]["artifact_id"]] == (
            image_artifacts_before
        )


async def test_lora_image_regeneration_cancel_retry_revision_switch_and_export(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    with SessionLocal() as session:
        definition = session.scalar(
            select(WorkflowDefinition).where(WorkflowDefinition.operation == "text_to_image")
        )
        assert definition and definition.current_revision_id
        revision = session.get(WorkflowRevision, definition.current_revision_id)
        assert revision
        graph = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "mock.safetensors"},
            },
            "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1]}},
            "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0]}},
        }
        extension = checkpoint_lora_extension(graph)
        assert extension
        revision.api_graph_json = graph
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "loras": {
                    "type": "array",
                    "default": [],
                    "maxItems": 8,
                }
            },
        }
        revision.dependencies_json = {"extensions": {"lora": extension}}
        asset = ModelAssetInstall(
            name="Integrated Ink",
            kind="lora",
            family="sdxl",
            local_path="C:/managed/integrated-ink",
            size_bytes=2048,
            manifest_json={
                "sha256": "e" * 64,
                "comfy_name": "integrated-ink.safetensors",
                "metadata": {"trigger_words": ["integrated ink"]},
            },
            active=True,
            verified_at=utcnow(),
        )
        session.add(asset)
        session.commit()
        asset_id = asset.id

    project = (await client.post("/api/projects", json={"name": "LoRA revision journey"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={"title": "LoRA history", "project_id": project["id"]},
        )
    ).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an ink illustration of an old workshop",
            "mode": "image",
            "settings": {
                "loras": [
                    {
                        "asset_id": asset_id,
                        "model_strength": 0.8,
                        "clip_strength": 0.7,
                        "enabled": True,
                    }
                ]
            },
        },
    )
    assert first.status_code == 202
    first_run = await _wait_for_run(client, first.json()["run"]["id"])
    assert first_run["provenance_json"]["auxiliary_assets"]["lora_stack"][0]["sha256"] == "e" * 64
    image_message_id = first.json()["assistant_message"]["id"]

    later = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Explain the atmosphere in one sentence", "mode": "text"},
    )
    assert later.status_code == 202
    await _wait_for_run(client, later.json()["run"]["id"])
    later_message_id = later.json()["assistant_message"]["id"]
    visible_before = [
        message["id"]
        for message in (await client.get(f"/api/chats/{chat['id']}")).json()["messages"]
        if message["transcript_visible"]
    ]

    original_generate = MockMediaAdapter.generate
    regeneration_started = asyncio.Event()

    async def hold_regeneration(
        adapter: MockMediaAdapter,
        request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        regeneration_started.set()
        while request.run_id not in adapter._cancelled:
            await asyncio.sleep(0.01)
        adapter._cancelled.discard(request.run_id)
        yield MediaEvent(type="cancelled", phase="cancelled")

    monkeypatch.setattr(MockMediaAdapter, "generate", hold_regeneration)
    regenerated = await client.post(
        f"/api/messages/{image_message_id}/regenerate",
        json={"settings": {}},
    )
    assert regenerated.status_code == 202
    replacement_run_id = regenerated.json()["run"]["id"]
    replacement_step_id = regenerated.json()["run"]["work_step_id"]
    await asyncio.wait_for(regeneration_started.wait(), timeout=8)
    cancelled = await client.post(f"/api/work-steps/{replacement_step_id}/cancel")
    assert cancelled.status_code == 200
    await _wait_for_run(client, replacement_run_id, "cancelled")

    after_cancel = (await client.get(f"/api/messages/{image_message_id}")).json()
    original_revision_id = after_cancel["active_response_revision_id"]
    assert [revision["status"] for revision in after_cancel["response_revisions"]] == [
        "complete",
        "cancelled",
    ]
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == later_message_id

    monkeypatch.setattr(MockMediaAdapter, "generate", original_generate)
    retried = await client.post(f"/api/work-steps/{replacement_step_id}/retry")
    assert retried.status_code == 200
    completed_replacement = await _wait_for_run(client, replacement_run_id)
    assert completed_replacement["settings_json"]["loras"][0]["asset_id"] == asset_id
    refreshed = (await client.get(f"/api/chats/{chat['id']}")).json()
    assert refreshed["active_head_message_id"] == later_message_id
    assert [
        message["id"] for message in refreshed["messages"] if message["transcript_visible"]
    ] == visible_before
    image_message = next(
        message for message in refreshed["messages"] if message["id"] == image_message_id
    )
    completed_revisions = [
        revision
        for revision in image_message["response_revisions"]
        if revision["status"] == "complete"
    ]
    assert len(completed_revisions) == 2
    replacement_revision_id = image_message["active_response_revision_id"]
    assert replacement_revision_id != original_revision_id

    selected_original = await client.post(
        f"/api/messages/{image_message_id}/revisions/{original_revision_id}/select"
    )
    assert selected_original.status_code == 200
    selected_replacement = await client.post(
        f"/api/messages/{image_message_id}/revisions/{replacement_revision_id}/select"
    )
    assert selected_replacement.status_code == 200
    assert (await client.get(f"/api/chats/{chat['id']}")).json()[
        "active_head_message_id"
    ] == later_message_id

    exported = await client.post(
        f"/api/projects/{project['id']}/export",
        params={"include_media": False},
    )
    archive_response = await client.get(exported.json()["url"])
    with zipfile.ZipFile(io.BytesIO(archive_response.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert all(not name.endswith(".safetensors") for name in archive.namelist())
    requirement = manifest["auxiliary_requirements"][0]
    assert requirement["id"] == f"auxiliary:lora:sha256:{'e' * 64}"
    assert "c:/managed" not in json.dumps(manifest).casefold()

    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": (
                "lora-revision-journey.lm-atelier.zip",
                archive_response.content,
                "application/zip",
            )
        },
    )
    assert imported.status_code == 201, imported.text
    imported_chat = (
        await client.get("/api/chats", params={"project_id": imported.json()["id"]})
    ).json()[0]
    imported_detail = (await client.get(f"/api/chats/{imported_chat['id']}")).json()
    imported_image = next(
        message
        for message in imported_detail["messages"]
        if message["role"] == "assistant"
        and any(part["type"] == "image" for part in message["parts"])
    )
    assert len(imported_image["response_revisions"]) == 2
    assert imported_image["active_response_revision_id"] == next(
        revision["id"]
        for revision in imported_image["response_revisions"]
        if revision["sequence"] == 2
    )


async def test_exhausted_media_storage_rejects_before_any_turn_is_written(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "local_lm.orchestrator.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    chat = (await client.post("/api/chats", json={"title": "No storage"})).json()
    rejected = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create an image of a clay teapot",
            "mode": "image",
        },
    )
    assert rejected.status_code == 422
    assert "not enough free storage" in rejected.json()["detail"]
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []
    assert (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json() == []


async def test_media_oom_fails_truthfully_then_retries_without_duplicate_output(
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    original_generate = MockMediaAdapter.generate

    async def fail_with_oom(
        _adapter: MockMediaAdapter,
        _request: MediaRequest,
    ) -> AsyncIterator[MediaEvent]:
        raise RuntimeError("CUDA out of memory")
        yield MediaEvent(type="complete")  # pragma: no cover

    monkeypatch.setattr(MockMediaAdapter, "generate", fail_with_oom)
    chat = (await client.post("/api/chats", json={"title": "OOM recovery"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create one image of a copper lantern", "mode": "image"},
    )
    assert accepted.status_code == 202
    run_id = accepted.json()["run"]["id"]
    failed = await _wait_for_run(client, run_id, "failed")
    assert failed["provenance_json"].get("outputs") in (None, [])
    message_id = accepted.json()["assistant_message"]["id"]
    failed_message = (await client.get(f"/api/messages/{message_id}")).json()
    assert not any(part["type"] == "image" for part in failed_message["parts"])

    monkeypatch.setattr(MockMediaAdapter, "generate", original_generate)
    step_id = accepted.json()["run"]["work_step_id"]
    retry = await client.post(f"/api/work-steps/{step_id}/retry")
    assert retry.status_code == 200
    completed = await _wait_for_run(client, run_id)
    assert len(completed["provenance_json"]["outputs"]) == 1
    completed_message = (await client.get(f"/api/messages/{message_id}")).json()
    assert sum(part["type"] == "image" for part in completed_message["parts"]) == 1
    with SessionLocal() as session:
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job and job.attempt == 2


async def test_video_postprocessing_never_holds_a_sqlite_write_transaction(
    app,
    client: AsyncClient,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    poster_checked = asyncio.Event()

    async def poster_with_concurrent_progress_write(_artifact) -> None:  # type: ignore[no-untyped-def]
        with SessionLocal() as concurrent:
            concurrent.execute(
                update(Job).where(Job.status == "running").values(updated_at=utcnow())
            )
            concurrent.commit()
        poster_checked.set()
        return None

    monkeypatch.setattr(
        app.state.services.artifacts,
        "video_poster",
        poster_with_concurrent_progress_write,
    )
    chat = (await client.post("/api/chats", json={"title": "Video transaction"})).json()
    accepted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": "Create a short video of a turning gear", "mode": "video"},
    )
    assert accepted.status_code == 202
    await _wait_for_run(client, accepted.json()["run"]["id"])
    assert poster_checked.is_set()
