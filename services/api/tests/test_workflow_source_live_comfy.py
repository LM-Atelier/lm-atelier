"""Install a constructed source and render it through an isolated real runtime."""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select
from test_workflow_review_live_comfy import settings as settings

from local_lm.db import SessionLocal
from local_lm.models import Job, WorkflowActivation, WorkflowInstallOffer, WorkflowRevision


def _source_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "EmptyImage",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
                "widgets_values": [64, 96, 1, 0x666666],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "widgets_values": ["constructed-installation"],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
    }


@pytest.mark.asyncio
async def test_one_source_approval_installs_and_renders_with_real_comfy(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    services = app.state.services
    real_replace = services.processes._replace

    async def cpu_replace(name: str, command: list[str], *args: Any, **kwargs: Any) -> None:
        if name == "media":
            command = [*command, "--cpu"]
        await real_replace(name, command, *args, **kwargs)

    monkeypatch.setattr(services.processes, "_replace", cpu_replace)
    async with services.scheduler.lease("primary"):
        await services.processes.start_media()
    initial_worker = next(item for item in services.processes.statuses() if item.name == "media")
    assert initial_worker.running and initial_worker.managed and initial_worker.pid
    graph = _source_graph()
    planned = await client.post(
        "/api/workflows/packages/install-plans",
        json={
            "name": "Constructed portrait installation",
            "operation": "text_to_image",
            "ui_graph": graph,
            "dependencies": {"version": 1, "slots": []},
            "selections": [],
        },
    )
    assert planned.status_code == 201, planned.text
    plan = planned.json()
    assert plan["can_accept"] and plan["blockers"] == [], plan
    assert plan["total_download_bytes"] == 0
    install_url = f"/api/workflow-install-offers/{plan['id']}/install"
    async with services.scheduler.lease("primary"):
        accepted = await client.post(install_url)
        assert accepted.status_code == 202, accepted.text
        jobs = accepted.json()
        assert [job["kind"] for job in jobs] == ["workflow_install"]
        with SessionLocal() as session:
            offer = session.scalar(
                select(WorkflowInstallOffer).where(
                    WorkflowInstallOffer.source_plan_id == plan["id"]
                )
            )
            assert offer is not None
            offer_id, draft_id = offer.id, offer.workflow_revision_id
            draft = session.get(WorkflowRevision, draft_id)
            assert draft is not None and not draft.trusted and draft.api_graph_json == {}
    pending = services.downloads._offer_tasks.get(offer_id)
    if pending is not None:
        await asyncio.wait_for(asyncio.shield(pending), timeout=300)
    progress = await client.get(f"/api/workflow-install-offers/{plan['id']}/progress")
    assert progress.status_code == 200, progress.text
    assert progress.json()["phase"] == "completed", progress.text
    revision_id = progress.json()["workflow_revision_id"]
    assert revision_id != draft_id
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None and revision.trusted and revision.ui_graph_json == graph
        assert revision.api_graph_json["1"]["inputs"]["width"] == 64
        assert revision.api_graph_json["1"]["inputs"]["height"] == 96
        activation = session.scalar(
            select(WorkflowActivation).where(WorkflowActivation.workflow_revision_id == revision_id)
        )
        assert activation is not None
        job = session.get(Job, jobs[0]["id"])
        assert job is not None and job.status == "complete"
    restored = next(item for item in services.processes.statuses() if item.name == "media")
    assert restored.running and restored.managed and restored.pid != initial_worker.pid
    repeated = await client.post(install_url)
    assert repeated.status_code == 202, repeated.text
    assert [item["id"] for item in repeated.json()] == [item["id"] for item in jobs]
    profile = await client.post(
        "/api/profiles",
        json={"name": "Constructed source CPU image", "role": "image", "engine": "comfyui"},
    )
    assert profile.status_code == 201, profile.text
    chat = (await client.post("/api/chats", json={"title": "Constructed source render"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}", json={"active_image_profile_id": profile.json()["id"]}
    )
    assert selected.status_code == 200, selected.text
    submitted = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Create a plain grey test image",
            "mode": "image",
            "workflow_revision_id": revision_id,
        },
    )
    assert submitted.status_code == 202, submitted.text
    run = submitted.json()["run"]
    assert run["workflow_revision_id"] == revision_id
    deadline = asyncio.get_running_loop().time() + 60
    while asyncio.get_running_loop().time() < deadline:
        current = (await client.get(f"/api/runs/{run['id']}")).json()
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.1)
    else:
        raise AssertionError("Constructed source render did not terminate")
    assert current["status"] == "complete", current
    outputs = current["provenance_json"]["outputs"]
    assert len(outputs) == 1
    content = await client.get(f"/api/artifacts/{outputs[0]['artifact_id']}/content")
    assert content.status_code == 200
    with Image.open(io.BytesIO(content.content)) as rendered:
        assert rendered.size == (64, 96) and rendered.mode == "RGB"
        assert rendered.getpixel((32, 48)) == (102, 102, 102)
