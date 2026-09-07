from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock

import pytest
from httpx2 import AsyncClient
from sqlalchemy import update
from test_custom_node_source_identity import installed_source as installed_source
from test_reviewed_custom_node_execution import (
    _approve_custom,
)
from test_reviewed_custom_node_execution import (
    custom_workflow as custom_workflow,
)
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime

from local_lm.custom_nodes import CustomNodeManager
from local_lm.db import SessionLocal
from local_lm.models import Chat, Run


@pytest.mark.parametrize("stage", ["http", "package", "selection_change"])
async def test_workflow_dispatch_releases_transactions_during_verification(
    custom_workflow, client: AsyncClient, app, settings, monkeypatch, stage: str
) -> None:
    workflow, _, _ = custom_workflow
    await _approve_custom(client, custom_workflow)
    services = app.state.services
    settings.media_engine = "comfyui"
    monkeypatch.setattr(services.orchestrator, "_ensure_media_worker", AsyncMock())
    profile = await client.post(
        "/api/profiles",
        json={"name": "Concurrent workflow model", "role": "image", "engine": "comfyui"},
    )
    assert profile.status_code == 201, profile.text
    chat = (await client.post("/api/chats", json={"title": "Concurrent workflow"})).json()
    selected = await client.patch(
        f"/api/chats/{chat['id']}",
        json={"active_image_profile_id": profile.json()["id"]},
    )
    assert selected.status_code == 200, selected.text
    active_sessions = []
    observed_transactions: list[bool] = []
    writes: list[str] = []
    captured = []
    original_factory = services.orchestrator.session_factory

    @contextmanager
    def tracked_session():
        with original_factory() as session:
            active_sessions.append(session)
            try:
                yield session
            finally:
                active_sessions.remove(session)

    original_generate = services.engines.media.generate

    async def generate(request):
        captured.append(request)
        async for event in original_generate(request):
            yield event

    original_info = services.engines.media.object_info
    original_verify = CustomNodeManager.verify
    run_id = ""

    async def probe():
        observed_transactions.append(any(session.in_transaction() for session in active_sessions))

        def write():
            with SessionLocal() as writer:
                writer.connection().exec_driver_sql("PRAGMA busy_timeout=500")
                writer.execute(
                    update(Chat).where(Chat.id == chat["id"]).values(title="Writer progressed")
                )
                if stage == "selection_change":
                    writer.execute(
                        update(Run).where(Run.id == run_id).values(workflow_revision_id=None)
                    )
                writer.commit()
                writes.append("committed")

        await asyncio.wait_for(asyncio.to_thread(write), timeout=2)

    async def object_info():
        if stage in {"http", "selection_change"}:
            await probe()
        return await original_info()

    async def verify(manager, install):
        if stage == "package":
            await probe()
        await original_verify(manager, install)

    async with services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Draw a blue bowl",
                "mode": "image",
                "workflow_revision_id": workflow["current_revision_id"],
            },
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run"]["id"]
        monkeypatch.setattr(services.orchestrator, "session_factory", tracked_session)
        monkeypatch.setattr(services.engines.media, "generate", generate)
        monkeypatch.setattr(services.engines.media, "object_info", object_info)
        monkeypatch.setattr(CustomNodeManager, "verify", verify)

    deadline = asyncio.get_running_loop().time() + 6
    while asyncio.get_running_loop().time() < deadline:
        current = (await client.get(f"/api/runs/{run_id}")).json()
        if current["status"] in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.03)
    else:
        raise AssertionError("workflow dispatch did not terminate")
    assert observed_transactions, "verification probe did not execute"
    assert writes and len(writes) == len(observed_transactions), current
    assert not any(observed_transactions), (
        "a workflow database transaction spanned verification I/O"
    )
    if stage == "selection_change":
        assert current["status"] == "failed", current
        assert captured == [], "dispatch used a workflow different from the reloaded run selection"
    else:
        assert current["status"] == "complete", current
        assert len(captured) == 1
    with SessionLocal() as session:
        assert session.get(Chat, chat["id"]).title == "Writer progressed"
