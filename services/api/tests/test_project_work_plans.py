from __future__ import annotations

import asyncio
import io
import json
import zipfile
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Project, Run, WorkPlan


async def _archive(
    app: FastAPI,
    client: AsyncClient,
    *,
    complete: bool,
    output_count: int = 1,
    workflow_revision_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    project = (await client.post("/api/projects", json={"name": "Portable edited versions"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Paper boats",
                "project_id": project["id"],
            },
        )
    ).json()
    async with app.state.services.scheduler.lease("primary"):
        source = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "Describe a blue paper boat",
                "mode": "text",
            },
        )
        assert source.status_code == 202, source.text
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={
                "text": "Describe a green paper boat",
                "idempotency_key": "portable-edit",
                **({"workflow_revision_id": workflow_revision_id} if workflow_revision_id else {}),
                **({"mode": "image", "output_count": output_count} if output_count > 1 else {}),
            },
        )
        assert edited.status_code == 202, edited.text
        if not complete:
            exported = await client.post(f"/api/projects/{project['id']}/export")
            assert exported.status_code == 201, exported.text
            content = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
            return source.json(), edited.json(), content.content
    deadline = asyncio.get_running_loop().time() + 10
    while asyncio.get_running_loop().time() < deadline:
        plans = (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json()
        if plans and all(plan["status"] == "complete" for plan in plans):
            break
        await asyncio.sleep(0.03)
    else:
        raise AssertionError("Constructed text plans did not finish")
    exported = await client.post(f"/api/projects/{project['id']}/export")
    assert exported.status_code == 201, exported.text
    content = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
    return source.json(), edited.json(), content.content


def _manifest(content: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return json.loads(archive.read("manifest.json"))


@pytest.mark.parametrize("output_count", [1, 3])
@pytest.mark.parametrize("complete", [True, False], ids=["completed", "interrupted"])
async def test_project_round_trip_preserves_edited_branch_graph_without_queue_authority(
    app: FastAPI, client: AsyncClient, complete: bool, output_count: int
) -> None:
    source, edited, content = await _archive(
        app, client, complete=complete, output_count=output_count
    )
    manifest = _manifest(content)
    plans = manifest["work_plans"]
    assert len(plans) == 2
    edited_plan = next(plan for plan in plans if plan["id"] == edited["work_plan_id"])
    assert edited_plan["summary_json"]["branch_head_message_id"] == edited["branch_head_message_id"]
    with SessionLocal() as session:
        jobs_before = set(session.scalars(select(Job.id)))
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": ("boats.lm-atelier.zip", content, "application/zip"),
        },
    )
    assert imported.status_code == 201, imported.text
    with SessionLocal() as session:
        chat = session.scalar(select(Chat).where(Chat.project_id == imported.json()["id"]))
        assert chat is not None
        chat_id = chat.id
        runs = list(session.scalars(select(Run).where(Run.chat_id == chat_id)))
        source_run = next(
            run
            for run in runs
            if run.provenance_json["imported_from_run_id"] == source["run"]["id"]
        )
        edited_run = next(
            run
            for run in runs
            if run.provenance_json["imported_from_run_id"] == edited["run"]["id"]
        )
        assert edited_run.work_plan_id and edited_run.work_step_id
        assert edited_run.work_plan_id != edited["work_plan_id"]
        plan = session.get(WorkPlan, edited_run.work_plan_id)
        assert plan is not None and plan.idempotency_key is None
        assert plan.steps[0].run_id == edited_run.id
        assert len(plan.steps) == output_count
        assert plan.summary_json["edit_source"]["source_message_id"] == source_run.user_message_id
        assert plan.summary_json["edit_source"]["source_run_id"] == source_run.id
        final_run = session.get(Run, plan.steps[-1].run_id)
        assert final_run is not None
        assert plan.summary_json["branch_head_message_id"] == final_run.assistant_message_id
        assert edited_run.provenance_json["edit_source"]["source_run_id"] == source_run.id
        assert chat.active_head_message_id == source_run.assistant_message_id
        assert plan.status == ("complete" if complete else "failed")
        assert set(session.scalars(select(Job.id))) == jobs_before
    discovered = await client.get(f"/api/chats/{chat_id}/edited-branches")
    assert discovered.status_code == 200, discovered.text
    assert len(discovered.json()["items"]) == 1
    branch = discovered.json()["items"][0]
    assert branch["source_available"] is True
    assert branch["can_continue"] is complete
    assert branch["jobs"] == []


@pytest.mark.parametrize(
    "corruption",
    ["duplicate_plan", "wrong_run", "missing_step", "wrong_head", "cycle", "wrong_source"],
)
async def test_project_import_rejects_invalid_work_plan_graph_before_mutation(
    app: FastAPI, client: AsyncClient, corruption: str
) -> None:
    source, edited, content = await _archive(app, client, complete=True)
    manifest = _manifest(content)
    plans = manifest["work_plans"]
    plan = next(item for item in plans if item["id"] == edited["work_plan_id"])
    if corruption == "duplicate_plan":
        plans.append(plan)
    elif corruption == "wrong_run":
        plan["steps"][0]["run_id"] = source["run"]["id"]
    elif corruption == "missing_step":
        plan["steps"] = []
    elif corruption == "cycle":
        first, second = (item["steps"][0]["id"] for item in plans)
        manifest["work_step_dependencies"] = [
            {"step_id": first, "depends_on_step_id": second},
            {"step_id": second, "depends_on_step_id": first},
        ]
    elif corruption == "wrong_source":
        plan["summary_json"]["edit_source"]["source_run_id"] = edited["run"]["id"]
    else:
        plan["summary_json"]["branch_head_message_id"] = source["assistant_message"]["id"]
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(content)) as original, zipfile.ZipFile(output, "w") as changed:
        for entry in original.infolist():
            changed.writestr(
                entry,
                json.dumps(manifest) if entry.filename == "manifest.json" else original.read(entry),
            )
    with SessionLocal() as session:
        before = session.scalar(select(func.count()).select_from(Project))
    imported = await client.post(
        "/api/projects/import",
        files={
            "archive": ("invalid.lm-atelier.zip", output.getvalue(), "application/zip"),
        },
    )
    assert imported.status_code == 422, imported.text
    assert imported.json()["code"] == "project-import-invalid"
    with SessionLocal() as session:
        assert session.scalar(select(func.count()).select_from(Project)) == before
