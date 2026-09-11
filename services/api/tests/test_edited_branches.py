from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Chat, Job, Message, Run, WorkPlan


async def _accepted_pair(
    client: AsyncClient,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    chat = (await client.post("/api/chats", json={"title": "Edited versions"})).json()
    source = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "A blue paper boat", "mode": "text"}
    )
    assert source.status_code == 202, source.text
    edit = await client.post(
        f"/api/messages/{source.json()['user_message']['id']}/edits",
        json={"text": "A green paper boat", "idempotency_key": "branch-discovery"},
    )
    assert edit.status_code == 202, edit.text
    return chat, source.json(), edit.json()


def _complete(plan_id: str) -> None:
    with SessionLocal() as session:
        plan = session.get(WorkPlan, plan_id)
        assert plan is not None
        plan.status = "complete"
        for step in plan.steps:
            step.status = "complete"
        for run in session.scalars(select(Run).where(Run.work_plan_id == plan_id)):
            run.status = "complete"
            for message_id in (run.user_message_id, run.assistant_message_id):
                message = session.get(Message, message_id)
                assert message is not None
                message.status = "complete"
        for job in session.scalars(select(Job).where(Job.work_plan_id == plan_id)):
            job.status = "complete"
        session.commit()


async def test_edited_branch_discovery_survives_paging_and_preserves_active_head(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This projection fixture owns progress; background jobs would rewrite it.
    monkeypatch.setattr(app.state.services.orchestrator, "start", lambda *_args: None)
    async with app.state.services.scheduler.lease("primary"):
        chat, source, first = await _accepted_pair(client)
        second = await client.post(
            f"/api/messages/{source['user_message']['id']}/edits",
            json={"text": "A red paper boat", "idempotency_key": "second-branch"},
        )
        assert second.status_code == 202, second.text
        second_id = second.json()["work_plan_id"]
        # Force equal timestamps: the ID tiebreaker must paginate without loss.
        with SessionLocal() as session:
            for plan_id in (first["work_plan_id"], second_id):
                plan = session.get(WorkPlan, plan_id)
                assert plan is not None
                plan.created_at = datetime(2026, 1, 1, tzinfo=UTC)
            job = session.scalar(select(Job).where(Job.work_plan_id == first["work_plan_id"]))
            assert job is not None
            job.progress_json = {"queue_position": 3, "queue_resource": "primary"}
            job_id = job.id
            session.commit()
        page = await client.get(f"/api/chats/{chat['id']}/edited-branches?limit=1")
        assert page.status_code == 200, page.text
        assert len(page.json()["items"]) == 1
        cursor = page.json()["next_cursor"]
        assert cursor
        next_page = await client.get(
            f"/api/chats/{chat['id']}/edited-branches", params={"limit": 1, "cursor": cursor}
        )
        assert next_page.status_code == 200, next_page.text
        assert next_page.json()["next_cursor"] is None
        items = page.json()["items"] + next_page.json()["items"]
        assert {item["plan"]["id"] for item in items} == {first["work_plan_id"], second_id}
        entry = next(item for item in items if item["plan"]["id"] == first["work_plan_id"])
        assert entry["source_message_id"] == source["user_message"]["id"]
        assert entry["source_run_id"] == source["run"]["id"]
        assert entry["branch_head_message_id"] == first["branch_head_message_id"]
        assert entry["can_continue"] is False
        assert [job["id"] for job in entry["jobs"]] == [job_id]
        assert entry["jobs"][0]["progress_json"]["queue_position"] == 3
        with SessionLocal() as session:
            plan = session.get(WorkPlan, first["work_plan_id"])
            assert plan is not None
            assert (
                plan.summary_json["edit_source"]["source_message_id"]
                == source["user_message"]["id"]
            )
            assert plan.summary_json["branch_head_message_id"] == first["branch_head_message_id"]
            stored = session.get(Chat, chat["id"])
            assert stored is not None
            assert stored.active_head_message_id == source["assistant_message"]["id"]
        other = (await client.post("/api/chats", json={"title": "Unrelated"})).json()
        wrong_cursor = await client.get(
            f"/api/chats/{other['id']}/edited-branches", params={"cursor": cursor}
        )
        assert wrong_cursor.status_code == 404


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "queued",
        "failed",
        "cancelled",
        "removed",
        "hidden",
        "wrong_head",
        "stale_head",
        "wrong_chat",
    ],
)
async def test_continue_edited_branch_validates_completion_and_active_head(
    app: FastAPI, client: AsyncClient, case: str
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        chat, source, edited = await _accepted_pair(client)
        if case != "queued":
            _complete(edited["work_plan_id"])
        with SessionLocal() as session:
            plan = session.get(WorkPlan, edited["work_plan_id"])
            head = session.get(Message, edited["branch_head_message_id"])
            assert plan is not None and head is not None
            if case in {"failed", "cancelled"}:
                plan.steps[0].status = case
            elif case == "removed":
                head.parts.clear()
                session.flush()
                head.content_removed_at = datetime.now(UTC)
            elif case == "hidden":
                head.transcript_visible = False
            elif case == "wrong_head":
                plan.summary_json = {
                    **plan.summary_json,
                    "branch_head_message_id": source["assistant_message"]["id"],
                }
            jobs_before = [
                (job.id, job.status, job.queue_ticket)
                for job in session.scalars(select(Job).order_by(Job.id))
            ]
            session.commit()
        target = chat["id"]
        if case == "wrong_chat":
            target = (await client.post("/api/chats", json={"title": "Another"})).json()["id"]
        payload = {
            "expected_active_head_message_id": "stale-head"
            if case == "stale_head"
            else source["assistant_message"]["id"]
        }
        response = await client.post(
            f"/api/chats/{target}/edited-branches/{edited['work_plan_id']}/activate", json=payload
        )
        assert response.status_code == (
            200 if case == "complete" else 404 if case == "wrong_chat" else 409
        ), response.text
        if case == "complete":
            assert response.json()["active_head_message_id"] == edited["branch_head_message_id"]
            replay = await client.post(
                f"/api/chats/{target}/edited-branches/{edited['work_plan_id']}/activate",
                json=payload,
            )
            assert replay.status_code == 200, replay.text
            assert replay.json() == response.json()
        with SessionLocal() as session:
            stored = session.get(Chat, chat["id"])
            assert stored is not None
            assert stored.active_head_message_id == (
                edited["branch_head_message_id"]
                if case == "complete"
                else source["assistant_message"]["id"]
            )
            assert jobs_before == [
                (job.id, job.status, job.queue_ticket)
                for job in session.scalars(select(Job).order_by(Job.id))
            ]


async def test_continue_edited_branch_rechecks_head_after_waiting_for_graph_guard(
    app: FastAPI, client: AsyncClient
) -> None:
    async with app.state.services.scheduler.lease("primary"):
        chat, source, edited = await _accepted_pair(client)
        _complete(edited["work_plan_id"])
        async with app.state.services.orchestrator.chat_guard(chat["id"]):
            pending = asyncio.create_task(
                client.post(
                    f"/api/chats/{chat['id']}/edited-branches/{edited['work_plan_id']}/activate",
                    json={"expected_active_head_message_id": source["assistant_message"]["id"]},
                )
            )
            await asyncio.sleep(0.05)
            assert not pending.done(), "Activation bypassed the graph guard"
            with SessionLocal() as session:
                stored = session.get(Chat, chat["id"])
                assert stored is not None
                stored.active_head_message_id = source["user_message"]["id"]
                session.commit()
        response = await pending
        assert response.status_code == 409, response.text
        with SessionLocal() as session:
            stored = session.get(Chat, chat["id"])
            assert stored is not None
            assert stored.active_head_message_id == source["user_message"]["id"]
