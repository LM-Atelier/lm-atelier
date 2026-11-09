"""Cancelling all of a chat's work at once, which deleting a busy chat starts with."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_until
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.domain import JobStatus
from local_lm.models import Chat, Job, WorkPlan
from local_lm.orchestrator import ConversationOrchestrator


@pytest.fixture
def work_never_finishes(monkeypatch: pytest.MonkeyPatch) -> tuple[set[str], set[str]]:
    """Every job starts and then waits, as long generations do; returns started and finished."""

    started: set[str] = set()
    finished: set[str] = set()

    async def wait_forever(
        _orchestrator: ConversationOrchestrator, job_id: str, _run_id: str
    ) -> None:
        started.add(job_id)
        try:
            await asyncio.Event().wait()
        finally:
            finished.add(job_id)

    monkeypatch.setattr(ConversationOrchestrator, "_execute", wait_forever)
    return started, finished


async def _busy_chat(client: AsyncClient, title: str, turns: int) -> tuple[str, set[str]]:
    chat = (await client.post("/api/chats", json={"title": title})).json()
    run_ids = set()
    for index in range(turns):
        accepted = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={"text": f"Neutral request {index}", "mode": "text"},
        )
        assert accepted.status_code == 202, accepted.text
        run_ids.add(accepted.json()["run"]["id"])
    return chat["id"], run_ids


def _jobs(run_ids: set[str]) -> list[Job]:
    with SessionLocal() as session:
        return list(session.scalars(select(Job).where(Job.run_id.in_(run_ids))).all())


async def _started(started: set[str], run_ids: set[str]) -> set[str]:
    async def read() -> set[str]:
        return {job.id for job in _jobs(run_ids)}

    return await wait_until(
        read,
        lambda ids: len(ids) == len(run_ids) and ids <= started,
        what="the chat's jobs starting",
        interval=0.01,
    )


async def _impact(client: AsyncClient, chat_id: str) -> dict[str, Any]:
    response = await client.get(f"/api/chats/{chat_id}/deletion-impact")
    assert response.status_code == 200, response.text
    impact: dict[str, Any] = response.json()
    return impact


async def test_all_of_a_chats_work_ends_and_the_chat_can_then_be_deleted(
    app: FastAPI,
    client: AsyncClient,
    work_never_finishes: tuple[set[str], set[str]],
) -> None:
    started, finished = work_never_finishes
    chat_id, run_ids = await _busy_chat(client, "Queued by mistake", turns=3)
    job_ids = await _started(started, run_ids)
    busy = await _impact(client, chat_id)
    assert busy["available_actions"] == []
    assert busy["conflicts"] == ["active_work"]

    response = await client.post(f"/api/chats/{chat_id}/cancel-all")

    assert response.status_code == 204, response.text
    assert finished == job_ids
    assert not (job_ids & app.state.services.orchestrator._tasks.keys())
    assert {job.status for job in _jobs(run_ids)} == {JobStatus.CANCELLED.value}
    idle = await _impact(client, chat_id)
    assert idle["counts"]["active_work"] == 0
    assert idle["available_actions"] == ["trash"]
    trashed = await client.post(
        f"/api/chats/{chat_id}/trash",
        json={
            "expected_revision": idle["revision"],
            "impact_sha256": idle["impact_sha256"],
            "operation_key": "trash-after-cancelling",
            "delete_generated_media": False,
        },
    )
    assert trashed.status_code == 200, trashed.text
    assert (await client.get(f"/api/chats/{chat_id}")).status_code == 404


async def test_pictures_queued_behind_a_busy_device_are_all_cancelled(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Too many pictures"})).json()
    async with app.state.services.scheduler.lease("primary"):
        plan_ids = set()
        for index in range(2):
            accepted = await client.post(
                f"/api/chats/{chat['id']}/turns",
                json={
                    "text": f"A plain grey square {index}",
                    "mode": "image",
                    "confirm_media": True,
                    "output_count": 3,
                },
            )
            assert accepted.status_code == 202, accepted.text
            plan_ids.add(accepted.json()["run"]["work_plan_id"])
        with SessionLocal() as session:
            queued = list(session.scalars(select(Job).where(Job.work_plan_id.in_(plan_ids))))
        assert len(queued) == 6
        assert {job.status for job in queued} == {JobStatus.QUEUED.value}
        assert (await _impact(client, chat["id"]))["conflicts"] == ["active_work"]

        response = await client.post(f"/api/chats/{chat['id']}/cancel-all")

        assert response.status_code == 204, response.text
        with SessionLocal() as session:
            jobs = list(session.scalars(select(Job).where(Job.work_plan_id.in_(plan_ids))))
            plans = list(session.scalars(select(WorkPlan).where(WorkPlan.id.in_(plan_ids))))
        assert {job.status for job in jobs} == {JobStatus.CANCELLED.value}
        assert {plan.status for plan in plans} == {JobStatus.CANCELLED.value}
        assert not ({job.id for job in jobs} & app.state.services.orchestrator._tasks.keys())
        idle = await _impact(client, chat["id"])
        assert idle["counts"]["active_work"] == 0
        assert idle["available_actions"] == ["trash"]


async def test_another_chats_work_is_left_running(
    client: AsyncClient, work_never_finishes: tuple[set[str], set[str]]
) -> None:
    started, finished = work_never_finishes
    chat_id, run_ids = await _busy_chat(client, "Cancelled", turns=1)
    other_id, other_runs = await _busy_chat(client, "Left alone", turns=2)
    await _started(started, run_ids | other_runs)

    response = await client.post(f"/api/chats/{chat_id}/cancel-all")

    assert response.status_code == 204, response.text
    assert {job.status for job in _jobs(run_ids)} == {JobStatus.CANCELLED.value}
    assert all(job.status != JobStatus.CANCELLED.value for job in _jobs(other_runs))
    assert not ({job.id for job in _jobs(other_runs)} & finished)
    other = await _impact(client, other_id)
    assert other["conflicts"] == ["active_work"]


async def test_a_chat_with_no_work_answers_the_same(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Quiet"})).json()

    response = await client.post(f"/api/chats/{chat['id']}/cancel-all")

    assert response.status_code == 204, response.text


async def test_only_a_chat_people_can_see_has_its_work_cancelled(client: AsyncClient) -> None:
    missing = await client.post("/api/chats/chat_missing/cancel-all")
    source = (await client.post("/api/chats", json={"title": "Helped"})).json()
    created = await client.post(
        "/api/prompt-helpers",
        json={"source_chat_id": source["id"], "draft_prompt": "A neutral draft"},
    )
    assert created.status_code == 201, created.text
    helper = created.json()
    hidden = await client.post(f"/api/chats/{helper['id']}/cancel-all")
    chat = (await client.post("/api/chats", json={"title": "Already deleted"})).json()
    impact = await _impact(client, chat["id"])
    trashed = await client.post(
        f"/api/chats/{chat['id']}/trash",
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": "trash-first",
            "delete_generated_media": False,
        },
    )
    assert trashed.status_code == 200, trashed.text

    deleted = await client.post(f"/api/chats/{chat['id']}/cancel-all")

    assert missing.status_code == 404
    assert hidden.status_code == 404
    assert deleted.status_code == 404
    for response in (missing, hidden, deleted):
        assert response.json()["code"] == "chat-not-found"
    with SessionLocal() as session:
        assert session.get(Chat, helper["id"]) is not None
