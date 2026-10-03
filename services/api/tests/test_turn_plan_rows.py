"""What both turn builders write around a plan: the chat they name and the runs they queue."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from sqlalchemy.orm import Session

from local_lm.db import SessionLocal
from local_lm.models import Job, Run, WorkStep
from local_lm.orchestrator import ConversationOrchestrator

ORDERED = {
    "text": (
        "Write a short story about a paper boat,\nthen create an image based on it, "
        "then animate the image into a video, then summarize the video"
    ),
    "mode": "auto",
    "confirm_media": True,
}


async def _chat(client: AsyncClient, title: str) -> str:
    return cast(str, (await client.post("/api/chats", json={"title": title})).json()["id"])


async def _turn(client: AsyncClient, chat_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(f"/api/chats/{chat_id}/turns", json=payload)
    assert response.status_code == 202, response.text
    plan_id = response.json()["run"]["work_plan_id"]
    plan = cast(dict[str, Any], (await client.get(f"/api/work-plans/{plan_id}")).json())
    for run_id in plan["summary_json"]["run_ids"]:

        async def read(run_id: str = run_id) -> dict[str, Any]:
            return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

        await wait_for_terminal_status(read, what=f"run {run_id}")
    return plan


async def _title(client: AsyncClient, chat_id: str) -> str:
    return cast(str, (await client.get(f"/api/chats/{chat_id}")).json()["title"])


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        (
            {"text": "Plan a small\ngarden by the fence", "mode": "text"},
            "Plan a small garden by the fence",
        ),
        (ORDERED, ORDERED["text"].replace("\n", " ")[:72]),
    ],
    ids=["one-step", "ordered"],
)
async def test_a_turn_names_a_new_chat_after_its_words_and_keeps_a_chosen_name(
    client: AsyncClient, payload: dict[str, Any], named: str
) -> None:
    fresh = await _chat(client, "New chat")
    chosen = await _chat(client, "Garden notes")

    await _turn(client, fresh, payload)
    await _turn(client, chosen, payload)

    assert await _title(client, fresh) == named
    assert await _title(client, chosen) == "Garden notes"


@pytest.mark.parametrize(
    "payload",
    [{"text": "Describe a paper boat", "mode": "text"}, ORDERED],
    ids=["one-step", "ordered"],
)
async def test_a_turn_s_plan_names_its_steps_runs_and_jobs_in_order(
    client: AsyncClient, payload: dict[str, Any]
) -> None:
    plan = await _turn(client, await _chat(client, "Plan rows"), payload)

    with SessionLocal() as session:
        steps = session.scalars(
            select(WorkStep).where(WorkStep.plan_id == plan["id"]).order_by(WorkStep.ordinal)
        ).all()
        jobs = [
            session.scalars(select(Job.id).where(Job.work_step_id == step.id)).one()
            for step in steps
        ]
    assert plan["summary_json"]["step_ids"] == [step.id for step in steps]
    assert plan["summary_json"]["run_ids"] == [step.run_id for step in steps]
    # The last job too, which nothing has flushed when the summary is written.
    assert plan["summary_json"]["job_ids"] == jobs


def test_a_run_whose_workflow_is_not_its_step_s_is_never_queued() -> None:
    step = WorkStep(workflow_revision_id="revision-a")
    run = Run(
        workflow_revision_id="revision-b",
        provenance_json={"workflow": {"revision_id": "revision-b"}},
    )

    session = Mock(spec=Session)

    with pytest.raises(RuntimeError, match="identity is inconsistent"):
        ConversationOrchestrator._add_queued_run(session, step, run)

    session.add.assert_not_called()
    session.flush.assert_not_called()
    assert step.run_id is None
