from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import Job, Run, RunContextSnapshot
from local_lm.schemas import TurnRequest

POLICY = "/api/queue/lanes/generation"


@pytest.mark.asyncio
@pytest.mark.parametrize("hold_after_acceptance", [False, True])
@pytest.mark.parametrize("freeze_context", [False, True])
async def test_accepts_turns_while_paused_and_resumes_the_same_snapshot(
    app: FastAPI,
    client: AsyncClient,
    hold_after_acceptance: bool,
    freeze_context: bool,
) -> None:
    paused = await client.post(
        POLICY + "/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "pause-before-acceptance"},
    )
    assert paused.status_code == 200 and paused.json()["dispatch_state"] == "paused"
    chat = (await client.post("/api/chats", json={"title": "Queued example"})).json()
    payload = {"text": "A blue square", "mode": "text"}
    if freeze_context:
        with SessionLocal() as session:
            result = await app.state.services.orchestrator.create_turn(
                session, chat["id"], TurnRequest.model_validate(payload), freeze_context=True
            )
            accepted = result.model_dump(mode="json")
    else:
        response = await client.post("/api/chats/" + chat["id"] + "/turns", json=payload)
        assert response.status_code == 202, response.text
        accepted = response.json()
    run_id = accepted["run"]["id"]
    plan_id = accepted["run"]["work_plan_id"]

    async def phase(expected: str) -> None:
        while True:
            with SessionLocal() as session:
                job = session.scalar(select(Job).where(Job.run_id == run_id))
                assert job is not None and job.status == "queued" and job.claim_owner is None
                if job.phase == expected:
                    return
            await asyncio.sleep(0.03)

    await asyncio.wait_for(phase("generation paused"), timeout=5)
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        accepted_settings = deepcopy(run.settings_json)
        snapshot = session.get(RunContextSnapshot, run_id)
        assert (snapshot is not None) == freeze_context
        accepted_digest = snapshot.sha256 if snapshot is not None else None
    if hold_after_acceptance:
        held = await client.post(
            "/api/queue/items/" + plan_id + "/hold",
            json={"expected_revision": 0, "idempotency_key": "hold-accepted"},
        )
        assert held.status_code == 200
    resumed = await client.post(
        POLICY + "/resume",
        json={"expected_revision": 1, "idempotency_key": "resume-accepted"},
    )
    assert resumed.status_code == 200 and resumed.json()["dispatch_state"] == "open"
    if hold_after_acceptance:
        await asyncio.wait_for(phase("held"), timeout=5)
        released = await client.post(
            "/api/queue/items/" + plan_id + "/release",
            json={"expected_revision": 1, "idempotency_key": "release-accepted"},
        )
        assert released.status_code == 200

    async def read() -> dict[str, Any]:
        current = await client.get("/api/runs/" + run_id)
        assert current.status_code == 200
        return current.json()

    finished = await wait_for_terminal_status(read, what="the resumed constructed generation")
    assert finished["id"] == run_id and finished["work_plan_id"] == plan_id
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None and run.settings_json == accepted_settings
        snapshot = session.get(RunContextSnapshot, run_id)
        if freeze_context:
            assert snapshot is not None and snapshot.sha256 == accepted_digest
