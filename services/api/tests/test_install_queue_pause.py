from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session, sessionmaker
from test_generation_queue_pause import client as client
from test_generation_queue_pause import queue_app as queue_app
from test_generation_queue_pause import sessions as sessions

from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler

POLICY = "/api/queue/lanes/install"
KINDS = ("activate", "registry_prepare", "workflow_install")
pytestmark = pytest.mark.asyncio


def seed_installs(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add_all(
            Job(
                id=kind,
                kind=kind,
                status="queued",
                queue_group="primary",
                queue_resource="media_compute",
                enqueued_at=datetime(2026, 1, 1, tzinfo=UTC),
                payload_json={
                    "display_label": "Neutral installation",
                    "fixture_token": "private-value",
                },
            )
            for kind in KINDS
        )
        session.commit()


async def command(client: AsyncClient, action: str, revision: int, key: str) -> dict[str, object]:
    response = await client.post(
        POLICY + "/" + action,
        json={"expected_revision": revision, "idempotency_key": key},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert isinstance(payload, dict)
    return payload


async def test_install_policy_is_open_and_does_not_expose_job_payloads(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_installs(sessions)
    response = await client.get(POLICY)
    assert response.status_code == 200
    assert response.json() == {
        "lane": "install",
        "dispatch_state": "open",
        "revision": 0,
        "running_jobs": 0,
        "allowed_actions": ["pause_after_current"],
    }
    assert "private-value" not in response.text
    assert "Neutral installation" not in response.text


async def test_install_pause_keeps_every_install_kind_queued_without_pausing_generation(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_installs(sessions)
    paused = await command(client, "pause-after-current", 0, "pause-installs")
    assert paused["dispatch_state"] == "paused"
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    assert scheduler.peek_next_eligible_job("primary") is None
    with sessions() as session:
        session.add(Job(id="generation", kind="image", status="queued", queue_group="primary"))
        session.add(
            Job(
                id="new-install",
                kind="activate",
                status="queued",
                queue_group="primary",
            )
        )
        session.commit()
    assert scheduler.peek_next_eligible_job("primary") == ("generation", None)
    with sessions() as session:
        for name in (*KINDS, "new-install"):
            job = session.get(Job, name)
            assert job is not None and job.status == "queued" and job.claim_owner is None
    resumed = await command(client, "resume", 1, "resume-installs")
    assert resumed["dispatch_state"] == "open"
    next_job = scheduler.peek_next_eligible_job("primary")
    assert next_job is not None and next_job[0] in KINDS


@pytest.mark.parametrize("kind", KINDS)
async def test_install_drain_keeps_the_claim_through_terminal_cleanup(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session], kind: str
) -> None:
    with sessions() as session:
        session.add(Job(id="current", kind=kind, status="queued", queue_group="primary"))
        session.commit()
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    async with scheduler.job_lease("current", resource="media_compute", group="primary"):
        draining = await command(client, "pause-after-current", 0, "drain-install")
        assert draining["dispatch_state"] == "draining"
        assert draining["running_jobs"] == 1
        with sessions() as session:
            current = session.get(Job, "current")
            assert current is not None and current.claim_owner is not None
            current.status = "complete"
            session.commit()
        assert (await client.get(POLICY)).json()["running_jobs"] == 1
    settled = (await client.get(POLICY)).json()
    assert settled["dispatch_state"] == "paused"
    assert settled["running_jobs"] == 0
    assert settled["revision"] == 2


async def test_install_command_replay_is_independent_of_other_lane_commands(
    client: AsyncClient,
) -> None:
    original = await command(client, "pause-after-current", 0, "shared-command")
    for lane in ("generation", "transfer"):
        assert (await client.get(f"/api/queue/lanes/{lane}")).json()["dispatch_state"] == "open"
        response = await client.post(
            f"/api/queue/lanes/{lane}/pause-after-current",
            json={"expected_revision": 0, "idempotency_key": "shared-command"},
        )
        assert response.status_code == 200
    await command(client, "resume", 1, "resume-install")
    assert await command(client, "pause-after-current", 0, "shared-command") == original
    assert (await client.get(POLICY)).json()["dispatch_state"] == "open"
    conflict = await client.post(
        POLICY + "/resume",
        json={"expected_revision": 1, "idempotency_key": "shared-command"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "queue-lane-conflict"
