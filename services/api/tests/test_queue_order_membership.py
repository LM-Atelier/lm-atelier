"""Queue moves respect installation cleanup and changes to plan eligibility."""

from __future__ import annotations

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from test_queue_manual_order import STAMP, audit, move_before, seed, seed_plans
from test_queue_manual_order import client as client
from test_queue_manual_order import queue_app as queue_app
from test_queue_manual_order import sessions as sessions

from local_lm.models import GenerationQueuePolicy, Job, QueueOrderEntry, QueueOrderReceipt
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("kind", ["activate", "registry_prepare", "workflow_install"])
async def test_installation_order_controls_the_real_scheduler_without_resuming_a_pause(
    client: AsyncClient, sessions: sessionmaker[Session], kind: str
) -> None:
    seed(sessions)
    with sessions() as session:
        for job in session.scalars(select(Job)):
            job.kind = kind
        session.add(GenerationQueuePolicy(lane="install", dispatch_state="paused", revision=3))
        session.commit()
    before = audit(sessions)
    response = await client.get("/api/queue/lanes/install/order")
    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 3
    moved = await client.post("/api/queue/lanes/install/reorder", json=move_before(page, 2, 0))
    assert moved.status_code == 200
    assert audit(sessions) == before
    with sessions() as session:
        policy = session.get(GenerationQueuePolicy, "install")
        assert policy is not None and policy.dispatch_state == "paused" and policy.revision == 4
        assert ResourceScheduler._eligible_jobs(session, "network", STAMP) == []
        policy.dispatch_state = "open"
        policy.revision += 1
        session.commit()
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "download-2",
            "download-0",
            "download-1",
        ]


@pytest.mark.parametrize("kind", ["activate", "registry_prepare", "workflow_install"])
async def test_terminal_install_cleanup_in_another_resource_prevents_every_move(
    client: AsyncClient, sessions: sessionmaker[Session], kind: str
) -> None:
    seed(sessions)
    with sessions() as session:
        for job in session.scalars(select(Job)):
            job.kind = "workflow_install"
        session.add(Job(id="cleaning-install", kind=kind, status="complete", claim_owner="live"))
        session.commit()
    response = await client.get("/api/queue/lanes/install/order")
    assert response.status_code == 200
    page = response.json()
    assert page["total"] == 3
    assert all(item["unavailable_reason"] == "lane-busy" for item in page["items"])
    moved = await client.post("/api/queue/lanes/install/reorder", json=move_before(page, 2, 0))
    assert moved.status_code == 409 and moved.json()["code"] == "queue-order-conflict"
    with sessions() as session:
        assert not list(session.scalars(select(QueueOrderEntry)))
        assert not list(session.scalars(select(QueueOrderReceipt)))


@pytest.mark.parametrize("release", [False, True])
async def test_hold_and_release_invalidate_an_earlier_order_snapshot(
    client: AsyncClient, sessions: sessionmaker[Session], release: bool
) -> None:
    seed_plans(sessions)
    response = await client.get("/api/queue/lanes/generation/order")
    assert response.status_code == 200
    page = response.json()
    held = await client.post(
        "/api/queue/items/plan-1/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-before-move"},
    )
    assert held.status_code == 200
    if release:
        released = await client.post(
            "/api/queue/items/plan-1/release",
            json={"expected_revision": 1, "idempotency_key": "release-before-move"},
        )
        assert released.status_code == 200
    moved = await client.post("/api/queue/lanes/generation/reorder", json=move_before(page, 1, 0))
    assert moved.status_code == 409 and moved.json()["code"] == "queue-order-conflict"
    with sessions() as session:
        assert not list(session.scalars(select(QueueOrderEntry)))
        assert not list(session.scalars(select(QueueOrderReceipt)))
        assert session.get(GenerationQueuePolicy, "generation") is None
