from __future__ import annotations

from datetime import UTC, datetime

from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy.orm import Session, sessionmaker
from test_generation_queue_pause import client as client
from test_generation_queue_pause import queue_app as queue_app
from test_generation_queue_pause import sessions as sessions

from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler

POLICY = "/api/queue/lanes/transfer"


def seed_downloads(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add_all(
            Job(
                id=name,
                kind="download",
                status=status,
                queue_group="network",
                queue_resource="network_transfer",
                enqueued_at=datetime(2026, 1, 1, tzinfo=UTC),
                payload_json={"remote_id": "example/model", "revision": "a" * 40},
            )
            for name, status in (("first", "queued"), ("second", "queued"), ("manual", "paused"))
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


async def test_transfer_policy_starts_open_without_exposing_download_metadata(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_downloads(sessions)
    response = await client.get(POLICY)
    assert response.status_code == 200
    assert response.json() == {
        "lane": "transfer",
        "dispatch_state": "open",
        "revision": 0,
        "running_jobs": 0,
        "allowed_actions": ["pause_after_current"],
    }
    assert "example/model" not in response.text
    assert "a" * 40 not in response.text


async def test_paused_transfers_keep_new_work_queued_and_manual_pause_unchanged(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_downloads(sessions)
    paused = await command(client, "pause-after-current", 0, "pause-idle")
    assert paused["dispatch_state"] == "paused"
    with sessions() as session:
        session.add(Job(id="new", kind="download", status="queued", queue_group="network"))
        session.commit()
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    assert scheduler.peek_next_eligible_job("network") is None
    resumed = await command(client, "resume", 1, "resume-idle")
    assert resumed["dispatch_state"] == "open"
    assert scheduler.peek_next_eligible_job("network") is not None
    with sessions() as session:
        manual = session.get(Job, "manual")
        first = session.get(Job, "first")
        assert manual is not None and manual.status == "paused"
        assert first is not None and first.status == "queued" and first.claim_owner is None
        assert first.payload_json == {"remote_id": "example/model", "revision": "a" * 40}


async def test_transfer_drain_waits_for_all_claims_and_nested_activation(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_downloads(sessions)
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    async with scheduler.job_lease(
        "first", resource="network_transfer", group="network", capacity=2
    ):
        async with scheduler.job_lease(
            "second", resource="network_transfer", group="network", capacity=2
        ):
            paused = await command(client, "pause-after-current", 0, "drain-two")
            assert paused["dispatch_state"] == "draining"
            assert paused["running_jobs"] == 2
            # Activation remains inside the transfer claim and can take its compute lease.
            async with scheduler.lease("primary"):
                during_activation = (await client.get(POLICY)).json()
                assert during_activation["running_jobs"] == 2
                with sessions() as session:
                    second = session.get(Job, "second")
                    assert second is not None
                    second.status = "complete"
                    session.commit()
                assert (await client.get(POLICY)).json()["running_jobs"] == 2
        remaining = (await client.get(POLICY)).json()
        assert remaining["dispatch_state"] == "draining"
        assert remaining["running_jobs"] == 1
    finished = (await client.get(POLICY)).json()
    assert finished["dispatch_state"] == "paused"
    assert finished["running_jobs"] == 0
    assert finished["revision"] == 2


async def test_transfer_and_generation_commands_have_independent_retry_namespaces(
    client: AsyncClient,
) -> None:
    paused = await command(client, "pause-after-current", 0, "shared-key")
    generation = (await client.get("/api/queue/lanes/generation")).json()
    assert generation["dispatch_state"] == "open"
    assert generation["revision"] == 0
    generation_pause = await client.post(
        "/api/queue/lanes/generation/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "shared-key"},
    )
    assert generation_pause.status_code == 200
    await command(client, "resume", 1, "later-resume")
    assert await command(client, "pause-after-current", 0, "shared-key") == paused
    assert (await client.get(POLICY)).json()["dispatch_state"] == "open"
    assert (await client.get("/api/queue/lanes/generation")).json()["dispatch_state"] == "paused"


async def test_transfer_conflicting_retry_leaves_policy_unchanged(client: AsyncClient) -> None:
    paused = await command(client, "pause-after-current", 0, "same-key")
    conflict = await client.post(
        POLICY + "/resume",
        json={"expected_revision": 1, "idempotency_key": "same-key"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "queue-lane-conflict"
    assert (await client.get(POLICY)).json() == paused
