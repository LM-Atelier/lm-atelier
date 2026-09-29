"""Current queue reads and exact retries remain independent of old ordering records."""

import sqlite3

import pytest
from httpx2 import AsyncClient
from sqlalchemy import insert, select
from sqlalchemy.orm import Session, sessionmaker
from test_queue_manual_order import REORDER, STAMP, move_before, seed, snapshot
from test_queue_manual_order import client as client
from test_queue_manual_order import queue_app as queue_app
from test_queue_manual_order import sessions as sessions

from local_lm.models import Job, QueueOrderEntry
from local_lm.queue_order import read_queue_order
from local_lm.scheduler import ResourceScheduler


def _read_steps(sessions: sessionmaker[Session], surface: str) -> tuple[list[str], int]:
    steps = 0

    def progress() -> int:
        nonlocal steps
        steps += 1
        return 0

    with sessions() as session:
        connection = session.connection().connection.driver_connection
        assert isinstance(connection, sqlite3.Connection)
        connection.set_progress_handler(progress, 100)
        try:
            if surface == "listing":
                identifiers = [
                    item.owner.id for item in read_queue_order(session, "transfer").items
                ]
            else:
                identifiers = [
                    job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)
                ]
        finally:
            connection.set_progress_handler(None, 0)
    return identifiers, steps


@pytest.mark.parametrize("surface", ["listing", "scheduler"])
async def test_current_queue_reads_do_not_scan_retained_ordering_history(
    client: AsyncClient, sessions: sessionmaker[Session], surface: str
) -> None:
    seed(sessions)
    response = await client.post(REORDER, json=move_before(await snapshot(client), 2, 0))
    assert response.status_code == 200
    before, baseline_steps = _read_steps(sessions, surface)
    assert before == ["download-2", "download-0", "download-1"]
    with sessions() as session:
        session.execute(
            insert(QueueOrderEntry),
            [
                {
                    "lane": "transfer",
                    "owner_type": "job",
                    "owner_id": f"completed-{index:05}",
                    "queue_group": "network",
                    "queue_resource": "network",
                    "priority": 0,
                    "position": index,
                }
                for index in range(20_000)
            ],
        )
        session.commit()
    after, expanded_steps = _read_steps(sessions, surface)
    assert after == before
    assert expanded_steps <= baseline_steps + 100, (baseline_steps, expanded_steps)


async def test_completed_work_keeps_the_exact_move_receipt_and_retry_position(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    command = move_before(await snapshot(client), 2, 0)
    first = await client.post(REORDER, json=command)
    assert first.status_code == 200
    with sessions() as session:
        for job in session.scalars(select(Job)):
            job.status = "complete"
        session.commit()
    assert (await snapshot(client))["items"] == []
    replay = await client.post(REORDER, json=command)
    assert replay.status_code == 200 and replay.json() == first.json()
    changed = {**command, "before": {"type": "job", "id": "download-1"}}
    refused = await client.post(REORDER, json=changed)
    assert refused.status_code == 409
    with sessions() as session:
        for job in session.scalars(select(Job)):
            job.status = "queued"
        session.commit()
    assert [item["owner"]["id"] for item in (await snapshot(client))["items"]] == [
        "download-2",
        "download-0",
        "download-1",
    ]
