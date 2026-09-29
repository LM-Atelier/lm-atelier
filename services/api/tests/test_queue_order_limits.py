"""Oversized queues give an actionable refusal without changing prior move receipts."""

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session, sessionmaker
from test_queue_manual_order import ORDER, REORDER, STAMP, move_before, seed, snapshot
from test_queue_manual_order import client as client
from test_queue_manual_order import queue_app as queue_app
from test_queue_manual_order import sessions as sessions

from local_lm.models import GenerationQueuePolicy, Job, QueueOrderEntry, QueueOrderReceipt


def _add_jobs(sessions: sessionmaker[Session], count: int) -> None:
    with sessions() as session:
        session.execute(
            insert(Job),
            [
                {
                    "id": f"queued-{index:05}",
                    "kind": "download",
                    "status": "queued",
                    "created_at": STAMP,
                    "enqueued_at": STAMP,
                    "queue_group": "network",
                    "queue_resource": "network",
                    "queue_priority": 0,
                    "queue_ticket": f"ticket-{index:05}",
                    "payload_json": {},
                }
                for index in range(count)
            ],
        )
        session.commit()


@pytest.mark.parametrize("count", [10_000, 10_001])
async def test_ordering_reports_the_actual_queue_size_boundary(
    client: AsyncClient, sessions: sessionmaker[Session], count: int
) -> None:
    _add_jobs(sessions, count)
    response = await client.get(ORDER, params={"limit": 50})
    if count == 10_000:
        assert response.status_code == 200
        page = response.json()
        assert page["total"] == count and len(page["items"]) == 50
        assert page["next_cursor"] is not None
    else:
        assert response.status_code == 409
        assert response.json()["code"] == "queue-order-limit-exceeded"
        assert response.json()["maximum_jobs"] == 10_000
        assert "10,000" in response.json()["detail"]


async def test_queue_growth_refuses_a_new_move_but_preserves_an_exact_retry(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    command = move_before(await snapshot(client), 2, 0)
    first = await client.post(REORDER, json=command)
    assert first.status_code == 200
    next_command = {
        **move_before(await snapshot(client), 2, 0),
        "idempotency_key": "move-after-growth",
    }
    with sessions() as session:
        before = list(session.execute(select(QueueOrderEntry.owner_id, QueueOrderEntry.position)))
    _add_jobs(sessions, 9_998)
    refused = await client.post(REORDER, json=next_command)
    assert refused.status_code == 409
    assert refused.json()["code"] == "queue-order-limit-exceeded"
    assert refused.json()["maximum_jobs"] == 10_000
    replay = await client.post(REORDER, json=command)
    assert replay.status_code == 200 and replay.json() == first.json()
    with sessions() as session:
        assert (
            list(session.execute(select(QueueOrderEntry.owner_id, QueueOrderEntry.position)))
            == before
        )
        assert session.scalar(select(func.count()).select_from(QueueOrderReceipt)) == 1
        policy = session.get(GenerationQueuePolicy, "transfer")
        assert policy is not None and policy.revision == first.json()["revision"]
