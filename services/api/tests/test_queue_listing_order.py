"""Dispatch pages retain scheduler order and fresh move comparisons."""

from datetime import timedelta

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker
from test_queue_manual_order import ORDER, REORDER, STAMP, move_before, seed, snapshot
from test_queue_manual_order import client as client
from test_queue_manual_order import queue_app as queue_app
from test_queue_manual_order import sessions as sessions

from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler


@pytest.mark.parametrize("case", ["priority", "aging", "equal-effective", "resources"])
async def test_dispatch_rows_follow_the_scheduler_across_distinct_cohorts(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    seed(sessions)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    with sessions() as session:
        jobs = list(session.scalars(select(Job).order_by(Job.id)))
        if case == "priority":
            for job, priority in zip(jobs, [0, 2, 1], strict=True):
                job.queue_priority = priority
        elif case == "aging":
            jobs[2].enqueued_at = STAMP - timedelta(seconds=60)
        elif case == "equal-effective":
            for job, priority, age in zip(jobs, [2, 0, 1], [0, 60, 30], strict=True):
                job.queue_priority = priority
                job.enqueued_at = STAMP - timedelta(seconds=age)
        else:
            jobs[1].queue_resource = "disk"
        session.commit()
        expected = [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)]
    page = await snapshot(client)
    assert [item["owner"]["id"] for item in page["items"]] == expected


async def test_manual_order_keeps_another_resources_slots_in_the_dispatch_listing(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(sessions)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    with sessions() as session:
        middle = session.get(Job, "download-1")
        assert middle is not None
        middle.queue_resource = "disk"
        session.commit()
    page = await snapshot(client)
    identifiers = [item["owner"]["id"] for item in page["items"]]
    command = move_before(page, identifiers.index("download-2"), identifiers.index("download-0"))
    response = await client.post(REORDER, json=command)
    assert response.status_code == 200
    page = await snapshot(client)
    with sessions() as session:
        expected = [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)]
    assert expected == ["download-2", "download-1", "download-0"]
    assert [item["owner"]["id"] for item in page["items"]] == expected


async def test_aging_keeps_unchanged_page_order_readable_with_fresh_move_comparisons(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    with sessions() as session:
        for index in range(51):
            session.add(
                Job(
                    id=f"download-{index:02}",
                    kind="download",
                    status="queued",
                    created_at=STAMP,
                    enqueued_at=STAMP + timedelta(seconds=index),
                    queue_group="network",
                    queue_resource="network",
                    queue_priority=0,
                    queue_ticket=f"ticket-{index:02}",
                    payload_json={},
                )
            )
        session.commit()
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP + timedelta(seconds=60))
    response = await client.get(ORDER, params={"limit": 50})
    assert response.status_code == 200
    first = response.json()
    old_cohorts = {item["owner"]["id"]: item["cohort_id"] for item in first["items"]}
    by_id = {item["owner"]["id"]: index for index, item in enumerate(first["items"])}
    stale_move = move_before(first, by_id["download-02"], by_id["download-01"])
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP + timedelta(seconds=61))
    second = await client.get(ORDER, params={"limit": 50, "cursor": first["next_cursor"]})
    assert second.status_code == 200
    assert [item["owner"]["id"] for item in second.json()["items"]] == ["download-50"]
    refreshed = await client.get(ORDER, params={"limit": 50})
    assert refreshed.status_code == 200
    new_cohorts = {item["owner"]["id"]: item["cohort_id"] for item in refreshed.json()["items"]}
    assert new_cohorts != old_cohorts
    refused = await client.post(REORDER, json=stale_move)
    assert refused.status_code == 409 and refused.json()["code"] == "queue-order-conflict"


async def test_a_real_dispatch_order_change_invalidates_the_page_cursor(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(sessions)
    with sessions() as session:
        older = session.get(Job, "download-0")
        newer = session.get(Job, "download-1")
        assert older is not None and newer is not None
        older.enqueued_at = STAMP - timedelta(seconds=29)
        newer.queue_priority = 1
        session.commit()
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    first = await client.get(ORDER, params={"limit": 1})
    assert first.status_code == 200
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP + timedelta(seconds=2))
    second = await client.get(ORDER, params={"limit": 1, "cursor": first.json()["next_cursor"]})
    assert second.status_code == 409 and second.json()["code"] == "queue-order-conflict"
