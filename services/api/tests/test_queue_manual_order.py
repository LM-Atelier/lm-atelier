from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker
from test_queue_control import make_queue_app

from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.models import (
    Chat,
    GenerationQueuePolicy,
    Job,
    QueueOrderEntry,
    QueueOrderReceipt,
    WorkPlan,
    WorkStep,
    WorkStepDependency,
)
from local_lm.queue_order import change_queue_order, read_queue_order
from local_lm.queue_order_v1 import QueueOrderCommand
from local_lm.scheduler import ResourceScheduler

ORDER = "/api/queue/lanes/transfer/order"
REORDER = "/api/queue/lanes/transfer/reorder"
STAMP = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def sessions(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> Iterator[sessionmaker[Session]]:
    # Move between cohorts only when a test deliberately advances the clock.
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()


@pytest.fixture
def queue_app(sessions: sessionmaker[Session]) -> FastAPI:
    return make_queue_app(sessions)


@pytest_asyncio.fixture
async def client(queue_app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=queue_app), base_url="http://testserver"
    ) as value:
        yield value


def seed(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        for index in range(3):
            session.add(
                Job(
                    id=f"download-{index}",
                    kind="download",
                    status="queued",
                    created_at=STAMP,
                    enqueued_at=STAMP,
                    queue_group="network",
                    queue_resource="network",
                    queue_priority=0,
                    queue_ticket=f"ticket-{index}",
                    payload_json={"source": "neutral-private-fixture"},
                )
            )
        session.commit()


def audit(sessions: sessionmaker[Session]) -> list[tuple[object, ...]]:
    with sessions() as session:
        return [
            tuple(row)
            for row in session.execute(
                select(
                    Job.id,
                    Job.status,
                    Job.created_at,
                    Job.enqueued_at,
                    Job.queue_ticket,
                    Job.queue_priority,
                    Job.claim_owner,
                    Job.payload_json,
                ).order_by(Job.id)
            )
        ]


async def snapshot(client: AsyncClient) -> dict[str, Any]:
    response = await client.get(ORDER)
    assert response.status_code == 200, response.text
    return response.json()


def move_before(page: dict[str, Any], item_index: int, anchor_index: int) -> dict[str, Any]:
    item, anchor = page["items"][item_index], page["items"][anchor_index]
    return {
        "expected_revision": page["revision"],
        "idempotency_key": "move-download",
        "cohort_id": item["cohort_id"],
        "owner": item["owner"],
        "before": anchor["owner"],
        "after": None,
        "expected_item_neighbors": item["neighbors"],
        "expected_anchor_neighbors": anchor["neighbors"],
    }


async def test_relative_order_changes_dispatch_without_rewriting_accepted_work(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    original = audit(sessions)
    page = await snapshot(client)
    assert [item["owner"] for item in page["items"]] == [
        {"type": "job", "id": f"download-{index}"} for index in range(3)
    ]
    command = move_before(page, 2, 0)
    response = await client.post(REORDER, json=command)
    assert response.status_code == 200, response.text
    assert response.json()["revision"] > page["revision"]
    assert audit(sessions) == original
    with sessions() as session:
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "download-2",
            "download-0",
            "download-1",
        ]
    replay = await client.post(REORDER, json=command)
    assert replay.status_code == 200 and replay.json() == response.json()
    assert audit(sessions) == original


@pytest.mark.parametrize("claim_status", ["running", "complete", "failed", "cancelled"])
async def test_a_claim_keeps_the_lane_nonidle_until_cleanup_releases_it(
    client: AsyncClient, sessions: sessionmaker[Session], claim_status: str
) -> None:
    seed(sessions)
    page = await snapshot(client)
    command = move_before(page, 2, 0)
    with sessions() as session:
        session.add(
            Job(
                id="cleanup",
                kind="download",
                status=claim_status,
                queue_group="other-network-group",
                queue_resource="network",
                claim_owner="worker-cleanup",
            )
        )
        session.commit()
    original = audit(sessions)
    response = await client.post(REORDER, json=command)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "queue-order-conflict"
    assert audit(sessions) == original
    with sessions() as session:
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "download-0",
            "download-1",
            "download-2",
        ]


async def test_stale_neighbours_refuse_without_overwriting_a_completed_move(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    page = await snapshot(client)
    first = await client.post(REORDER, json=move_before(page, 2, 0))
    assert first.status_code == 200, first.text
    stale = move_before(page, 1, 0)
    stale["idempotency_key"] = "stale-neighbours"
    response = await client.post(REORDER, json=stale)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "queue-order-conflict"
    current = await snapshot(client)
    assert [item["owner"]["id"] for item in current["items"]] == [
        "download-2",
        "download-0",
        "download-1",
    ]


async def test_an_older_priority_bucket_still_dispatches_before_manual_order(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    page = await snapshot(client)
    response = await client.post(REORDER, json=move_before(page, 2, 0))
    assert response.status_code == 200, response.text
    with sessions() as session:
        first = session.get(Job, "download-0")
        assert first is not None
        first.enqueued_at = STAMP - timedelta(seconds=31)
        session.commit()
        assert ResourceScheduler._eligible_jobs(session, "network", STAMP)[0].id == first.id


@pytest.mark.parametrize("kind,priority,age", [("download", -1, 30), ("activate", 0, 1)])
async def test_manual_order_does_not_take_another_cohorts_existing_dispatch_slot(
    client: AsyncClient, sessions: sessionmaker[Session], kind: str, priority: int, age: int
) -> None:
    seed(sessions)
    with sessions() as session:
        session.add(
            Job(
                id="other-cohort",
                kind=kind,
                status="queued",
                queue_group="network",
                queue_resource="network",
                queue_priority=priority,
                created_at=STAMP - timedelta(seconds=age),
                enqueued_at=STAMP - timedelta(seconds=age),
                queue_ticket="first",
            )
        )
        session.commit()
        assert ResourceScheduler._eligible_jobs(session, "network", STAMP)[0].id == "other-cohort"
    page = await snapshot(client)
    positions = {item["owner"]["id"]: index for index, item in enumerate(page["items"])}
    response = await client.post(
        REORDER, json=move_before(page, positions["download-2"], positions["download-0"])
    )
    assert response.status_code == 200, response.text
    with sessions() as session:
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "other-cohort",
            "download-2",
            "download-0",
            "download-1",
        ]


@pytest.mark.parametrize("restore_order", [False, True])
@pytest.mark.parametrize("initial_revision", [0, 2])
async def test_reorder_wins_at_the_actual_claim_update_even_if_order_is_restored(
    client: AsyncClient,
    queue_app: FastAPI,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    restore_order: bool,
    initial_revision: int,
) -> None:
    seed(sessions)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    monkeypatch.setattr("local_lm.scheduler.utcnow", lambda: STAMP)
    if initial_revision:
        with sessions() as session:
            session.add(
                GenerationQueuePolicy(
                    lane="transfer", dispatch_state="open", revision=initial_revision
                )
            )
            session.commit()
    page = await snapshot(client)
    command = QueueOrderCommand.model_validate(move_before(page, 2, 0))
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    intercepted = False
    passes = 0

    class OnePass(Exception):
        pass

    def one_pass(_group: str) -> list[str]:
        nonlocal passes
        passes += 1
        if passes > 1:
            raise OnePass
        return []

    def before(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        nonlocal intercepted
        if (
            intercepted
            or not statement.startswith("UPDATE jobs SET")
            or "attempt=" not in statement
        ):
            return
        intercepted = True
        with sessions() as other:
            moved = change_queue_order(other, "transfer", command)
            assert moved.revision == initial_revision + 1
        if restore_order:
            with sessions() as other:
                current = read_queue_order(other, "transfer").model_dump(mode="json")
                other.rollback()
                reverse = move_before(current, 1, 0)
                reverse["idempotency_key"] = "restore-first-position"
                change_queue_order(other, "transfer", QueueOrderCommand.model_validate(reverse))

    monkeypatch.setattr(scheduler, "_expire_foreign_claims", one_pass)
    engine = sessions.kw["bind"]
    event.listen(engine, "before_cursor_execute", before)
    try:
        with pytest.raises(OnePass):
            await scheduler._acquire_job(
                "download-0",
                resource="network",
                group="network",
                priority=0,
                capacity=1,
                local_lock=asyncio.Semaphore(1),
            )
    finally:
        event.remove(engine, "before_cursor_execute", before)
    assert intercepted, "The move never raced the actual scheduler claim."
    with sessions() as session:
        job = session.get(Job, "download-0")
        assert job is not None and job.status == "queued" and job.claim_owner is None
        policy = session.get(GenerationQueuePolicy, "transfer")
        assert policy is not None and policy.revision == initial_revision + (
            2 if restore_order else 1
        )


async def test_an_actual_winning_claim_prevents_all_order_writes(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    command = move_before(await snapshot(client), 2, 0)
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    async with scheduler.job_lease("download-0", resource="network", group="network"):
        original = audit(sessions)
        response = await client.post(REORDER, json=command)
        assert response.status_code == 409
        assert response.json()["code"] == "queue-order-conflict"
        assert audit(sessions) == original
        with sessions() as session:
            assert not list(session.scalars(select(QueueOrderEntry)))
            assert not list(session.scalars(select(QueueOrderReceipt)))


async def test_order_and_exact_command_receipt_survive_a_new_database_engine(
    client: AsyncClient, sessions: sessionmaker[Session], settings: Settings
) -> None:
    seed(sessions)
    command = move_before(await snapshot(client), 2, 0)
    response = await client.post(REORDER, json=command)
    assert response.status_code == 200
    new_engine = create_database_engine(settings)
    try:
        with Session(new_engine) as session:
            order = read_queue_order(session, "transfer")
            assert [item.owner.id for item in order.items] == [
                "download-2",
                "download-0",
                "download-1",
            ]
            session.rollback()
            replay = change_queue_order(
                session, "transfer", QueueOrderCommand.model_validate(command)
            )
            assert replay.model_dump(mode="json") == response.json()
    finally:
        new_engine.dispose()


def seed_plans(sessions: sessionmaker[Session]) -> None:
    with sessions() as session:
        session.add(Chat(id="chat-plans", title="Example", scope="standard"))
        session.flush()
        for plan_index in range(2):
            plan_id = f"plan-{plan_index}"
            session.add(
                WorkPlan(
                    id=plan_id,
                    chat_id="chat-plans",
                    transcript_sequence=plan_index + 1,
                    persistence_scope="durable",
                    status="queued",
                )
            )
            session.flush()
            for index in range(2):
                step_id = f"step-{plan_index}-{index}"
                session.add(
                    WorkStep(
                        id=step_id,
                        plan_id=plan_id,
                        ordinal=index,
                        operation="text_to_image",
                        status="queued",
                    )
                )
                session.flush()
                session.add(
                    Job(
                        id=f"image-{plan_index}-{index}",
                        kind="image",
                        status="queued",
                        work_plan_id=plan_id,
                        work_step_id=step_id,
                        queue_group="compute",
                        queue_resource="media_compute",
                        enqueued_at=STAMP,
                        created_at=STAMP,
                        queue_ticket=step_id,
                    )
                )
            session.flush()
            session.add(
                WorkStepDependency(
                    step_id=f"step-{plan_index}-1", depends_on_step_id=f"step-{plan_index}-0"
                )
            )
        session.commit()


async def test_plan_order_keeps_descendants_grouped_and_dependencies_intact(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plans(sessions)
    original = audit(sessions)
    response = await client.get("/api/queue/lanes/generation/order")
    assert response.status_code == 200
    page = response.json()
    assert [item["owner"] for item in page["items"]] == [
        {"type": "work_plan", "id": "plan-0"},
        {"type": "work_plan", "id": "plan-1"},
    ]
    moved = await client.post("/api/queue/lanes/generation/reorder", json=move_before(page, 1, 0))
    assert moved.status_code == 200, moved.text
    assert audit(sessions) == original
    with sessions() as session:
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "compute", STAMP)] == [
            "image-1-0",
            "image-0-0",
        ]
        first_step = session.get(WorkStep, "step-1-0")
        first_job = session.get(Job, "image-1-0")
        assert first_step is not None and first_job is not None
        first_step.status = first_job.status = "complete"
        session.commit()
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "compute", STAMP)] == [
            "image-1-1",
            "image-0-0",
        ]
        assert list(
            session.execute(
                select(WorkStep.plan_id, WorkStep.ordinal).order_by(WorkStep.id)
            ).tuples()
        ) == [("plan-0", 0), ("plan-0", 1), ("plan-1", 0), ("plan-1", 1)]
        assert list(
            session.execute(
                select(WorkStepDependency.step_id, WorkStepDependency.depends_on_step_id).order_by(
                    WorkStepDependency.step_id
                )
            ).tuples()
        ) == [("step-0-1", "step-0-0"), ("step-1-1", "step-1-0")]


@pytest.mark.parametrize("scope", ["private", "incognito"])
async def test_private_plans_do_not_enter_the_controllable_order_projection(
    client: AsyncClient, sessions: sessionmaker[Session], scope: str
) -> None:
    with sessions() as session:
        session.add(Chat(id="private-chat", title="Neutral private title", scope=scope))
        session.flush()
        session.add(
            WorkPlan(
                id="private-plan",
                chat_id="private-chat",
                transcript_sequence=1,
                persistence_scope="durable",
                status="queued",
            )
        )
        session.flush()
        session.add(
            Job(
                id="private-job",
                kind="image",
                status="queued",
                work_plan_id="private-plan",
                queue_group="compute",
                queue_resource="media_compute",
                enqueued_at=STAMP,
                payload_json={"prompt": "Neutral private fixture"},
            )
        )
        session.commit()
    response = await client.get("/api/queue/lanes/generation/order")
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert "private" not in response.text.lower()


async def test_paged_order_keeps_neighbours_and_detects_a_changed_snapshot(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(sessions)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    first = await client.get(ORDER, params={"limit": 1})
    assert first.status_code == 200
    first_page = first.json()
    assert len(first_page["items"]) == 1 and first_page["total"] == 3
    assert first_page["items"][0]["neighbors"] == {
        "before": None,
        "after": {"type": "job", "id": "download-1"},
    }
    second = await client.get(ORDER, params={"limit": 1, "cursor": first_page["next_cursor"]})
    assert second.status_code == 200
    assert second.json()["items"][0]["owner"]["id"] == "download-1"
    page = await snapshot(client)
    assert (await client.post(REORDER, json=move_before(page, 2, 0))).status_code == 200
    stale = await client.get(ORDER, params={"limit": 1, "cursor": first_page["next_cursor"]})
    assert stale.status_code == 409
    assert stale.json()["code"] == "queue-order-conflict"


async def test_one_page_supplies_the_neighbour_comparison_for_a_move_across_its_edge(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    response = await client.get(ORDER, params={"limit": 1})
    assert response.status_code == 200
    page = response.json()
    assert len(page["items"]) == 1
    item = page["items"][0]
    command = {
        "expected_revision": page["revision"],
        "idempotency_key": "move-beyond-page",
        "cohort_id": item["cohort_id"],
        "owner": item["owner"],
        "before": None,
        "after": item["neighbors"]["after"],
        "expected_item_neighbors": item["neighbors"],
        "expected_anchor_neighbors": item["after_neighbors"],
    }
    moved = await client.post(REORDER, json=command)
    assert moved.status_code == 200, moved.text
    with sessions() as session:
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "download-1",
            "download-0",
            "download-2",
        ]


async def test_paused_lane_can_be_ordered_without_resuming_dispatch(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    with sessions() as session:
        session.add(GenerationQueuePolicy(lane="transfer", dispatch_state="paused", revision=4))
        session.commit()
    page = await snapshot(client)
    assert all(item["unavailable_reason"] is None for item in page["items"])
    moved = await client.post(REORDER, json=move_before(page, 2, 0))
    assert moved.status_code == 200, moved.text
    with sessions() as session:
        policy = session.get(GenerationQueuePolicy, "transfer")
        assert policy is not None and policy.dispatch_state == "paused" and policy.revision == 5
        assert ResourceScheduler._eligible_jobs(session, "network", STAMP) == []
        policy.dispatch_state = "open"
        policy.revision += 1
        session.commit()
        assert [job.id for job in ResourceScheduler._eligible_jobs(session, "network", STAMP)] == [
            "download-2",
            "download-0",
            "download-1",
        ]


async def test_aging_changes_the_cohort_even_without_a_lane_revision_change(
    client: AsyncClient, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed(sessions)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP)
    page = await snapshot(client)
    command = move_before(page, 2, 0)
    monkeypatch.setattr("local_lm.queue_order.utcnow", lambda: STAMP + timedelta(seconds=31))
    moved = await client.post(REORDER, json=command)
    assert moved.status_code == 409 and moved.json()["code"] == "queue-order-conflict"
    with sessions() as session:
        assert not list(session.scalars(select(QueueOrderEntry)))
        assert not list(session.scalars(select(QueueOrderReceipt)))
        assert session.get(GenerationQueuePolicy, "transfer") is None


async def test_hidden_verification_claim_blocks_the_entire_generation_lane(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plans(sessions)
    with sessions() as session:
        session.add(
            Job(id="hidden-check", kind="edit_verify", status="complete", claim_owner="cleanup")
        )
        session.commit()
    response = await client.get("/api/queue/lanes/generation/order")
    assert response.status_code == 200
    page = response.json()
    assert len(page["items"]) == 2
    assert all(item["unavailable_reason"] == "lane-busy" for item in page["items"])
    moved = await client.post("/api/queue/lanes/generation/reorder", json=move_before(page, 1, 0))
    assert moved.status_code == 409
    with sessions() as session:
        assert not list(session.scalars(select(QueueOrderEntry)))
        assert not list(session.scalars(select(QueueOrderReceipt)))


async def test_order_projection_never_loads_job_or_plan_payload_objects(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plans(sessions)

    def refuse_payload_object(_instance: object, _context: object) -> None:
        raise AssertionError("Order projection loaded a payload-bearing object")

    event.listen(Job, "load", refuse_payload_object)
    event.listen(WorkPlan, "load", refuse_payload_object)
    try:
        response = await client.get("/api/queue/lanes/generation/order")
        assert response.status_code == 200
        assert response.json()["total"] == 2
    finally:
        event.remove(Job, "load", refuse_payload_object)
        event.remove(WorkPlan, "load", refuse_payload_object)


async def test_receipt_write_failure_rolls_back_order_and_revision_together(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    command = QueueOrderCommand.model_validate(move_before(await snapshot(client), 2, 0))
    engine = sessions.kw["bind"]

    def refuse_receipt(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        if statement.lstrip().startswith("INSERT INTO queue_order_receipts"):
            raise RuntimeError("Receipt write interrupted")

    event.listen(engine, "before_cursor_execute", refuse_receipt)
    try:
        with sessions() as session, pytest.raises(RuntimeError, match="Receipt write interrupted"):
            change_queue_order(session, "transfer", command)
    finally:
        event.remove(engine, "before_cursor_execute", refuse_receipt)
    with sessions() as session:
        assert not list(session.scalars(select(QueueOrderEntry)))
        assert not list(session.scalars(select(QueueOrderReceipt)))
        assert session.get(GenerationQueuePolicy, "transfer") is None
    assert (await client.post(REORDER, json=command.model_dump(mode="json"))).status_code == 200


async def test_reusing_a_move_key_for_different_placement_refuses_without_writes(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    command = move_before(await snapshot(client), 2, 0)
    assert (await client.post(REORDER, json=command)).status_code == 200
    next_command = move_before(await snapshot(client), 2, 0)
    response = await client.post(REORDER, json=next_command)
    assert response.status_code == 409
    current = await snapshot(client)
    assert current["revision"] == 1
    assert [item["owner"]["id"] for item in current["items"]] == [
        "download-2",
        "download-0",
        "download-1",
    ]
    with sessions() as session:
        assert len(list(session.scalars(select(QueueOrderReceipt)))) == 1


@pytest.mark.parametrize("cursor", ["!", "not-base64", "e30=", "WzEsMiwzXQ==", "W10="])
async def test_malformed_or_stale_page_cursor_refuses_as_a_typed_conflict(
    client: AsyncClient, sessions: sessionmaker[Session], cursor: str
) -> None:
    seed(sessions)
    response = await client.get(ORDER, params={"cursor": cursor})
    assert response.status_code == 409
    assert response.json()["code"] == "queue-order-conflict"


async def test_unowned_generation_is_not_a_standalone_orderable_item(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    with sessions() as session:
        session.add(
            Job(
                id="unowned-generation",
                kind="image",
                status="queued",
                queue_group="compute",
                queue_resource="media_compute",
            )
        )
        session.commit()
    response = await client.get("/api/queue/lanes/generation/order")
    assert response.status_code == 200
    assert response.json()["items"] == []


async def test_one_read_snapshot_cannot_mix_a_policy_revision_with_later_membership(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    with sessions() as session:
        session.add(GenerationQueuePolicy(lane="transfer", dispatch_state="open", revision=5))
        session.commit()
    intercepted = False
    engine = sessions.kw["bind"]

    def change_membership(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        nonlocal intercepted
        if intercepted or not statement.lstrip().startswith("SELECT jobs.id"):
            return
        intercepted = True
        with sessions() as writer:
            policy = writer.get(GenerationQueuePolicy, "transfer")
            job = writer.get(Job, "download-2")
            assert policy is not None and job is not None
            policy.revision = 6
            job.status = "paused"
            writer.commit()

    event.listen(engine, "before_cursor_execute", change_membership)
    try:
        page = await snapshot(client)
    finally:
        event.remove(engine, "before_cursor_execute", change_membership)
    assert intercepted
    assert page["revision"] == 5
    assert all(item["unavailable_reason"] is None for item in page["items"])
    fresh = await snapshot(client)
    assert fresh["revision"] == 6
    paused = next(item for item in fresh["items"] if item["owner"]["id"] == "download-2")
    assert paused["unavailable_reason"] == "blocked"


async def test_order_projection_loads_only_entries_for_current_visible_owners(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed(sessions)
    assert (
        await client.post(REORDER, json=move_before(await snapshot(client), 2, 0))
    ).status_code == 200
    with sessions() as session:
        for owner_type, owner_id in [("job", "completed-download"), ("work_plan", "download-0")]:
            session.add(
                QueueOrderEntry(
                    lane="transfer",
                    owner_type=owner_type,
                    owner_id=owner_id,
                    queue_group="network",
                    queue_resource="network",
                    priority=0,
                    position=0,
                )
            )
        session.commit()
    loaded: list[tuple[str, str]] = []

    def record_entry(entry: QueueOrderEntry, _context: object) -> None:
        loaded.append((entry.owner_type, entry.owner_id))

    event.listen(QueueOrderEntry, "load", record_entry)
    try:
        page = await snapshot(client)
    finally:
        event.remove(QueueOrderEntry, "load", record_entry)
    assert [item["owner"]["id"] for item in page["items"]] == [
        "download-2",
        "download-0",
        "download-1",
    ]
    assert sorted(loaded) == [("job", f"download-{index}") for index in range(3)]
