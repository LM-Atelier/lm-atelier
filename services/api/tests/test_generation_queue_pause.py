from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import event, select
from sqlalchemy.orm import Session, sessionmaker

from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.models import Job
from local_lm.scheduler import ResourceScheduler
from tests.test_queue_control import job_audit, make_queue_app, seed_plan

POLICY = "/api/queue/lanes/generation"


@pytest.fixture
def sessions(settings: Settings) -> Iterator[sessionmaker[Session]]:
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


async def command(
    client: AsyncClient, action: str, *, revision: int, key: str
) -> dict[str, object]:
    response = await client.post(
        POLICY + "/" + action,
        json={"expected_revision": revision, "idempotency_key": key},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_generation_dispatch_starts_open_with_a_content_free_projection(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions)
    response = await client.get(POLICY)
    assert response.status_code == 200
    assert response.json() == {
        "lane": "generation",
        "dispatch_state": "open",
        "revision": 0,
        "running_jobs": 0,
        "allowed_actions": ["pause_after_current"],
    }
    assert "A blue square" not in response.text
    assert "chat-a" not in response.text


@pytest.mark.asyncio
async def test_idle_pause_keeps_every_accepted_job_and_hold_unchanged(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions)
    held = await client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "existing-hold"},
    )
    assert held.status_code == 200
    before = job_audit(sessions)
    paused = await command(client, "pause-after-current", revision=0, key="pause-idle")
    assert paused == {
        "lane": "generation",
        "dispatch_state": "paused",
        "revision": 1,
        "running_jobs": 0,
        "allowed_actions": ["resume"],
    }
    assert job_audit(sessions) == before
    page = (await client.get("/api/queue/activity")).json()
    assert page["items"][0]["control_state"] == "held"
    assert page["items"][0]["control_revision"] == 1


@pytest.mark.asyncio
async def test_pause_drains_a_claim_until_release_even_after_its_job_completes(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions, claimed=True)
    result = await command(client, "pause-after-current", revision=0, key="drain-current")
    assert result["dispatch_state"] == "draining"
    assert result["running_jobs"] == 1
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = "complete"
        session.commit()
    still_claimed = (await client.get(POLICY)).json()
    assert still_claimed["dispatch_state"] == "draining"
    assert still_claimed["running_jobs"] == 1
    await queue_app.state.services.scheduler._release_job("first", "existing-worker", "primary")
    finished = (await client.get(POLICY)).json()
    assert finished["dispatch_state"] == "paused"
    assert finished["running_jobs"] == 0
    assert finished["revision"] == 2


@pytest.mark.asyncio
async def test_resume_preserves_plan_hold_and_original_acceptance_order(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions)
    held = await client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-before-pause"},
    )
    assert held.status_code == 200
    before = job_audit(sessions)
    await command(client, "pause-after-current", revision=0, key="pause-before-resume")
    reopened = await command(client, "resume", revision=1, key="resume-once")
    assert reopened["dispatch_state"] == "open"
    assert reopened["revision"] == 2
    assert reopened["allowed_actions"] == ["pause_after_current"]
    assert job_audit(sessions) == before
    page = (await client.get("/api/queue/activity")).json()
    assert page["items"][0]["control_state"] == "held"


@pytest.mark.asyncio
async def test_identical_retry_replays_the_original_response_after_state_moves_on(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions)
    original = await command(client, "pause-after-current", revision=0, key="retry-pause")
    await command(client, "resume", revision=1, key="later-resume")
    replay = await command(client, "pause-after-current", revision=0, key="retry-pause")
    assert replay == original
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "open"
    assert current["revision"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "revision", "key"),
    [("resume", 0, "stale"), ("resume", 1, "same-key"), ("pause-after-current", 1, "same-key")],
)
async def test_stale_or_reused_command_cannot_mutate_the_lane(
    client: AsyncClient, action: str, revision: int, key: str
) -> None:
    original = await command(client, "pause-after-current", revision=0, key="same-key")
    response = await client.post(
        POLICY + "/" + action,
        json={"expected_revision": revision, "idempotency_key": key},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-lane-conflict"
    assert (await client.get(POLICY)).json() == original


@pytest.mark.asyncio
async def test_a_new_app_recovers_drained_state_and_durable_retry(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions, claimed=True)
    original = await command(client, "pause-after-current", revision=0, key="restart-pause")
    assert original["dispatch_state"] == "draining"
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = "interrupted"
        job.claim_owner = None
        session.commit()
    from local_lm.generation_queue import recover_generation_queue

    # The product performs recovery during startup, never in a polled read.
    with sessions() as session:
        recover_generation_queue(session)
        session.commit()
    restarted = make_queue_app(sessions)
    async with AsyncClient(
        transport=ASGITransport(app=restarted), base_url="http://testserver"
    ) as other:
        current = (await other.get(POLICY)).json()
        assert current["dispatch_state"] == "paused"
        assert current["revision"] == 2
        assert (
            await command(other, "pause-after-current", revision=0, key="restart-pause") == original
        )


@pytest.mark.asyncio
async def test_paused_generation_does_not_claim_but_the_same_resource_install_can(
    client: AsyncClient,
    queue_app: FastAPI,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.add(
            Job(
                id="install",
                kind="registry_prepare",
                status="queued",
                queue_group="primary",
                queue_resource="media_compute",
            )
        )
        session.commit()
    await command(client, "pause-after-current", revision=0, key="pause-generation")
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    generation_started = asyncio.Event()
    attempted = asyncio.Event()
    original = scheduler._fresh_eligible_job_ids

    def observed(session: Session, group: str, now: datetime) -> tuple[str, ...]:
        result = original(session, group, now)
        attempted.set()
        return result

    monkeypatch.setattr(scheduler, "_fresh_eligible_job_ids", observed)

    async def generate() -> None:
        async with scheduler.job_lease("first", resource="media_compute", group="primary"):
            generation_started.set()

    task = asyncio.create_task(generate())
    try:
        await asyncio.wait_for(attempted.wait(), timeout=5)
        async with scheduler.job_lease("install", resource="media_compute", group="primary"):
            assert not generation_started.is_set()
            with sessions() as session:
                assert session.scalar(select(Job.status).where(Job.id == "first")) == "queued"
                assert session.scalar(select(Job.status).where(Job.id == "install")) == "running"
        assert not generation_started.is_set()
        await command(client, "resume", revision=1, key="resume-generation")
        await asyncio.wait_for(generation_started.wait(), timeout=5)
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("resume_before_claim", [False, True])
@pytest.mark.parametrize("initial_revision", [0, 2])
async def test_pause_at_the_final_claim_wins_even_if_resumed_before_the_update(
    client: AsyncClient,
    queue_app: FastAPI,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
    resume_before_claim: bool,
    initial_revision: int,
) -> None:
    seed_plan(sessions)
    assert (await client.get(POLICY)).status_code == 200
    from local_lm.generation_queue import change_generation_queue
    from local_lm.schemas import QueueControlCommand

    if initial_revision:
        await command(client, "pause-after-current", revision=0, key="establish-pause")
        await command(client, "resume", revision=1, key="establish-open")
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    intercepted = False
    passes = 0

    class OnePass(Exception):
        pass

    def one_pass(group: str) -> list[str]:
        nonlocal passes
        passes += 1
        if passes > 1:
            raise OnePass
        return []

    def before(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        many: bool,
    ) -> None:
        nonlocal intercepted
        if (
            not intercepted
            and statement.startswith("UPDATE jobs SET")
            and "claim_owner=" in statement
            and "attempt=" in statement
        ):
            intercepted = True
            with sessions() as other:
                paused = change_generation_queue(
                    other,
                    "pause_after_current",
                    QueueControlCommand(
                        expected_revision=initial_revision, idempotency_key="claim-race-pause"
                    ),
                )
                assert paused.dispatch_state == "paused"
            if resume_before_claim:
                with sessions() as other:
                    change_generation_queue(
                        other,
                        "resume",
                        QueueControlCommand(
                            expected_revision=initial_revision + 1,
                            idempotency_key="claim-race-resume",
                        ),
                    )

    monkeypatch.setattr(scheduler, "_expire_foreign_claims", one_pass)
    engine = sessions.kw["bind"]
    event.listen(engine, "before_cursor_execute", before)
    try:
        with pytest.raises(OnePass):
            await scheduler._acquire_job(
                "first",
                resource="media_compute",
                group="primary",
                priority=0,
                capacity=1,
                local_lock=asyncio.Semaphore(1),
            )
    finally:
        event.remove(engine, "before_cursor_execute", before)
    assert intercepted, "The competing command never reached the actual final claim UPDATE."
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None and job.status == "queued" and job.claim_owner is None
    current = (await client.get(POLICY)).json()
    assert current["revision"] == initial_revision + (2 if resume_before_claim else 1)


@pytest.mark.asyncio
async def test_a_real_claim_that_wins_before_pause_is_allowed_to_finish(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions)
    assert (await client.get(POLICY)).status_code == 200
    scheduler: ResourceScheduler = queue_app.state.services.scheduler
    async with scheduler.job_lease("first", resource="media_compute", group="primary") as claim:
        response = await command(client, "pause-after-current", revision=0, key="after-claim")
        assert response["dispatch_state"] == "draining"
        with sessions() as session:
            job = session.get(Job, "first")
            assert job is not None and job.claim_owner == claim.token and job.status == "running"
            job.status = "complete"
            session.commit()
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "paused"
    assert current["running_jobs"] == 0


@pytest.mark.asyncio
async def test_a_valid_hidden_verification_job_obeys_generation_pause(
    client: AsyncClient,
    settings: Settings,
    sessions: sessionmaker[Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert (await client.get(POLICY)).status_code == 200
    from local_lm.domain import utcnow
    from tests.test_queue_control import (
        finish_source_and_queue_verifier,
        prepare_verification_source,
    )

    orchestrator, run_id, artifacts = prepare_verification_source(settings, sessions, monkeypatch)
    with sessions() as session:
        child_id = finish_source_and_queue_verifier(orchestrator, session, run_id, artifacts)
        source = session.get(Job, "first")
        assert source is not None
        source.claim_owner = None
        session.commit()
        assert child_id in [
            job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())
        ]
    paused = await command(client, "pause-after-current", revision=0, key="pause-verification")
    assert paused["dispatch_state"] == "paused"
    with sessions() as session:
        assert child_id not in [
            job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())
        ]
        child = session.get(Job, child_id)
        assert child is not None and child.status == "queued" and child.claim_owner is None
    page = (await client.get("/api/queue/activity")).json()
    assert not any(item["owner_id"] == child_id for item in page["items"])
    await command(client, "resume", revision=1, key="resume-verification")
    with sessions() as session:
        assert child_id in [
            job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())
        ]


@pytest.mark.asyncio
async def test_foreign_claim_expiry_finishes_draining_without_starting_queued_work(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session]
) -> None:
    seed_plan(sessions, claimed=True)
    assert (await client.get(POLICY)).status_code == 200
    from datetime import timedelta

    from local_lm.domain import utcnow

    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.claim_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
    paused = await command(client, "pause-after-current", revision=0, key="pause-expiring")
    assert paused["dispatch_state"] == "draining"
    assert queue_app.state.services.scheduler._expire_foreign_claims("primary") == ["first"]
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "paused" and current["revision"] == 2
    with sessions() as session:
        waiting = session.get(Job, "second")
        assert waiting is not None and waiting.status == "queued" and waiting.claim_owner is None


@pytest.mark.asyncio
@pytest.mark.parametrize("lane", ["transfer", "install"])
async def test_unsupported_lane_actions_do_not_change_generation(
    client: AsyncClient,
    lane: str,
) -> None:
    baseline = await client.get(POLICY)
    assert baseline.status_code == 200
    response = await client.post(
        "/api/queue/lanes/" + lane + "/pause-after-current",
        json={"expected_revision": 0, "idempotency_key": "unsupported"},
    )
    assert response.status_code == 404
    assert (await client.get(POLICY)).json() == baseline.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_key", [True, False])
async def test_concurrent_generation_commands_have_one_durable_transition(
    client: AsyncClient,
    queue_app: FastAPI,
    sessions: sessionmaker[Session],
    same_key: bool,
) -> None:
    assert (await client.get(POLICY)).status_code == 200
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from sqlalchemy import text

    ready = Barrier(2)
    engine = sessions.kw["bind"]

    def before(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        many: bool,
    ) -> None:
        if statement == "BEGIN IMMEDIATE":
            ready.wait(timeout=5)

    def post(key: str) -> tuple[int, object]:
        async def request() -> tuple[int, object]:
            async with AsyncClient(
                transport=ASGITransport(app=queue_app), base_url="http://testserver"
            ) as other:
                response = await other.post(
                    POLICY + "/pause-after-current",
                    json={"expected_revision": 0, "idempotency_key": key},
                )
                return response.status_code, response.json()

        return asyncio.run(request())

    event.listen(engine, "before_cursor_execute", before)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(post, "concurrent-a")
            second = pool.submit(post, "concurrent-a" if same_key else "concurrent-b")
            outcomes = [first.result(timeout=10), second.result(timeout=10)]
    finally:
        event.remove(engine, "before_cursor_execute", before)
    assert sorted(status for status, _ in outcomes) == ([200, 200] if same_key else [200, 409])
    if same_key:
        assert outcomes[0][1] == outcomes[1][1]
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "paused" and current["revision"] == 1
    with sessions() as session:
        assert session.scalar(text("SELECT count(*) FROM generation_queue_receipts")) == 1


@pytest.mark.asyncio
async def test_policy_read_finishes_while_retention_holds_the_writer(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    from threading import Event, Thread

    from local_lm.artifact_library import begin_artifact_write_fence

    assert (await client.get(POLICY)).status_code == 200
    ready, release, escaped = Event(), Event(), Event()

    def hold() -> None:
        with sessions() as session:
            begin_artifact_write_fence(session)
            ready.set()
            if not release.wait(5):
                escaped.set()
            session.rollback()

    holder = Thread(target=hold)
    holder.start()
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        # This existing read is the control under the same live writer.
        assert (await client.get("/api/jobs/activity")).status_code == 200
        response = await client.get(POLICY)
        assert response.status_code == 200
        assert not escaped.is_set(), "The read waited for the writer's emergency timeout."
        assert response.json()["dispatch_state"] == "open"
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 6)
    assert not holder.is_alive()


@pytest.mark.asyncio
async def test_policy_command_waiting_for_a_writer_does_not_block_the_event_loop(
    client: AsyncClient, sessions: sessionmaker[Session]
) -> None:
    from threading import Event, Thread, get_ident

    from local_lm.artifact_library import begin_artifact_write_fence

    ready, release, attempted, escaped = Event(), Event(), Event(), Event()
    writer_id: list[int] = []

    def hold() -> None:
        writer_id.append(get_ident())
        with sessions() as session:
            begin_artifact_write_fence(session)
            ready.set()
            if not release.wait(5):
                escaped.set()
            session.rollback()

    def before(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        many: bool,
    ) -> None:
        if statement == "BEGIN IMMEDIATE" and get_ident() not in writer_id:
            attempted.set()

    holder = Thread(target=hold)
    holder.start()
    engine = sessions.kw["bind"]
    event.listen(engine, "before_cursor_execute", before)

    async def pulse() -> None:
        assert await asyncio.to_thread(attempted.wait, 5)
        release.set()

    heartbeat = asyncio.create_task(pulse())
    try:
        assert await asyncio.to_thread(ready.wait, 5)
        response = await client.post(
            POLICY + "/pause-after-current",
            json={"expected_revision": 0, "idempotency_key": "writer-contention"},
        )
        await heartbeat
        assert not escaped.is_set(), "The event loop could not release the contending writer."
        assert response.status_code == 200
    finally:
        release.set()
        await asyncio.to_thread(holder.join, 6)
        heartbeat.cancel()
        await asyncio.gather(heartbeat, return_exceptions=True)
        event.remove(engine, "before_cursor_execute", before)
    assert not holder.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["complete", "failed", "cancelled", "interrupted"])
async def test_expired_terminal_claim_releases_drain_without_rewriting_the_result(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session], status: str
) -> None:
    from datetime import timedelta

    from local_lm.domain import utcnow

    seed_plan(sessions, claimed=True)
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = status
        job.error = "An existing result"
        job.result_json = {"summary": "Recorded result"}
        job.completed_at = utcnow().replace(tzinfo=None)
        job.claim_expires_at = utcnow() - timedelta(seconds=1)
        session.commit()
        before = (job.status, job.error, job.result_json, job.completed_at, job.phase)
    paused = await command(client, "pause-after-current", revision=0, key="terminal-pause")
    assert paused["dispatch_state"] == "draining"
    assert queue_app.state.services.scheduler._expire_foreign_claims("primary") == ["first"]
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        assert (job.status, job.error, job.result_json, job.completed_at, job.phase) == before
        assert job.claim_owner is None and job.claim_expires_at is None and job.heartbeat_at is None
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "paused" and current["running_jobs"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("owner_kind", ["current", "unexpired-foreign"])
async def test_a_live_terminal_handoff_keeps_its_claim_and_draining_state(
    client: AsyncClient, queue_app: FastAPI, sessions: sessionmaker[Session], owner_kind: str
) -> None:
    from datetime import timedelta

    from local_lm.domain import utcnow

    seed_plan(sessions, claimed=True)
    scheduler = queue_app.state.services.scheduler
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = "complete"
        job.claim_owner = scheduler._owner + "_active" if owner_kind == "current" else "foreign"
        job.claim_expires_at = utcnow() + timedelta(seconds=60 if owner_kind != "current" else -1)
        session.commit()
    await command(client, "pause-after-current", revision=0, key="live-handoff")
    assert scheduler._expire_foreign_claims("primary") == []
    current = (await client.get(POLICY)).json()
    assert current["dispatch_state"] == "draining" and current["running_jobs"] == 1


@pytest.mark.asyncio
async def test_terminal_handoff_renews_its_lease_until_resource_release(
    queue_app: FastAPI, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from local_lm import scheduler as scheduler_module
    from local_lm.domain import utcnow

    seed_plan(sessions, claimed=True)
    scheduler = queue_app.state.services.scheduler
    old = utcnow() - timedelta(seconds=1)
    with sessions() as session:
        job = session.get(Job, "first")
        assert job is not None
        job.status = "complete"
        job.claim_owner = scheduler._owner + "_handoff"
        job.claim_expires_at = old
        session.commit()
        token = job.claim_owner
    updated = asyncio.Event()

    def after(
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        many: bool,
    ) -> None:
        if statement.startswith("UPDATE jobs SET") and "heartbeat_at=" in statement:
            updated.set()

    monkeypatch.setattr(scheduler_module, "_HEARTBEAT_SECONDS", 0)
    engine = sessions.kw["bind"]
    event.listen(engine, "after_cursor_execute", after)
    task = asyncio.create_task(scheduler._heartbeat("first", token))
    try:
        await asyncio.wait_for(updated.wait(), 5)
        with sessions() as session:
            job = session.get(Job, "first")
            assert job is not None and job.status == "complete" and job.claim_owner == token
            assert (
                job.claim_expires_at is not None
                and job.claim_expires_at.replace(tzinfo=UTC) > utcnow()
            )
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        event.remove(engine, "after_cursor_execute", after)


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_real_startup_releases_abandoned_terminal_claims_with_no_waiting_jobs(
    settings: Settings, expired: bool
) -> None:
    from datetime import timedelta

    from local_lm import db
    from local_lm.database_migrations import upgrade_database
    from local_lm.domain import utcnow
    from local_lm.main import create_app
    from local_lm.models import GenerationQueuePolicy

    app = create_app(settings)
    upgrade_database(settings)
    with db.SessionLocal() as session:
        session.add(
            Job(
                id="finished-before-stop",
                kind="image",
                status="complete",
                queue_group="primary",
                claim_owner="departed-dispatcher",
                claim_expires_at=utcnow() + timedelta(seconds=-1 if expired else 60),
                result_json={"summary": "Recorded result"},
            )
        )
        session.add(GenerationQueuePolicy(lane="generation", dispatch_state="draining", revision=1))
        session.commit()
    async with app.router.lifespan_context(app):
        with db.SessionLocal() as session:
            job = session.get(Job, "finished-before-stop")
            policy = session.get(GenerationQueuePolicy, "generation")
            assert job is not None and job.status == "complete"
            assert job.result_json == {"summary": "Recorded result"}
            assert job.claim_owner is None and job.claim_expires_at is None
            assert policy is not None and policy.dispatch_state == "paused" and policy.revision == 2
