from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy import delete, event, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from local_lm import api, db
from local_lm.api_errors import register_api_error_handler
from local_lm.artifacts import ArtifactStore
from local_lm.backups import BackupManager
from local_lm.chat_deletion import delete_exchange
from local_lm.config import Settings
from local_lm.db import Base, SessionLocal, create_database_engine
from local_lm.domain import ArtifactKind, Operation, utcnow
from local_lm.exports import ProjectExporter
from local_lm.image_edit_verification import ImageEditVerificationJobPayload
from local_lm.models import Chat, Job, Message, ModelProfile, Project, Run, WorkPlan, WorkStep
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim, ResourceScheduler

STAMP = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def sessions(settings: Settings) -> Iterator[sessionmaker[Session]]:
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    try:
        yield sessionmaker(engine, expire_on_commit=False)
    finally:
        engine.dispose()


def make_queue_app(sessions: sessionmaker[Session]) -> FastAPI:
    app = FastAPI()
    register_api_error_handler(app)

    def signing_key(purpose: bytes) -> bytes:
        assert purpose == b"user-queue-activity"
        return b"neutral-queue-control-tests"

    app.state.services = SimpleNamespace(
        scheduler=ResourceScheduler(session_factory=sessions),
        security=SimpleNamespace(local_state_signing_key=signing_key),
    )
    app.include_router(api.router)

    async def session_dependency() -> AsyncIterator[Session]:
        with sessions() as session:
            yield session

    app.dependency_overrides[api.get_conversation_session] = session_dependency
    return app


@pytest.fixture
def queue_app(sessions: sessionmaker[Session]) -> FastAPI:
    return make_queue_app(sessions)


@pytest_asyncio.fixture
async def queue_client(queue_app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=queue_app), base_url="http://testserver"
    ) as client:
        yield client


def seed_plan(sessions: sessionmaker[Session], *, claimed: bool = False) -> None:
    with sessions() as session:
        session.add(Chat(id="chat-a", title="Example", scope="standard"))
        session.flush()
        session.add(
            WorkPlan(
                id="plan-a",
                chat_id="chat-a",
                transcript_sequence=1,
                persistence_scope="durable",
                status="queued",
            )
        )
        session.flush()
        for identifier, status in [
            ("done", "complete"),
            ("first", "running" if claimed else "queued"),
            ("second", "queued"),
        ]:
            session.add(
                Job(
                    id=identifier,
                    kind="image",
                    status=status,
                    work_plan_id="plan-a",
                    enqueued_at=STAMP,
                    created_at=STAMP,
                    queue_group="primary",
                    queue_resource="primary",
                    payload_json={"prompt": "A blue square"},
                    claim_owner="existing-worker" if claimed and identifier == "first" else None,
                )
            )
        session.commit()


def job_audit(sessions: sessionmaker[Session]) -> list[tuple[object, ...]]:
    with sessions() as session:
        return [
            tuple(row)
            for row in session.execute(
                select(
                    Job.id,
                    Job.status,
                    Job.enqueued_at,
                    Job.created_at,
                    Job.claim_owner,
                    Job.payload_json,
                ).order_by(Job.id)
            )
        ]


async def test_hold_release_preserves_work_and_replays_the_original_result(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    original_jobs = job_audit(sessions)
    command = {"expected_revision": 0, "idempotency_key": "first-hold"}
    held = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert held.status_code == 200
    assert held.json()["control_state"] == "held"
    assert held.json()["control_revision"] == 1
    assert job_audit(sessions) == original_jobs

    duplicate = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert duplicate.status_code == 200 and duplicate.json() == held.json()
    released = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 1, "idempotency_key": "first-release"},
    )
    assert released.status_code == 200
    assert released.json()["control_state"] == "eligible"
    assert released.json()["control_revision"] == 2
    assert job_audit(sessions) == original_jobs

    historical = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert historical.status_code == 200 and historical.json() == held.json()


async def test_a_winning_descendant_claim_refuses_hold_without_changing_siblings(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions, claimed=True)
    original_jobs = job_audit(sessions)
    response = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "racing-hold"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-control-conflict"
    assert job_audit(sessions) == original_jobs


@contextmanager
def claimant(
    sessions: sessionmaker[Session],
    *,
    job_id: str = "first",
) -> Iterator[threading.Event]:
    entered = threading.Event()
    ready = threading.Event()
    state: dict[str, Any] = {}
    errors: list[BaseException] = []

    async def run() -> None:
        state["loop"] = asyncio.get_running_loop()
        state["task"] = asyncio.current_task()
        ready.set()
        scheduler = ResourceScheduler(session_factory=sessions)
        try:
            async with scheduler.job_lease(job_id, resource="primary", group="primary"):
                entered.set()
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=lambda: asyncio.run(run()), daemon=True)
    thread.start()
    try:
        assert ready.wait(5), "The real scheduler task did not start."
        yield entered
    finally:
        if ready.is_set():
            state["loop"].call_soon_threadsafe(state["task"].cancel)
        thread.join(5)
        assert not thread.is_alive(), "The scheduler test left a live claimant."
        assert not errors, errors


@contextmanager
def before_first_claim(
    sessions: sessionmaker[Session],
) -> Iterator[
    tuple[threading.Event, threading.Event, threading.Event, list[int], sessionmaker[Session]]
]:
    reached = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    rowcounts: list[int] = []
    engine = sessions.kw["bind"]

    def is_claim(statement: str) -> bool:
        return statement.startswith("UPDATE jobs SET") and "attempt=(jobs.attempt +" in statement

    def before(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        if is_claim(statement) and not reached.is_set():
            reached.set()
            assert release.wait(5), "The claimant's database-write barrier was not released."

    class ObservedSession(Session):
        def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
            result = super().execute(statement, *args, **kwargs)
            # Observe the same returned rowcount the production claimant uses,
            # after SQLAlchemy has consumed UPDATE RETURNING. A cursor callback
            # before that consumption reports zero even for a winning write.
            if is_claim(str(statement)) and not completed.is_set():
                rowcounts.append(result.rowcount)
                completed.set()
            return result

    observed = sessionmaker(engine, class_=ObservedSession, expire_on_commit=False)
    event.listen(engine, "before_cursor_execute", before)
    try:
        yield reached, release, completed, rowcounts, observed
    finally:
        release.set()
        event.remove(engine, "before_cursor_execute", before)


async def test_hold_wins_at_the_real_final_claim_write(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with (
        before_first_claim(sessions) as (reached, release, completed, rowcounts, observed),
        claimant(observed) as entered,
    ):
        try:
            assert await asyncio.to_thread(reached.wait, 5)
            held = await queue_client.post(
                "/api/queue/items/plan-a/hold",
                json={"expected_revision": 0, "idempotency_key": "hold-before-claim"},
            )
            assert held.status_code == 200
            release.set()
            assert await asyncio.to_thread(completed.wait, 5)
            assert rowcounts == [0]
            assert not entered.is_set()
            with sessions() as session:
                remaining = session.execute(
                    select(Job.status, Job.claim_owner).where(Job.id.in_(["first", "second"]))
                ).all()
            assert remaining == [("queued", None), ("queued", None)]
            response = await queue_client.post(
                "/api/queue/items/plan-a/release",
                json={"expected_revision": 1, "idempotency_key": "resume-after-refusal"},
            )
            assert response.status_code == 200
            assert await asyncio.to_thread(entered.wait, 5)
        finally:
            release.set()


async def test_real_claim_wins_before_the_hold_transaction(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with (
        before_first_claim(sessions) as (reached, release, completed, rowcounts, observed),
        claimant(observed) as entered,
    ):
        try:
            assert await asyncio.to_thread(reached.wait, 5)
            release.set()
            assert await asyncio.to_thread(completed.wait, 5)
            assert rowcounts == [1]
            assert await asyncio.to_thread(entered.wait, 5)
            original = job_audit(sessions)
            response = await queue_client.post(
                "/api/queue/items/plan-a/hold",
                json={"expected_revision": 0, "idempotency_key": "hold-after-claim"},
            )
            assert response.status_code == 409
            assert response.json()["code"] == "queue-control-conflict"
            assert job_audit(sessions) == original
        finally:
            release.set()


async def test_hold_and_release_invalidate_the_pre_hold_ranking_revision(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.add(
            Job(
                id="younger",
                kind="image",
                status="queued",
                queue_group="primary",
                enqueued_at=utcnow() - timedelta(seconds=90),
            )
        )
        session.commit()
    with (
        before_first_claim(sessions) as (reached, release, completed, rowcounts, observed),
        claimant(observed) as entered,
    ):
        try:
            assert await asyncio.to_thread(reached.wait, 5)
            held = await queue_client.post(
                "/api/queue/items/plan-a/hold",
                json={"expected_revision": 0, "idempotency_key": "hold-during-ranking"},
            )
            assert held.status_code == 200
            resumed = await queue_client.post(
                "/api/queue/items/plan-a/release",
                json={"expected_revision": 1, "idempotency_key": "release-during-ranking"},
            )
            assert resumed.status_code == 200
            release.set()
            assert await asyncio.to_thread(completed.wait, 5)
            assert rowcounts == [0]
            assert not entered.is_set()
            with sessions() as session:
                ordered = ResourceScheduler._eligible_jobs(session, "primary", utcnow())
                assert ordered[0].id == "younger"
                assert session.get(Job, "first").claim_owner is None
        finally:
            release.set()


async def test_reusing_a_key_for_a_changed_command_does_not_make_another_transition(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    response = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "one-command"},
    )
    assert response.status_code == 200
    for action, revision in [("release", 1), ("hold", 1)]:
        conflict = await queue_client.post(
            "/api/queue/items/plan-a/" + action,
            json={"expected_revision": revision, "idempotency_key": "one-command"},
        )
        assert conflict.status_code == 409
    replay = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "one-command"},
    )
    assert replay.json() == response.json()


async def test_a_receipt_failure_rolls_back_the_tentative_hold(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    engine = sessions.kw["bind"]

    def fail_receipt(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        if statement.startswith("INSERT INTO work_plan_control_receipts"):
            raise RuntimeError("Neutral receipt failure")

    event.listen(engine, "before_cursor_execute", fail_receipt)
    try:
        with pytest.raises(RuntimeError, match="Neutral receipt failure"):
            await queue_client.post(
                "/api/queue/items/plan-a/hold",
                json={"expected_revision": 0, "idempotency_key": "retry-rolled-back"},
            )
    finally:
        event.remove(engine, "before_cursor_execute", fail_receipt)
    # The failed transaction is tested before a later successful retry can hide it.
    with sessions() as session:
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 0
    retried = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "retry-rolled-back"},
    )
    assert retried.status_code == 200 and retried.json()["control_revision"] == 1


async def test_duplicate_concurrent_delivery_has_one_durable_result(
    sessions: sessionmaker[Session], queue_app: FastAPI
) -> None:
    seed_plan(sessions)
    command = {"expected_revision": 0, "idempotency_key": "concurrent-hold"}

    def post() -> tuple[int, dict[str, Any]]:
        async def request() -> tuple[int, dict[str, Any]]:
            async with AsyncClient(
                transport=ASGITransport(app=queue_app), base_url="http://testserver"
            ) as client:
                response = await client.post("/api/queue/items/plan-a/hold", json=command)
                return response.status_code, response.json()

        return asyncio.run(request())

    with ThreadPoolExecutor(max_workers=2) as pool:
        barrier = threading.Barrier(2)

        def together() -> tuple[int, dict[str, Any]]:
            barrier.wait(timeout=5)
            return post()

        first = pool.submit(together)
        second = pool.submit(together)
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert results[0][0] == 200 and results[1] == results[0]
    with sessions() as session:
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 1
        assert session.scalar(text("SELECT revision FROM work_plan_controls")) == 1


async def test_different_concurrent_commands_cannot_share_one_revision(
    sessions: sessionmaker[Session], queue_app: FastAPI
) -> None:
    seed_plan(sessions)
    barrier = threading.Barrier(2)

    def post(key: str) -> int:
        async def request() -> int:
            async with AsyncClient(
                transport=ASGITransport(app=queue_app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/queue/items/plan-a/hold",
                    json={"expected_revision": 0, "idempotency_key": key},
                )
                return response.status_code

        barrier.wait(timeout=5)
        return asyncio.run(request())

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(post, key) for key in ("first-command", "second-command")]
        statuses = [future.result(timeout=10) for future in futures]
    assert sorted(statuses) == [200, 409]
    with sessions() as session:
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 1
        assert session.scalar(text("SELECT revision FROM work_plan_controls")) == 1


async def test_completed_plan_replays_its_receipt_until_the_owner_is_deleted(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    command = {"expected_revision": 0, "idempotency_key": "durable-response"}
    held = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert held.status_code == 200
    with sessions() as session:
        session.execute(update(Job).values(status="complete"))
        session.execute(update(WorkPlan).where(WorkPlan.id == "plan-a").values(status="complete"))
        session.commit()
    replay = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert replay.status_code == 200 and replay.json() == held.json()
    with sessions() as session:
        session.execute(delete(Chat))
        session.commit()
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 0
    deleted = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert deleted.status_code == 404


def attach_completed_run(session: Session) -> Run:
    user = Message(chat_id="chat-a", role="user")
    session.add(user)
    session.flush()
    assistant = Message(chat_id="chat-a", parent_id=user.id, role="assistant", status="complete")
    session.add(assistant)
    session.flush()
    run = Run(
        chat_id="chat-a",
        user_message_id=user.id,
        assistant_message_id=assistant.id,
        work_plan_id="plan-a",
        operation="image_edit",
        status="complete",
    )
    session.add(run)
    session.flush()
    source = session.get(Job, "first")
    assert source is not None
    source.run_id = run.id
    source.status = "complete"
    return run


async def test_command_keys_are_scoped_to_their_plan(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.add(Chat(id="chat-b", title="Another plan"))
        session.flush()
        session.add(WorkPlan(id="plan-b", chat_id="chat-b", transcript_sequence=1))
        session.flush()
        session.add(Job(id="other-job", work_plan_id="plan-b", status="queued"))
        session.commit()
    command = {"expected_revision": 0, "idempotency_key": "shared-key"}
    first = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    second = await queue_client.post("/api/queue/items/plan-b/hold", json=command)
    assert first.status_code == second.status_code == 200
    assert first.json()["owner_id"] == "plan-a" and second.json()["owner_id"] == "plan-b"
    with sessions() as session:
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 2
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 2


async def test_identical_owner_and_key_in_another_database_make_an_independent_transition(
    sessions: sessionmaker[Session], queue_client: AsyncClient, tmp_path: Path
) -> None:
    seed_plan(sessions)
    command = {"expected_revision": 0, "idempotency_key": "same-owner-key"}
    held = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert held.status_code == 200
    released = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 1, "idempotency_key": "source-release"},
    )
    assert released.status_code == 200
    other_settings = Settings(data_dir=tmp_path / "other-database", dev=True)
    other_settings.prepare()
    engine = create_database_engine(other_settings)
    try:
        Base.metadata.create_all(engine)
        other_sessions = sessionmaker(engine, expire_on_commit=False)
        seed_plan(other_sessions)
        async with AsyncClient(
            transport=ASGITransport(app=make_queue_app(other_sessions)),
            base_url="http://testserver",
        ) as other:
            result = await other.post("/api/queue/items/plan-a/hold", json=command)
        assert result.status_code == 200 and result.json()["control_revision"] == 1
        with other_sessions() as session:
            assert session.execute(
                text("SELECT state, revision FROM work_plan_controls")
            ).one() == ("held", 1)
            assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 1
        with sessions() as session:
            assert session.execute(
                text("SELECT state, revision FROM work_plan_controls")
            ).one() == ("eligible", 2)
    finally:
        engine.dispose()


async def test_a_late_internal_descendant_obeys_hold_after_its_source_job_is_deleted(
    settings: Settings, sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-before-verification"},
    )
    assert held.status_code == 200
    with sessions() as session:
        run = attach_completed_run(session)
        store = ArtifactStore(settings)
        artifacts = []
        for color in ("blue", "green"):
            content = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(content, format="PNG")
            artifacts.append(
                store.ingest_bytes(
                    session, content.getvalue(), kind=ArtifactKind.IMAGE, media_type="image/png"
                )
            )
        session.flush()
        payload = ImageEditVerificationJobPayload(
            chat_id="chat-a",
            source_run_id=run.id,
            source_job_id="first",
            source_artifact_id=artifacts[0].id,
            result_artifact_id=artifacts[1].id,
            vision_profile_id="neutral-vision",
        )
        session.add(
            Job(
                id="verification",
                kind="edit_verify",
                status="queued",
                queue_group="primary",
                payload_json=payload.model_dump(mode="json"),
            )
        )
        session.execute(update(Job).where(Job.id == "second").values(status="complete"))
        session.commit()
        session.execute(delete(Job).where(Job.id == "first"))
        session.commit()
    with claimant(sessions, job_id="verification") as entered:

        async def wait_for_held_progress() -> None:
            while True:
                with sessions() as session:
                    job = session.get(Job, "verification")
                    assert job is not None
                    if job.phase == "held":
                        assert job.status == "queued" and job.claim_owner is None
                        return
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_held_progress(), timeout=5)
        assert not entered.is_set()
        released = await queue_client.post(
            "/api/queue/items/plan-a/release",
            json={"expected_revision": 1, "idempotency_key": "release-verification"},
        )
        assert released.status_code == 200
        assert await asyncio.to_thread(entered.wait, 5)


async def test_projection_advertises_only_supported_actions_and_reconciles_hold(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.add(Job(id="download", kind="download", status="queued"))
        session.commit()
    before = await queue_client.get("/api/queue/activity")
    assert before.status_code == 200
    items = {item["owner_id"]: item for item in before.json()["items"]}
    assert items["plan-a"]["allowed_actions"] == ["hold"]
    assert items["plan-a"]["control_revision"] == 0
    assert items["download"]["allowed_actions"] == []
    assert items["download"]["control_state"] is None
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-for-projection"},
    )
    assert held.status_code == 200
    after = await queue_client.get("/api/queue/activity")
    item = next(item for item in after.json()["items"] if item["owner_id"] == "plan-a")
    assert item["control_state"] == "held"
    assert item["status"] == "queued" and item["queued_jobs"] == 2
    assert item["control_revision"] == 1 and item["allowed_actions"] == ["release"]


async def test_malformed_internal_ownership_omits_actions_and_refuses_start(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.add(
            Job(
                id="malformed",
                kind="edit_verify",
                status="queued",
                work_plan_id="plan-a",
                queue_group="primary",
                payload_json={"source_run_id": "missing-run"},
            )
        )
        session.commit()
    response = await queue_client.get("/api/queue/activity")
    assert response.status_code == 200
    assert response.json()["items"][0]["allowed_actions"] == []
    refused = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "malformed-owner"},
    )
    assert refused.status_code == 409
    with sessions() as session:
        assert "malformed" not in [
            job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())
        ]
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0


async def test_new_descendants_age_from_their_own_later_enqueue_time(
    sessions: sessionmaker[Session], queue_client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed_plan(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-for-aging"},
    )
    assert held.status_code == 200
    from local_lm import queue_control

    released_at = STAMP + timedelta(days=1)
    monkeypatch.setattr(queue_control, "utcnow", lambda: released_at)
    response = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 1, "idempotency_key": "release-for-aging"},
    )
    assert response.status_code == 200
    with sessions() as session:
        session.add(
            Job(
                id="aaa-new",
                kind="image",
                status="queued",
                work_plan_id="plan-a",
                queue_group="primary",
                enqueued_at=released_at + timedelta(seconds=30),
            )
        )
        session.commit()
        ordered = ResourceScheduler._eligible_jobs(
            session, "primary", released_at + timedelta(seconds=60)
        )
        assert [job.id for job in ordered] == ["first", "second", "aaa-new"]
        assert session.get(Job, "first").enqueued_at == STAMP.replace(tzinfo=None)


async def test_real_exchange_deletion_removes_controls_and_receipts(
    sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "hold-before-exchange-deletion"},
    )
    assert held.status_code == 200
    with sessions() as session:
        run = attach_completed_run(session)
        user_id = run.user_message_id
        session.execute(update(Job).values(status="complete"))
        session.commit()
        result = delete_exchange(session, user_id)
        assert result.work_plan_ids == ["plan-a"]
        session.commit()
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 0


async def test_portable_project_round_trip_excludes_live_controls_and_receipts(
    settings: Settings, sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "portable-hold-key"},
    )
    assert held.status_code == 200
    settings.prepare()
    store = ArtifactStore(settings)
    exporter = ProjectExporter(settings, store)
    with sessions() as session:
        session.add(
            WorkStep(
                plan_id="plan-a",
                ordinal=1,
                operation=Operation.TEXT_TO_IMAGE.value,
                status="queued",
            )
        )
        project = Project(name="Portable queue history")
        session.add(project)
        session.flush()
        session.get(Chat, "chat-a").project_id = project.id
        session.commit()
        artifact = exporter.export(session, project.id, include_media=False)
        session.commit()
        archive = store.resolve(artifact).read_bytes()
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            manifest = json.loads(package.read("manifest.json"))
        serialized = json.dumps(manifest)
        assert "portable-hold-key" not in serialized
        assert "control_state" not in serialized and "control_revision" not in serialized
        assert "work_plan_control" not in serialized
        imported = exporter.import_archive(session, io.BytesIO(archive))
        session.commit()
        plans = session.scalars(
            select(WorkPlan).join(Chat).where(Chat.project_id == imported.id)
        ).all()
        assert len(plans) == 1 and plans[0].id != "plan-a" and plans[0].status == "failed"
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 1
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 1


async def test_database_backup_restores_a_consistent_control_and_receipt_snapshot(
    settings: Settings, sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    command = {"expected_revision": 0, "idempotency_key": "snapshot-hold"}
    held = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert held.status_code == 200
    settings.prepare()
    manager = BackupManager(settings)
    backup = manager.create(include_media=False)
    assert manager.verify(backup.name).verified
    released = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 1, "idempotency_key": "after-snapshot"},
    )
    assert released.status_code == 200
    manager.request_restore(backup.name)
    # Restore is a startup operation; close every constructed database pool first.
    sessions.kw["bind"].dispose()
    db.engine.dispose()
    assert manager.apply_pending_restore()
    with sessions() as session:
        assert session.execute(text("SELECT state, revision FROM work_plan_controls")).one() == (
            "held",
            1,
        )
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 1
        assert ResourceScheduler._eligible_jobs(session, "primary", utcnow()) == []
    replay = await queue_client.post("/api/queue/items/plan-a/hold", json=command)
    assert replay.status_code == 200 and replay.json() == held.json()


async def test_held_work_still_counts_toward_the_real_chat_admission_limit(
    app: FastAPI, client: AsyncClient
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Held admission"})).json()
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            "/api/chats/" + chat["id"] + "/turns",
            json={"text": "A blue square", "mode": "text"},
        )
        assert response.status_code == 202
        accepted = response.json()
        plan_id = accepted["run"]["work_plan_id"]
        held = await client.post(
            "/api/queue/items/" + plan_id + "/hold",
            json={"expected_revision": 0, "idempotency_key": "hold-admission"},
        )
        assert held.status_code == 200
        with SessionLocal() as session:
            session.add_all(
                [
                    Job(
                        id="held-admission-" + str(index),
                        kind="chat",
                        status="queued",
                        run_id=accepted["run"]["id"],
                        work_plan_id=plan_id,
                    )
                    for index in range(31)
                ]
            )
            session.commit()
        rejected = await client.post(
            "/api/chats/" + chat["id"] + "/turns",
            json={"text": "Another blue square", "mode": "text"},
        )
        assert rejected.status_code == 422 and "32 pending items" in rejected.json()["detail"]
        cancelled = await client.post("/api/work-plans/" + plan_id + "/cancel")
        assert cancelled.status_code == 200


async def test_a_fresh_process_observes_the_hold_before_replaying_its_receipt(
    settings: Settings, sessions: sessionmaker[Session], queue_client: AsyncClient
) -> None:
    seed_plan(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "restart-hold"},
    )
    assert held.status_code == 200
    repository = Path(__file__).resolve().parents[3]
    script = """
import asyncio, json, os
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy.orm import sessionmaker
from local_lm.config import Settings
from local_lm.db import create_database_engine
from local_lm.domain import ArtifactKind, Operation, utcnow
from local_lm.orchestrator import ConversationOrchestrator
from local_lm.scheduler import JobClaim, ResourceScheduler
from test_queue_control import make_queue_app
settings = Settings()
engine = create_database_engine(settings)
sessions = sessionmaker(engine, expire_on_commit=False)
with sessions() as session:
    before = [job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())]
async def replay():
    async with AsyncClient(transport=ASGITransport(app=make_queue_app(sessions)),
                           base_url="http://testserver") as client:
        response = await client.post("/api/queue/items/plan-a/hold",
            json={"expected_revision": 0, "idempotency_key": "restart-hold"})
        return response.status_code, response.json()
status, response = asyncio.run(replay())
engine.dispose()
print(json.dumps({"pid": os.getpid(), "before": before, "status": status, "response": response}))
"""
    environment = {
        **os.environ,
        "LOCAL_LM_DATA_DIR": str(settings.data_dir),
        "PYTHONPATH": os.pathsep.join(
            [str(repository / "services/api"), str(repository / "services/api/tests")]
        ),
    }
    result = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", script],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    value = json.loads(result.stdout)
    assert value["pid"] != os.getpid()
    assert value["before"] == []
    assert value["status"] == 200 and value["response"] == held.json()


@pytest.mark.parametrize("delete_first", [False, True])
async def test_owner_deletion_and_hold_are_ordered_by_the_actual_database_writer(
    sessions: sessionmaker[Session], queue_app: FastAPI, delete_first: bool
) -> None:
    seed_plan(sessions)
    writer_ready = threading.Event()
    contender_attempted = threading.Event()
    engine = sessions.kw["bind"]

    def before(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        if (delete_first and statement == "BEGIN IMMEDIATE") or (
            not delete_first and statement.startswith("DELETE FROM chats")
        ):
            contender_attempted.set()

    def after(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        first_write = statement.startswith(
            "DELETE FROM chats" if delete_first else "INSERT INTO work_plan_controls"
        )
        if first_write:
            writer_ready.set()
            assert contender_attempted.wait(5), "The second real database writer never arrived."

    def remove_owner() -> int:
        with sessions() as session:
            result = session.execute(delete(Chat).where(Chat.id == "chat-a"))
            session.commit()
            return result.rowcount

    def hold() -> int:
        async def post() -> int:
            async with AsyncClient(
                transport=ASGITransport(app=queue_app), base_url="http://testserver"
            ) as client:
                response = await client.post(
                    "/api/queue/items/plan-a/hold",
                    json={"expected_revision": 0, "idempotency_key": "racing-deletion"},
                )
                return response.status_code

        return asyncio.run(post())

    event.listen(engine, "before_cursor_execute", before)
    event.listen(engine, "after_cursor_execute", after)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(remove_owner if delete_first else hold)
            assert writer_ready.wait(5), "The first writer never acquired the database."
            second = pool.submit(hold if delete_first else remove_owner)
            outcomes = [first.result(timeout=10), second.result(timeout=10)]
        assert contender_attempted.is_set()
        assert outcomes == ([1, 404] if delete_first else [200, 1])
    finally:
        contender_attempted.set()
        event.remove(engine, "before_cursor_execute", before)
        event.remove(engine, "after_cursor_execute", after)
    with sessions() as session:
        assert session.get(Chat, "chat-a") is None
        assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0
        assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 0


@pytest.mark.parametrize(
    ("scope", "chat_scope"), [("ephemeral", "standard"), ("durable", "private")]
)
async def test_unsupported_owner_scope_exposes_no_control_or_receipt(
    sessions: sessionmaker[Session], queue_client: AsyncClient, scope: str, chat_scope: str
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.execute(update(WorkPlan).values(persistence_scope=scope))
        session.execute(update(Chat).values(scope=chat_scope))
        session.commit()
    page = await queue_client.get("/api/queue/activity")
    assert page.status_code == 200
    item = page.json()["items"][0]
    assert item["allowed_actions"] == []
    assert item["control_state"] is None and item["control_revision"] is None
    command = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "unsupported-owner"},
    )
    assert command.status_code == 404


def prepare_verification_source(
    settings: Settings, sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> tuple[ConversationOrchestrator, str, list[str]]:
    seed_plan(sessions, claimed=True)
    store = ArtifactStore(settings)
    with sessions() as session:
        run = attach_completed_run(session)
        run.operation = Operation.IMAGE_TO_IMAGE.value
        run.status = "running"
        source = session.get(Job, "first")
        source.status = "running"
        source.attempt = 1
        source.claim_owner = "existing-worker"
        step = WorkStep(
            plan_id="plan-a", ordinal=1, operation=run.operation, status="running", run_id=run.id
        )
        session.add(step)
        session.flush()
        run.work_step_id = step.id
        source.work_step_id = step.id
        session.add(
            WorkStep(
                plan_id="plan-a",
                ordinal=2,
                operation=Operation.TEXT_TO_IMAGE.value,
                status="queued",
            )
        )
        artifacts = []
        for color in ("blue", "green"):
            content = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(content, format="PNG")
            artifacts.append(
                store.ingest_bytes(
                    session, content.getvalue(), kind=ArtifactKind.IMAGE, media_type="image/png"
                )
            )
        session.flush()
        run.provenance_json = {"input_artifact_ids": [artifacts[0].id]}
        session.get(Chat, "chat-a").vision_settings_json = {"verify_image_edits": True}
        profile = ModelProfile(id="neutral-vision", name="Example vision", engine="mock")
        session.add(profile)
        session.commit()
        run_id, artifact_ids = run.id, [artifact.id for artifact in artifacts]
    orchestrator = ConversationOrchestrator(
        engines=SimpleNamespace(settings=settings),
        artifacts=store,
        events=SimpleNamespace(),
        scheduler=ResourceScheduler(session_factory=sessions),
        processes=SimpleNamespace(),
        session_factory=sessions,
    )
    # Vision qualification is independent of the queue transaction under test.
    monkeypatch.setattr(
        orchestrator,
        "_vision_profile_for_chat",
        lambda session, chat, selected: session.get(ModelProfile, "neutral-vision"),
    )
    return orchestrator, run_id, artifact_ids


def finish_source_and_queue_verifier(
    orchestrator: ConversationOrchestrator, session: Session, run_id: str, artifacts: list[str]
) -> str:
    run = session.get(Run, run_id)
    source = session.get(Job, "first")
    assert run is not None and source is not None
    assert orchestrator._complete(
        session,
        run,
        source,
        {"artifact_ids": artifacts[1:]},
        claim=JobClaim(token="existing-worker", attempt=1),
    )
    identifier = orchestrator._queue_image_edit_verification(session, run, source.id, artifacts[1:])
    assert identifier is not None
    session.flush()
    return identifier


@pytest.mark.parametrize("already_claimed", [False, True])
async def test_an_existing_internal_child_participates_in_the_plan_command(
    settings: Settings,
    sessions: sessionmaker[Session],
    queue_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    already_claimed: bool,
) -> None:
    orchestrator, run_id, artifacts = prepare_verification_source(settings, sessions, monkeypatch)
    with sessions() as session:
        identifier = finish_source_and_queue_verifier(orchestrator, session, run_id, artifacts)
        if already_claimed:
            session.execute(
                update(Job)
                .where(Job.id == identifier)
                .values(status="running", claim_owner="verification-worker", attempt=1)
            )
        session.commit()
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "existing-verification"},
    )
    assert held.status_code == (409 if already_claimed else 200)
    with sessions() as session:
        child = session.get(Job, identifier)
        assert child is not None and child.work_plan_id is None and child.run_id is None
        assert session.get(Job, "first").status == "complete"
        assert session.get(Job, "second").claim_owner is None
        if already_claimed:
            assert child.claim_owner == "verification-worker"
            assert session.scalar(text("SELECT count(*) FROM work_plan_controls")) == 0
            assert session.scalar(text("SELECT count(*) FROM work_plan_control_receipts")) == 0
        else:
            assert identifier not in [
                job.id for job in ResourceScheduler._eligible_jobs(session, "primary", utcnow())
            ]


async def test_hold_waits_for_atomic_source_completion_and_internal_child_insertion(
    settings: Settings,
    sessions: sessionmaker[Session],
    queue_app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator, run_id, artifacts = prepare_verification_source(settings, sessions, monkeypatch)
    child_inserted = threading.Event()
    hold_attempted = threading.Event()
    engine = sessions.kw["bind"]

    def before(
        connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, many: bool
    ) -> None:
        if statement == "BEGIN IMMEDIATE":
            hold_attempted.set()

    def complete() -> str:
        with sessions() as session:
            identifier = finish_source_and_queue_verifier(orchestrator, session, run_id, artifacts)
            child_inserted.set()
            assert hold_attempted.wait(5), "Hold never attempted the real writer reservation."
            session.commit()
            return identifier

    def hold() -> int:
        async def post() -> int:
            async with AsyncClient(
                transport=ASGITransport(app=queue_app), base_url="http://testserver"
            ) as client:
                return (
                    await client.post(
                        "/api/queue/items/plan-a/hold",
                        json={"expected_revision": 0, "idempotency_key": "source-completion"},
                    )
                ).status_code

        return asyncio.run(post())

    event.listen(engine, "before_cursor_execute", before)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            completed = pool.submit(complete)
            assert child_inserted.wait(5), "The real source completion did not insert its child."
            held = pool.submit(hold)
            identifier = completed.result(timeout=10)
            assert held.result(timeout=10) == 200
    finally:
        hold_attempted.set()
        event.remove(engine, "before_cursor_execute", before)
    with sessions() as session:
        assert session.get(Job, "first").status == "complete"
        assert session.get(Job, identifier).status == "queued"
        assert session.get(Job, identifier).claim_owner is None
        assert session.scalar(text("SELECT state FROM work_plan_controls")) == "held"
        assert ResourceScheduler._eligible_jobs(session, "primary", utcnow()) == []


def assert_refused_control_left_no_receipt(sessions: sessionmaker[Session]) -> None:
    from local_lm.models import WorkPlanControl, WorkPlanControlReceipt

    with sessions() as session:
        assert session.get(WorkPlanControl, "plan-a") is None
        assert session.scalar(select(WorkPlanControlReceipt)) is None


async def test_a_stale_revision_is_refused_even_when_the_plan_is_eligible(
    sessions: sessionmaker[Session],
    queue_client: AsyncClient,
) -> None:
    from local_lm.models import WorkPlanControl, WorkPlanControlReceipt

    seed_plan(sessions)
    before = job_audit(sessions)
    held = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "advance-to-held"},
    )
    assert held.status_code == 200 and held.json()["control_revision"] == 1
    released = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 1, "idempotency_key": "advance-to-eligible"},
    )
    assert released.status_code == 200 and released.json()["control_revision"] == 2
    response = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "stale-first-hold"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-control-conflict"
    with sessions() as session:
        control = session.get(WorkPlanControl, "plan-a")
        assert control.state == "eligible" and control.revision == 2
        assert session.get(WorkPlanControlReceipt, ("plan-a", "stale-first-hold")) is None
        assert len(session.scalars(select(WorkPlanControlReceipt)).all()) == 2
    assert job_audit(sessions) == before
    valid = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 2, "idempotency_key": "current-hold"},
    )
    assert valid.status_code == 200 and valid.json()["control_revision"] == 3


async def test_release_cannot_create_a_transition_for_a_plan_that_was_never_held(
    sessions: sessionmaker[Session],
    queue_client: AsyncClient,
) -> None:
    seed_plan(sessions)
    before = job_audit(sessions)
    response = await queue_client.post(
        "/api/queue/items/plan-a/release",
        json={"expected_revision": 0, "idempotency_key": "release-without-hold"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-control-conflict"
    assert_refused_control_left_no_receipt(sessions)
    assert job_audit(sessions) == before


@pytest.mark.parametrize("status", ["complete", "failed", "cancelled", "interrupted"])
async def test_a_terminal_plan_cannot_hold_still_queued_descendants(
    sessions: sessionmaker[Session],
    queue_client: AsyncClient,
    status: str,
) -> None:
    seed_plan(sessions)
    with sessions() as session:
        session.get(WorkPlan, "plan-a").status = status
        session.commit()
    before = job_audit(sessions)
    response = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "terminal-owner-hold"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-control-conflict"
    assert_refused_control_left_no_receipt(sessions)
    assert job_audit(sessions) == before


async def test_hold_refuses_a_running_descendant_after_its_real_lease_releases_ownership(
    sessions: sessionmaker[Session],
    queue_client: AsyncClient,
) -> None:
    seed_plan(sessions)
    scheduler = ResourceScheduler(session_factory=sessions)
    async with (
        asyncio.timeout(30),
        scheduler.job_lease(
            "first",
            resource="primary",
            group="primary",
        ) as claim,
    ):
        assert claim is not None
        with sessions() as session:
            job = session.get(Job, "first")
            assert job.status == "running" and job.claim_owner == claim.token
    with sessions() as session:
        job = session.get(Job, "first")
        assert job.status == "running" and job.claim_owner is None
        assert job.claim_expires_at is None and job.heartbeat_at is None
    before = job_audit(sessions)
    response = await queue_client.post(
        "/api/queue/items/plan-a/hold",
        json={"expected_revision": 0, "idempotency_key": "released-running-descendant"},
    )
    assert response.status_code == 409
    assert response.json()["code"] == "queue-control-conflict"
    assert_refused_control_left_no_receipt(sessions)
    assert job_audit(sessions) == before
