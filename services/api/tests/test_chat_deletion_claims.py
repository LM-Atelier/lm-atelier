from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event, select, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm.chat_deletion import ExchangeBusy, delete_exchange
from local_lm.db import SessionLocal, database_is_contended
from local_lm.domain import JobStatus, RunStatus, utcnow
from local_lm.models import Job, Message, Run, RunContextSnapshot
from local_lm.schemas import TurnRequest


@pytest.mark.parametrize(
    "status",
    [JobStatus.COMPLETE, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.INTERRUPTED],
)
@pytest.mark.parametrize("expired_timestamp", [False, True])
async def test_finished_exchange_waits_for_its_actual_claim_scope_to_release(
    app: FastAPI, client: AsyncClient, status: JobStatus, expired_timestamp: bool
) -> None:
    response = await client.post("/api/chats", json={"title": "Finishing work"})
    assert response.status_code == 201
    payload = response.json()
    assert isinstance(payload, dict)
    chat_id = payload.get("id")
    assert isinstance(chat_id, str)
    with SessionLocal() as session:
        accepted = await app.state.services.orchestrator.create_turn(
            session,
            chat_id,
            TurnRequest.model_validate({"text": "A neutral exchange", "mode": "text"}),
            freeze_context=True,
        )
        run_id = accepted.run.id
        user_message_id = accepted.run.user_message_id
        assert isinstance(run_id, str) and isinstance(user_message_id, str)
        job_id = session.scalar(select(Job.id).where(Job.run_id == run_id))
        assert job_id is not None
        context = session.get(RunContextSnapshot, run_id)
        assert context is not None
        context_digest = context.sha256

    async with app.state.services.scheduler.job_lease(
        job_id, resource="interactive_compute", group="primary"
    ) as claim:
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None and job.claim_owner == claim.token
            run = session.get(Run, run_id)
            assert run is not None
            job.status = status.value
            run.status = RunStatus.FAILED.value if status is JobStatus.INTERRUPTED else status.value
            if expired_timestamp:
                job.claim_expires_at = utcnow() - timedelta(seconds=1)
            session.commit()

        refused = await client.delete(f"/api/messages/{user_message_id}/exchange")
        assert refused.status_code == 409
        error = refused.json()
        assert isinstance(error, dict)
        assert error.get("code") == "exchange-busy"
        assert error.get("job_count") == 1

        with SessionLocal() as session:
            with pytest.raises(ExchangeBusy) as failure:
                delete_exchange(session, user_message_id)
            assert failure.value.job_count == 1
            session.rollback()
            job = session.get(Job, job_id)
            run = session.get(Run, run_id)
            context = session.get(RunContextSnapshot, run_id)
            assert job is not None and job.status == status.value
            assert job.claim_owner == claim.token and job.attempt == claim.attempt
            assert run is not None and run.user_message_id == user_message_id
            assert session.get(Message, user_message_id) is not None
            assert context is not None and context.sha256 == context_digest

    removed = await client.delete(f"/api/messages/{user_message_id}/exchange")
    assert removed.status_code == 200
    with SessionLocal() as session:
        assert session.get(Job, job_id) is None
        assert session.get(Run, run_id) is None
        assert session.get(Message, user_message_id) is None
        assert session.get(RunContextSnapshot, run_id) is None


async def test_retry_cannot_queue_a_job_between_deletion_inspection_and_commit(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await client.post("/api/chats", json={"title": "Finishing history"})
    assert response.status_code == 201
    payload = response.json()
    assert isinstance(payload, dict)
    chat_id = payload.get("id")
    assert isinstance(chat_id, str)
    with SessionLocal() as session:
        accepted = await app.state.services.orchestrator.create_turn(
            session,
            chat_id,
            TurnRequest.model_validate({"text": "A neutral exchange", "mode": "text"}),
            freeze_context=True,
        )
        run_id = accepted.run.id
        user_message_id = accepted.run.user_message_id
        assert isinstance(run_id, str) and isinstance(user_message_id, str)
        job = session.scalar(select(Job).where(Job.run_id == run_id))
        assert job is not None
        job_id = job.id
        job.status = JobStatus.FAILED.value
        accepted.run.status = RunStatus.FAILED.value
        session.commit()

    attempted = False
    refused = False

    def retry_after_job_read(
        _connection: Connection,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal attempted, refused
        if attempted or not statement.lstrip().startswith("SELECT") or "FROM jobs" not in statement:
            return
        attempted = True
        with binding.engine.connect() as connection:
            previous_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
            assert isinstance(previous_timeout, int)
            try:
                connection.exec_driver_sql("PRAGMA busy_timeout=1")
                connection.commit()
                with Session(bind=connection) as retry_session:
                    try:
                        retry_session.execute(
                            update(Job)
                            .where(Job.id == job_id)
                            .values(status=JobStatus.QUEUED.value)
                        )
                        retry_session.commit()
                    except OperationalError as error:
                        assert database_is_contended(error)
                        refused = True
                        retry_session.rollback()
            finally:
                connection.exec_driver_sql(f"PRAGMA busy_timeout={previous_timeout}")

    with SessionLocal() as session:
        binding = session.get_bind()
        event.listen(binding, "after_cursor_execute", retry_after_job_read)
        try:
            removed = delete_exchange(session, user_message_id)
            assert attempted and refused
            assert removed.job_ids == [job_id]
            session.commit()
        finally:
            event.remove(binding, "after_cursor_execute", retry_after_job_read)
    with SessionLocal() as session:
        assert session.get(Job, job_id) is None
        assert session.get(Run, run_id) is None
        assert session.get(RunContextSnapshot, run_id) is None


async def test_cached_finished_job_cannot_hide_a_newly_held_claim(
    app: FastAPI, client: AsyncClient
) -> None:
    response = await client.post("/api/chats", json={"title": "Repeated work"})
    assert response.status_code == 201
    payload = response.json()
    assert isinstance(payload, dict)
    chat_id = payload.get("id")
    assert isinstance(chat_id, str)
    with SessionLocal() as cached_session:
        accepted = await app.state.services.orchestrator.create_turn(
            cached_session,
            chat_id,
            TurnRequest.model_validate({"text": "A neutral exchange", "mode": "text"}),
            freeze_context=True,
        )
        run_id = accepted.run.id
        user_message_id = accepted.run.user_message_id
        assert isinstance(run_id, str) and isinstance(user_message_id, str)
        cached_job = cached_session.scalar(select(Job).where(Job.run_id == run_id))
        assert cached_job is not None
        job_id = cached_job.id
        cached_job.status = JobStatus.COMPLETE.value
        accepted.run.status = RunStatus.COMPLETE.value
        cached_session.commit()
        assert cached_job.claim_owner is None

        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            job.status = JobStatus.QUEUED.value
            session.commit()
        async with app.state.services.scheduler.job_lease(
            job_id, resource="interactive_compute", group="primary"
        ) as claim:
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner == claim.token
                job.status = JobStatus.COMPLETE.value
                session.commit()
            assert cached_job.status == JobStatus.COMPLETE.value
            assert cached_job.claim_owner is None
            with pytest.raises(ExchangeBusy) as failure:
                delete_exchange(cached_session, user_message_id)
            assert failure.value.job_count == 1
            cached_session.rollback()
            with SessionLocal() as session:
                job = session.get(Job, job_id)
                assert job is not None and job.claim_owner == claim.token
                assert session.get(Run, run_id) is not None
                assert session.get(RunContextSnapshot, run_id) is not None

    removed = await client.delete(f"/api/messages/{user_message_id}/exchange")
    assert removed.status_code == 200
    with SessionLocal() as session:
        assert session.get(Job, job_id) is None
        assert session.get(Run, run_id) is None
        assert session.get(RunContextSnapshot, run_id) is None
