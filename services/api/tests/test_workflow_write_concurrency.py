from __future__ import annotations

import asyncio
import threading

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, func, select

from local_lm import api
from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.schemas import WorkflowCreate, WorkflowRevisionCreate


@pytest.mark.parametrize("revision", [False, True], ids=["workflow", "revision"])
@pytest.mark.parametrize("contended", [False, True], ids=["available", "busy"])
async def test_workflow_writes_allow_the_event_loop_to_release_another_writer(
    client: AsyncClient, revision: bool, contended: bool
) -> None:
    payload = {
        "name": "Concurrent write",
        "operation": "text_to_image",
        "engine": "mock",
        "api_graph": {"loader": {"class_type": "PortableLoader"}},
    }
    endpoint = "/api/workflows"
    table = "workflow_definitions"
    if revision:
        created = await client.post(endpoint, json=payload)
        assert created.status_code == 201, created.text
        endpoint = f"/api/workflows/{created.json()['id']}/revisions"
        payload.pop("name")
        payload.pop("operation")
        payload.pop("engine")
        table = "workflow_revisions"

    loop = asyncio.get_running_loop()
    attempts: list[str] = []
    released = asyncio.Event()
    with SessionLocal() as writer:
        connection = writer.connection()
        engine = connection.engine
        if contended:
            connection.exec_driver_sql("BEGIN IMMEDIATE")

        def release_writer() -> None:
            writer.rollback()
            released.set()

        def before_write(conn, cursor, statement, parameters, context, executemany):
            writes_workflow = statement.startswith(f"INSERT INTO {table} ") or (
                revision and statement.startswith("UPDATE workflow_revisions ")
            )
            if not writes_workflow or attempts:
                return
            attempts.append(statement)
            cursor.execute("PRAGMA busy_timeout=250")
            # Once the request reaches SQLite, unrelated event-loop work must
            # remain runnable so it can release an existing write transaction.
            loop.call_soon_threadsafe(release_writer)

        event.listen(engine, "before_cursor_execute", before_write)
        try:
            response = await client.post(endpoint, json=payload)
            assert response.status_code == 201, response.text
            assert attempts, "the workflow write was not observed"
            await asyncio.wait_for(released.wait(), timeout=2)
        finally:
            event.remove(engine, "before_cursor_execute", before_write)
            writer.rollback()


@pytest.mark.parametrize("revision", [False, True], ids=["workflow", "revision"])
@pytest.mark.parametrize("fail", [False, True], ids=["commit", "rollback"])
async def test_cancelling_a_workflow_write_keeps_its_session_until_the_worker_finishes(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, revision: bool, fail: bool
) -> None:
    payload = WorkflowCreate(
        name="Cancelled request",
        operation="text_to_image",
        engine="mock",
        api_graph={"loader": {"class_type": "PortableLoader"}},
    )
    created = await client.post("/api/workflows", json=payload.model_dump(mode="json"))
    assert created.status_code == 201, created.text
    workflow_id = created.json()["id"]
    loop = asyncio.get_running_loop()
    event_loop_thread = threading.get_ident()
    entered = asyncio.Event()
    closed = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    original_contract = api.workflow_artifact_contract

    def contract(**kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert threading.get_ident() != event_loop_thread, "workflow writes block the event loop"
        try:
            assert release.wait(timeout=10), "the test did not release the workflow writer"
            assert not closed.is_set(), "the request closed a session still used by its writer"
            if fail:
                raise ValueError("Neutral workflow write failure")
            return original_contract(**kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(api, "workflow_artifact_contract", contract)

    async def request():
        with SessionLocal() as session:
            try:
                if revision:
                    return await api.create_workflow_revision(
                        workflow_id, WorkflowRevisionCreate(api_graph=payload.api_graph), session
                    )
                return await api.create_workflow(payload, session)
            finally:
                closed.set()

    with SessionLocal() as session:
        count = session.scalar(select(func.count()).select_from(WorkflowRevision))
    task = asyncio.create_task(request())
    try:
        await asyncio.wait_for(entered.wait(), timeout=10)
        assert not task.done(), "the workflow write did not yield to the event loop"
        for _ in range(2):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not closed.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set() and closed.is_set()
        with SessionLocal() as session:
            assert session.scalar(select(func.count()).select_from(WorkflowRevision)) == (
                count if fail else count + 1
            )
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_concurrent_revision_writes_choose_distinct_versions(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Concurrent revisions",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"loader": {"class_type": "PortableLoader"}},
        },
    )
    assert created.status_code == 201, created.text
    workflow_id = created.json()["id"]
    loop = asyncio.get_running_loop()
    event_loop_thread = threading.get_ident()
    first_read = asyncio.Event()
    second_entered = asyncio.Event()
    release = threading.Event()
    reader_threads: list[int] = []
    with SessionLocal() as session:
        engine = session.get_bind()

    def before_statement(conn, cursor, statement, parameters, context, executemany):
        if (
            statement.startswith("UPDATE workflow_revisions SET version = version WHERE 0")
            and reader_threads
            and threading.get_ident() != reader_threads[0]
        ):
            loop.call_soon_threadsafe(second_entered.set)

    def after_statement(conn, cursor, statement, parameters, context, executemany):
        if not statement.startswith("SELECT max(workflow_revisions.version)"):
            return
        reader_threads.append(threading.get_ident())
        if len(reader_threads) == 1:
            loop.call_soon_threadsafe(first_read.set)
            if threading.get_ident() != event_loop_thread:
                assert release.wait(timeout=10), (
                    "the test did not release the first revision writer"
                )
        else:
            loop.call_soon_threadsafe(second_entered.set)

    endpoint = f"/api/workflows/{workflow_id}/revisions"
    payload = {"api_graph": {"loader": {"class_type": "PortableLoader"}}}
    event.listen(engine, "before_cursor_execute", before_statement)
    event.listen(engine, "after_cursor_execute", after_statement)
    first = asyncio.create_task(client.post(endpoint, json=payload))
    second = None
    try:
        await asyncio.wait_for(first_read.wait(), timeout=10)
        second = asyncio.create_task(client.post(endpoint, json=payload))
        await asyncio.wait_for(second_entered.wait(), timeout=10)
        release.set()
        responses = await asyncio.gather(first, second, return_exceptions=True)
        assert all(getattr(response, "status_code", None) == 201 for response in responses), (
            responses
        )
        with SessionLocal() as session:
            versions = session.scalars(
                select(WorkflowRevision.version)
                .where(WorkflowRevision.workflow_id == workflow_id)
                .order_by(WorkflowRevision.version)
            ).all()
            assert versions == [1, 2, 3]
            definition = session.get(WorkflowDefinition, workflow_id)
            current = session.get(WorkflowRevision, definition.current_revision_id)
            assert current.version == 3
    finally:
        release.set()
        await asyncio.gather(
            first, *([second] if second is not None else []), return_exceptions=True
        )
        event.remove(engine, "before_cursor_execute", before_statement)
        event.remove(engine, "after_cursor_execute", after_statement)
