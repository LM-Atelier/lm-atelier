"""A render keeps going when one of its progress writes finds the database held.

SQLite has a single writer. While another connection holds it, a write waits
out the busy timeout and then fails with "database is locked". A progress
update, a preview frame or the engine's completion stamp only says how far
the render has got; the next one says it again with newer numbers, so a held
database is a reason to skip one, never a reason to end the render.

The writer is held for real, from a second connection, for exactly the event
under test, and the application's own connections wait a tenth of a second
instead of five so a contended write fails quickly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import QueuePool
from test_scheduler import _real_sqlite_error

from local_lm import db as database_module
from local_lm import orchestrator as orchestrator_module
from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import JobStatus
from local_lm.models import Job, Run
from local_lm.progress import apply_engine_progress

SHORT_BUSY_WAIT_MS = 100


@contextmanager
def _short_busy_wait() -> Iterator[None]:
    """Make every new application connection give up after a tenth of a second."""

    engine = database_module.engine

    def configure(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA busy_timeout={SHORT_BUSY_WAIT_MS}")
        cursor.close()

    event.listen(engine, "connect", configure)
    engine.dispose()
    try:
        yield
    finally:
        event.remove(engine, "connect", configure)
        engine.dispose()


@contextmanager
def _writer_held(database: Path) -> Iterator[None]:
    holder = sqlite3.connect(database, timeout=5)
    try:
        holder.execute("BEGIN IMMEDIATE")
        yield
    finally:
        holder.rollback()
        holder.close()


def _hold_the_writer_during(
    monkeypatch: pytest.MonkeyPatch,
    database: Path,
    chosen: Callable[[MediaEvent], bool],
) -> list[str]:
    """Hold SQLite's writer while the render handles each chosen event.

    The engine's generator yields an event and the render writes it before
    asking for the next one, so a writer taken just before the yield and
    released when the generator resumes covers exactly that event's writes.
    """

    held: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        async for media_event in original(self, request):
            if chosen(media_event):
                with _writer_held(database):
                    held.append(media_event.type)
                    yield media_event
                continue
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    return held


@contextmanager
def _old_reads_on_every_pooled_connection(database: Path) -> Iterator[None]:
    """Leave a result open on each pooled connection, then commit elsewhere.

    The results outlive their connections' return to the pool, as a stream
    read partway would. A connection that kept its result's read would make
    the next write through it fail at once with SQLITE_BUSY_SNAPSHOT.
    """

    engine = database_module.engine
    pool = engine.pool
    assert isinstance(pool, QueuePool)
    connections = [engine.connect() for _ in range(pool.size())]
    kept = []
    try:
        for connection in connections:
            result = connection.exec_driver_sql("SELECT name FROM sqlite_master")
            assert result.fetchone() is not None
            kept.append(result)
        for connection in connections:
            connection.close()
        other = sqlite3.connect(database, timeout=5)
        try:
            # A row has to change: a write that changes nothing leaves the
            # old reads current.
            other.execute("UPDATE jobs SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now')")
            other.commit()
        finally:
            other.close()
        yield
    finally:
        for connection in connections:
            connection.close()
        kept.clear()


def _old_reads_during(
    monkeypatch: pytest.MonkeyPatch,
    database: Path,
    chosen: Callable[[MediaEvent], bool],
) -> list[str]:
    met: list[str] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        async for media_event in original(self, request):
            if chosen(media_event):
                with _old_reads_on_every_pooled_connection(database):
                    met.append(media_event.type)
                    yield media_event
                continue
            yield media_event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    return met


async def _render(client: AsyncClient, key: str) -> tuple[dict[str, Any], str]:
    chat = await client.post("/api/chats", json={"title": "Render beside a held database"})
    assert chat.status_code == 201
    chat_id = chat.json()["id"]
    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "Create an image of a purple observatory",
            "mode": "image",
            "idempotency_key": key,
        },
    )
    assert response.status_code == 202
    run_id = response.json()["run"]["id"]

    async def read() -> dict[str, Any] | None:
        current = await client.get(f"/api/chats/{chat_id}")
        assert current.status_code == 200
        assistants = [
            message for message in current.json()["messages"] if message["role"] == "assistant"
        ]
        return assistants[-1] if assistants else None

    assistant = await wait_for_terminal_status(
        read, what=f"the assistant run in chat {chat_id}", expected=None
    )
    return dict(assistant), run_id


def _job_status(run_id: str) -> str:
    with SessionLocal() as session:
        run = session.get(Run, run_id)
        assert run is not None
        job = session.scalars(select(Job).where(Job.run_id == run.id)).one()
        return job.status


def _finished_with_its_picture(assistant: dict[str, Any], run_id: str) -> None:
    assert assistant["status"] == "complete", json.dumps(assistant["parts"])[:400]
    image = next(part for part in assistant["parts"] if part["type"] == "image")
    assert image["artifact"]["kind"] == "image"
    assert image["artifact"]["metadata_json"].get("temporary_preview") is None
    assert "locked" not in json.dumps(assistant)
    assert _job_status(run_id) == JobStatus.COMPLETE.value


@pytest.mark.parametrize(
    ("label", "chosen"),
    [
        ("progress", lambda media_event: media_event.phase == "sampling"),
        ("preview", lambda media_event: media_event.type == "preview"),
        ("completion stamp", lambda media_event: media_event.type == "complete"),
    ],
)
async def test_a_render_finishes_when_a_write_about_its_progress_finds_the_database_held(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    chosen: Callable[[MediaEvent], bool],
) -> None:
    database = settings.state_dir / "local-lm.sqlite3"
    held = _hold_the_writer_during(monkeypatch, database, chosen)

    with _short_busy_wait():
        assistant, run_id = await _render(client, f"held-{label.replace(' ', '-')}")

    _finished_with_its_picture(assistant, run_id)
    # Once, and only once: a failed render is retried, and every retry would
    # meet the held database again.
    assert len(held) == 1, f"the {label} write has to have met the held database once"


async def test_a_render_finishes_when_its_pooled_connections_still_hold_old_reads(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A result left open on a pooled connection does not fail the next write at once.

    The pool closes what a connection still has open when it comes back, so
    the progress write lands outright: it is not even skipped.
    """

    database = settings.state_dir / "local-lm.sqlite3"
    met = _old_reads_during(monkeypatch, database, lambda event: event.phase == "sampling")

    with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        assistant, run_id = await _render(client, "old-reads")

    _finished_with_its_picture(assistant, run_id)
    assert met == ["progress"]
    assert not [
        record for record in caplog.records if "skipped a progress update" in record.getMessage()
    ]


@pytest.mark.parametrize("stage", ["sampling", "preview", "complete"])
async def test_a_progress_write_that_fails_for_another_reason_still_ends_the_render(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
) -> None:
    """Only a held database is skipped; a broken one still ends the render.

    Without this the repair could swallow every database failure, and a render
    would carry on over damage nobody hears about.
    """

    broken_database = tmp_path / "broken.sqlite3"
    with sqlite3.connect(broken_database) as connection:
        connection.execute("CREATE TABLE present (id INTEGER PRIMARY KEY)")
        try:
            connection.execute("SELECT missing FROM present")
        except sqlite3.OperationalError as error:
            broken = OperationalError("statement", {}, error)
        else:
            raise AssertionError("sqlite did not fail as this test requires")
    refused: list[str] = []

    def refuse_one_stage(session: Any, **values: Any) -> Any:
        if values.get("stage") == stage:
            refused.append(stage)
            raise broken
        return apply_engine_progress(session, **values)

    monkeypatch.setattr(orchestrator_module, "apply_engine_progress", refuse_one_stage)

    assistant, run_id = await _render(client, f"broken-{stage}")

    # A failed render is retried up to its allowance, and each try is refused.
    assert refused and set(refused) == {stage}
    assert assistant["status"] == "failed"
    assert _job_status(run_id) == JobStatus.FAILED.value
    # The database's own text names the statement and every value it carried;
    # the conversation shows a plain reason instead.
    shown = next(part for part in assistant["parts"] if part["type"] == "error")
    assert shown["text"] == "This run stopped because of a database error."
    assert "missing" not in json.dumps(assistant)
    with SessionLocal() as session:
        job = session.scalars(select(Job).where(Job.run_id == run_id)).one()
        assert job.error == shown["text"]


async def test_a_held_progress_update_gives_up_quickly_and_the_next_ones_wait_quietly(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two progress events in a row while another connection holds the writer.

    Each wait blocks the event loop, which reads the engine's stream: a render
    that waited out the full busy timeout on every event starved the stream's
    keepalive and lost the connection all the same. The first update gives up
    in a fraction of a second, the next is left out without trying, and
    neither is broadcast.
    """

    database = settings.state_dir / "local-lm.sqlite3"
    tried: list[str] = []

    def counting(session: Any, **values: Any) -> Any:
        tried.append(str(values.get("stage")))
        return apply_engine_progress(session, **values)

    monkeypatch.setattr(orchestrator_module, "apply_engine_progress", counting)
    held = _hold_the_writer_during(
        monkeypatch, database, lambda event: event.phase in {"encoding prompt", "sampling"}
    )
    broadcast: list[str] = []
    publish = app.state.services.events.publish

    async def observed(event_type: str, entity_id: Any = None, payload: Any = None) -> Any:
        if event_type == "generation.progress" and isinstance(payload, dict):
            broadcast.append(str(payload.get("phase")))
        return await publish(event_type, entity_id, payload)

    monkeypatch.setattr(app.state.services.events, "publish", observed)

    with caplog.at_level(logging.INFO, logger="local_lm.orchestrator"):
        assistant, run_id = await _render(client, "held-in-a-row")

    _finished_with_its_picture(assistant, run_id)
    assert len(held) == 2
    skips = [
        record.getMessage()
        for record in caplog.records
        if "skipped a progress" in record.getMessage()
    ]
    assert len(skips) == 1, skips
    waited_ms = float(skips[0].rsplit(" after ", 1)[1].split(" ms", 1)[0])
    assert waited_ms < 1000, skips[0]
    assert "sampling" not in tried, "the update right after a held one tried the database again"
    assert not {"encoding prompt", "sampling"} & set(broadcast)


def _busy_timeout(dbapi_connection: Any) -> int:
    cursor = dbapi_connection.execute("PRAGMA busy_timeout")
    try:
        return int(cursor.fetchone()[0])
    finally:
        cursor.close()


async def test_a_render_gives_each_connection_back_its_usual_busy_timeout(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A progress write's shorter wait ends before its connection goes back to the pool.

    The pool hands a returned connection to whichever session asks next, on
    any thread. One that still carried a render's quarter-second wait would
    make that session's writes give up on a held database twenty times sooner
    than any other write does.
    """

    engine = database_module.engine
    during: list[int] = []
    returned: list[int] = []

    def observed(session: Any, **values: Any) -> Any:
        during.append(_busy_timeout(session.connection().connection.driver_connection))
        return apply_engine_progress(session, **values)

    def checked_in(dbapi_connection: Any, _record: Any) -> None:
        if dbapi_connection is not None:
            returned.append(_busy_timeout(dbapi_connection))

    monkeypatch.setattr(orchestrator_module, "apply_engine_progress", observed)
    event.listen(engine, "checkin", checked_in)
    try:
        assistant, run_id = await _render(client, "usual-busy-timeout")
    finally:
        event.remove(engine, "checkin", checked_in)

    _finished_with_its_picture(assistant, run_id)
    assert orchestrator_module.MEDIA_UPDATE_BUSY_WAIT_MS in during
    assert returned
    assert set(returned) == {database_module.SQLITE_BUSY_TIMEOUT_MS}


async def test_a_run_stopped_by_a_database_held_too_long_says_so_in_plain_words(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contended = _real_sqlite_error(tmp_path / "busy", contended=True)

    def held(*_args: Any, **_kwargs: Any) -> Any:
        raise contended

    monkeypatch.setattr(orchestrator_module.ConversationOrchestrator, "_complete", held)
    # The completion is tried again after each pause, on every retry of the run.
    pauses = orchestrator_module.HELD_DATABASE_RETRY_DELAYS_SECONDS
    monkeypatch.setattr(
        orchestrator_module, "HELD_DATABASE_RETRY_DELAYS_SECONDS", (0.0,) * len(pauses)
    )

    assistant, run_id = await _render(client, "held-at-completion")

    assert assistant["status"] == "failed"
    shown = next(part for part in assistant["parts"] if part["type"] == "error")
    assert shown["text"] == (
        "This run stopped because another task held the database for too long. Try again."
    )
    assert "sqlite3" not in json.dumps(assistant) and "[SQL:" not in json.dumps(assistant)
    with SessionLocal() as session:
        job = session.scalars(select(Job).where(Job.run_id == run_id)).one()
        assert job.error == shown["text"]
