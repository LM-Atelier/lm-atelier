from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event, update
from sqlalchemy.orm import Session

from local_lm import api
from local_lm.api_errors import register_api_error_handler
from local_lm.db import Base
from local_lm.models import Chat, Job, WorkPlan, WorkStep, WorkStepDependency

STAMP = datetime(2026, 9, 1, tzinfo=UTC)


class Security:
    def local_state_signing_key(self, purpose: bytes) -> bytes:
        assert purpose == b"user-queue-activity"
        return b"neutral-test-signing-key"


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as value:
        yield value
    engine.dispose()


@pytest_asyncio.fixture
async def client(session: Session) -> AsyncIterator[AsyncClient]:
    app = FastAPI()
    register_api_error_handler(app)
    app.state.services = SimpleNamespace(security=Security())
    app.include_router(api.router)

    async def override() -> AsyncIterator[Session]:
        yield session

    app.dependency_overrides[api.get_conversation_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as value:
        yield value


def job(
    session: Session,
    identifier: str,
    *,
    kind: str = "image",
    status: str = "queued",
    plan: str | None = None,
    created: int = 0,
) -> None:
    session.add(
        Job(
            id=identifier,
            kind=kind,
            status=status,
            work_plan_id=plan,
            created_at=STAMP + timedelta(seconds=created),
            updated_at=STAMP + timedelta(seconds=created),
            progress=0.25,
            phase="neutral-private-phase",
            payload_json={"prompt": "neutral-private-payload"},
            result_json={"path": "neutral-private-result"},
        )
    )


def plan(
    session: Session,
    identifier: str,
    *,
    scope: str = "durable",
    status: str = "queued",
    created: int = 0,
    chat_scope: str = "standard",
) -> None:
    session.add(Chat(id="chat-" + identifier, title="Example " + identifier, scope=chat_scope))
    session.flush()
    session.add(
        WorkPlan(
            id=identifier,
            chat_id="chat-" + identifier,
            transcript_sequence=1,
            persistence_scope=scope,
            status=status,
            created_at=STAMP + timedelta(seconds=created),
            updated_at=STAMP + timedelta(seconds=created),
            summary_json={"prompt": "neutral-private-plan"},
        )
    )
    session.flush()


async def test_groups_plans_and_omits_internal_jobs_and_private_payloads(
    session: Session,
    client: AsyncClient,
) -> None:
    plan(session, "plan-a")
    session.add_all(
        [
            WorkStep(
                id="first",
                plan_id="plan-a",
                ordinal=0,
                operation="image",
                status="complete",
                prompt="neutral-private-prompt",
                settings_json={"path": "neutral-private-settings"},
            ),
            WorkStep(id="second", plan_id="plan-a", ordinal=1, operation="image", status="queued"),
            WorkStep(id="third", plan_id="plan-a", ordinal=2, operation="image", status="queued"),
        ]
    )
    session.flush()
    session.add(WorkStepDependency(step_id="third", depends_on_step_id="second"))
    job(session, "running", plan="plan-a", status="running")
    job(session, "queued", plan="plan-a")
    job(session, "hidden", plan="plan-a", kind="edit_verify", status="running")
    job(session, "download", kind="download", status="paused", created=1)
    job(session, "install", kind="activate", created=2)
    job(session, "hidden-alone", kind="edit_verify", created=3)
    job(session, "done", status="complete", created=4)
    session.commit()
    session.expunge_all()
    loaded: list[object] = []

    def track(_session: Session, instance: object) -> None:
        loaded.append(instance)

    event.listen(session, "loaded_as_persistent", track)
    try:
        response = await client.get("/api/queue/activity")
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    value = response.json()
    assert value["total"] == 3
    assert value["lane_counts"] == {"generation": 1, "transfer": 1, "install": 1}
    assert [row["owner_id"] for row in value["items"]] == ["plan-a", "download", "install"]
    row = value["items"][0]
    assert row["owner_type"] == "work_plan"
    assert row["status"] == "running"
    assert row["chat_title"] == "Example plan-a"
    assert row["step_count"] == 3 and row["completed_steps"] == 1 and row["blocked_steps"] == 1
    assert row["active_jobs"] == 2 and row["running_jobs"] == 1 and row["queued_jobs"] == 1
    assert row["progress"] is None
    assert value["items"][1]["progress"] == 0.25
    assert row["created_at"].endswith("Z") or row["created_at"].endswith("+00:00")
    assert "neutral-private" not in json.dumps(value)
    assert "hidden" not in json.dumps(value)
    assert loaded == []
    assert not session.new and not session.dirty and not session.deleted


async def test_pages_tied_owners_and_excludes_newer_acceptance(
    session: Session,
    client: AsyncClient,
) -> None:
    for identifier in ["c", "a", "b"]:
        job(session, identifier)
    plan(session, "plan-z")
    session.commit()
    first = await client.get("/api/queue/activity", params={"limit": 2})
    assert first.status_code == 200
    assert [row["owner_id"] for row in first.json()["items"]] == ["a", "b"]
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    job(session, "later", created=1)
    session.commit()
    second = await client.get("/api/queue/activity", params={"limit": 2, "cursor": cursor})
    assert second.status_code == 200
    assert [row["owner_id"] for row in second.json()["items"]] == ["c", "plan-z"]
    assert second.json()["next_cursor"] is None and second.json()["total"] == 4
    fresh = await client.get("/api/queue/activity")
    assert fresh.json()["total"] == 5


async def test_lane_filter_and_cursor_context_fail_closed(
    session: Session, client: AsyncClient
) -> None:
    for index in range(3):
        job(session, "download-" + str(index), kind="download", created=index)
    job(session, "generation")
    session.commit()
    response = await client.get("/api/queue/activity", params={"lane": "transfer", "limit": 1})
    assert response.status_code == 200
    value = response.json()
    assert value["total"] == 3 and value["lane_counts"]["generation"] == 1
    assert value["items"][0]["lane"] == "transfer"
    cursor = value["next_cursor"]
    for params in [
        {"lane": "generation", "cursor": cursor},
        {"cursor": cursor},
        {"lane": "transfer", "cursor": cursor[:-1] + ("a" if cursor[-1] != "a" else "b")},
        {"cursor": "neutral-invalid-cursor"},
    ]:
        invalid = await client.get("/api/queue/activity", params=params)
        assert invalid.status_code == 422
        assert "queue-activity-cursor-invalid" in invalid.text
        assert cursor not in invalid.text and "neutral-invalid-cursor" not in invalid.text


@pytest.mark.parametrize("scope,chat_scope", [("ephemeral", "standard"), ("durable", "incognito")])
async def test_private_plan_omits_chat_identity(
    session: Session, client: AsyncClient, scope: str, chat_scope: str
) -> None:
    plan(session, "private", scope=scope, chat_scope=chat_scope)
    session.commit()
    response = await client.get("/api/queue/activity")
    assert response.status_code == 200
    row = response.json()["items"][0]
    assert row["chat_id"] is None and row["chat_title"] is None
    assert "Example private" not in response.text


async def test_active_child_survives_terminal_plan_summary_and_pending_writes_do_not_flush(
    session: Session,
    client: AsyncClient,
) -> None:
    plan(session, "lagging", status="complete")
    job(session, "active", plan="lagging", status="running")
    session.commit()
    job(session, "uncommitted")
    response = await client.get("/api/queue/activity")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["owner_id"] == "lagging"
    assert response.json()["items"][0]["status"] == "running"
    assert len(session.new) == 1


@pytest.mark.parametrize(
    "params",
    [{"limit": "0"}, {"limit": "101"}, {"limit": "-1"}, {"lane": "invalid"}, {"cursor": ""}],
)
async def test_queue_query_is_bounded(client: AsyncClient, params: dict[str, str]) -> None:
    response = await client.get("/api/queue/activity", params=params)
    assert response.status_code == 422


async def test_empty_queue_is_explicit(client: AsyncClient) -> None:
    response = await client.get("/api/queue/activity")
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    assert response.json()["next_cursor"] is None


async def test_large_queue_has_bounded_scalar_reads(session: Session, client: AsyncClient) -> None:
    for index in range(601):
        job(session, f"job-{index:04}", created=index)
    session.commit()
    statements: list[str] = []

    def track(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        statements.append(statement)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", track)
    try:
        response = await client.get("/api/queue/activity", params={"limit": 25})
    finally:
        event.remove(engine, "before_cursor_execute", track)
    assert response.status_code == 200
    value = response.json()
    assert value["total"] == 601 and len(value["items"]) == 25
    reads = [
        statement
        for statement in statements
        if statement.lstrip().upper().startswith(("SELECT", "WITH"))
    ]
    assert len(reads) == 3
    assert all(
        "payload_json" not in statement
        and "summary_json" not in statement
        and "settings_json" not in statement
        for statement in reads
    )


async def test_count_and_rows_share_a_read_snapshot(tmp_path: Path, client: AsyncClient) -> None:
    probe = await client.get("/api/queue/activity")
    assert probe.status_code == 200
    from local_lm.user_queue_activity import list_queue_activity

    engine = create_engine("sqlite:///" + str(tmp_path / "queue.db"))
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
    Base.metadata.create_all(engine)
    with Session(engine) as writer:
        job(writer, "accepted")
        writer.commit()
    changed = False

    def advance(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal changed
        if changed or not statement.lstrip().upper().startswith("WITH"):
            return
        changed = True
        with Session(engine) as writer:
            writer.execute(update(Job).where(Job.id == "accepted").values(status="complete"))
            writer.commit()

    event.listen(engine, "after_cursor_execute", advance)
    try:
        with Session(engine) as reader:
            result = list_queue_activity(reader, signing_key=b"neutral-snapshot-key")
            assert changed
            assert result.total == 1
            assert [row.owner_id for row in result.items] == ["accepted"]
            assert result.items[0].status == "queued"
    finally:
        event.remove(engine, "after_cursor_execute", advance)
    with Session(engine) as reader:
        result = list_queue_activity(reader, signing_key=b"neutral-snapshot-key")
        assert result.total == 0 and result.items == []
    engine.dispose()
