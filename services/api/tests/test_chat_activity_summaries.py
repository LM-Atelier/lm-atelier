from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import Connection, create_engine, event
from sqlalchemy.orm import Session

from local_lm import api
from local_lm.api_errors import register_api_error_handler
from local_lm.chat_summary_reads import list_chat_summary_rows
from local_lm.db import Base
from local_lm.models import (
    Chat,
    ChatActivityEvent,
    Job,
    Message,
    Project,
    ResponseRevision,
    ResponseRevisionPart,
    Run,
)


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
    app.include_router(api.router)

    async def override() -> AsyncIterator[Session]:
        yield session

    app.dependency_overrides[api.get_conversation_session] = override
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as value:
        yield value


async def test_chat_summaries_never_load_settings_or_hidden_chat_rows(
    session: Session, client: AsyncClient
) -> None:
    session.add_all(
        [
            Chat(
                id="visible-chat",
                title="Color study",
                draft_prompt="neutral-private-draft",
                generation_settings_json={"sentinel": "neutral-private-settings"},
                origin_json={"sentinel": "neutral-private-origin"},
            ),
            Chat(id="hidden-chat", title="Neutral helper title", scope="studio", origin_json={}),
        ]
    )
    session.commit()
    session.expunge_all()
    hydrated: list[object] = []

    def loaded(_session: Session, instance: object) -> None:
        hydrated.append(instance)

    event.listen(session, "loaded_as_persistent", loaded)
    try:
        response = await client.get("/api/chats/summaries")
    finally:
        event.remove(session, "loaded_as_persistent", loaded)
    assert response.status_code == 200
    rows = response.json()
    assert [row["id"] for row in rows] == ["visible-chat"]
    assert set(rows[0]) == {
        "id",
        "project_id",
        "title",
        "archived",
        "pinned",
        "created_at",
        "updated_at",
        "activity",
    }
    assert not hydrated
    assert "neutral-private" not in response.text
    assert "Neutral helper title" not in response.text
    assert rows[0]["activity"]["active_work_count"] == 0
    assert rows[0]["activity"]["unresolved_failed_count"] == 0


async def test_chat_summary_pages_keep_pins_and_archived_filter(
    session: Session, client: AsyncClient
) -> None:
    stamp = datetime(2026, 9, 1, tzinfo=UTC)
    session.add_all(
        [
            Chat(id="pinned", title="Pinned", pinned=True, updated_at=stamp),
            Chat(id="recent", title="Recent", updated_at=stamp + timedelta(seconds=2)),
            Chat(id="older", title="Older", updated_at=stamp + timedelta(seconds=1)),
            Chat(id="archived", title="Archived", archived=True, updated_at=stamp),
        ]
    )
    session.commit()
    first = await client.get("/api/chats/summaries", params={"limit": 2})
    second = await client.get("/api/chats/summaries", params={"limit": 2, "offset": 2})
    assert first.status_code == second.status_code == 200
    assert [row["id"] for row in first.json()] == ["pinned", "recent"]
    assert [row["id"] for row in second.json()] == ["older"]
    archived = await client.get("/api/chats/summaries", params={"include_archived": True})
    assert archived.status_code == 200
    assert {row["id"] for row in archived.json()} == {"pinned", "recent", "older", "archived"}


async def test_chat_summary_search_keeps_unicode_project_matches(
    session: Session, client: AsyncClient
) -> None:
    session.add(Project(id="project", name="ÉTUDE"))
    session.flush()
    session.add_all(
        [
            Chat(id="in-project", title="Color study", project_id="project"),
            Chat(id="other", title="Another study"),
        ]
    )
    session.commit()
    response = await client.get(
        "/api/chats/summaries", params={"query": "étude", "search_projects": True}
    )
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == ["in-project"]


async def test_chat_summary_page_size_is_bounded(client: AsyncClient) -> None:
    response = await client.get("/api/chats/summaries", params={"limit": 201})
    assert response.status_code == 422


def test_summary_reader_filters_before_paging_without_hydrating_chats(session: Session) -> None:
    stamp = datetime(2026, 9, 1, tzinfo=UTC)
    session.add(Project(id="project", name="ÉTUDE"))
    session.flush()
    session.add_all(
        [
            Chat(id="pinned", title="First", pinned=True, project_id="project", updated_at=stamp),
            Chat(
                id="recent",
                title="Second",
                project_id="project",
                updated_at=stamp + timedelta(seconds=2),
            ),
            Chat(
                id="archived",
                title="Archived",
                project_id="project",
                archived=True,
                updated_at=stamp,
            ),
            Chat(id="unrelated", title="Third", updated_at=stamp + timedelta(seconds=4)),
            Chat(id="hidden", title="ÉTUDE", scope="studio", origin_json={}),
        ]
    )
    session.commit()
    session.expunge_all()
    hydrated: list[object] = []

    def loaded(_session: Session, instance: object) -> None:
        hydrated.append(instance)

    event.listen(session, "loaded_as_persistent", loaded)
    try:
        first = list_chat_summary_rows(session, query="étude", search_projects=True, limit=1)
        second = list_chat_summary_rows(
            session, query="étude", search_projects=True, limit=1, offset=1
        )
        archived = list_chat_summary_rows(session, project_id="project", include_archived=True)
    finally:
        event.remove(session, "loaded_as_persistent", loaded)
    assert [row.id for row in first] == ["pinned"]
    assert [row.id for row in second] == ["recent"]
    assert {row.id for row in archived} == {"pinned", "recent", "archived"}
    assert not hydrated


async def test_populated_summary_pages_use_bounded_queries_without_loading_content(
    session: Session, client: AsyncClient
) -> None:
    stamp = datetime(2026, 9, 1, tzinfo=UTC)
    chats = [
        Chat(
            id=f"chat-{index}",
            title=f"Color study {index}",
            updated_at=stamp + timedelta(seconds=index),
            generation_settings_json={"sentinel": "neutral-private-settings"},
        )
        for index in range(210)
    ]
    session.add_all(chats)
    session.flush()
    for chat in chats:
        for attempt, status in enumerate(("complete", "complete", "failed", "queued")):
            identity = f"{chat.id}-{attempt}"
            user = Message(id=f"user-{identity}", chat_id=chat.id, role="user")
            answer = Message(id=f"answer-{identity}", chat_id=chat.id, role="assistant")
            session.add_all([user, answer])
            session.flush()
            run = Run(
                id=f"run-{identity}",
                chat_id=chat.id,
                user_message_id=user.id,
                assistant_message_id=answer.id,
                status=status,
            )
            session.add(run)
            session.flush()
            session.add(Job(id=f"job-{identity}", run_id=run.id, kind="chat", status=status))
            if status == "queued":
                continue
            revision = ResponseRevision(
                id=f"revision-{identity}",
                message_id=answer.id,
                run_id=run.id,
                sequence=1,
                status=status,
                parts=[
                    ResponseRevisionPart(position=0, type="text", text="neutral-private-response")
                ],
            )
            session.add(revision)
            session.flush()
            session.add(
                ChatActivityEvent(
                    id=f"event-{identity}",
                    chat_id=chat.id,
                    message_id=answer.id,
                    response_revision_id=revision.id,
                    job_id=f"job-{identity}",
                    attempt=1,
                    kind="failure" if status == "failed" else "output",
                    occurred_at=stamp + timedelta(seconds=attempt),
                )
            )
    session.commit()
    session.expunge_all()
    statements: list[str] = []
    hydrated: list[object] = []

    def statement(
        _connection: Connection,
        _cursor: object,
        sql: str,
        _parameters: object,
        _context: object,
        _many: bool,
    ) -> None:
        statements.append(sql)

    def loaded(_session: Session, instance: object) -> None:
        hydrated.append(instance)

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", statement)
    event.listen(session, "loaded_as_persistent", loaded)
    try:
        for size in (1, 50, 200):
            statements.clear()
            response = await client.get("/api/chats/summaries", params={"limit": size, "offset": 7})
            assert response.status_code == 200
            rows = response.json()
            assert [row["id"] for row in rows] == [
                f"chat-{index}" for index in range(202, 202 - size, -1)
            ]
            assert len(statements) <= 3
            assert all(sql.lstrip().upper().startswith("SELECT") for sql in statements)
            assert not hydrated and len(session.identity_map) == 0
            assert "neutral-private" not in response.text
            for row in rows:
                activity = row["activity"]
                assert activity["active_work_count"] == 1
                assert activity["unresolved_failed_count"] == 1
                assert activity["last_output"]["id"] == f"event-{row['id']}-1"
                assert activity["last_failure"]["id"] == f"event-{row['id']}-2"
    finally:
        event.remove(engine, "before_cursor_execute", statement)
        event.remove(session, "loaded_as_persistent", loaded)
