from __future__ import annotations

from collections.abc import AsyncIterator, Generator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx2 import ASGITransport, AsyncClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.orm import Session

from local_lm import api
from local_lm.db import Base
from local_lm.models import Job

STAMP = datetime(2026, 9, 1, tzinfo=UTC)


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
    app.include_router(api.router)

    async def override_session() -> AsyncIterator[Session]:
        yield session

    app.dependency_overrides[api.get_conversation_session] = override_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as value:
        yield value


def add_job(
    session: Session,
    identifier: str,
    status: str,
    *,
    kind: str = "image",
    created: int = 0,
    updated: int | None = None,
) -> None:
    session.add(
        Job(
            id=identifier,
            kind=kind,
            status=status,
            created_at=STAMP + timedelta(seconds=created),
            updated_at=STAMP + timedelta(seconds=created if updated is None else updated),
        )
    )


@pytest.mark.parametrize("status", ["queued", "running", "paused"])
async def test_recent_completed_history_does_not_hide_active_work(
    session: Session, client: AsyncClient, status: str
) -> None:
    add_job(session, "old-active", status)
    for index in range(101):
        add_job(session, f"complete-{index:03}", "complete", created=index + 1)
    session.commit()
    legacy = await client.get("/api/jobs")
    assert legacy.status_code == 200
    assert len(legacy.json()) == 100
    assert all(row["id"] != "old-active" for row in legacy.json())
    response = await client.get("/api/jobs/activity")
    assert response.status_code == 200
    result = response.json()
    assert result["active_count"] == 1
    assert [row["id"] for row in result["active"]] == ["old-active"]
    assert result["active"][0]["status"] == status
    assert result["recent_issues"] == []
    assert session.scalar(select(func.count(Job.id))) == 102
    active = session.get(Job, "old-active")
    assert active is not None and active.status == status


async def test_activity_limits_hydration_and_reports_all_active_jobs(
    session: Session, client: AsyncClient
) -> None:
    for index, status in enumerate(["running", "queued", "paused", "queued", "running"]):
        add_job(session, f"active-{index}", status, created=index)
    for index, status in enumerate(["running", "queued", "paused", "failed"]):
        add_job(session, f"hidden-{index}", status, kind="edit_verify", created=-10)
    for index in range(50):
        add_job(session, f"complete-{index}", "complete", created=index + 20)
    session.commit()
    session.expunge_all()
    loaded: list[str] = []

    def track(_session: Session, instance: object) -> None:
        if isinstance(instance, Job):
            loaded.append(instance.id)

    event.listen(session, "loaded_as_persistent", track)
    try:
        response = await client.get("/api/jobs/activity", params={"active_limit": 2})
    finally:
        event.remove(session, "loaded_as_persistent", track)
    assert response.status_code == 200
    result = response.json()
    assert result["active_count"] == 5
    assert [row["id"] for row in result["active"]] == ["active-0", "active-1"]
    assert result["recent_issues"] == []
    assert set(loaded) == {"active-0", "active-1"}
    expanded = await client.get("/api/jobs/activity", params={"active_limit": 5})
    assert [row["id"] for row in expanded.json()["active"]] == [f"active-{i}" for i in range(5)]


async def test_recent_issues_follow_update_time_and_do_not_displace_active_work(
    session: Session, client: AsyncClient
) -> None:
    add_job(session, "active", "queued", created=-100)
    add_job(session, "failed", "failed", created=20, updated=100)
    add_job(session, "cancelled", "cancelled", created=50, updated=110)
    add_job(session, "interrupted", "interrupted", created=1, updated=120)
    add_job(session, "older-issue", "failed", created=80, updated=90)
    add_job(session, "hidden-issue", "failed", kind="edit_verify", updated=200)
    session.commit()
    response = await client.get("/api/jobs/activity", params={"active_limit": 1})
    assert response.status_code == 200
    result = response.json()
    assert result["active_count"] == 1
    assert [row["id"] for row in result["active"]] == ["active"]
    assert [row["id"] for row in result["recent_issues"]] == ["interrupted", "cancelled", "failed"]


async def test_empty_and_tied_activity_is_deterministic(
    session: Session, client: AsyncClient
) -> None:
    response = await client.get("/api/jobs/activity")
    assert response.status_code == 200
    assert response.json() == {"active": [], "active_count": 0, "recent_issues": []}
    for identifier in ["active-c", "active-a", "active-b"]:
        add_job(session, identifier, "queued")
    session.commit()
    response = await client.get("/api/jobs/activity", params={"active_limit": 2})
    assert [row["id"] for row in response.json()["active"]] == ["active-a", "active-b"]


@pytest.mark.parametrize("limit", ["0", "501", "-1", "invalid"])
async def test_activity_limit_is_bounded(client: AsyncClient, limit: str) -> None:
    response = await client.get("/api/jobs/activity", params={"active_limit": limit})
    assert response.status_code == 422
