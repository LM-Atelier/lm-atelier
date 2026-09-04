"""The manual cleanup endpoint runs one bounded batch off the event loop.

POST /api/artifacts/cleanup used to run the whole retention sweep on the
event-loop thread inside the request's session: a press on a store with
thousands of aged previews froze every other request for the duration, and
contended with the background sweep for the write fence. It now runs the same
budgeted batch the sweep runs, in a worker thread, and says when more remains.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from local_lm import api as api_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact


def _aged_temporary(store: ArtifactStore, session: Session, index: int) -> Artifact:
    """A temporary preview two days old, unreferenced, and so eligible."""

    artifact = store.ingest_bytes(
        session,
        f"preview {index}".encode(),
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        metadata={"temporary_preview": True},
    )
    artifact.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=48)
    session.flush()
    return artifact


def _count(session: Session) -> int:
    return int(session.scalar(select(func.count(Artifact.id))) or 0)


def _seed(settings: Settings, count: int) -> None:
    store = ArtifactStore(settings)
    with SessionLocal() as session:
        for index in range(count):
            _aged_temporary(store, session, index)
        session.commit()


async def _cleanup(client: AsyncClient, *, dry_run: bool) -> tuple[int, bool]:
    response = await client.post("/api/artifacts/cleanup", json={"dry_run": dry_run})
    assert response.status_code == 200
    body = response.json()
    return body["removed_count"], body["truncated"]


async def test_a_real_run_is_one_bounded_batch_that_says_more_remains(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(settings, 5)
    monkeypatch.setattr(api_module, "RETENTION_BATCH_DELETIONS", 2)

    results = [await _cleanup(client, dry_run=False) for _ in range(3)]

    assert results == [(2, True), (2, True), (1, False)]
    with SessionLocal() as session:
        assert _count(session) == 0


async def test_a_dry_run_reports_the_whole_eligible_set_and_removes_nothing(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(settings, 5)
    monkeypatch.setattr(api_module, "RETENTION_BATCH_DELETIONS", 2)

    assert await _cleanup(client, dry_run=True) == (5, False)
    with SessionLocal() as session:
        assert _count(session) == 5


async def test_a_zero_time_budget_still_removes_one_per_call(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is consulted after the first deletion, never before it.

    A store whose fixed per-pass work alone exhausts the budget would otherwise
    answer every press with nothing removed. The last real deletion is still
    reported as truncated, because the bound stopped the run before the orphan
    walk could confirm that nothing remains; the call after it confirms.
    """

    _seed(settings, 3)
    monkeypatch.setattr(api_module, "RETENTION_BATCH_SECONDS", 0.0)

    results = [await _cleanup(client, dry_run=False) for _ in range(4)]

    assert results == [(1, True), (1, True), (1, True), (0, False)]


async def test_the_batch_runs_off_the_event_loop_for_both_modes(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(settings, 1)
    threads: list[int | None] = []
    real_cleanup = ArtifactStore.cleanup_retention

    def spying_cleanup(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return real_cleanup(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", spying_cleanup)

    assert await _cleanup(client, dry_run=True) == (1, False)
    assert await _cleanup(client, dry_run=False) == (1, False)

    assert len(threads) == 2
    assert threading.main_thread().ident not in threads


async def test_a_batch_that_raises_rolls_back_and_keeps_every_row(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(settings, 3)
    real_delete = ArtifactStore._delete_artifact
    deletions = 0

    def failing_delete(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        nonlocal deletions
        deletions += 1
        if deletions == 2:
            raise RuntimeError("the second deletion fails")
        return real_delete(self, *args, **kwargs)

    monkeypatch.setattr(ArtifactStore, "_delete_artifact", failing_delete)

    with pytest.raises(RuntimeError, match="second deletion"):
        await client.post("/api/artifacts/cleanup", json={"dry_run": False})

    with SessionLocal() as session:
        assert _count(session) == 3
    monkeypatch.setattr(ArtifactStore, "_delete_artifact", real_delete)
    assert await _cleanup(client, dry_run=False) == (3, False)
