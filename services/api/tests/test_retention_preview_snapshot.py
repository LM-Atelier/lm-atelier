"""Retention previews stay responsive and cannot borrow old eligibility counts."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy.orm import Session
from test_retention_policy import _kept, _unused_for

from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings


async def test_clearing_old_eligible_files_does_not_mean_shortening_has_no_new_effect(
    client: AsyncClient, settings: Settings
) -> None:
    old = [_unused_for(settings, 40, f"Neutral old file {index}") for index in range(2)]
    newly_eligible = _unused_for(settings, 10, "Neutral retained file")
    displayed = (await client.get("/api/artifacts/storage")).json()
    assert displayed["eligible_count"] == 2

    cleared = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleared.status_code == 200 and cleared.json()["removed_count"] == 2
    assert all(not _kept(identifier) for identifier in old)
    assert _kept(newly_eligible)
    current = (await client.get("/api/artifacts/storage")).json()
    proposed = await client.post(
        "/api/artifacts/retention/preview", json={"media_days": 7, "temporary_hours": 24}
    )
    assert proposed.status_code == 200
    assert current["eligible_count"] == 0
    assert proposed.json()["removed_count"] == 1 <= displayed["eligible_count"]
    assert (await client.get("/api/artifacts/retention")).json()["revision"] == 0

    chosen = await client.put(
        "/api/artifacts/retention",
        json={"expected_revision": 0, "media_days": 7, "temporary_hours": 24},
    )
    assert chosen.status_code == 200
    next_clear = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert next_clear.status_code == 200 and next_clear.json()["removed_count"] == 1
    assert not _kept(newly_eligible)


async def test_proposed_retention_scanning_uses_the_existing_off_thread_cleanup_boundary(
    client: AsyncClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _unused_for(settings, 10, "Neutral preview thread check")
    event_loop_thread = threading.get_ident()
    calls: list[int] = []
    original = ArtifactStore.cleanup_retention

    def observed(self: ArtifactStore, session: Session, **kwargs: Any) -> Any:
        calls.append(threading.get_ident())
        return original(self, session, **kwargs)

    monkeypatch.setattr(ArtifactStore, "cleanup_retention", observed)
    response = await client.post(
        "/api/artifacts/retention/preview", json={"media_days": 7, "temporary_hours": 24}
    )
    assert response.status_code == 200 and response.json()["removed_count"] == 1
    assert _kept(artifact)
    assert len(calls) == 1
    assert event_loop_thread not in calls
