"""Retention windows chosen in Settings, and every clearing pass that uses them."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from annotated_types import Ge, Le
from httpx2 import AsyncClient

from local_lm import main as main_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, RetentionPolicy
from local_lm.retention_policy import MEDIA_SCOPE, read_policy, windows_for, write_policy
from local_lm.schemas import (
    MAX_RETENTION_DAYS,
    MAX_TEMPORARY_RETENTION_HOURS,
    RetentionWindowsIn,
)


def _unused_for(settings: Settings, days: int, label: str) -> str:
    """A picture nothing uses, which retention first found unused `days` ago."""

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session,
            label.encode(),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            metadata={"unreferenced_at": (datetime.now(UTC) - timedelta(days=days)).isoformat()},
        )
        session.commit()
        return artifact.id


def _preview_made(settings: Settings, hours: int, label: str) -> str:
    """A preview nothing uses, made `hours` ago."""

    store = ArtifactStore(settings)
    with SessionLocal() as session:
        artifact = store.ingest_bytes(
            session,
            label.encode(),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
            metadata={"temporary_preview": True},
        )
        artifact.created_at = datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=hours)
        session.commit()
        return artifact.id


def _kept(artifact_id: str) -> bool:
    with SessionLocal() as session:
        return session.get(Artifact, artifact_id) is not None


def _choose(settings: Settings, media_days: int, temporary_hours: int) -> None:
    with SessionLocal() as session:
        write_policy(
            session,
            settings,
            0,
            RetentionWindowsIn(media_days=media_days, temporary_hours=temporary_hours),
        )
        session.commit()


async def _put(client: AsyncClient, expected: int, days: Any, hours: Any) -> Any:
    return await client.put(
        "/api/artifacts/retention",
        json={"expected_revision": expected, "media_days": days, "temporary_hours": hours},
    )


async def test_the_installation_windows_apply_until_somebody_chooses(
    client: AsyncClient, settings: Settings
) -> None:
    response = await client.get("/api/artifacts/retention")

    assert response.status_code == 200
    assert response.json() == {
        "media_days": settings.artifact_retention_days,
        "temporary_hours": settings.temporary_retention_hours,
        "revision": 0,
        "default_media_days": settings.artifact_retention_days,
        "default_temporary_hours": settings.temporary_retention_hours,
    }


async def test_a_choice_is_kept_and_one_based_on_an_older_revision_is_refused(
    client: AsyncClient,
) -> None:
    first = await _put(client, 0, 7, 6)
    assert first.status_code == 200
    assert (first.json()["media_days"], first.json()["revision"]) == (7, 1)

    late = await _put(client, 0, 90, 48)
    assert late.status_code == 409
    assert late.json()["code"] == "retention-policy-stale"
    assert late.json()["current_revision"] == 1

    second = await _put(client, 1, 14, 12)
    assert second.status_code == 200 and second.json()["revision"] == 2
    stale = await _put(client, 1, 90, 48)
    assert stale.status_code == 409 and stale.json()["current_revision"] == 2

    kept = (await client.get("/api/artifacts/retention")).json()
    assert (kept["media_days"], kept["temporary_hours"], kept["revision"]) == (14, 12, 2)
    assert (kept["default_media_days"], kept["default_temporary_hours"]) == (30, 24)


def test_a_choice_is_reported_as_saved_by_a_session_that_read_the_one_before(
    settings: Settings,
) -> None:
    _choose(settings, 7, 24)

    with SessionLocal() as session:
        # Holding the row keeps it in the session, as a caller that had read it would.
        held = session.get(RetentionPolicy, MEDIA_SCOPE)
        assert held is not None and read_policy(session, settings).revision == 1
        chosen = write_policy(
            session, settings, 1, RetentionWindowsIn(media_days=14, temporary_hours=12)
        )
        assert (chosen.media_days, chosen.temporary_hours, chosen.revision) == (14, 12, 2)
        assert windows_for(session, settings) == (14, 12)
        session.commit()


@pytest.mark.parametrize(
    ("days", "hours"),
    [(0, 24), (3651, 24), (30, 0), (30, 169), ("30", 24), (30.5, 24), (30, None)],
)
async def test_windows_the_installation_could_not_set_are_refused(
    client: AsyncClient, days: Any, hours: Any
) -> None:
    response = await _put(client, 0, days, hours)

    assert response.status_code == 422
    assert (await client.get("/api/artifacts/retention")).json()["revision"] == 0


def test_the_bounds_are_the_installation_settings_own() -> None:
    def bounds(field: str) -> tuple[int, int]:
        metadata = Settings.model_fields[field].metadata
        low = next(item.ge for item in metadata if isinstance(item, Ge))
        high = next(item.le for item in metadata if isinstance(item, Le))
        assert isinstance(low, int) and isinstance(high, int)
        return int(low), int(high)

    assert bounds("artifact_retention_days") == (1, MAX_RETENTION_DAYS)
    assert bounds("temporary_retention_hours") == (1, MAX_TEMPORARY_RETENTION_HOURS)


async def test_storage_figures_and_clear_now_follow_the_chosen_windows(
    client: AsyncClient, settings: Settings
) -> None:
    unused = _unused_for(settings, 10, "unused for ten days")
    preview = _preview_made(settings, 8, "a preview from this morning")

    before = (await client.get("/api/artifacts/storage")).json()
    assert (before["eligible_count"], before["retention_days"]) == (0, 30)

    assert (await _put(client, 0, 7, 6)).status_code == 200
    after = (await client.get("/api/artifacts/storage")).json()
    assert after["eligible_count"] == 2
    assert (after["retention_days"], after["temporary_retention_hours"]) == (7, 6)

    cleared = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert cleared.status_code == 200 and cleared.json()["removed_count"] == 2
    assert not _kept(unused) and not _kept(preview)


async def test_a_preview_counts_proposed_windows_without_clearing_or_choosing(
    client: AsyncClient, settings: Settings
) -> None:
    unused = _unused_for(settings, 10, "unused for ten days")

    shorter = await client.post(
        "/api/artifacts/retention/preview", json={"media_days": 7, "temporary_hours": 24}
    )
    current = await client.post(
        "/api/artifacts/retention/preview", json={"media_days": 30, "temporary_hours": 24}
    )

    assert shorter.status_code == 200
    assert (shorter.json()["dry_run"], shorter.json()["removed_count"]) == (True, 1)
    assert current.json()["removed_count"] == 0
    assert _kept(unused)
    assert (await client.get("/api/artifacts/retention")).json()["revision"] == 0
    refused = await client.post(
        "/api/artifacts/retention/preview", json={"media_days": 0, "temporary_hours": 24}
    )
    assert refused.status_code == 422


async def test_clearing_at_start_follows_the_chosen_windows(settings: Settings) -> None:
    unused = _unused_for(settings, 10, "unused for ten days")
    store = ArtifactStore(settings)

    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0)
    assert _kept(unused)

    _choose(settings, 7, 24)
    await main_module.sweep_artifact_retention(store, settings, pause_seconds=0)
    assert not _kept(unused)


def test_a_choice_saved_while_a_pass_waits_for_the_writer_is_the_one_it_uses(
    settings: Settings,
) -> None:
    """The pass reads the windows after taking the writer, never before.

    The choice is committed at the last moment before the pass asks for the
    writer. A pass that had already read the windows would still hold the
    installation's thirty days and keep the file.
    """

    unused = _unused_for(settings, 10, "unused for ten days")
    store = ArtifactStore(settings)

    def choose_just_before_the_writer(name: str) -> None:
        if name == "acquire-writer":
            _choose(settings, 7, 24)

    with SessionLocal() as session:
        summary = store.cleanup_retention(
            session,
            windows_from=lambda held: windows_for(held, settings),
            dry_run=False,
            report_phase=choose_just_before_the_writer,
        )
        session.commit()

    assert summary.removed_count == 1
    assert not _kept(unused)


@pytest.mark.parametrize(
    "windows",
    [
        {},
        {"retention_days": 30},
        {"temporary_hours": 24},
        {"retention_days": 30, "temporary_hours": 24, "windows_from": lambda _held: (30, 24)},
        {"retention_days": 30, "windows_from": lambda _held: (30, 24)},
    ],
)
def test_a_pass_is_given_the_windows_one_way_exactly(
    settings: Settings, windows: dict[str, Any]
) -> None:
    with SessionLocal() as session, pytest.raises(TypeError):
        ArtifactStore(settings).cleanup_retention(session, dry_run=True, **windows)


async def test_diagnostics_report_the_windows_retention_uses(client: AsyncClient) -> None:
    assert (await _put(client, 0, 7, 6)).status_code == 200

    created = await client.post("/api/diagnostics")
    archive = await client.get(created.json()["url"])

    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        configuration = json.loads(bundle.read("diagnostics.json"))["configuration"]
    assert configuration["artifact_retention_days"] == 7
    assert configuration["temporary_retention_hours"] == 6
