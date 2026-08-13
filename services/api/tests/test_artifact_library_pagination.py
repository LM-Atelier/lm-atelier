from __future__ import annotations

from datetime import UTC, datetime, timedelta
from threading import Thread

import pytest
from httpx2 import AsyncClient
from sqlalchemy import event, text
from sqlalchemy.orm import Session

from local_lm import db
from local_lm.artifacts import ArtifactStore
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, ArtifactLibraryEntry


def _add_entry(
    index: int,
    created_at: datetime,
    *,
    kind: ArtifactKind = ArtifactKind.IMAGE,
    favorite: bool = False,
    state: str = "visible",
) -> tuple[str, str]:
    digest = f"{index:064x}"
    artifact_id = f"sha256:{digest}"
    entry_id = f"libentry:sha256:{digest}"
    deleted_at = created_at if state == "trashed" else None
    recovery_id = f"recovery:{digest}" if state == "trashed" else None
    with db.SessionLocal() as session:
        session.add(
            Artifact(
                id=artifact_id,
                sha256=digest,
                kind=kind.value,
                media_type="image/png" if kind is ArtifactKind.IMAGE else "video/mp4",
                size_bytes=index + 1,
                relative_path=f"{digest[:2]}/{digest[2:4]}/{digest}",
                original_name=f"private-original-{index}",
                metadata_json={"private": f"metadata-{index}"},
                favorite=favorite,
                created_at=created_at,
                updated_at=created_at,
            )
        )
        session.flush()
        session.add(
            ArtifactLibraryEntry(
                id=entry_id,
                artifact_id=artifact_id,
                display_name=f"Library item {index}",
                favorite=favorite,
                state=state,
                deleted_at=deleted_at,
                recovery_id=recovery_id,
                version=1,
                created_at=created_at,
                updated_at=created_at,
            )
        )
        session.commit()
    return entry_id, artifact_id


async def test_entry_feed_is_bounded_ordered_private_and_cursor_stable(
    client: AsyncClient,
) -> None:
    now = datetime(2026, 8, 12, 12, tzinfo=UTC)
    entry_1, _ = _add_entry(1, now)
    entry_2, _ = _add_entry(2, now)
    entry_3, _ = _add_entry(3, now - timedelta(minutes=1), favorite=True)
    entry_4, _ = _add_entry(
        4,
        now - timedelta(minutes=2),
        kind=ArtifactKind.VIDEO,
    )

    first = await client.get("/api/artifact-library", params={"limit": 2})
    assert first.status_code == 200
    body = first.json()
    assert [item["id"] for item in body["items"]] == [entry_2, entry_1]
    assert body["next_cursor"]
    assert set(body) == {"items", "next_cursor"}
    assert set(body["items"][0]) == {
        "id",
        "artifact_id",
        "version",
        "state",
        "display_name",
        "favorite",
        "kind",
        "media_type",
        "size_bytes",
        "created_at",
        "updated_at",
    }
    assert "private-original" not in first.text
    assert "metadata-" not in first.text
    assert "relative_path" not in first.text
    assert '"url"' not in first.text
    assert "recovery:" not in first.text

    _add_entry(5, now + timedelta(minutes=1))
    second = await client.get(
        "/api/artifact-library",
        params={"limit": 2, "cursor": body["next_cursor"]},
    )
    assert second.status_code == 200
    assert [item["id"] for item in second.json()["items"]] == [entry_3, entry_4]
    assert second.json()["next_cursor"] is None

    favorites = await client.get(
        "/api/artifact-library",
        params={"favorite": "true", "query": " ITEM 3 "},
    )
    assert favorites.status_code == 200
    assert [item["id"] for item in favorites.json()["items"]] == [entry_3]
    videos = await client.get("/api/artifact-library", params={"kind": "video"})
    assert [item["id"] for item in videos.json()["items"]] == [entry_4]
    literal_wildcard = await client.get("/api/artifact-library", params={"query": "%"})
    assert literal_wildcard.status_code == 200
    assert literal_wildcard.json()["items"] == []


async def test_feed_query_parameters_are_bounded_and_exact(client: AsyncClient) -> None:
    for params in (
        {"limit": 0},
        {"limit": 101},
        {"limit": "true"},
        {"kind": "other"},
        {"state": "all"},
        {"favorite": "1"},
        {"favorite": "TRUE"},
        {"query": "x" * 201},
    ):
        response = await client.get("/api/artifact-library", params=params)
        assert response.status_code == 422


async def test_cursor_tamper_cross_filter_and_missing_anchor_are_fixed_422(
    client: AsyncClient,
) -> None:
    now = datetime(2026, 8, 12, 13, tzinfo=UTC)
    anchor_id, _ = _add_entry(10, now)
    _add_entry(11, now - timedelta(minutes=1))
    first = await client.get("/api/artifact-library", params={"limit": 1})
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)

    tampered = cursor[:-1] + ("A" if cursor[-1] != "A" else "B")
    for params in (
        {"limit": 1, "cursor": tampered},
        {"limit": 2, "cursor": cursor},
        {"limit": 1, "cursor": cursor, "kind": "image"},
        {"limit": 1, "cursor": cursor, "favorite": "true"},
    ):
        response = await client.get("/api/artifact-library", params=params)
        assert response.status_code == 422
        assert response.json() == {
            "code": "artifact-library-cursor-invalid",
            "detail": (
                "The Media Library page request is invalid. Start again from the first page."
            ),
        }

    with db.SessionLocal() as session:
        session.execute(text("DROP TRIGGER artifact_library_entry_delete_guard"))
        entry = session.get(ArtifactLibraryEntry, anchor_id)
        assert entry is not None
        session.delete(entry)
        session.commit()
    missing = await client.get("/api/artifact-library", params={"limit": 1, "cursor": cursor})
    assert missing.status_code == 422
    assert missing.json()["code"] == "artifact-library-cursor-invalid"


async def test_trashed_entries_are_explicit_and_separate(
    client: AsyncClient,
) -> None:
    now = datetime(2026, 8, 12, 14, tzinfo=UTC)
    visible, _ = _add_entry(20, now)
    trashed, _ = _add_entry(21, now + timedelta(minutes=1), state="trashed")

    default_page = await client.get("/api/artifact-library")
    assert [item["id"] for item in default_page.json()["items"]] == [visible]
    trash_page = await client.get("/api/artifact-library", params={"state": "trashed"})
    assert trash_page.status_code == 200
    assert [item["id"] for item in trash_page.json()["items"]] == [trashed]
    assert trash_page.json()["items"][0]["state"] == "trashed"
    assert "recovery:" not in trash_page.text


async def test_corrupt_selected_lookahead_or_anchor_fails_without_partial_items(
    client: AsyncClient,
) -> None:
    now = datetime(2026, 8, 12, 14, 30, tzinfo=UTC)
    newest_entry, newest_artifact = _add_entry(25, now)
    _, lookahead_artifact = _add_entry(26, now - timedelta(minutes=1))
    with db.SessionLocal() as session:
        artifact = session.get(Artifact, lookahead_artifact)
        assert artifact is not None
        artifact.size_bytes = -1
        session.commit()

    corrupt_lookahead = await client.get("/api/artifact-library", params={"limit": 1})
    assert corrupt_lookahead.status_code == 409
    assert corrupt_lookahead.json() == {
        "code": "artifact-library-conflict",
        "detail": "The Media Library could not be read safely. Refresh and try again.",
    }
    assert newest_entry not in corrupt_lookahead.text

    with db.SessionLocal() as session:
        artifact = session.get(Artifact, lookahead_artifact)
        assert artifact is not None
        artifact.size_bytes = 27
        session.commit()
    first = await client.get("/api/artifact-library", params={"limit": 1})
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    with db.SessionLocal() as session:
        anchor = session.get(Artifact, newest_artifact)
        assert anchor is not None
        anchor.media_type = "private\nmarker"
        session.commit()
    corrupt_anchor = await client.get(
        "/api/artifact-library", params={"limit": 1, "cursor": cursor}
    )
    assert corrupt_anchor.status_code == 409
    assert corrupt_anchor.json()["code"] == "artifact-library-conflict"
    assert "private" not in corrupt_anchor.text


async def test_noncanonical_artifact_id_and_alias_path_fail_closed(
    client: AsyncClient,
) -> None:
    now = datetime(2026, 8, 12, 14, 45, tzinfo=UTC)
    digest = "a" * 64
    with db.SessionLocal() as session:
        session.add(
            Artifact(
                id="not-content-addressed",
                sha256=digest,
                kind=ArtifactKind.IMAGE.value,
                media_type="image/png",
                size_bytes=10,
                relative_path=f"aa/aa/{digest}",
                original_name="private-noncanonical-name.png",
                metadata_json={"private": "noncanonical-marker"},
                created_at=now,
                updated_at=now,
            )
        )
        session.flush()
        session.add(
            ArtifactLibraryEntry(
                id=f"libentry:sha256:{digest}",
                artifact_id="not-content-addressed",
                display_name="Noncanonical item",
                favorite=False,
                state="visible",
                version=1,
                created_at=now,
                updated_at=now,
            )
        )
        session.commit()

    invalid_id = await client.get("/api/artifact-library")
    assert invalid_id.status_code == 409
    assert invalid_id.json()["code"] == "artifact-library-conflict"
    assert "not-content-addressed" not in invalid_id.text
    assert "noncanonical-marker" not in invalid_id.text

    with db.SessionLocal() as session:
        session.execute(text("DROP TRIGGER artifact_library_entry_delete_guard"))
        entry = session.get(ArtifactLibraryEntry, f"libentry:sha256:{digest}")
        assert entry is not None
        session.delete(entry)
        artifact = session.get(Artifact, "not-content-addressed")
        assert artifact is not None
        session.delete(artifact)
        session.commit()
    valid_entry, valid_artifact = _add_entry(27, now)
    with db.SessionLocal() as session:
        artifact = session.get(Artifact, valid_artifact)
        assert artifact is not None
        artifact.relative_path = f"alias/{artifact.sha256}"
        session.commit()
    invalid_path = await client.get("/api/artifact-library")
    assert invalid_path.status_code == 409
    assert invalid_path.json()["code"] == "artifact-library-conflict"
    assert valid_entry not in invalid_path.text
    assert "alias" not in invalid_path.text


async def test_cursor_page_uses_one_snapshot_and_performs_no_writes_or_file_reads(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 12, 15, tzinfo=UTC)
    entry_1, _ = _add_entry(30, now)
    entry_2, _ = _add_entry(31, now - timedelta(minutes=1))
    first = await client.get("/api/artifact-library", params={"limit": 1})
    cursor = first.json()["next_cursor"]
    assert isinstance(cursor, str)
    assert first.json()["items"][0]["id"] == entry_1

    def forbidden_file_read(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("the Media Library feed must not open Artifact bytes")

    monkeypatch.setattr(ArtifactStore, "resolve", forbidden_file_read)
    monkeypatch.setattr(ArtifactStore, "verified_path", forbidden_file_read)
    monkeypatch.setattr(ArtifactStore, "delivery_metadata", forbidden_file_read)
    statements: list[str] = []
    commits = 0
    flushes = 0
    injected = False
    writer_errors: list[BaseException] = []
    request_connection: object | None = None

    def record_statement(
        connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal request_connection
        normalized = statement.strip()
        if request_connection is None and normalized.upper() == "BEGIN":
            request_connection = connection
        if connection is request_connection:
            statements.append(normalized)

    def count_commit(_session: Session) -> None:
        nonlocal commits
        commits += 1

    def count_flush(_session: Session, _context: object) -> None:
        nonlocal flushes
        flushes += 1

    def inject_between_reads(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        nonlocal injected
        if (
            injected
            or not statement.lstrip().upper().startswith("SELECT")
            or "artifact_library_entries.id =" not in statement
        ):
            return
        injected = True

        def writer() -> None:
            try:
                _add_entry(32, now - timedelta(seconds=30))
            except BaseException as exc:  # pragma: no cover - asserted below
                writer_errors.append(exc)

        thread = Thread(target=writer)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()

    event.listen(db.engine, "before_cursor_execute", record_statement)
    event.listen(db.engine, "after_cursor_execute", inject_between_reads)
    event.listen(Session, "after_commit", count_commit)
    event.listen(Session, "after_flush", count_flush)
    try:
        page = await client.get("/api/artifact-library", params={"limit": 1, "cursor": cursor})
    finally:
        event.remove(db.engine, "before_cursor_execute", record_statement)
        event.remove(db.engine, "after_cursor_execute", inject_between_reads)
        event.remove(Session, "after_commit", count_commit)
        event.remove(Session, "after_flush", count_flush)

    assert injected is True
    assert writer_errors == []
    assert page.status_code == 200
    assert [item["id"] for item in page.json()["items"]] == [entry_2]
    assert any(statement.upper() == "BEGIN" for statement in statements)
    assert all(
        statement.upper() == "BEGIN" or statement.upper().startswith("SELECT")
        for statement in statements
    )
    # The commit and two helper flushes belong to the deliberately concurrent writer.
    assert commits == 1
    assert flushes == 2

    refreshed = await client.get("/api/artifact-library", params={"limit": 100})
    assert [item["id"] for item in refreshed.json()["items"]][:3] == [
        entry_1,
        f"libentry:sha256:{32:064x}",
        entry_2,
    ]
