from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm import artifact_library
from local_lm import artifacts as artifacts_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind


@pytest.fixture
def sweepable(tmp_path: Path) -> tuple[ArtifactStore, Session]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{settings.data_dir / 'retention.sqlite3'}")
    Base.metadata.create_all(engine)
    store = ArtifactStore(settings)
    with Session(engine) as session:
        yield store, session


def test_the_sweep_walks_the_reference_graph_once_not_once_per_deletion(
    sweepable: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep must pay for the reference graph ONCE, through every path.

    cleanup_retention computes the graph once before its loop. Passing that
    snapshot into _delete_artifact removes one recompute per deletion, but it is
    not sufficient on its own: _delete_artifact flushes for every artifact, and
    the registered before_flush listener independently walks the same graph each
    time. An earlier candidate was rejected for exactly that (codex/R2326), and
    its test could not see the problem because it patched only the ArtifactStore
    wrapper while the listener resolves the module-level function in
    artifact_library.

    So this counts BOTH bindings. Each module resolves its own global, so
    patching one proves nothing about the other, and counting only the wrapper is
    what made the previous instrument false.
    """

    store, session = sweepable
    for index in range(8):
        store.ingest_bytes(
            session,
            f"disposable {index}".encode(),
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
    session.commit()

    # first pass only marks them unreferenced; nothing is eligible yet
    store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()

    calls = 0
    real = artifact_library.referenced_artifact_ids

    def counted(*args: Any, **kwargs: Any) -> set[str]:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_library, "referenced_artifact_ids", counted)
    monkeypatch.setattr(artifacts_module, "referenced_artifact_ids", counted)

    summary = store.cleanup_retention(session, retention_days=0, temporary_hours=0, dry_run=False)
    session.commit()

    assert summary.removed_count == 8, "the sweep must actually delete, or nothing is measured"
    assert calls == 1, (
        f"the reference graph was walked {calls} times to delete 8 artifacts; it must "
        f"be walked once. A count near 9 means the flush listener is still walking it "
        f"per deletion behind a snapshot that only the wrapper honours."
    )


async def test_repeated_cleanup_distinguishes_newly_marked_from_total_pending(
    client: AsyncClient,
) -> None:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("orphan.bin", b"recoverable", "application/octet-stream")},
    )
    assert uploaded.status_code == 201

    first = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert first.status_code == 200
    assert first.json()["marked_count"] == 1
    assert first.json()["retention_pending_count"] == 1
    assert first.json()["removed_count"] == 0

    repeated = await client.post("/api/artifacts/cleanup", json={"dry_run": False})
    assert repeated.status_code == 200
    assert repeated.json()["marked_count"] == 0
    assert repeated.json()["retention_pending_count"] == 1
    assert repeated.json()["removed_count"] == 0

    storage = await client.get("/api/artifacts/storage")
    assert storage.status_code == 200
    assert storage.json()["retention_pending_count"] == 1
    assert storage.json()["eligible_count"] == 0


def test_a_poster_bearing_artifact_does_not_buy_a_second_graph_walk(
    sweepable: tuple[ArtifactStore, Session],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sweep's own final flush must lend the snapshot too.

    Marking `unreferenced_at` makes an artifact dirty. If its metadata names a
    poster then `_pending_json_reference_ids` is non-empty at the sweep's final
    flush, so the listener fires there as well as on the deletion flushes. That
    is a constant extra walk rather than the per-deletion N+1 that was rejected,
    but "the graph is computed once per sweep" is not true while it happens, and
    a claim that is nearly true is the kind that gets believed.

    Found by Codex before routing, at codex/R2330.
    """

    store, session = sweepable
    poster = store.ingest_bytes(
        session,
        b"poster target",
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
    )
    artifact = store.ingest_bytes(
        session,
        b"poster bearing",
        kind=ArtifactKind.VIDEO,
        media_type="video/mp4",
    )
    # The poster must EXIST. Naming an absent one is a genuinely dangling
    # reference and the listener is right to refuse it, which is a different
    # test from this one.
    artifact.metadata_json = {**artifact.metadata_json, "poster_artifact_id": poster.id}
    session.commit()

    calls = 0
    real = artifact_library.referenced_artifact_ids

    def counted(*args: Any, **kwargs: Any) -> set[str]:
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(artifact_library, "referenced_artifact_ids", counted)
    monkeypatch.setattr(artifacts_module, "referenced_artifact_ids", counted)

    store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
    session.commit()

    assert calls == 1, (
        f"one sweep walked the reference graph {calls} times. A count of 2 means the "
        f"sweep's own final flush is outside the lend, so a dirty poster-bearing "
        f"artifact buys an extra whole-graph walk."
    )
