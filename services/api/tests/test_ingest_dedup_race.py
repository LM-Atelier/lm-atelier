from __future__ import annotations

from pathlib import Path
from typing import IO, Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm import artifacts as artifacts_module
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact

PAYLOAD = b"bytes that already exist in the store"


@pytest.fixture
def store_and_engine(tmp_path: Path) -> tuple[ArtifactStore, Any]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{settings.data_dir / 'ingest.sqlite3'}")
    Base.metadata.create_all(engine)
    return ArtifactStore(settings), engine


def _make_sweepable(store: ArtifactStore, engine: Any) -> None:
    """Ingest one artifact and bring it to the point of being swept."""

    with Session(engine) as session:
        store.ingest_bytes(
            session,
            PAYLOAD,
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
        session.commit()
        # The first pass only marks it unreferenced, exactly as production does.
        store.cleanup_retention(session, retention_days=30, temporary_hours=24, dry_run=False)
        session.commit()


def test_a_sweep_cannot_run_between_publication_and_the_row_that_protects_it(
    store_and_engine: tuple[ArtifactStore, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window that made a deduplicated ingest unsafe is now closed.

    `ingest_stream` publishes the file under its digest and only then looks the
    row up, and on the deduplication path it could return having written nothing:
    no insert, no update, no write fence. A sweep that had already decided the
    artifact was unreferenced could therefore delete the row and the freshly
    republished bytes in between, and the request would return an Artifact that
    resolved to nothing - or, as measured, miss its own lookup and insert a new
    row pointing at a file that had just been deleted.

    The reservation is now taken before publication and held through the row
    work, so a sweep cannot be inside that window at all. The assertion is that
    the sweep is REFUSED the writer slot there. In production the sweep is a
    different thread and simply waits for the ingest to commit; driving it from
    this thread surfaces the same exclusion immediately, which is what makes the
    test deterministic rather than a race the runner might win.
    """

    store, engine = store_and_engine
    _make_sweepable(store, engine)

    real_publish = ArtifactStore._publish_under_its_digest
    sweep_excluded: list[bool] = []

    def publish_then_sweep(
        self: ArtifactStore, source: IO[bytes], session: Session
    ) -> tuple[str, int]:
        published = real_publish(self, source, session)
        with Session(engine) as sweeper:
            try:
                store.cleanup_retention(sweeper, retention_days=0, temporary_hours=0, dry_run=False)
                sweeper.commit()
                sweep_excluded.append(False)
            except OperationalError:
                sweeper.rollback()
                sweep_excluded.append(True)
        return published

    monkeypatch.setattr(ArtifactStore, "_publish_under_its_digest", publish_then_sweep)

    with Session(engine) as session:
        returned = store.ingest_bytes(
            session,
            PAYLOAD,
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
        session.commit()
        returned_id = returned.id

    assert sweep_excluded == [True], (
        "the sweep was able to run between publication and the row that protects "
        "the bytes, which is the window this reservation exists to close"
    )

    with Session(engine) as session:
        row = session.get(Artifact, returned_id)
        assert row is not None, "the request's artifact row did not survive"
        assert store.resolve(row).is_file(), "the request's artifact bytes did not survive"


def test_the_sweep_still_removes_a_genuinely_unreferenced_artifact(
    store_and_engine: tuple[ArtifactStore, Any],
) -> None:
    """The control. A reservation that blocked the sweep outright would pass the
    test above and break retention entirely, so the ordinary path is asserted
    too: with no ingest in flight, an eligible artifact is still swept.
    """

    store, engine = store_and_engine
    _make_sweepable(store, engine)

    with Session(engine) as session:
        summary = store.cleanup_retention(
            session, retention_days=0, temporary_hours=0, dry_run=False
        )
        session.commit()

    assert summary.removed_count >= 1
    with Session(engine) as session:
        assert session.scalars(select(Artifact)).all() == []


def test_the_reservation_is_held_before_the_bytes_appear_not_merely_before_the_row(
    store_and_engine: tuple[ArtifactStore, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Where the reservation is taken is the guarantee, not that it is taken.

    Taking it anywhere inside publication makes the test above pass, because by
    the time that hook runs the reservation is held either way. The defect this
    closes lives in the gap between the bytes appearing under their digest and
    anything holding the writer slot, so the probe has to be inside that gap:
    it runs at the moment of the rename that publishes them.

    A mutation moving the reservation after publication left the earlier test
    green, which is why this one exists.

    `rename_entry` is patched on the artifacts module rather than on
    filesystem_links, because that module imported the name and a caller
    resolves its own global.
    """

    store, engine = store_and_engine
    _make_sweepable(store, engine)

    real_rename = artifacts_module.rename_entry
    excluded_at_publication: list[bool] = []

    def rename_then_probe(*args: Any, **kwargs: Any) -> Any:
        published = real_rename(*args, **kwargs)
        with Session(engine) as sweeper:
            try:
                store.cleanup_retention(sweeper, retention_days=0, temporary_hours=0, dry_run=False)
                sweeper.commit()
                excluded_at_publication.append(False)
            except OperationalError:
                sweeper.rollback()
                excluded_at_publication.append(True)
        return published

    monkeypatch.setattr(artifacts_module, "rename_entry", rename_then_probe)

    with Session(engine) as session:
        returned = store.ingest_bytes(
            session,
            PAYLOAD,
            kind=ArtifactKind.IMAGE,
            media_type="image/png",
        )
        session.commit()
        returned_id = returned.id

    assert excluded_at_publication == [True], (
        "a sweep ran at the moment the bytes were published, so the reservation "
        "is being taken after publication rather than before it"
    )

    with Session(engine) as session:
        row = session.get(Artifact, returned_id)
        assert row is not None
        assert store.resolve(row).is_file()
