"""Each retention sweep step reads only the artifact rows it needs.

A sweep batch read every artifact row as an object, metadata parsed, to recover
interrupted deletions even when nothing was staged, and again to list the paths
its file walk compares. On a large store that was fixed work in every batch,
done while the writer was held. The flush guard's existence check named every
pending id in one statement, and one flush may name more ids than a default
SQLite build lets one statement carry.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from local_lm.artifact_library import ArtifactReferenceDataError
from local_lm.artifacts import ArtifactStore
from local_lm.config import Settings
from local_lm.db import Base
from local_lm.domain import ArtifactKind
from local_lm.models import Artifact, Job


@pytest.fixture
def library(tmp_path: Path) -> Iterator[tuple[ArtifactStore, Session]]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    engine = create_engine(f"sqlite:///{tmp_path / 'library.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield ArtifactStore(settings), session
    finally:
        session.close()
        engine.dispose()


def _picture(store: ArtifactStore, session: Session, content: bytes) -> Artifact:
    return store.ingest_bytes(session, content, kind=ArtifactKind.IMAGE, media_type="image/png")


def _row(serial: int) -> Artifact:
    digest = f"{serial:064x}"
    return Artifact(
        id=f"sha256:{digest}",
        sha256=digest,
        kind="image",
        media_type="image/png",
        size_bytes=1,
        relative_path=f"{digest[:2]}/{digest[2:4]}/{digest}",
        metadata_json={},
    )


@dataclass
class _Trace:
    """What one sweep read, by the phase it reported while reading it."""

    statements: dict[str, list[tuple[str, int]]]
    loaded: dict[str, list[str]]

    def reading_artifacts(self, phase: str) -> list[tuple[str, int]]:
        return [
            (sql, parameters) for sql, parameters in self.statements[phase] if "artifacts" in sql
        ]


def _sweep(store: ArtifactStore, session: Session) -> _Trace:
    current = ["before"]
    trace = _Trace(defaultdict(list), defaultdict(list))

    def statement(
        _connection: Connection,
        _cursor: object,
        sql: str,
        parameters: tuple[object, ...] | None,
        _context: object,
        _many: bool,
    ) -> None:
        trace.statements[current[0]].append((sql, len(parameters or ())))

    def loaded(artifact: Artifact, _context: object) -> None:
        trace.loaded[current[0]].append(artifact.id)

    def report(name: str) -> None:
        current[0] = name

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", statement)
    event.listen(Artifact, "load", loaded)
    try:
        store.cleanup_retention(
            session,
            retention_days=30,
            temporary_hours=24,
            dry_run=False,
            report_phase=report,
        )
        session.commit()
    finally:
        event.remove(engine, "before_cursor_execute", statement)
        event.remove(Artifact, "load", loaded)
    return trace


@pytest.mark.parametrize("staging", ["absent", "empty"])
def test_recovering_staged_deletions_reads_no_artifact_row_when_nothing_is_staged(
    library: tuple[ArtifactStore, Session], staging: str
) -> None:
    store, session = library
    for content in (b"first picture", b"second picture", b"third picture"):
        _picture(store, session, content)
    session.commit()
    if staging == "empty":
        (store.root / ".delete-pending").mkdir()
    session.expunge_all()

    trace = _sweep(store, session)

    assert trace.reading_artifacts("recover-staged-deletions") == []
    assert trace.loaded["recover-staged-deletions"] == []
    assert not (store.root / ".delete-pending").exists()


def test_recovery_reads_only_the_rows_its_staged_files_name(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    interrupted = _picture(store, session, b"a deletion that never committed")
    for content in (b"an untouched picture", b"another untouched picture"):
        _picture(store, session, content)
    session.commit()
    original = store.resolve(interrupted)
    trash = store.root / ".delete-pending"
    trash.mkdir()
    os.replace(original, trash / f"{interrupted.sha256}.{'a' * 32}")
    committed = trash / f"{'c' * 64}.{'b' * 32}"
    committed.write_bytes(b"bytes whose row is already gone")
    session.expunge_all()

    trace = _sweep(store, session)

    reads = trace.reading_artifacts("recover-staged-deletions")
    assert [parameters for _sql, parameters in reads] == [2]
    assert trace.loaded["recover-staged-deletions"] == [interrupted.id]
    assert original.read_bytes() == b"a deletion that never committed"
    assert not committed.exists()
    assert not trash.exists()


def test_recovery_restores_every_staged_file_across_lookup_chunks(
    library: tuple[ArtifactStore, Session],
) -> None:
    # One interrupted batch can stage more files than one lookup names. A row a
    # later lookup missed would read as a committed deletion and lose its bytes.
    store, session = library
    rows = [_row(serial) for serial in range(1, 502)]
    session.add_all(rows)
    session.commit()
    trash = store.root / ".delete-pending"
    trash.mkdir()
    for row in rows:
        (trash / f"{row.sha256}.{'a' * 32}").write_bytes(row.sha256.encode())
    committed = trash / f"{'f' * 64}.{'b' * 32}"
    committed.write_bytes(b"bytes whose row is already gone")
    session.expunge_all()

    trace = _sweep(store, session)

    reads = trace.reading_artifacts("recover-staged-deletions")
    assert sorted(parameters for _sql, parameters in reads) == [2, 500]
    for row in rows:
        assert (store.root / row.relative_path).read_bytes() == row.sha256.encode()
    assert not committed.exists()
    assert not trash.exists()


def test_the_file_walk_reads_only_the_paths_it_compares(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    kept = _picture(store, session, b"an indexed picture")
    session.commit()
    content = b"bytes no row points at"
    unindexed = store.root / "ab" / "cd" / ("abcd" + "0" * 60)
    unindexed.parent.mkdir(parents=True)
    unindexed.write_bytes(content)
    aged = (datetime.now(UTC) - timedelta(hours=25)).timestamp()
    os.utime(unindexed, (aged, aged))

    trace = _sweep(store, session)

    reads = trace.reading_artifacts("cleanup-orphan-files")
    assert len(reads) == 1
    selected = reads[0][0].split("FROM", 1)[0]
    assert "relative_path" in selected
    assert "metadata_json" not in selected
    assert not unindexed.exists()
    assert store.resolve(kept).read_bytes() == b"an indexed picture"


def test_the_flush_guard_checks_pending_ids_a_bounded_chunk_at_a_time(
    library: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library
    rows = [_row(serial) for serial in range(1, 1202)]
    session.add_all(rows)
    session.commit()
    checks: list[int] = []

    def statement(
        _connection: Connection,
        _cursor: object,
        sql: str,
        parameters: tuple[object, ...] | None,
        _context: object,
        _many: bool,
    ) -> None:
        if "FROM artifacts" in sql and "artifacts.id IN" in sql:
            checks.append(len(parameters or ()))

    engine = session.get_bind()
    event.listen(engine, "before_cursor_execute", statement)
    try:
        session.add(Job(payload_json={}, result_json={"artifact_ids": [row.id for row in rows]}))
        session.commit()
        named = sorted(checks)
        checks.clear()

        # The last chunk is checked as closely as the first: an id with no
        # artifact, sorting after every other, is still refused.
        missing = "sha256:" + "f" * 64
        session.add(
            Job(payload_json={}, result_json={"artifact_ids": [row.id for row in rows] + [missing]})
        )
        with pytest.raises(ArtifactReferenceDataError):
            session.flush()
        session.rollback()
    finally:
        event.remove(engine, "before_cursor_execute", statement)

    assert named == [201, 500, 500]
    assert sorted(checks) == [202, 500, 500]
