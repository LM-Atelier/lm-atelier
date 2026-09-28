"""Keep the reference checks that run under the database writer to what they need.

A flush that recorded an artifact reference walked every stored reference
before it could write, and a deletion re-validated every stored document with
a depth rule whose cost grew with the square of each document. Both held the
writer, so on a large library every other writer waited seconds and could give
up. These cases hold the repairs: a flush that deletes nothing does not walk the
graph; the walk reads each level a chunk at a time and still honours what the
session has not flushed; and the deletion-time check measures depth in
proportion to the document while agreeing exactly with the triggers' rule.
"""

from __future__ import annotations

import json
import random
import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from local_lm import artifact_library
from local_lm.artifact_deletion_authority import ArtifactDeletionProofError
from local_lm.artifact_library import (
    ArtifactReferenceDataError,
    begin_artifact_write_fence,
    referenced_artifact_ids,
)
from local_lm.artifact_library_schema import (
    MAX_JSON_DEPTH,
    STORED_JSON_INVALID_SQL,
    _json_tree_depth,
    _member_depth,
)
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


def _bytes(store: ArtifactStore, session: Session, content: bytes) -> Artifact:
    return store.ingest_bytes(
        session, content, kind=ArtifactKind.OTHER, media_type="application/octet-stream"
    )


def _row(serial: int, *, kind: str = "image", metadata: dict[str, Any] | None = None) -> Artifact:
    digest = f"{serial:064x}"
    return Artifact(
        id=f"sha256:{digest}",
        sha256=digest,
        kind=kind,
        media_type="video/mp4" if kind == "video" else "image/png",
        size_bytes=1,
        relative_path=f"{digest[:2]}/{digest[2:4]}/{digest}",
        metadata_json=metadata or {},
    )


def _nested(levels: int) -> object:
    value: object = None
    for _ in range(levels):
        value = [value]
    return value


def _plant_metadata(session: Session, serial: int, metadata: object) -> None:
    """Store an artifact whose metadata the write trigger would refuse or slow down."""

    trigger = session.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE name = 'artifacts_artifact_reference_insert_guard'"
        )
    ).scalar_one()
    session.execute(text("DROP TRIGGER artifacts_artifact_reference_insert_guard"))
    row = _row(serial)
    session.execute(
        text(
            "INSERT INTO artifacts (id, sha256, kind, media_type, size_bytes, relative_path, "
            "metadata_json, favorite, created_at, updated_at) VALUES "
            "(:id, :sha256, 'image', 'image/png', 1, :path, :metadata, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "id": row.id,
            "sha256": row.sha256,
            "path": row.relative_path,
            "metadata": json.dumps(metadata),
        },
    )
    session.execute(text(trigger))
    session.commit()


def test_a_flush_that_deletes_nothing_does_not_walk_the_reference_graph(
    library: tuple[ArtifactStore, Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    store, session = library
    target = _bytes(store, session, b"named by a job")
    spare = _bytes(store, session, b"named by nothing")
    session.commit()
    walks: list[str] = []
    walk = artifact_library.referenced_artifact_ids

    def counted(*args: Any, **kwargs: Any) -> Any:
        walks.append("walk")
        return walk(*args, **kwargs)

    monkeypatch.setattr(artifact_library, "referenced_artifact_ids", counted)

    session.add(Job(payload_json={}, result_json={"artifact_ids": [target.id]}))
    session.commit()
    assert walks == []

    # Naming bytes that do not exist is still refused, without the walk.
    session.add(Job(payload_json={}, result_json={"artifact_ids": ["sha256:" + "0" * 64]}))
    with pytest.raises(ArtifactReferenceDataError):
        session.flush()
    session.rollback()
    assert walks == []

    # Deleting is what the graph answers for, and a deleting flush still asks.
    session.delete(spare)
    session.commit()
    assert walks == ["walk"]


def test_the_walk_follows_every_link_across_chunks(
    library: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library
    count = 1203  # more than two chunks of the reads the walk makes
    posters = [_row(serial) for serial in range(1, count + 1)]
    videos = [
        _row(10_000 + index, kind="video", metadata={"poster_artifact_id": poster.id})
        for index, poster in enumerate(posters)
    ]
    session.add_all(posters)
    session.flush()
    session.add_all(videos)
    session.flush()
    session.add(Job(payload_json={}, result_json={"artifact_ids": [video.id for video in videos]}))
    session.commit()
    session.expunge_all()

    graph = referenced_artifact_ids(session)

    assert {video.id for video in videos} <= graph
    assert {poster.id for poster in posters} <= graph


def test_a_change_the_session_has_not_flushed_still_counts_in_the_walk(
    library: tuple[ArtifactStore, Session],
) -> None:
    store, session = library
    old_poster = _bytes(store, session, b"old poster")
    new_poster = _bytes(store, session, b"new poster")
    video = _row(1, kind="video", metadata={"poster_artifact_id": old_poster.id})
    session.add(video)
    session.flush()
    session.add(Job(payload_json={}, result_json={"artifact_id": video.id}))
    session.commit()

    # One flush moves the retained video to a new poster and deletes the old
    # one. The stored video still names the old poster until the flush writes,
    # so only the session's own copy says it is free.
    video.metadata_json = {"poster_artifact_id": new_poster.id}
    session.delete(old_poster)
    session.commit()

    assert session.get(Artifact, old_poster.id) is None
    assert video.id in referenced_artifact_ids(session)
    assert new_poster.id in referenced_artifact_ids(session)


def test_the_deletion_time_check_holds_the_depth_bound_exactly(
    library: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library
    # Root object, then the key's list, then one level per nested list.
    _plant_metadata(session, 1, {"nested": _nested(MAX_JSON_DEPTH - 1)})
    begin_artifact_write_fence(session)
    referenced_artifact_ids(session, for_deletion=True)
    session.rollback()

    _plant_metadata(session, 2, {"nested": _nested(MAX_JSON_DEPTH)})
    begin_artifact_write_fence(session)
    with pytest.raises(ArtifactDeletionProofError, match="invalid"):
        referenced_artifact_ids(session, for_deletion=True)
    session.rollback()


def _instructions(session: Session, sql: str) -> int:
    """How many thousand SQLite virtual-machine steps one statement takes."""

    driver = session.connection().connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    steps = [0]

    def count() -> int:
        steps[0] += 1
        return 0

    driver.set_progress_handler(count, 1000)
    try:
        session.connection().exec_driver_sql(sql).scalar_one()
    finally:
        driver.set_progress_handler(None, 0)
    return steps[0]


def test_the_deletion_time_check_costs_in_proportion_to_the_documents(
    library: tuple[ArtifactStore, Session],
) -> None:
    _store, session = library
    costs = []
    for serial, size in ((1, 250), (2, 1000)):
        _plant_metadata(session, serial, {"items": [0] * size})
        costs.append(_instructions(session, STORED_JSON_INVALID_SQL))
        session.execute(text("DELETE FROM artifacts"))
        session.commit()

    # Four times the document should cost about four times the steps. The
    # depth rule the triggers use costs about sixteen times here, which is
    # what held the writer for seconds on a large library.
    assert costs[1] < 8 * costs[0], costs


def _depth(value: object, level: int = 0) -> int:
    children: list[object] = []
    if isinstance(value, dict):
        children = list(value.values())
    elif isinstance(value, list):
        children = list(value)
    return max([level, *(_depth(child, level + 1) for child in children)])


def _document(randomness: random.Random, level: int, limit: int) -> object:
    if level >= limit:
        return randomness.choice([None, True, 0, 2.5, "", "a.b[c]", 'q"x', "üñ"])
    side = [
        randomness.choice([None, 1, "s", {}, [], {"k": [1, {"z": None}]}])
        for _ in range(randomness.choice([0, 1, 2]))
    ]
    spine = _document(randomness, level + 1, limit)
    if randomness.random() < 0.5:
        items = [*side, spine]
        randomness.shuffle(items)
        return items
    keys = ["a", "b.c", "d[0]", 'q"uote', "sp ace", "ü", "", "$", "[]"]
    members = {randomness.choice(keys) + str(index): value for index, value in enumerate(side)}
    members[randomness.choice(keys) + "spine"] = spine
    return members


@pytest.mark.parametrize("rule", [_json_tree_depth, _member_depth])
def test_both_depth_rules_measure_the_same_depth(rule: Callable[[str], str]) -> None:
    randomness = random.Random(20260923)
    documents = [
        _document(randomness, 0, limit) for limit in range(MAX_JSON_DEPTH + 4) for _ in range(12)
    ]
    documents = [value if isinstance(value, (dict, list)) else [value] for value in documents]
    documents += [{}, [], [[]], {"a": {}}, [None] * 20, {"k": [[[[]]]]}]
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, value TEXT)")
        connection.executemany(
            "INSERT INTO documents (value) VALUES (?)",
            [(json.dumps(document),) for document in documents],
        )
        measured = connection.execute(
            f"SELECT {rule('documents.value')} FROM documents ORDER BY id"
        ).fetchall()
    finally:
        connection.close()

    assert [depth for (depth,) in measured] == [_depth(document) for document in documents]
