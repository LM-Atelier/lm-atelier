"""The artifact reference index stays exactly what the reference walk reads.

Each source table records a changed row in the same statement as the change,
whoever writes it, and draining reads that row again with the walk's own
readers. The stamp says when the index may be trusted at all: after anything
that could have changed rows unseen, it goes stale and the store is rebuilt.
These cases hold each part of that, and hold the index to the walk itself.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from sqlalchemy import create_engine, delete, text, update
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from local_lm import artifact_reference_extractors as extractors
from local_lm import artifact_reference_index
from local_lm.artifact_deletion_authority import (
    ArtifactReferenceDataError,
    open_reference_index_writer,
)
from local_lm.artifact_library import begin_artifact_write_fence, referenced_artifact_ids
from local_lm.artifact_library_schema import (
    ARTIFACT_METADATA_REFERENCE_KEYS,
    JSON_WRITE_TRIGGER_SQL,
    MAX_JSON_DEPTH,
)
from local_lm.artifact_reference_index import (
    MaintenanceStep,
    ReferenceIndexUncertified,
    audit_sample,
    certify_and_seed,
    drain_changes,
    indexed_references,
    maintain_reference_index,
    maintenance_step,
    prepare_reference_index,
    record_changes,
    stamp_state,
    start_reference_index,
)
from local_lm.artifact_reference_index_schema import (
    CREATE_INDEX_SQL,
    CREATE_TABLE_SQL,
    REQUIRED_TRIGGER_SQL,
    ROW_INVALID_SQL,
    trigger_name,
    with_keys,
)
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.models import (
    Artifact,
    Chat,
    Job,
    Message,
    MessagePart,
    MessageReference,
    Run,
    WorkPlan,
    WorkStep,
)

JOBS, RUNS, STEPS, REFERENCES, CHATS, ARTIFACTS = range(1, 7)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Session]:
    engine = create_engine(f"sqlite:///{tmp_path / 'index.sqlite3'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _artifact(session: Session, serial: int, *, metadata: dict[str, Any] | None = None) -> str:
    digest = f"{serial:064x}"
    session.add(
        Artifact(
            id=f"sha256:{digest}",
            sha256=digest,
            kind="image",
            media_type="image/png",
            size_bytes=1,
            relative_path=f"{digest[:2]}/{digest[2:4]}/{digest}",
            metadata_json=metadata or {},
        )
    )
    session.flush()
    return f"sha256:{digest}"


#: A job row written without the ORM, so its document is stored as given.
RAW_JOB = (
    "INSERT INTO jobs (id, kind, status, progress, phase, progress_json, queue_priority, "
    "payload_json, result_json, attempt, cancellable, created_at, updated_at) VALUES "
    "(?, 'chat', 'queued', 0, 'queued', '{}', 0, ?, '{}', 0, 1, "
    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
)


def _raw_job(session: Session, job_id: str, payload: str) -> None:
    session.connection().exec_driver_sql(RAW_JOB, (job_id, payload))


def _changes(session: Session) -> set[tuple[int, str]]:
    session.flush()
    return {
        (int(source), str(key))
        for source, key in session.execute(
            text("SELECT source, source_key FROM artifact_reference_changes")
        )
    }


def _edges(session: Session) -> set[tuple[str, int, str]]:
    return {
        (str(name), int(source), str(key))
        for name, source, key in session.execute(
            text("SELECT artifact_id, source, source_key FROM artifact_reference_edges")
        )
    }


def _drain(session: Session) -> None:
    session.flush()
    begin_artifact_write_fence(session)
    drain_changes(session.connection(), limit=None)
    session.commit()


def _settled(session: Session) -> None:
    """Drain whatever creating the fixture rows recorded."""

    _drain(session)
    assert _changes(session) == set()


def _current(session: Session) -> bool:
    return stamp_state(session.connection()).matches


def test_a_created_store_is_current_and_empty(store: Session) -> None:
    assert _current(store)
    assert _edges(store) == set()
    assert _changes(store) == set()


def test_each_source_records_its_inserts_watched_updates_and_deletes(store: Session) -> None:
    first, second, third, fourth = (_artifact(store, serial) for serial in range(1, 5))
    _settled(store)

    job = Job(payload_json={"artifact_id": first}, result_json={})
    store.add(job)
    assert _changes(store) == {(JOBS, job.id)}
    _drain(store)
    assert _edges(store) == {(first, JOBS, job.id)}
    job.result_json = {"nested": [{"artifact_ids": [second]}]}
    assert _changes(store) == {(JOBS, job.id)}
    _drain(store)
    assert _edges(store) == {(first, JOBS, job.id), (second, JOBS, job.id)}
    store.execute(update(Job).where(Job.id == job.id).values(status="completed"))
    assert _changes(store) == set(), "a column no document lives in records nothing"

    run = Run(
        chat_id="chat",
        user_message_id="user",
        assistant_message_id="assistant",
        provenance_json={
            "input_artifact_ids": [first],
            "outputs": [{"artifact_id": second, "poster_artifact_id": third}],
        },
        settings_json={"mask": {"artifact_id": fourth}},
    )
    step = WorkStep(
        plan_id="plan",
        ordinal=0,
        operation="image_edit",
        input_bindings_json=[{"artifact_id": second}],
        settings_json={"mask": {"artifact_id": third}},
    )
    reference = MessageReference(
        message_id="message",
        position=0,
        reference_subject_id="subject",
        mention_slug="subject",
        subject_name="Subject",
        subject_kind="person",
        artifact_ids_json=[fourth, first],
    )
    store.add_all((run, step, reference))
    assert _changes(store) == {(RUNS, run.id), (STEPS, step.id), (REFERENCES, reference.id)}
    _drain(store)
    assert {edge for edge in _edges(store) if edge[1] != JOBS} == {
        (first, RUNS, run.id),
        (second, RUNS, run.id),
        (third, RUNS, run.id),
        (fourth, RUNS, run.id),
        (second, STEPS, step.id),
        (third, STEPS, step.id),
        (fourth, REFERENCES, reference.id),
        (first, REFERENCES, reference.id),
    }

    store.delete(job)
    store.delete(step)
    assert _changes(store) == {(JOBS, job.id), (STEPS, step.id)}
    _drain(store)
    assert {edge[1] for edge in _edges(store)} == {RUNS, REFERENCES}


def test_only_a_studio_chat_names_its_source(store: Session) -> None:
    source = _artifact(store, 1)
    _settled(store)
    chat = Chat(title="Plain", origin_json={"source_artifact_id": source})
    store.add(chat)
    assert _changes(store) == set(), "a chat outside the studio names nothing"
    chat.title = "Renamed"
    chat.origin_json = {}
    assert _changes(store) == set()

    chat.scope = "studio"
    chat.origin_json = {"source_artifact_id": source}
    assert _changes(store) == {(CHATS, chat.id)}
    _drain(store)
    assert _edges(store) == {(source, CHATS, chat.id)}

    chat.scope = "standard"
    assert _changes(store) == {(CHATS, chat.id)}
    _drain(store)
    assert _edges(store) == set(), "leaving the studio drops what the chat named"


def test_an_artifact_names_its_poster_and_proxy(store: Session) -> None:
    poster = _artifact(store, 1)
    proxy = _artifact(store, 2)
    video = _artifact(
        store, 3, metadata={"poster_artifact_id": poster, "browser_proxy_artifact_id": proxy}
    )
    assert (ARTIFACTS, video) in _changes(store)
    _drain(store)
    assert _edges(store) == {(poster, ARTIFACTS, video), (proxy, ARTIFACTS, video)}


def test_core_raw_and_cascading_writes_are_recorded(store: Session, tmp_path: Path) -> None:
    named = _artifact(store, 1)
    chat = Chat(title="Owner")
    store.add(chat)
    store.flush()
    user = Message(chat_id=chat.id, role="user")
    assistant = Message(chat_id=chat.id, role="assistant")
    store.add_all((user, assistant))
    store.flush()
    run = Run(chat_id=chat.id, user_message_id=user.id, assistant_message_id=assistant.id)
    plan = WorkPlan(chat_id=chat.id, transcript_sequence=1)
    store.add_all((run, plan))
    store.flush()
    step = WorkStep(
        plan_id=plan.id,
        ordinal=0,
        operation="image_edit",
        input_bindings_json=[{"artifact_id": named}],
    )
    store.add(step)
    _settled(store)

    store.execute(
        update(Run).where(Run.id == run.id).values(provenance_json={"input_artifact_ids": [named]})
    )
    assert _changes(store) == {(RUNS, run.id)}, "a Core update is recorded"
    _drain(store)
    assert _edges(store) == {(named, RUNS, run.id), (named, STEPS, step.id)}

    # Another connection, with none of the application's functions, still
    # writes the source tables, and its change is still recorded.
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        raw.execute(RAW_JOB, ("raw_job", json.dumps({"artifact_id": named})))
    store.rollback()
    assert _changes(store) == {(JOBS, "raw_job")}
    _drain(store)
    assert (named, STEPS, step.id) in _edges(store)

    store.connection().exec_driver_sql("PRAGMA foreign_keys=ON")
    store.execute(delete(WorkPlan).where(WorkPlan.id == plan.id))
    assert _changes(store) == {(STEPS, step.id)}, "a cascade's deletions are recorded"
    _drain(store)
    assert _edges(store) == {(named, RUNS, run.id), (named, JOBS, "raw_job")}


def test_a_row_the_readers_refuse_names_only_the_unreadable_marker(store: Session) -> None:
    first = _artifact(store, 1)
    second = _artifact(store, 2)
    _settled(store)
    # Both names exist, so the write guard lets it through; SQLite reads both
    # copies of the repeated name while Python keeps only the last.
    _raw_job(store, "twice", f'{{"artifact_id": "{first}", "artifact_id": "{second}"}}')
    _drain(store)
    assert _edges(store) == {("", JOBS, "twice")}
    with pytest.raises(ArtifactReferenceDataError):
        indexed_references(store.connection())


@pytest.mark.parametrize(
    "document",
    ['{"artifact_id": NaN}', '{"a": {"b": 1, "b": 2}}', '{"artifact_ids": "not a list"}'],
)
def test_the_readers_refuse_what_the_walk_cannot_trust(document: str) -> None:
    assert extractors.row_edges(JOBS, (document, "{}")) == {extractors.UNDERIVABLE}


def test_a_lone_surrogate_id_is_unreadable_rather_than_unstorable() -> None:
    assert extractors.row_edges(REFERENCES, ('["\\ud800"]',)) == {extractors.UNDERIVABLE}


def _plant_too_deep(store: Session) -> str:
    """An artifact whose metadata only the database's own check refuses.

    The metadata nests past the depth the write guard allows, under a key no
    reader follows, so the Python readers find nothing wrong with it. It is
    stored with its guard dropped and then restored, as an older schema could
    have left it.
    """

    deep: object = None
    for _ in range(MAX_JSON_DEPTH):
        deep = [deep]
    metadata = json.dumps({"nested": deep})
    assert extractors.row_edges(ARTIFACTS, (metadata,)) == set()
    digest = "d" * 64
    guard = store.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE name = 'artifacts_artifact_reference_insert_guard'"
        )
    ).scalar_one()
    store.execute(text("DROP TRIGGER artifacts_artifact_reference_insert_guard"))
    store.execute(
        text(
            "INSERT INTO artifacts (id, sha256, kind, media_type, size_bytes, relative_path, "
            "metadata_json, favorite, created_at, updated_at) VALUES "
            "(:id, :sha256, 'image', 'image/png', 1, :path, :metadata, 0, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "id": f"sha256:{digest}",
            "sha256": digest,
            "path": f"dd/dd/{digest}",
            "metadata": metadata,
        },
    )
    store.execute(text(guard))
    store.commit()
    return f"sha256:{digest}"


def test_a_rebuild_marks_a_row_only_the_stored_document_check_refuses(store: Session) -> None:
    planted = _plant_too_deep(store)
    assert not _current(store), "dropping a guard moved the schema cookie"
    assert prepare_reference_index(store)
    # Only the database check refuses its depth, exactly as a deletion's
    # whole-store check did, and the rebuild runs that check row by row.
    assert ("", ARTIFACTS, planted) in _edges(store)
    with pytest.raises(ArtifactReferenceDataError):
        indexed_references(store.connection())


def test_a_document_the_walk_refuses_is_refused_by_the_index_too(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk's whole-store check and the rebuild's row check refuse the same rows.

    A deletion that trusted an index answering here would go ahead over a
    document the walk refuses, so an index that answers is a disagreement.
    """

    _plant_too_deep(store)
    assert prepare_reference_index(store)
    monkeypatch.setattr(artifact_reference_index, "shadow_comparisons", 0)
    monkeypatch.setattr(artifact_reference_index, "shadow_disagreements", 0)
    begin_artifact_write_fence(store)
    with pytest.raises(ArtifactReferenceDataError):
        referenced_artifact_ids(store, for_deletion=True)
    counts = (
        artifact_reference_index.shadow_comparisons,
        artifact_reference_index.shadow_disagreements,
    )
    assert counts == (1, 0)
    store.rollback()

    with open_reference_index_writer(store.connection()):
        store.execute(text("DELETE FROM artifact_reference_edges WHERE artifact_id = ''"))
    store.commit()
    begin_artifact_write_fence(store)
    with pytest.raises(ArtifactReferenceDataError):
        referenced_artifact_ids(store, for_deletion=True)
    assert artifact_reference_index.shadow_disagreements == 1


def test_the_stamp_goes_stale_when_anything_it_records_changes(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _current(store)
    monkeypatch.setattr(artifact_reference_index, "REFERENCE_INDEX_FORMAT", 2)
    assert not _current(store), "a reader format change"
    monkeypatch.undo()
    monkeypatch.setattr(sqlite3, "sqlite_version", "0.0.0")
    assert not _current(store), "another SQLite library"
    monkeypatch.undo()
    assert _current(store)
    store.execute(text("CREATE TABLE throwaway (id INTEGER)"))
    store.execute(text("DROP TABLE throwaway"))
    store.commit()
    assert not _current(store), "any DDL"
    assert prepare_reference_index(store)
    assert _current(store)
    store.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
    store.execute(text("INSERT INTO alembic_version VALUES ('later')"))
    store.commit()
    assert prepare_reference_index(store)
    assert stamp_state(store.connection()).current[3] == "later"
    store.execute(text("UPDATE alembic_version SET version_num = 'migrated'"))
    store.commit()
    assert not _current(store), "a migration"


def test_a_large_stale_store_is_rebuilt_in_the_background_not_at_startup(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    named = _artifact(store, 1)
    jobs = [Job(payload_json={"artifact_ids": [named]}, result_json={}) for _ in range(7)]
    store.add_all(jobs)
    store.execute(text("CREATE TABLE throwaway (id INTEGER)"))
    store.commit()
    monkeypatch.setattr(artifact_reference_index, "MAX_INLINE_REBUILD_ROWS", 3)
    monkeypatch.setattr(artifact_reference_index, "DRAIN_CHUNK", 4)

    assert prepare_reference_index(store) is False
    assert not _current(store)

    first = maintenance_step(store)
    assert first.rebuilt and first.drained == 4
    steps: list[MaintenanceStep] = [first]
    while steps[-1].remaining:
        steps.append(maintenance_step(store))
    assert not any(step.rebuilt for step in steps[1:]), "a rebuild in progress is not restarted"
    assert _current(store)
    assert _edges(store) == {(named, JOBS, job.id) for job in jobs}


def test_a_copy_through_the_backup_interface_is_rebuilt_not_trusted(
    store: Session, tmp_path: Path
) -> None:
    named = _artifact(store, 1)
    job = Job(payload_json={"artifact_id": named}, result_json={})
    store.add(job)
    _drain(store)
    copy_path = tmp_path / "copy.sqlite3"
    with sqlite3.connect(tmp_path / "index.sqlite3") as source, sqlite3.connect(copy_path) as copy:
        source.backup(copy)
        assert copy.execute("PRAGMA schema_version").fetchone() == (1,)
    engine = create_engine(f"sqlite:///{copy_path}")
    try:
        with Session(engine) as restored:
            assert not stamp_state(restored.connection()).matches
            restored.rollback()
            assert prepare_reference_index(restored)
            assert stamp_state(restored.connection()).schema_version > 1
            assert _current(restored)
            assert _edges(restored) == {(named, JOBS, job.id)}
    finally:
        engine.dispose()


def test_only_the_writer_window_may_change_the_index(store: Session, tmp_path: Path) -> None:
    named = _artifact(store, 1)
    store.add(Job(payload_json={"artifact_id": named}, result_json={}))
    _drain(store)
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        for statement in (
            "DELETE FROM artifact_reference_edges",
            "INSERT INTO artifact_reference_edges VALUES ('x', 1, 'y')",
            "DELETE FROM artifact_reference_changes",
            "UPDATE artifact_reference_index SET format_version = 1",
        ):
            with pytest.raises(sqlite3.OperationalError, match="no such function"):
                raw.execute(statement)
        # Recording a change needs no authority.
        raw.execute("INSERT INTO artifact_reference_changes VALUES (1, 'forged', 0)")
    store.rollback()
    for statement in (
        "DELETE FROM artifact_reference_edges",
        "UPDATE artifact_reference_edges SET artifact_id = 'other'",
        "DELETE FROM artifact_reference_changes",
        "UPDATE artifact_reference_changes SET audit = 1",
        "INSERT OR REPLACE INTO artifact_reference_index (id) VALUES (1)",
        "DELETE FROM artifact_reference_index",
    ):
        with pytest.raises(IntegrityError, match="not authorized"):
            store.execute(text(statement))
        store.rollback()
    with open_reference_index_writer(store.connection()):
        store.execute(text("UPDATE artifact_reference_changes SET audit = 1"))
    store.rollback()
    # The forged change costs a reading of a row that does not exist.
    _drain(store)
    assert _changes(store) == set()


def test_certification_refuses_a_guard_that_does_not_read_as_written(
    store: Session, tmp_path: Path
) -> None:
    weakened = JSON_WRITE_TRIGGER_SQL[0]
    name = trigger_name(weakened)
    store.execute(text(f"DROP TRIGGER {name}"))
    store.execute(text(f"CREATE TRIGGER {name}\nBEFORE INSERT ON jobs\nBEGIN\n  SELECT 1;\nEND"))
    store.commit()
    begin_artifact_write_fence(store)
    with pytest.raises(ReferenceIndexUncertified, match=name):
        certify_and_seed(store.connection())
    store.rollback()
    engine = store.get_bind()
    assert start_reference_index(lambda: Session(engine)) is False
    assert not _current(store)


def test_every_required_trigger_is_installed_as_written(store: Session) -> None:
    installed = {
        name: sql
        for name, sql in store.execute(
            text("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
        )
    }
    for statement in REQUIRED_TRIGGER_SQL:
        assert installed[trigger_name(statement)].strip() == statement.strip()


def _graph(store: Session) -> dict[str, str]:
    """A store naming artifacts every way the walk follows, with posters, a cycle and a gap."""

    ids = {name: _artifact(store, serial) for serial, name in enumerate("abcdefgh", start=1)}
    ids["video"] = _artifact(store, 20, metadata={"poster_artifact_id": ids["a"]})
    ids["loop_one"] = _artifact(store, 21)
    ids["loop_two"] = _artifact(store, 22, metadata={"poster_artifact_id": ids["loop_one"]})
    loop_one = store.get(Artifact, ids["loop_one"])
    assert loop_one is not None
    loop_one.metadata_json = {"browser_proxy_artifact_id": ids["loop_two"]}
    # Retained, and naming a poster that is retained through it.
    ids["kept_video"] = _artifact(store, 23, metadata={"poster_artifact_id": ids["b"]})
    store.add(Job(payload_json={"artifact_id": ids["kept_video"]}, result_json={}))
    store.add(Job(payload_json={}, result_json={"artifact_ids": [ids["loop_two"]]}))
    store.add(
        Run(
            chat_id="chat",
            user_message_id="user",
            assistant_message_id="assistant",
            provenance_json={"outputs": [{"artifact_id": ids["c"]}]},
        )
    )
    store.add(
        MessageReference(
            message_id="message",
            position=0,
            reference_subject_id="subject",
            mention_slug="subject",
            subject_name="Subject",
            subject_kind="person",
            artifact_ids_json=[ids["d"]],
        )
    )
    store.add(Chat(title="Studio", scope="studio", origin_json={"source_artifact_id": ids["e"]}))
    store.add(MessagePart(message_id="message", position=0, type="image", artifact_id=ids["f"]))
    store.flush()
    store.execute(
        text(
            "INSERT INTO artifact_library_entries (id, artifact_id, display_name, favorite, "
            "state, version, created_at, updated_at) VALUES (:id, :artifact, 'Kept', 0, "
            "'visible', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {"id": f"libentry:{ids['g']}", "artifact": ids["g"]},
    )
    store.commit()
    return ids


def test_the_index_answers_what_a_complete_walk_answers(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _graph(store)
    _drain(store)
    monkeypatch.setattr(artifact_reference_index, "shadow_comparisons", 0)
    begin_artifact_write_fence(store)
    walked = referenced_artifact_ids(store, for_deletion=True)
    assert artifact_reference_index.shadow_comparisons == 1, "the walk was compared"

    indexed = indexed_references(store.connection())
    assert indexed is not None
    assert indexed.retained == frozenset(walked)
    expected = {
        ids[name] for name in ("b", "c", "d", "e", "f", "g", "kept_video", "loop_one", "loop_two")
    }
    assert indexed.retained == expected, "h, the unreferenced video and its poster are free"
    assert indexed.namers[ids["a"]] == {ids["video"]}, "a free poster still has its namer"
    assert indexed.namers[ids["loop_one"]] == {ids["loop_two"]}


def test_recorded_changes_are_read_again_without_writing(store: Session) -> None:
    ids = _graph(store)
    _drain(store)
    store.add(Job(payload_json={"artifact_id": ids["h"]}, result_json={}))
    store.flush()
    connection = store.connection()
    driver = connection.connection.driver_connection
    assert isinstance(driver, sqlite3.Connection)
    before = driver.total_changes
    indexed = indexed_references(connection)
    assert indexed is not None
    assert ids["h"] in indexed.retained, "the undrained job is read again"
    assert driver.total_changes == before, "reading the index writes nothing"


def test_a_disagreement_with_the_walk_is_counted(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = _graph(store)
    _drain(store)
    with open_reference_index_writer(store.connection()):
        store.execute(
            text("DELETE FROM artifact_reference_edges WHERE artifact_id = :id"), {"id": ids["d"]}
        )
    monkeypatch.setattr(artifact_reference_index, "shadow_disagreements", 0)
    begin_artifact_write_fence(store)
    referenced_artifact_ids(store, for_deletion=True)
    assert artifact_reference_index.shadow_disagreements == 1


def test_the_audit_finds_an_entry_the_index_lost_and_has_it_read_again(store: Session) -> None:
    ids = _graph(store)
    _drain(store)
    with open_reference_index_writer(store.connection()):
        store.execute(
            text("DELETE FROM artifact_reference_edges WHERE artifact_id = :id"), {"id": ids["c"]}
        )
    store.commit()
    result, disagreed = audit_sample(store.connection(), {})
    assert (result.missing, result.extra) == (1, 0)
    assert list(disagreed) == [RUNS]
    begin_artifact_write_fence(store)
    record_changes(store.connection(), disagreed)
    store.commit()
    _drain(store)
    clean, again = audit_sample(store.connection(), {})
    assert (clean.missing, clean.extra, again) == (0, 0, {})
    assert clean.examined == result.examined


def test_startup_makes_a_small_migrated_store_current(settings: Settings) -> None:
    from local_lm import db
    from local_lm.db import configure_database

    configure_database(settings)
    try:
        assert start_reference_index(db.SessionLocal)
        with db.SessionLocal() as session:
            assert stamp_state(session.connection()).matches
    finally:
        db.engine.dispose()


def test_the_migration_adds_an_empty_untrusted_index_and_removes_it(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "2a07d89a3b46")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        guards = dict(
            connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
        )
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"artifact_reference_edges", "artifact_reference_changes"} <= tables
        assert connection.execute("SELECT * FROM artifact_reference_index").fetchall() == [
            (1, None, None, None, None)
        ]
        installed = dict(
            connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
        )
    for statement in REQUIRED_TRIGGER_SQL:
        assert installed[trigger_name(statement)].strip() == statement.strip()
    command.downgrade(config, "2a07d89a3b46")
    with sqlite3.connect(database) as connection:
        assert not {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'artifact_reference_%'"
            )
        }
        assert (
            dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"))
            == guards
        )
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert (
            dict(connection.execute("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"))
            == installed
        )


#: A digest of every reader the index derives entries with. When it changes,
#: what the index should hold may have changed: raise REFERENCE_INDEX_FORMAT
#: so every store rebuilds, and record the new digest with it.
READER_DIGEST = {1: "20c0cf22c34d23b55b83e6e9364c7bde9842706cc829ed8452d23f4c4dbc8b47"}


def test_the_reader_format_names_the_readers_it_was_built_with() -> None:
    readers = (
        extractors._mapping,
        extractors._ids,
        extractors._optional_id,
        extractors._run_ids,
        extractors._work_step_ids,
        extractors._settings_ids,
        extractors._job_ids,
        extractors._metadata_ids,
        extractors._unique_members,
        extractors._refuse_constant,
        extractors.parse_stored_json,
        extractors._encodable,
        extractors.document_edges,
        extractors.row_edges,
    )
    material = "\n".join(inspect.getsource(reader) for reader in readers)
    material += repr(
        (
            sorted(extractors._SCALAR_ARTIFACT_KEYS),
            sorted(extractors._LIST_ARTIFACT_KEYS),
            extractors.MAX_REFERENCE_LIST,
            extractors.MAX_REFERENCE_VALUES,
            extractors.MAX_REFERENCE_DEPTH,
            dict(extractors.SOURCE_COLUMNS),
            ARTIFACT_METADATA_REFERENCE_KEYS,
            extractors.UNDERIVABLE,
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    assert READER_DIGEST.get(artifact_reference_index.REFERENCE_INDEX_FORMAT) == digest


_SOURCE_TABLES = "jobs|runs|work_steps|message_references|chats|artifacts"


def test_nothing_writes_a_source_table_in_a_way_that_skips_its_record() -> None:
    """Nothing removes a source row in a way that leaves no record of it.

    REPLACE, in an INSERT or an UPDATE, deletes the row it conflicts with and
    fires no delete trigger, so the index would keep that row's names until it
    is next rebuilt. Turning triggers off would skip every record. Other
    conflict clauses are harmless: a record never conflicts with anything.
    """

    root = Path(artifact_reference_index.__file__).resolve().parent
    forbidden = re.compile(
        r"\b(?:REPLACE\s+INTO|INSERT\s+OR\s+REPLACE\s+INTO|UPDATE\s+OR\s+REPLACE)\s+"
        rf"[\"']?(?:{_SOURCE_TABLES})\b"
        r"|prefix_with\(\s*[\"']OR\s+REPLACE"
        r"|setconfig|recursive_triggers|writable_schema",
        re.IGNORECASE,
    )
    offending = [
        f"{path.relative_to(root)}:{number}"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if not line.lstrip().startswith("#") and forbidden.search(line)
    ]
    assert offending == []


async def test_upkeep_rebuilds_a_stale_store_and_then_says_it_is_ready(store: Session) -> None:
    named = _artifact(store, 1)
    jobs = [Job(payload_json={"artifact_ids": [named]}, result_json={}) for _ in range(3)]
    store.add_all(jobs)
    store.execute(text("CREATE TABLE throwaway (id INTEGER)"))
    store.commit()
    assert not _current(store)
    engine = store.get_bind()
    ready = asyncio.Event()
    upkeep = asyncio.create_task(
        maintain_reference_index(
            lambda: Session(engine), ready=ready, poll_seconds=0.01, pause_seconds=0
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=30)
    finally:
        upkeep.cancel()
        with suppress(asyncio.CancelledError):
            await upkeep
    store.rollback()
    assert _current(store)
    assert _edges(store) == {(named, JOBS, job.id) for job in jobs}


async def test_upkeep_waits_out_another_writer(store: Session, tmp_path: Path) -> None:
    store.execute(text("CREATE TABLE throwaway (id INTEGER)"))
    store.commit()
    assert not _current(store), "the store needs a rebuild, so upkeep needs the writer"
    other = sqlite3.connect(tmp_path / "index.sqlite3")
    engine = create_engine(f"sqlite:///{tmp_path / 'index.sqlite3'}", connect_args={"timeout": 0})
    ready = asyncio.Event()
    try:
        other.execute("BEGIN IMMEDIATE")
        upkeep = asyncio.create_task(
            maintain_reference_index(
                lambda: Session(engine), ready=ready, poll_seconds=0.01, pause_seconds=0
            )
        )
        await asyncio.sleep(0.3)
        assert not upkeep.done(), "a held writer is waited out, not fatal"
        assert not ready.is_set()
        other.rollback()
        await asyncio.wait_for(ready.wait(), timeout=30)
        upkeep.cancel()
        with suppress(asyncio.CancelledError):
            await upkeep
    finally:
        other.close()
        engine.dispose()
    store.rollback()
    assert _current(store)


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE artifacts SET id = :moved WHERE id = :planted",
        "UPDATE artifacts SET id = id WHERE id = :planted",
        "INSERT INTO artifact_reference_changes (source, source_key, audit) "
        "VALUES (6, :planted, 0)",
        "INSERT OR REPLACE INTO artifact_reference_changes (source, source_key, audit) "
        "VALUES (6, :planted, 0)",
    ],
)
def test_a_refused_row_stays_refused_whatever_another_connection_records(
    store: Session, tmp_path: Path, statement: str
) -> None:
    """No record another connection can make lifts a refusal the row's stored text earns.

    The write guards do not watch `id`, so a moved row's documents reach their
    new key unchecked, and any connection may add a record. Neither may drain
    the row without the stored-document check.
    """

    planted = _plant_too_deep(store)
    assert prepare_reference_index(store)
    assert ("", ARTIFACTS, planted) in _edges(store)
    moved = "sha256:" + "c" * 64
    # Another connection, with none of the application's functions.
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        raw.execute(statement, {"moved": moved, "planted": planted})
    store.rollback()
    _drain(store)
    holder = moved if ":moved" in statement else planted
    assert ("", ARTIFACTS, holder) in _edges(store)
    with pytest.raises(ArtifactReferenceDataError):
        indexed_references(store.connection())


def test_a_chat_leaving_the_studio_names_nothing_however_its_origin_is_stored(
    store: Session, tmp_path: Path
) -> None:
    source = _artifact(store, 1)
    chat = Chat(title="Studio", scope="studio", origin_json={"source_artifact_id": source})
    store.add(chat)
    _drain(store)
    assert _edges(store) == {(source, CHATS, chat.id)}
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        raw.execute(
            "UPDATE chats SET scope = 'standard', origin_json = 'not a document' WHERE id = ?",
            (chat.id,),
        )
    store.rollback()
    _drain(store)
    assert _edges(store) == set(), "a chat outside the studio is not read, so it is not refused"


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("jobs", "payload_json", '{"note": "changed"}'),
        ("jobs", "result_json", '{"note": "changed"}'),
        ("runs", "settings_json", '{"note": "changed"}'),
        ("runs", "provenance_json", '{"note": "changed"}'),
        ("work_steps", "settings_json", '{"note": "changed"}'),
        ("work_steps", "input_bindings_json", "[]"),
        ("message_references", "artifact_ids_json", "[]"),
        ("chats", "origin_json", '{"note": "changed"}'),
        ("chats", "scope", "standard"),
        ("artifacts", "metadata_json", '{"note": "changed"}'),
    ],
)
def test_every_column_a_reader_reads_records_its_row(
    store: Session, table: str, column: str, value: str
) -> None:
    named = _artifact(store, 1)
    sources = {
        "jobs": JOBS,
        "runs": RUNS,
        "work_steps": STEPS,
        "message_references": REFERENCES,
        "chats": CHATS,
        "artifacts": ARTIFACTS,
    }
    rows: dict[str, Any] = {
        "jobs": lambda: Job(payload_json={"artifact_id": named}, result_json={}),
        "runs": lambda: Run(
            chat_id="chat", user_message_id="user", assistant_message_id="assistant"
        ),
        "work_steps": lambda: WorkStep(plan_id="plan", ordinal=0, operation="image_edit"),
        "message_references": lambda: MessageReference(
            message_id="message",
            position=0,
            reference_subject_id="subject",
            mention_slug="subject",
            subject_name="Subject",
            subject_kind="person",
        ),
        "chats": lambda: Chat(title="Studio", scope="studio", origin_json={}),
    }
    key = named
    if table != "artifacts":
        row = rows[table]()
        store.add(row)
        store.flush()
        key = str(row.id)
    _settled(store)
    store.execute(
        text(f"UPDATE {table} SET {column} = :value WHERE id = :id"), {"value": value, "id": key}
    )
    assert _changes(store) == {(sources[table], key)}


def test_a_rebuild_still_being_checked_is_not_answered_from(store: Session) -> None:
    store.add(Job(payload_json={}, result_json={}))
    store.commit()
    begin_artifact_write_fence(store)
    certify_and_seed(store.connection())
    store.commit()
    assert _current(store), "the stamp is written with the rebuild's records"
    assert indexed_references(store.connection()) is None
    _drain(store)
    assert indexed_references(store.connection()) is not None


def test_the_row_check_refuses_by_document_shape_and_by_what_a_document_names(
    store: Session,
) -> None:
    """Both halves of the whole-store check, the structural and the semantic, run per row."""

    guard = store.execute(
        text("SELECT sql FROM sqlite_master WHERE name = 'jobs_artifact_reference_insert_guard'")
    ).scalar_one()
    store.execute(text("DROP TRIGGER jobs_artifact_reference_insert_guard"))
    deep: object = None
    for _ in range(MAX_JSON_DEPTH + 1):
        deep = [deep]
    _raw_job(store, "shape", json.dumps({"nested": deep}))
    _raw_job(store, "names", json.dumps({"artifact_id": 5}))
    _raw_job(store, "fine", json.dumps({"note": "nothing named"}))
    store.execute(text(guard))
    store.commit()
    keys = ("shape", "names", "fine")
    refused = set(
        store.connection()
        .exec_driver_sql(with_keys(ROW_INVALID_SQL[JOBS], len(keys)), keys)
        .scalars()
    )
    assert refused == {"shape", "names"}


@pytest.mark.parametrize(
    "dropped",
    ["runs_artifact_reference_change_update", "artifact_reference_edges_delete_authority"],
)
def test_certification_refuses_a_missing_change_or_authority_trigger(
    store: Session, dropped: str
) -> None:
    store.execute(text(f"DROP TRIGGER {dropped}"))
    store.commit()
    begin_artifact_write_fence(store)
    with pytest.raises(ReferenceIndexUncertified, match=dropped):
        certify_and_seed(store.connection())
    store.rollback()


def test_upkeep_rebuilds_a_backup_copy_itself(store: Session, tmp_path: Path) -> None:
    """A copy restored after startup, or one startup could not reset, is reset by upkeep."""

    named = _artifact(store, 1)
    job = Job(payload_json={"artifact_id": named}, result_json={})
    store.add(job)
    _drain(store)
    copy_path = tmp_path / "copy.sqlite3"
    with sqlite3.connect(tmp_path / "index.sqlite3") as source, sqlite3.connect(copy_path) as copy:
        source.backup(copy)
    engine = create_engine(f"sqlite:///{copy_path}")
    try:
        with Session(engine) as restored:
            step = maintenance_step(restored)
            assert step.rebuilt
            while step.remaining:
                step = maintenance_step(restored)
            assert stamp_state(restored.connection()).schema_version > 1
            assert _current(restored)
            assert _edges(restored) == {(named, JOBS, job.id)}
    finally:
        engine.dispose()


async def test_upkeep_counts_a_failure_that_is_not_another_writer(
    store: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def failing(_session: Session) -> MaintenanceStep:
        nonlocal calls
        calls += 1
        raise OperationalError("SELECT 1", {}, sqlite3.OperationalError("no such table: missing"))

    monkeypatch.setattr(artifact_reference_index, "maintenance_step", failing)
    monkeypatch.setattr(artifact_reference_index, "upkeep_failures", 0)
    engine = store.get_bind()
    upkeep = asyncio.create_task(
        maintain_reference_index(lambda: Session(engine), poll_seconds=0.01, pause_seconds=0)
    )
    try:
        async with asyncio.timeout(30):
            while artifact_reference_index.upkeep_failures == 0:
                await asyncio.sleep(0.01)
    finally:
        upkeep.cancel()
        with suppress(asyncio.CancelledError):
            await upkeep
    assert artifact_reference_index.upkeep_failures == 1


def test_the_index_tables_are_written_as_create_all_writes_them(store: Session) -> None:
    """The migration recognizes these tables by their text, so both ways write the same text."""

    stored = {
        name: str(sql)
        for name, sql in store.execute(
            text(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type IN ('table', 'index') AND name LIKE '%artifact_reference%'"
            )
        )
        if sql is not None
    }
    expected = {
        statement.split(None, 3)[2]: statement
        for statement in (*CREATE_TABLE_SQL, *CREATE_INDEX_SQL)
    }
    assert {name: sql.strip() for name, sql in stored.items()} == {
        name: sql.strip() for name, sql in expected.items()
    }


def _records(session: Session) -> set[tuple[int, str, object]]:
    session.flush()
    return {
        (int(source), str(key), audit)
        for source, key, audit in session.execute(
            text("SELECT source, source_key, audit FROM artifact_reference_changes")
        )
    }


@pytest.mark.parametrize("audit", ["", "x", b"", 2, -1, 0.5])
def test_a_record_of_any_type_but_zero_asks_for_the_check(
    store: Session, tmp_path: Path, audit: object
) -> None:
    """A record whose audit is anything but exactly 0 has its row checked.

    Any connection may record a change, with any value: SQLite keeps text and
    blobs as given and ranks them above every number, so a key's records are
    read as asking for the check unless every one of them is 0.
    """

    planted = _plant_too_deep(store)
    begin_artifact_write_fence(store)
    certify_and_seed(store.connection())
    store.commit()
    assert (ARTIFACTS, planted, 1) in _records(store), "the rebuild has not reached it yet"
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        raw.execute(
            "INSERT INTO artifact_reference_changes (source, source_key, audit) VALUES (6, ?, ?)",
            (planted, audit),
        )
    store.rollback()
    _drain(store)
    assert ("", ARTIFACTS, planted) in _edges(store)


def test_a_waiting_record_lifts_no_refusal_for_a_reader(store: Session, tmp_path: Path) -> None:
    """A reader reads a recorded row the way a drain would, so its marker still refuses."""

    planted = _plant_too_deep(store)
    assert prepare_reference_index(store)
    with sqlite3.connect(tmp_path / "index.sqlite3") as raw:
        raw.execute("UPDATE artifacts SET id = id WHERE id = ?", (planted,))
    store.rollback()
    assert (ARTIFACTS, planted) in _changes(store), "the record waits for the next drain"
    with pytest.raises(ArtifactReferenceDataError):
        indexed_references(store.connection())


def test_an_id_change_records_both_keys_and_checks_the_new_one(store: Session) -> None:
    named = _artifact(store, 1)
    job = Job(payload_json={"artifact_id": named}, result_json={})
    store.add(job)
    _drain(store)
    store.execute(text("UPDATE jobs SET id = 'moved' WHERE id = :id"), {"id": job.id})
    assert _records(store) == {(JOBS, "moved", 0), (JOBS, "moved", 1), (JOBS, job.id, 0)}
    _drain(store)
    assert _edges(store) == {(named, JOBS, "moved")}


def test_a_conflict_clause_on_a_source_write_has_no_record_to_act_on(store: Session) -> None:
    """Writes whose own conflict clause SQLite carries into the triggers still record."""

    named = _artifact(store, 1)
    job = Job(payload_json={"artifact_id": named}, result_json={})
    store.add(job)
    _drain(store)
    for value in ('{"note": "first"}', '{"note": "second"}'):
        store.execute(
            text("UPDATE OR ROLLBACK jobs SET payload_json = :value WHERE id = :id"),
            {"value": value, "id": job.id},
        )
    store.execute(
        text(
            "INSERT INTO jobs (id, kind, status, progress, phase, progress_json, queue_priority, "
            "payload_json, result_json, attempt, cancellable, created_at, updated_at) VALUES "
            "(:id, 'chat', 'queued', 0, 'queued', '{}', 0, :payload, '{}', 0, 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) "
            "ON CONFLICT (id) DO UPDATE SET payload_json = excluded.payload_json"
        ),
        {"id": job.id, "payload": json.dumps({"artifact_id": named})},
    )
    assert _changes(store) == {(JOBS, job.id)}
    _drain(store)
    assert _edges(store) == {(named, JOBS, job.id)}
