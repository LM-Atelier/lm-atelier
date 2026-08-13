"""Durable Media Library membership and fail-closed artifact retention authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NoReturn, cast

from sqlalchemy import func, select, update
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from .domain import ArtifactKind, utcnow
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    MessagePart,
    MessageReference,
    ReferenceAsset,
    ReferenceSubject,
    ResponseRevisionPart,
    Run,
    SetupVerification,
    WorkStep,
)

MAX_REFERENCE_ROWS = 100_000
MAX_REFERENCE_LIST = 4_096
MAX_REFERENCE_VALUES = 100_000
MAX_REFERENCE_DEPTH = 16
REFERENCE_CORRUPT = "Stored artifact reference data is invalid."


class ArtifactReferenceDataError(RuntimeError):
    pass


class ArtifactLibraryConflict(ValueError):
    pass


def begin_artifact_write_fence(session: Session) -> None:
    """Take SQLite's writer reservation before proving deletion authority."""

    connection = session.connection()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")


def library_entry_id(artifact: Artifact) -> str:
    return f"libentry:sha256:{artifact.sha256}"


def ensure_library_entry(session: Session, artifact: Artifact) -> ArtifactLibraryEntry | None:
    """Publish durable image/video membership; generic ingest deliberately does not call this."""

    if artifact.kind not in {ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}:
        return None
    display_name = (artifact.original_name or "").strip() or artifact.sha256
    statement = (
        insert(ArtifactLibraryEntry)
        .values(
            id=library_entry_id(artifact),
            artifact_id=artifact.id,
            display_name=display_name,
            favorite=artifact.favorite,
            state="visible",
            deleted_at=None,
            recovery_id=None,
            version=1,
            created_at=artifact.created_at,
            updated_at=artifact.updated_at,
        )
        .on_conflict_do_nothing(index_elements=["artifact_id"])
    )
    session.execute(statement)
    entry = session.scalar(
        select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
    )
    if entry is None or entry.id != library_entry_id(artifact):
        raise RuntimeError("Artifact library membership is inconsistent.")
    return entry


def set_library_favorite(
    session: Session, artifact: Artifact, favorite: bool
) -> ArtifactLibraryEntry:
    entry = ensure_library_entry(session, artifact)
    if entry is None:
        raise ValueError("only image and video artifacts can enter the Media Library")
    desired = bool(favorite)
    observed_version = entry.version
    changed = cast(
        CursorResult[Any],
        session.execute(
            update(ArtifactLibraryEntry)
            .where(
                ArtifactLibraryEntry.id == entry.id,
                ArtifactLibraryEntry.version == observed_version,
                ArtifactLibraryEntry.favorite != desired,
            )
            .values(favorite=desired, version=observed_version + 1, updated_at=utcnow())
        ),
    )
    session.expire(entry)
    session.refresh(entry)
    if changed.rowcount != 1 and entry.favorite != desired:
        raise ArtifactLibraryConflict("Media Library entry changed; refresh and try again.")
    session.execute(update(Artifact).where(Artifact.id == artifact.id).values(favorite=desired))
    session.expire(artifact)
    session.refresh(artifact)
    return entry


def _fail() -> NoReturn:
    raise ArtifactReferenceDataError(REFERENCE_CORRUPT)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _fail()
    if len(value) > MAX_REFERENCE_LIST:
        _fail()
    return value


def _ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    if any(not isinstance(item, str) or not item or len(item) > 80 for item in value):
        _fail()
    return set(value)


def _optional_id(value: object) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, str) or not value or len(value) > 80:
        _fail()
    return {value}


def _run_ids(value: object) -> set[str]:
    row = _mapping(value)
    found: set[str] = set()
    for key in ("input_artifact_ids", "resolved_dependency_artifact_ids"):
        if key in row:
            found.update(_ids(row[key]))
    outputs = row.get("outputs")
    if outputs is not None:
        if not isinstance(outputs, list) or len(outputs) > MAX_REFERENCE_LIST:
            _fail()
        for output in outputs:
            item = _mapping(output)
            for key in ("artifact_id", "poster_artifact_id", "browser_proxy_artifact_id"):
                if key in item:
                    found.update(_optional_id(item[key]))
    return found


def _work_step_ids(value: object) -> set[str]:
    if not isinstance(value, list) or len(value) > MAX_REFERENCE_LIST:
        _fail()
    found: set[str] = set()
    for raw in value:
        item = _mapping(raw)
        if "artifact_id" in item:
            found.update(_optional_id(item["artifact_id"]))
    return found


def _settings_ids(value: object) -> set[str]:
    row = _mapping(value)
    mask = row.get("mask")
    if mask is None:
        return set()
    mask_row = _mapping(mask)
    if "artifact_id" not in mask_row:
        _fail()
    return _optional_id(mask_row["artifact_id"])


_SCALAR_ARTIFACT_KEYS = {
    "artifact_id",
    "source_artifact_id",
    "result_artifact_id",
    "input_artifact_id",
    "poster_artifact_id",
    "browser_proxy_artifact_id",
}
_LIST_ARTIFACT_KEYS = {"artifact_ids", "input_artifact_ids"}


def _job_ids(value: object, *, depth: int = 0, budget: list[int] | None = None) -> set[str]:
    if budget is None:
        budget = [0]
    budget[0] += 1
    if budget[0] > MAX_REFERENCE_VALUES or depth > MAX_REFERENCE_DEPTH:
        _fail()
    found: set[str] = set()
    if isinstance(value, Mapping):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for key, child in value.items():
            if not isinstance(key, str):
                _fail()
            if key in _SCALAR_ARTIFACT_KEYS:
                found.update(_optional_id(child))
            elif key in _LIST_ARTIFACT_KEYS:
                found.update(_ids(child))
            else:
                found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif isinstance(value, list):
        if len(value) > MAX_REFERENCE_LIST:
            _fail()
        for child in value:
            found.update(_job_ids(child, depth=depth + 1, budget=budget))
    elif value is not None and not isinstance(value, str | int | float | bool):
        _fail()
    return found


def _pending_json_reference_ids(session: Session) -> set[str]:
    """Parse every new or changed JSON reference producer before it is written."""

    found: set[str] = set()

    def retain(values: set[str]) -> None:
        found.update(values)
        if len(found) > MAX_REFERENCE_VALUES:
            _fail()

    for value in session.new.union(session.dirty):
        if isinstance(value, MessageReference):
            retain(_ids(value.artifact_ids_json or []))
        elif isinstance(value, Run):
            retain(_run_ids(value.provenance_json or {}))
            retain(_settings_ids(value.settings_json or {}))
        elif isinstance(value, WorkStep):
            retain(_work_step_ids(value.input_bindings_json or []))
            retain(_settings_ids(value.settings_json or {}))
        elif isinstance(value, Chat) and value.scope == "studio":
            retain(_optional_id(_mapping(value.origin_json).get("source_artifact_id")))
        elif isinstance(value, Job):
            retain(_job_ids(value.payload_json or {}))
            retain(_job_ids(value.result_json or {}))
        elif isinstance(value, Artifact):
            metadata = _mapping(value.metadata_json or {})
            for key in ("poster_artifact_id", "browser_proxy_artifact_id"):
                if key in metadata:
                    retain(_optional_id(metadata[key]))
    return found


def guard_artifact_reference_flush(
    session: Session,
    _flush_context: object,
    _instances: object,
) -> None:
    """Serialize JSON reference publication with deletion and refuse dangling ids."""

    referenced = _pending_json_reference_ids(session)
    deleted = {value.id for value in session.deleted if isinstance(value, Artifact)}
    if not referenced and not deleted:
        return
    begin_artifact_write_fence(session)
    if deleted & referenced_artifact_ids(session):
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT)
    available = {
        value.id for value in session.new if isinstance(value, Artifact) and value.id not in deleted
    }
    available.update(
        session.scalars(select(Artifact.id).where(Artifact.id.in_(sorted(referenced)))).all()
    )
    available.difference_update(deleted)
    if referenced - available:
        raise ArtifactReferenceDataError(REFERENCE_CORRUPT)


def referenced_artifact_ids(session: Session) -> set[str]:
    """Return the complete strong-reference graph or fail closed on corrupt JSON."""

    found: set[str] = set()
    counted_tables = (
        MessagePart,
        ResponseRevisionPart,
        ReferenceSubject,
        ReferenceAsset,
        SetupVerification,
        ArtifactLibraryEntry,
        MessageReference,
        Run,
        WorkStep,
        Chat,
        Job,
    )
    row_count = 0
    for table in counted_tables:
        row_count += session.scalar(select(func.count()).select_from(table)) or 0
        if row_count > MAX_REFERENCE_ROWS:
            _fail()

    def retain(values: set[str]) -> None:
        found.update(values)
        if len(found) > MAX_REFERENCE_VALUES:
            _fail()

    direct_columns = (
        MessagePart.artifact_id,
        ResponseRevisionPart.artifact_id,
        ReferenceSubject.cover_artifact_id,
        ReferenceAsset.artifact_id,
        SetupVerification.input_artifact_id,
        ArtifactLibraryEntry.artifact_id,
    )
    for column in direct_columns:
        retain({value for value in session.scalars(select(column)) if value})

    for value in session.scalars(select(MessageReference.artifact_ids_json)):
        retain(_ids(value))
    for value in session.scalars(select(Run.provenance_json)):
        retain(_run_ids(value))
    for value in session.scalars(select(Run.settings_json)):
        retain(_settings_ids(value))
    for value in session.scalars(select(WorkStep.input_bindings_json)):
        retain(_work_step_ids(value))
    for value in session.scalars(select(WorkStep.settings_json)):
        retain(_settings_ids(value))
    for scope, value in session.execute(select(Chat.scope, Chat.origin_json)):
        if scope == "studio":
            row = _mapping(value)
            retain(_optional_id(row.get("source_artifact_id")))
    for payload, result in session.execute(select(Job.payload_json, Job.result_json)):
        retain(_job_ids(payload))
        retain(_job_ids(result))

    pending = list(found)
    visited: set[str] = set()
    while pending:
        artifact_id = pending.pop()
        if artifact_id in visited:
            continue
        visited.add(artifact_id)
        if len(visited) > MAX_REFERENCE_VALUES:
            _fail()
        artifact = session.get(Artifact, artifact_id)
        if artifact is None:
            continue
        metadata = _mapping(artifact.metadata_json)
        for key in ("poster_artifact_id", "browser_proxy_artifact_id"):
            if key in metadata:
                linked = _optional_id(metadata[key])
                for linked_id in linked - found:
                    retain({linked_id})
                    pending.append(linked_id)
    return found
