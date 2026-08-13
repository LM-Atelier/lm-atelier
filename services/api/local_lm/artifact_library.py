"""Durable Media Library membership and fail-closed artifact retention authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from .domain import ArtifactKind
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
    entry.favorite = favorite
    entry.version += 1
    artifact.favorite = favorite
    session.flush()
    return entry


def release_library_entry(session: Session, artifact: Artifact) -> bool:
    """Drop durable membership so an authorized cleanup may delete bytes.

    User-facing DELETE must not call this while Phase A has no Trash path;
    system cleanups that intentionally retire synthetic or chat-scoped media may.
    """

    entry = session.scalar(
        select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == artifact.id)
    )
    if entry is None:
        return False
    session.delete(entry)
    session.flush()
    return True


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
    for value in session.scalars(select(WorkStep.input_bindings_json)):
        retain(_work_step_ids(value))
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
