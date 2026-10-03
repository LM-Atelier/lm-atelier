"""Inspect a conversation's canonical attachments without copying them into recovery data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement, TextClause

from .artifact_library import (
    ArtifactReferenceDataError,
    _ids,
    _job_ids,
    _mapping,
    _optional_id,
    _run_ids,
    _settings_ids,
    _work_step_ids,
)
from .artifact_library_schema import ARTIFACT_METADATA_REFERENCE_KEYS
from .chat_recovery_schema import CHAT_OWNER_SQL, ROW_COLUMNS
from .db import Base
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollectionMembership,
    MediaTagAssignment,
    RecoveryItem,
)
from .recovery_previews import RecoverySnapshot
from .recovery_v1 import RecoveryConflict, RecoveryCountsV1

MAX_GRAPH_ROWS = 100_000
MAX_GRAPH_BYTES = 16 * 1024 * 1024
_TERMINAL = frozenset({"complete", "failed", "cancelled", "interrupted"})


class ChatRecoveryGraphError(ValueError):
    def __init__(self) -> None:
        super().__init__("chat-recovery-graph-invalid")


@dataclass(frozen=True)
class ChatRecoveryGraph:
    snapshot: RecoverySnapshot
    owned_job_ids: tuple[str, ...]
    artifact_ids: frozenset[str]
    rows: Mapping[str, tuple[Mapping[str, Any], ...]] = field(repr=False)


class _GraphReader:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.digest = hashlib.sha256(b"chat-recovery-canonical-graph-v1\0")
        self.row_count = 0
        self.byte_count = 0

    def read(
        self,
        name: str,
        predicate: ColumnElement[bool] | TextClause,
        parameters: dict[str, object] | None = None,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        table = Base.metadata.tables[name]
        statement = (
            (select(*(table.c[column] for column in columns)) if columns else select(table))
            .where(predicate)
            .order_by(*table.primary_key.columns)
        )
        result: list[Mapping[str, Any]] = []
        try:
            for row in self.session.execute(statement, parameters or {}).mappings():
                values = dict(row)
                normalized = {
                    key: value.isoformat() if isinstance(value, datetime) else value
                    for key, value in values.items()
                }
                encoded = json.dumps(
                    [name, normalized], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
                self.row_count += 1
                self.byte_count += len(encoded)
                if self.row_count > MAX_GRAPH_ROWS or self.byte_count > MAX_GRAPH_BYTES:
                    raise ChatRecoveryGraphError()
                self.digest.update(len(encoded).to_bytes(8, "big"))
                self.digest.update(encoded)
                result.append(values)
        except (TypeError, ValueError, RecursionError):
            raise ChatRecoveryGraphError() from None
        return tuple(result)


def _artifact_roots(rows: Mapping[str, tuple[Mapping[str, Any], ...]]) -> set[str]:
    found: set[str] = set()
    for name, column in (
        ("message_parts", "artifact_id"),
        ("response_revision_parts", "artifact_id"),
        ("chat_composer_draft_attachments", "artifact_id"),
        ("run_context_artifacts", "artifact_id"),
        ("setup_verifications", "input_artifact_id"),
    ):
        for row in rows[name]:
            found.update(_optional_id(row[column]))
    for row in rows["message_references"]:
        found.update(_ids(row["artifact_ids_json"]))
    for row in rows["runs"]:
        found.update(_run_ids(row["provenance_json"]))
        found.update(_settings_ids(row["settings_json"]))
    for row in rows["work_steps"]:
        found.update(_work_step_ids(row["input_bindings_json"]))
        found.update(_settings_ids(row["settings_json"]))
    for row in rows["jobs"]:
        found.update(_job_ids(row["payload_json"]))
        found.update(_job_ids(row["result_json"]))
    for row in rows["chats"]:
        if row["scope"] == "studio":
            found.update(_optional_id(_mapping(row["origin_json"]).get("source_artifact_id")))
    return found


def _owned_jobs(
    reader: _GraphReader, chat_id: str, jobs: tuple[Mapping[str, Any], ...]
) -> tuple[str, ...]:
    """A historical source link does not transfer another conversation's job ownership."""

    identifiers: dict[str, set[str]] = {"runs": set(), "work_plans": set(), "work_steps": set()}
    for job in jobs:
        for name, key in (
            ("runs", "run_id"),
            ("work_plans", "work_plan_id"),
            ("work_steps", "work_step_id"),
        ):
            identifiers[name].update(_optional_id(job[key]))
        payload = _mapping(job["payload_json"])
        if (
            all(job[key] is None for key in ("run_id", "work_plan_id", "work_step_id"))
            and payload.get("chat_id") is None
        ):
            identifiers["runs"].update(_optional_id(payload.get("source_run_id")))

    def read_parents(name: str) -> dict[str, Mapping[str, Any]]:
        parents: dict[str, Mapping[str, Any]] = {}
        table = Base.metadata.tables[name]
        values = sorted(identifiers[name])
        for start in range(0, len(values), 500):
            for row in reader.read(name, table.c.id.in_(values[start : start + 500])):
                parents[str(row["id"])] = row
        if len(parents) != len(values):
            raise ChatRecoveryGraphError()
        return parents

    steps = read_parents("work_steps")
    identifiers["work_plans"].update(str(step["plan_id"]) for step in steps.values())
    runs = read_parents("runs")
    plans = read_parents("work_plans")
    found: list[str] = []
    for job in jobs:
        owners: set[str] = set()
        for parents, key in ((runs, "run_id"), (plans, "work_plan_id")):
            if job[key] is not None:
                owners.add(str(parents[job[key]]["chat_id"]))
        if job["work_step_id"] is not None:
            owners.add(str(plans[steps[job["work_step_id"]]["plan_id"]]["chat_id"]))
        payload = _mapping(job["payload_json"])
        named_chat = payload.get("chat_id")
        if named_chat is not None:
            if not isinstance(named_chat, str) or not named_chat:
                raise ChatRecoveryGraphError()
            owners.add(named_chat)
        source_run_id = payload.get("source_run_id")
        if not owners and source_run_id is not None:
            if not isinstance(source_run_id, str):
                raise ChatRecoveryGraphError()
            owners.add(str(runs[source_run_id]["chat_id"]))
        if owners == {chat_id}:
            found.append(str(job["id"]))
    return tuple(found)


def inspect_chat_recovery_graph(session: Session, chat_id: str) -> ChatRecoveryGraph:
    """Read a bounded graph; its caller reserves the writer before using the result."""

    reader = _GraphReader(session)
    rows: dict[str, tuple[Mapping[str, Any], ...]] = {}
    with session.no_autoflush:
        for name, predicate in CHAT_OWNER_SQL.items():
            if ROW_COLUMNS[name] != tuple(Base.metadata.tables[name].columns.keys()):
                raise ChatRecoveryGraphError()
            rows[name] = reader.read(
                name, text(predicate.format(row=name, chat=":chat_id")), {"chat_id": chat_id}
            )
        if len(rows["chats"]) != 1:
            raise ChatRecoveryGraphError()
        reader.read(
            "recovery_items",
            (RecoveryItem.kind == "chat") & (RecoveryItem.subject_id == chat_id),
        )
        # A fork remains independently owned, but creating one changes the impact.
        reader.read(
            "chats",
            text("json_extract(chats.origin_json, '$.source_chat_id') = :chat_id"),
            {"chat_id": chat_id},
        )
        try:
            found = _artifact_roots(rows)
            visited: set[str] = set()
            retained_bytes = 0
            while frontier := sorted(found - visited):
                if len(found) > MAX_GRAPH_ROWS:
                    raise ChatRecoveryGraphError()
                for start in range(0, len(frontier), 500):
                    batch = frontier[start : start + 500]
                    artifacts = reader.read("artifacts", Artifact.id.in_(batch))
                    if len(artifacts) != len(batch):
                        raise ChatRecoveryGraphError()
                    for artifact in artifacts:
                        size = artifact["size_bytes"]
                        if type(size) is not int or size < 0:
                            raise ChatRecoveryGraphError()
                        retained_bytes += size
                        metadata = _mapping(artifact["metadata_json"])
                        for key in ARTIFACT_METADATA_REFERENCE_KEYS:
                            if key in metadata:
                                found.update(_optional_id(metadata[key]))
                    entries = reader.read(
                        "artifact_library_entries", ArtifactLibraryEntry.artifact_id.in_(batch)
                    )
                    entry_ids = [entry["id"] for entry in entries]
                    reader.read(
                        "media_collection_memberships",
                        MediaCollectionMembership.entry_id.in_(entry_ids),
                    )
                    reader.read("media_tag_assignments", MediaTagAssignment.entry_id.in_(entry_ids))
                    visited.update(batch)
            owned_jobs = _owned_jobs(reader, chat_id, rows["jobs"])
        except ArtifactReferenceDataError:
            raise ChatRecoveryGraphError() from None
        active_work = len(rows["turn_creation_claims"])
        for name in ("messages", "response_revisions", "runs", "work_plans", "work_steps", "jobs"):
            terminal = _TERMINAL | {"partial"} if name == "work_plans" else _TERMINAL
            active_work += sum(
                row["status"] not in terminal or (name == "jobs" and row["claim_owner"] is not None)
                for row in rows[name]
            )
        counts = RecoveryCountsV1(
            messages=len(rows["messages"]),
            message_parts=len(rows["message_parts"]),
            response_revisions=len(rows["response_revisions"]),
            runs=len(rows["runs"]),
            jobs=len(rows["jobs"]),
            work_plans=len(rows["work_plans"]),
            references=len(rows["message_references"]),
            artifacts=len(visited),
            active_work=active_work,
            retained_bytes=retained_bytes,
            # Every counted byte is still referenced by this canonical graph.
            reclaimable_bytes=0,
        )
        return ChatRecoveryGraph(
            snapshot=RecoverySnapshot(
                fingerprint=reader.digest.hexdigest(),
                counts=counts,
                conflicts=(RecoveryConflict.ACTIVE_WORK,) if active_work else (),
            ),
            owned_job_ids=owned_jobs,
            artifact_ids=frozenset(visited),
            rows=rows,
        )
