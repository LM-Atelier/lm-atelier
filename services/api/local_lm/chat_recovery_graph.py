"""Inspect a conversation's canonical attachments without copying them into recovery data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import Result, select, text
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

# Distinct rows read, and the canonical bytes an inspection keeps in memory.
MAX_GRAPH_ROWS = 100_000
MAX_GRAPH_BYTES = 16 * 1024 * 1024
# Canonical bytes hashed, each row once: how long one inspection holds the reserved writer.
MAX_GRAPH_READ_BYTES = 24 * 1024 * 1024
_TERMINAL = frozenset({"complete", "failed", "cancelled", "interrupted"})
# The only owned columns read after inspection; every column is still hashed.
_KEEP: Mapping[str, tuple[str, ...]] = {
    "chats": ("id",),
    "messages": ("id", "role", "status"),
    "message_parts": ("message_id", "type", "artifact_id"),
    "response_revisions": ("id", "message_id", "status"),
    "response_revision_parts": ("response_revision_id", "type", "artifact_id"),
    "runs": ("id", "status"),
    "work_plans": ("id", "status"),
    "work_steps": ("id", "status"),
    "jobs": ("id", "status", "claim_owner", "run_id", "work_plan_id", "work_step_id"),
}
_PARENT_KEEP: Mapping[str, tuple[str, ...]] = {
    "runs": ("id", "chat_id"),
    "work_plans": ("id", "chat_id"),
    "work_steps": ("id", "plan_id"),
}
_ARTIFACT_COLUMNS: Mapping[str, str] = {
    "message_parts": "artifact_id",
    "response_revision_parts": "artifact_id",
    "chat_composer_draft_attachments": "artifact_id",
    "run_context_artifacts": "artifact_id",
    "setup_verifications": "input_artifact_id",
}


class ChatRecoveryGraphError(ValueError):
    def __init__(self, code: str = "chat-recovery-graph-invalid") -> None:
        super().__init__(code)


class ChatRecoveryGraphTooLarge(ChatRecoveryGraphError):
    """A valid conversation larger than one inspection may read under the writer."""

    def __init__(self) -> None:
        super().__init__("chat-recovery-graph-too-large")


@dataclass(frozen=True)
class ChatRecoveryGraph:
    snapshot: RecoverySnapshot
    owned_job_ids: tuple[str, ...]
    artifact_ids: frozenset[str]
    # Only the columns later code reads, never whole rows.
    rows: Mapping[str, tuple[Mapping[str, Any], ...]] = field(repr=False)


def _encoded(name: str, values: Mapping[str, Any]) -> bytes:
    normalized = {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in values.items()
    }
    return json.dumps(
        [name, normalized], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class _GraphReader:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.digest = hashlib.sha256(b"chat-recovery-canonical-graph-v1\0")
        self.row_count = 0
        self.byte_count = 0
        self.read_bytes = 0

    def read(
        self,
        name: str,
        predicate: ColumnElement[bool] | TextClause,
        parameters: dict[str, object] | None = None,
        *,
        columns: tuple[str, ...] | None = None,
        keep: Callable[[Mapping[str, Any]], dict[str, Any]] | None = None,
        counted: Collection[object] = (),
        facts: Callable[[Mapping[str, Any]], Mapping[str, int]] | None = None,
    ) -> tuple[Mapping[str, Any], ...]:
        """Hash every selected value, but keep and count as kept only what `keep` returns.

        A row whose id is in `counted` was already counted by this inspection: it is hashed
        again, in order, without being counted twice. Values that `facts` computes from a row
        join it before it is encoded, so they are counted and hashed with it; a fact named
        like one of its columns refuses the read rather than replace that column.
        """
        table = Base.metadata.tables[name]
        statement = (
            (select(*(table.c[column] for column in columns)) if columns else select(table))
            .where(predicate)
            .order_by(*table.primary_key.columns)
        )
        result: list[Mapping[str, Any]] = []
        rows: Result[Any] | None = None
        try:
            rows = self.session.execute(statement, parameters or {})
            for row in rows.mappings():
                values = dict(row)
                if facts is not None:
                    computed = facts(values)
                    if not values.keys().isdisjoint(computed):
                        raise ChatRecoveryGraphError()
                    values.update(computed)
                encoded = _encoded(name, values)
                if not counted or values["id"] not in counted:
                    self.row_count += 1
                    self.read_bytes += len(encoded)
                kept = values if keep is None else keep(values)
                self.byte_count += len(encoded) if keep is None else len(_encoded(name, kept))
                if (
                    self.row_count > MAX_GRAPH_ROWS
                    or self.byte_count > MAX_GRAPH_BYTES
                    or self.read_bytes > MAX_GRAPH_READ_BYTES
                ):
                    raise ChatRecoveryGraphTooLarge()
                self.digest.update(len(encoded).to_bytes(8, "big"))
                self.digest.update(encoded)
                result.append(kept)
        except ChatRecoveryGraphError:
            raise
        except (TypeError, ValueError, RecursionError):
            raise ChatRecoveryGraphError() from None
        finally:
            # A refusal partway through leaves no statement open on the reserved connection.
            if rows is not None:
                rows.close()
        return tuple(result)

    def leaves_room(self) -> bool:
        """Whether every count stays within fifteen sixteenths of its bound."""
        return all(
            16 * used <= 15 * limit
            for used, limit in (
                (self.row_count, MAX_GRAPH_ROWS),
                (self.byte_count, MAX_GRAPH_BYTES),
                (self.read_bytes, MAX_GRAPH_READ_BYTES),
            )
        )


def _artifact_roots(name: str, row: Mapping[str, Any]) -> set[str]:
    """Name the artifacts one owned row links, checked as the library's reference walk is."""
    found = _optional_id(row[_ARTIFACT_COLUMNS[name]]) if name in _ARTIFACT_COLUMNS else set()
    if name == "message_references":
        found.update(_ids(row["artifact_ids_json"]))
    elif name == "runs":
        found.update(_run_ids(row["provenance_json"]))
        found.update(_settings_ids(row["settings_json"]))
    elif name == "work_steps":
        found.update(_work_step_ids(row["input_bindings_json"]))
        found.update(_settings_ids(row["settings_json"]))
    elif name == "jobs":
        found.update(_job_ids(row["payload_json"]))
        found.update(_job_ids(row["result_json"]))
    elif name == "chats" and row["scope"] == "studio":
        found.update(_optional_id(_mapping(row["origin_json"]).get("source_artifact_id")))
    return found


def _organized(session: Session, batch: Collection[str]) -> frozenset[str]:
    """The entries an album or tag holds: all that release reads of organization.

    Their rows have no limit, so the graph binds one digit per entry instead.
    """
    entries = select(ArtifactLibraryEntry.id).where(ArtifactLibraryEntry.artifact_id.in_(batch))
    held: set[str] = set()
    for column in (MediaCollectionMembership.entry_id, MediaTagAssignment.entry_id):
        held.update(session.scalars(select(column).where(column.in_(entries)).distinct()))
    return frozenset(held)


def _owned_jobs(
    reader: _GraphReader, chat_id: str, rows: Mapping[str, tuple[Mapping[str, Any], ...]]
) -> tuple[str, ...]:
    """A historical source link does not transfer another conversation's job ownership."""

    jobs = rows["jobs"]
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
        columns = _PARENT_KEEP[name]
        # An owned parent was counted when this graph read it; only a foreign one counts now.
        owned = frozenset(row["id"] for row in rows[name])
        for start in range(0, len(values), 500):
            for row in reader.read(
                name,
                table.c.id.in_(values[start : start + 500]),
                keep=lambda row: {column: row[column] for column in columns},
                counted=owned,
            ):
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
    found: set[str] = set()

    def keeper(name: str) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
        columns = _KEEP.get(name, ())

        def keep(row: Mapping[str, Any]) -> dict[str, Any]:
            found.update(_artifact_roots(name, row))
            kept = {column: row[column] for column in columns}
            if name == "jobs":
                payload = _mapping(row["payload_json"])
                kept["payload_json"] = {
                    key: payload[key] for key in ("chat_id", "source_run_id") if key in payload
                }
            return kept

        return keep

    with session.no_autoflush:
        try:
            for name, predicate in CHAT_OWNER_SQL.items():
                if ROW_COLUMNS[name] != tuple(Base.metadata.tables[name].columns.keys()):
                    raise ChatRecoveryGraphError()
                rows[name] = reader.read(
                    name,
                    text(predicate.format(row=name, chat=":chat_id")),
                    {"chat_id": chat_id},
                    keep=keeper(name),
                )
            if len(rows["chats"]) != 1:
                raise ChatRecoveryGraphError()
            trashed = reader.read(
                "recovery_items",
                (RecoveryItem.kind == "chat") & (RecoveryItem.subject_id == chat_id),
            )
            # A fork remains independently owned, but creating one changes the impact.
            reader.read(
                "chats",
                text("json_extract(chats.origin_json, '$.source_chat_id') = :chat_id"),
                {"chat_id": chat_id},
            )
            visited: set[str] = set()
            organized: set[str] = set()
            retained_bytes = 0
            while frontier := sorted(found - visited):
                if len(found) > MAX_GRAPH_ROWS:
                    raise ChatRecoveryGraphTooLarge()
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
                    organized.update(_organized(session, batch))
                    reader.read(
                        "artifact_library_entries",
                        ArtifactLibraryEntry.artifact_id.in_(batch),
                        facts=lambda row: {"organized": int(row["id"] in organized)},
                    )
                    visited.update(batch)
            owned_jobs = _owned_jobs(reader, chat_id, rows)
        except ArtifactReferenceDataError:
            raise ChatRecoveryGraphError() from None
        # Admit a conversation only with room for its deletion record and later library or
        # schema growth, so one already in Recently Deleted can still be listed and purged.
        if not trashed and not reader.leaves_room():
            raise ChatRecoveryGraphTooLarge()
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
