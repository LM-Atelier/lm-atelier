"""Apply a confirmed, non-owning organization impact in one writer transaction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, timedelta
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .artifact_library import (
    _validate_library_row,
    begin_artifact_write_fence,
    set_library_favorite,
)
from .domain import new_id, utcnow
from .media_organization import add_manual_membership, assign_media_tag, media_tag_slug
from .media_organization_receipts_v1 import (
    OrganizationAction,
    OrganizationPreviewV1,
    organization_input_digest,
    organization_result_digest,
    validate_saved_preview,
    validate_saved_result,
)
from .media_recovery import preview_media_trash, trash_media
from .models import (
    Artifact,
    ArtifactLibraryEntry,
    MediaCollection,
    MediaCollectionMembership,
    MediaOrganizationImpact,
    MediaTag,
    MediaTagAssignment,
)
from .recovery_v1 import RecoveryAction, RecoveryCommandV1

INVALID = "The Media Library organization request is invalid."
STALE = "The selected media or organization changed. Review the selection again."
TAG_CONFLICT = "This tag name is already in use. Choose a different name."
MERGE_TRASH = (
    "Restore media in Recently Deleted before merging this tag, then review the merge again."
)
MAX_REFERENCES = 100_000


class OrganizationInvalid(ValueError):
    pass


class OrganizationStale(ValueError):
    pass


class OrganizationTagConflict(ValueError):
    pass


class OrganizationMergeTrash(ValueError):
    pass


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SelectedEntry(StrictModel):
    id: Annotated[StrictStr, Field(pattern=r"^libentry:sha256:[0-9a-f]{64}$")]
    version: Annotated[StrictInt, Field(ge=1, le=9_223_372_036_854_775_806)]


class OrganizationTarget(StrictModel):
    id: Annotated[StrictStr, Field(pattern=r"^(collection_[0-9a-f]{32}|mediatag_[0-9a-f]{32})$")]
    version: Annotated[StrictInt, Field(ge=1, le=9_223_372_036_854_775_806)]


class OrganizationRequest(StrictModel):
    action: OrganizationAction
    target: OrganizationTarget | None = None
    destination: OrganizationTarget | None = None
    entries: Annotated[list[SelectedEntry], Field(max_length=100)] = Field(default_factory=list)
    changes: dict[StrictStr, StrictStr | None] | None = None
    favorite: StrictBool | None = None


class ApplyOrganization(StrictModel):
    operation_key: Annotated[
        StrictStr, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9:_-]+$")
    ]


def _text(value: object, maximum: int, *, empty: bool = False) -> str:
    if type(value) is not str or len(value) > maximum or (not empty and not value.strip()):
        raise OrganizationInvalid(INVALID)
    if any(ord(char) < 32 or ord(char) > 126 for char in value) or value != value.strip():
        raise OrganizationInvalid(INVALID)
    return value


def parse_request(payload: object) -> OrganizationRequest:
    try:
        request = OrganizationRequest.model_validate(payload)
    except ValidationError as exc:
        raise OrganizationInvalid(INVALID) from exc
    action = request.action
    required = {"action"}
    if action in {"set-favorite", "trash"}:
        required.add("entries")
        if action == "set-favorite":
            required.add("favorite")
        if action == "set-favorite" and request.favorite is None:
            raise OrganizationInvalid(INVALID)
    else:
        required.add("target")
        if request.target is None:
            raise OrganizationInvalid(INVALID)
        prefix = "collection_" if "album" in action else "mediatag_"
        if not request.target.id.startswith(prefix):
            raise OrganizationInvalid(INVALID)
        if action in {
            "add-to-album",
            "remove-from-album",
            "reorder-album",
            "add-tag",
            "remove-tag",
        }:
            required.add("entries")
        elif action.startswith("rename-"):
            required.add("changes")
            changes = request.changes
            if changes is None:
                raise OrganizationInvalid(INVALID)
            if action == "rename-album":
                if set(changes) != {"name", "description"}:
                    raise OrganizationInvalid(INVALID)
                _text(changes["name"], 200)
                _text(changes["description"], 2_000, empty=True)
            else:
                if set(changes) != {"label", "color"}:
                    raise OrganizationInvalid(INVALID)
                _text(changes["label"], 200)
                media_tag_slug(changes["label"])
                color = changes["color"]
                if color is not None and (
                    len(color) != 7
                    or color[0] != "#"
                    or any(char not in "0123456789abcdef" for char in color[1:])
                ):
                    raise OrganizationInvalid(INVALID)
        elif action == "merge-tags":
            required.add("destination")
            if (
                request.destination is None
                or not request.destination.id.startswith("mediatag_")
                or request.destination.id == request.target.id
            ):
                raise OrganizationInvalid(INVALID)
    if request.model_fields_set != required:
        raise OrganizationInvalid(INVALID)
    if "entries" in required and (
        not request.entries or len({entry.id for entry in request.entries}) != len(request.entries)
    ):
        raise OrganizationInvalid(INVALID)
    if action == "reorder-album" and len(request.entries) < 2:
        raise OrganizationInvalid(INVALID)
    return request


@dataclass
class OrganizationSnapshot:
    request: OrganizationRequest
    entries: dict[str, tuple[ArtifactLibraryEntry, Artifact]]
    target: MediaCollection | MediaTag | None
    destination: MediaTag | None
    memberships: dict[str, MediaCollectionMembership]
    assignments: dict[str, MediaTagAssignment]
    changed_count: int
    fingerprint: str


def _snapshot(session: Session, request: OrganizationRequest) -> OrganizationSnapshot:
    target: MediaCollection | MediaTag | None = None
    destination = None
    if request.target is not None:
        target = (
            session.get(MediaCollection, request.target.id)
            if "album" in request.action
            else session.get(MediaTag, request.target.id)
        )
        if target is None or target.version != request.target.version:
            raise OrganizationStale(STALE)
    if request.destination is not None:
        destination = session.get(MediaTag, request.destination.id)
        if destination is None or destination.version != request.destination.version:
            raise OrganizationStale(STALE)
    if request.action == "rename-tag":
        assert isinstance(target, MediaTag) and request.changes is not None
        slug = media_tag_slug(request.changes["label"])
        if session.scalar(
            select(MediaTag.id).where(MediaTag.slug == slug, MediaTag.id != target.id)
        ):
            raise OrganizationTagConflict(TAG_CONFLICT)
    memberships: dict[str, MediaCollectionMembership] = {}
    assignments: dict[str, MediaTagAssignment] = {}
    ids = [entry.id for entry in request.entries]
    if isinstance(target, MediaCollection):
        statement = select(MediaCollectionMembership).where(
            MediaCollectionMembership.collection_id == target.id
        )
        if ids:
            statement = statement.where(MediaCollectionMembership.entry_id.in_(ids))
        members = session.scalars(statement.limit(MAX_REFERENCES + 1)).all()
        if len(members) > MAX_REFERENCES:
            raise OrganizationInvalid(INVALID)
        memberships = {member.entry_id: member for member in members}
    elif isinstance(target, MediaTag):
        tag_ids = [target.id] + ([destination.id] if destination is not None else [])
        assignment_statement = select(MediaTagAssignment).where(
            MediaTagAssignment.tag_id.in_(tag_ids)
        )
        if ids:
            assignment_statement = assignment_statement.where(MediaTagAssignment.entry_id.in_(ids))
        assigned = session.scalars(assignment_statement.limit(MAX_REFERENCES + 1)).all()
        if len(assigned) > MAX_REFERENCES:
            raise OrganizationInvalid(INVALID)
        assignments = {item.entry_id: item for item in assigned if item.tag_id == target.id}
        if not ids:
            ids = sorted({item.entry_id for item in assigned})
    if not ids:
        ids = list(memberships)
    entries: dict[str, tuple[ArtifactLibraryEntry, Artifact]] = {}
    for start in range(0, len(ids), 500):
        rows = session.execute(
            select(ArtifactLibraryEntry, Artifact)
            .join(Artifact, Artifact.id == ArtifactLibraryEntry.artifact_id)
            .where(ArtifactLibraryEntry.id.in_(ids[start : start + 500]))
        ).all()
        entries.update({entry.id: (entry, artifact) for entry, artifact in rows})
    if set(entries) != set(ids):
        raise OrganizationStale(STALE)
    expected = {entry.id: entry.version for entry in request.entries}
    for entry, artifact in entries.values():
        _validate_library_row(entry, artifact)
        if expected and (entry.state != "visible" or entry.version != expected[entry.id]):
            raise OrganizationStale(STALE)
    if request.action == "merge-tags" and any(
        entries[entry_id][0].state != "visible" for entry_id in assignments
    ):
        raise OrganizationMergeTrash(MERGE_TRASH)
    action = request.action
    if action == "add-to-album":
        changed = len(ids) - len(memberships)
    elif action == "remove-from-album":
        changed = len(memberships)
    elif action == "reorder-album":
        if set(memberships) != set(ids):
            raise OrganizationStale(STALE)
        positions = sorted(member.position for member in memberships.values())
        changed = sum(
            memberships[entry_id].position != position
            for entry_id, position in zip(ids, positions, strict=True)
        )
    elif action == "add-tag":
        changed = len(ids) - len(assignments)
    elif action == "remove-tag":
        changed = len(assignments)
    elif action == "set-favorite":
        changed = sum(entry.favorite != request.favorite for entry, _artifact in entries.values())
    elif action == "trash":
        changed = len(entries)
    elif action.startswith("rename-"):
        assert target is not None and request.changes is not None
        changed = int(
            any(getattr(target, field) != value for field, value in request.changes.items())
        )
    elif action == "merge-tags":
        changed = len(assignments)
    else:
        changed = 1
    material = {
        "request": request.model_dump(exclude_unset=True),
        "entries": [
            [key, entry.version, entry.state, entry.favorite, artifact.id, artifact.size_bytes]
            for key, (entry, artifact) in sorted(entries.items())
        ],
        "memberships": [
            [key, value.position, value.note, value.added_at.isoformat()]
            for key, value in sorted(memberships.items())
        ],
        "assignments": sorted(assignments),
        "target_version": target.version if target is not None else None,
        "destination_version": destination.version if destination is not None else None,
    }
    fingerprint = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return OrganizationSnapshot(
        request, entries, target, destination, memberships, assignments, changed, fingerprint
    )


def preview_organization(session: Session, payload: object) -> dict[str, Any]:
    request = parse_request(payload)
    begin_artifact_write_fence(session)
    snapshot = _snapshot(session, request)
    now = utcnow()
    expired = (
        select(MediaOrganizationImpact.id)
        .where(
            MediaOrganizationImpact.expires_at <= now,
            MediaOrganizationImpact.operation_key.is_(None),
            MediaOrganizationImpact.response_json.is_(None),
        )
        .order_by(MediaOrganizationImpact.expires_at, MediaOrganizationImpact.id)
        .limit(1000)
    )
    session.execute(delete(MediaOrganizationImpact).where(MediaOrganizationImpact.id.in_(expired)))
    expires = now + timedelta(minutes=15)
    body = {
        "id": new_id("orgimp"),
        "action": request.action,
        "selected_count": len(snapshot.entries),
        "size_bytes": sum(artifact.size_bytes for _entry, artifact in snapshot.entries.values()),
        "changed_count": snapshot.changed_count,
        "expires_at": expires.isoformat(),
    }
    recovery: dict[str, dict[str, str]] | None = None
    if request.action == "trash":
        recovery = {}
        for selected in request.entries:
            impact = preview_media_trash(session, selected.id, now)
            if RecoveryAction.TRASH not in impact.available_actions:
                raise OrganizationStale(STALE)
            recovery[selected.id] = {
                "expected_revision": impact.revision,
                "impact_sha256": impact.impact_sha256,
            }
    record = MediaOrganizationImpact(
        id=body["id"],
        request_json=request.model_dump(exclude_unset=True),
        preview_json=body,
        fingerprint=snapshot.fingerprint,
        expires_at=expires,
        recovery_json=recovery,
    )
    record.input_sha256 = organization_input_digest(record)
    session.add(record)
    session.flush()
    return body


def _trash_snapshot(
    session: Session,
    snapshot: OrganizationSnapshot,
    impact: MediaOrganizationImpact,
    operation_key: str,
) -> list[str]:
    """Join canonical recovery writes to the transaction that confirms the whole selection."""

    proofs = impact.recovery_json
    if type(proofs) is not dict or set(proofs) != set(snapshot.entries):
        raise OrganizationStale(STALE)
    commands: list[tuple[str, RecoveryCommandV1]] = []
    for selected in snapshot.request.entries:
        proof = proofs[selected.id]
        if type(proof) is not dict or set(proof) != {"expected_revision", "impact_sha256"}:
            raise OrganizationStale(STALE)
        key = hashlib.sha256(
            json.dumps([impact.id, operation_key, selected.id], separators=(",", ":")).encode()
        ).hexdigest()
        try:
            commands.append(
                (selected.id, RecoveryCommandV1.model_validate({**proof, "operation_key": key}))
            )
        except ValidationError as exc:
            raise OrganizationStale(STALE) from exc
    now = utcnow()
    # A later refusal rolls back every earlier transition through the caller's
    # one transaction. The same frozen proofs are used on every retry.
    return [
        trash_media(session, entry_id, command, now).deletion_id for entry_id, command in commands
    ]


def _apply_snapshot(session: Session, snapshot: OrganizationSnapshot) -> int | None:
    request, target = snapshot.request, snapshot.target
    action = request.action
    if action == "set-favorite":
        assert request.favorite is not None
        for entry, artifact in snapshot.entries.values():
            if entry.favorite != request.favorite:
                set_library_favorite(session, artifact, request.favorite)
        return None
    assert target is not None
    if action == "add-to-album":
        for selected in request.entries:
            if selected.id not in snapshot.memberships:
                add_manual_membership(
                    session,
                    collection_id=target.id,
                    entry_id=selected.id,
                    expected_version=target.version,
                )
    elif action == "add-tag":
        for selected in request.entries:
            if selected.id not in snapshot.assignments:
                assign_media_tag(
                    session, tag_id=target.id, entry_id=selected.id, expected_version=target.version
                )
    elif action in {"remove-from-album", "remove-tag"}:
        members = (
            snapshot.memberships.values()
            if action == "remove-from-album"
            else snapshot.assignments.values()
        )
        for member in members:
            session.delete(member)
        session.flush()
    elif action == "reorder-album" and snapshot.changed_count:
        positions = sorted(member.position for member in snapshot.memberships.values())
        retained = {
            key: (member.note, member.added_at) for key, member in snapshot.memberships.items()
        }
        for member in snapshot.memberships.values():
            session.delete(member)
        session.flush()
        for selected, position in zip(request.entries, positions, strict=True):
            note, added_at = retained[selected.id]
            session.add(
                MediaCollectionMembership(
                    collection_id=target.id,
                    entry_id=selected.id,
                    position=position,
                    note=note,
                    added_at=added_at,
                )
            )
        session.flush()
    elif action.startswith("rename-") and snapshot.changed_count:
        assert request.changes is not None
        for field, value in request.changes.items():
            setattr(target, field, value)
        if isinstance(target, MediaTag):
            target.slug = media_tag_slug(target.label)
        target.version += 1
        target.updated_at = utcnow()
        session.flush()
    elif action in {"delete-album", "delete-tag", "merge-tags"}:
        if action == "merge-tags":
            assert snapshot.destination is not None
            destination = snapshot.destination
            for entry_id in snapshot.assignments:
                if session.get(MediaTagAssignment, (destination.id, entry_id)) is None:
                    entry = snapshot.entries[entry_id][0]
                    if entry.state != "visible":
                        raise OrganizationStale(STALE)
                    assign_media_tag(
                        session,
                        tag_id=destination.id,
                        entry_id=entry_id,
                        expected_version=destination.version,
                    )
        session.delete(target)
        session.flush()
        return None
    session.refresh(target)
    return target.version


def _saved_input(
    impact: MediaOrganizationImpact,
) -> tuple[OrganizationRequest, OrganizationPreviewV1]:
    try:
        request = parse_request(impact.request_json)
        preview = validate_saved_preview(impact, request.action)
        if request.entries and preview.selected_count != len(request.entries):
            raise ValueError("organization-count-invalid")
        return request, preview
    except (ValueError, TypeError, RecursionError) as exc:
        raise OrganizationStale(STALE) from exc


def _result_version(request: OrganizationRequest, preview: OrganizationPreviewV1) -> int | None:
    if request.target is None or request.action in {"delete-album", "delete-tag", "merge-tags"}:
        return None
    increment = preview.changed_count
    if request.action == "reorder-album":
        increment = 2 * preview.selected_count if preview.changed_count else 0
    return request.target.version + increment


def apply_organization(session: Session, impact_id: str, payload: object) -> dict[str, Any]:
    try:
        command = ApplyOrganization.model_validate(payload)
    except ValidationError as exc:
        raise OrganizationInvalid(INVALID) from exc
    begin_artifact_write_fence(session)
    replay = session.scalar(
        select(MediaOrganizationImpact).where(
            MediaOrganizationImpact.operation_key == command.operation_key
        )
    )
    if replay is not None:
        if replay.id != impact_id or replay.response_json is None:
            raise OrganizationStale(STALE)
        request, preview = _saved_input(replay)
        try:
            return validate_saved_result(replay, preview, _result_version(request, preview))
        except (ValueError, TypeError, RecursionError) as exc:
            raise OrganizationStale(STALE) from exc
    impact = session.get(MediaOrganizationImpact, impact_id)
    if (
        impact is None
        or impact.operation_key is not None
        or impact.expires_at.replace(tzinfo=UTC) <= utcnow()
    ):
        raise OrganizationStale(STALE)
    request, preview = _saved_input(impact)
    snapshot = _snapshot(session, request)
    if (
        snapshot.fingerprint != impact.fingerprint
        or preview.selected_count != len(snapshot.entries)
        or preview.changed_count != snapshot.changed_count
        or preview.size_bytes
        != sum(artifact.size_bytes for _, artifact in snapshot.entries.values())
        or impact.expires_at.replace(tzinfo=UTC).isoformat() != preview.expires_at
    ):
        raise OrganizationStale(STALE)
    deletion_ids = None
    if snapshot.request.action == "trash":
        deletion_ids = _trash_snapshot(session, snapshot, impact, command.operation_key)
        version = None
    else:
        version = _apply_snapshot(session, snapshot)
    response: dict[str, Any] = {
        "id": impact.id,
        "action": snapshot.request.action,
        "selected_count": len(snapshot.entries),
        "changed_count": snapshot.changed_count,
        "target_version": version,
    }
    if deletion_ids is not None:
        response["deletion_ids"] = deletion_ids
    impact.operation_key = command.operation_key
    impact.response_json = response
    impact.response_sha256 = organization_result_digest(impact)
    try:
        validate_saved_result(impact, preview, _result_version(request, preview))
    except (ValueError, TypeError, RecursionError) as exc:
        raise OrganizationStale(STALE) from exc
    session.flush()
    return response
