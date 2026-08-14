"""Strict backend foundation for manual Media Library collections and tags."""

from __future__ import annotations

import re
from typing import NoReturn

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .artifact_library import begin_artifact_write_fence
from .domain import new_id, utcnow
from .models import (
    ArtifactLibraryEntry,
    MediaCollection,
    MediaCollectionMembership,
    MediaTag,
    MediaTagAssignment,
)

MEDIA_ORGANIZATION_INVALID = "The Media Library organization request is invalid."
MEDIA_ORGANIZATION_CONFLICT = "The Media Library organization changed. Refresh and try again."
_TAG_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class MediaOrganizationError(ValueError):
    pass


class MediaOrganizationConflict(MediaOrganizationError):
    pass


def _invalid() -> NoReturn:
    raise MediaOrganizationError(MEDIA_ORGANIZATION_INVALID)


def _text(value: object, *, maximum: int, empty: bool = False) -> str:
    if type(value) is not str:
        _invalid()
    if len(value) > maximum or (not empty and not value):
        _invalid()
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        _invalid()
    if not empty and value != value.strip():
        _invalid()
    return value


def media_tag_slug(value: object) -> str:
    label = _text(value, maximum=200)
    normalized = "-".join(label.casefold().split())
    if len(normalized) > 80 or _TAG_NAME.fullmatch(normalized) is None:
        _invalid()
    return normalized


def _color(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or re.fullmatch(r"#[0-9a-f]{6}", value) is None:
        _invalid()
    return value


def create_manual_collection(
    session: Session, *, name: object, description: object = ""
) -> MediaCollection:
    collection = MediaCollection(
        id=new_id("collection"),
        kind="manual",
        name=_text(name, maximum=200),
        description=_text(description, maximum=2_000, empty=True),
        version=1,
    )
    session.add(collection)
    session.flush()
    return collection


def create_media_tag(session: Session, *, label: object, color: object = None) -> MediaTag:
    exact_label = _text(label, maximum=200)
    tag = MediaTag(
        id=new_id("mediatag"),
        slug=media_tag_slug(exact_label),
        label=exact_label,
        color=_color(color),
        version=1,
    )
    session.add(tag)
    try:
        session.flush()
    except IntegrityError as exc:
        raise MediaOrganizationConflict(MEDIA_ORGANIZATION_CONFLICT) from exc
    return tag


def _visible_entry(session: Session, entry_id: object) -> ArtifactLibraryEntry:
    if type(entry_id) is not str or re.fullmatch(r"libentry:sha256:[0-9a-f]{64}", entry_id) is None:
        _invalid()
    entry = session.get(ArtifactLibraryEntry, entry_id)
    if entry is None or entry.state != "visible":
        _invalid()
    return entry


def add_manual_membership(
    session: Session,
    *,
    collection_id: object,
    entry_id: object,
    expected_version: object,
    note: object = None,
) -> MediaCollectionMembership:
    if (
        type(collection_id) is not str
        or re.fullmatch(r"collection_[0-9a-f]{32}", collection_id) is None
        or type(expected_version) is not int
        or expected_version < 1
    ):
        _invalid()
    exact_note = None if note is None else _text(note, maximum=1_000)
    begin_artifact_write_fence(session)
    collection = session.get(MediaCollection, collection_id)
    _visible_entry(session, entry_id)
    if collection is None or collection.kind != "manual" or collection.version != expected_version:
        raise MediaOrganizationConflict(MEDIA_ORGANIZATION_CONFLICT)
    if session.get(MediaCollectionMembership, (collection_id, entry_id)) is not None:
        raise MediaOrganizationConflict(MEDIA_ORGANIZATION_CONFLICT)
    position = session.scalar(
        select(func.max(MediaCollectionMembership.position)).where(
            MediaCollectionMembership.collection_id == collection_id
        )
    )
    membership = MediaCollectionMembership(
        collection_id=collection_id,
        entry_id=entry_id,
        position=0 if position is None else position + 1,
        note=exact_note,
        added_at=utcnow(),
    )
    session.add(membership)
    session.flush()
    session.expire(collection)
    session.refresh(collection)
    return membership


def assign_media_tag(
    session: Session,
    *,
    tag_id: object,
    entry_id: object,
    expected_version: object,
) -> MediaTagAssignment:
    if (
        type(tag_id) is not str
        or re.fullmatch(r"mediatag_[0-9a-f]{32}", tag_id) is None
        or type(expected_version) is not int
        or expected_version < 1
    ):
        _invalid()
    begin_artifact_write_fence(session)
    tag = session.get(MediaTag, tag_id)
    _visible_entry(session, entry_id)
    if tag is None or tag.version != expected_version:
        raise MediaOrganizationConflict(MEDIA_ORGANIZATION_CONFLICT)
    if session.get(MediaTagAssignment, (tag_id, entry_id)) is not None:
        raise MediaOrganizationConflict(MEDIA_ORGANIZATION_CONFLICT)
    assignment = MediaTagAssignment(tag_id=tag_id, entry_id=entry_id, added_at=utcnow())
    session.add(assignment)
    session.flush()
    session.expire(tag)
    session.refresh(tag)
    return assignment
