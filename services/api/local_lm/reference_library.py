"""Creating, renaming, archiving and deleting the subjects a user can name.

Two rules drive most of this and neither is obvious from the data model alone.

**Archiving is the removal a user wants; deletion is the one they have to mean.**
A subject is referenced by past runs, and those references are history rather
than pointers - a run recorded what it used, and that record has to stay true
after the subject is gone. So deletion reports what it would cost before it
happens, and archiving exists so that the common case never needs that
conversation at all.

**A rename is not a new subject.** The display name changes, the mention may
change with it, and neither rewrites what an old run meant. Only the addressing
token is unique; the human-facing name never is, because two people can
reasonably be called the same thing.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from .models import ReferenceAsset, ReferenceSubject
from .references import (
    MAX_NAME,
    ReferenceError,
    ReferenceKind,
    parse_kind,
    slugify_mention,
    valid_mention_slug,
)

MAX_PAGE = 200
DEFAULT_PAGE = 50


@dataclass(frozen=True)
class DeletionImpact:
    """What permanently deleting a subject would take with it."""

    reference_subject_id: str
    name: str
    asset_count: int
    # Artifacts this subject is the only user of. Everything else it holds is
    # shared, and shared bytes are not this subject's to destroy.
    exclusive_artifact_ids: tuple[str, ...]

    @property
    def shared_artifact_count(self) -> int:
        return self.asset_count - len(self.exclusive_artifact_ids)


def _available_slug(session: Session, desired: str, *, exclude_id: str | None = None) -> str:
    """The canonical slug, or a numbered variant when it is already taken.

    Suffixing rather than refusing, because the collision is usually two people
    with the same name rather than a mistake, and making the second one fail
    would push the user into inventing a worse name themselves.
    """

    taken = select(ReferenceSubject.mention_slug).where(
        ReferenceSubject.mention_slug.startswith(desired)
    )
    if exclude_id is not None:
        taken = taken.where(ReferenceSubject.id != exclude_id)
    existing = set(session.scalars(taken).all())
    if desired not in existing:
        return desired
    for suffix in range(2, 1000):
        candidate = f"{desired}-{suffix}"
        if candidate not in existing:
            return candidate
    raise ReferenceError(f"too many subjects already answer to {desired!r}")


def create_subject(
    session: Session,
    *,
    name: str,
    kind: ReferenceKind | str,
    mention_slug: str | None = None,
    description: str | None = None,
    aliases: list[str] | None = None,
    tags: list[str] | None = None,
) -> ReferenceSubject:
    """Add a subject, giving it a mention nobody else answers to."""

    cleaned = (name or "").strip()
    if not cleaned:
        raise ReferenceError("a subject needs a name")
    if len(cleaned) > MAX_NAME:
        raise ReferenceError(f"a name is at most {MAX_NAME} characters")

    if mention_slug is None:
        slug = _available_slug(session, slugify_mention(cleaned))
    else:
        slug = mention_slug.strip()
        if not valid_mention_slug(slug):
            raise ReferenceError(
                f"{slug!r} is not a usable mention; use letters, digits and hyphens"
            )
        if session.scalar(select(ReferenceSubject.id).where(ReferenceSubject.mention_slug == slug)):
            # An explicitly chosen mention is refused rather than silently
            # renumbered: the user asked for that exact one.
            raise ReferenceError(f"another subject already answers to @{slug}")

    subject = ReferenceSubject(
        name=cleaned,
        mention_slug=slug,
        kind=parse_kind(kind).value,
        description=(description or None),
        aliases_json=list(aliases or []),
        tags_json=list(tags or []),
    )
    session.add(subject)
    session.flush()
    return subject


def rename_subject(
    session: Session, subject: ReferenceSubject, *, name: str, follow_mention: bool = False
) -> ReferenceSubject:
    """Change what a subject is called, and optionally how it is addressed.

    The mention deliberately does not follow by default. Past turns recorded a
    display name for provenance, but a live chat draft may hold the mention, and
    silently changing the addressing token underneath someone mid-sentence is
    worse than letting the two drift apart.
    """

    cleaned = (name or "").strip()
    if not cleaned:
        raise ReferenceError("a subject needs a name")
    if len(cleaned) > MAX_NAME:
        raise ReferenceError(f"a name is at most {MAX_NAME} characters")
    subject.name = cleaned
    if follow_mention:
        subject.mention_slug = _available_slug(
            session, slugify_mention(cleaned), exclude_id=subject.id
        )
    session.flush()
    return subject


def set_archived(session: Session, subject: ReferenceSubject, archived: bool) -> ReferenceSubject:
    """Archive or restore. Reversible on purpose - this is the normal removal."""

    subject.archived = bool(archived)
    session.flush()
    return subject


def set_favorite(session: Session, subject: ReferenceSubject, favorite: bool) -> ReferenceSubject:
    """Mark for the user's own convenience.

    Organisation only. Nothing may read this as a quality signal or feed it into
    ranking - it says someone wanted this near the top of a list, not that its
    images are good.
    """

    subject.favorite = bool(favorite)
    session.flush()
    return subject


def list_subjects(
    session: Session,
    *,
    kind: ReferenceKind | str | None = None,
    include_archived: bool = False,
    search: str | None = None,
    limit: int = DEFAULT_PAGE,
    offset: int = 0,
) -> tuple[list[ReferenceSubject], int]:
    """A page of subjects and the total that matched.

    Archived subjects are excluded unless asked for, because archiving is the
    removal a user expects to be honoured everywhere by default.
    """

    if limit < 1 or limit > MAX_PAGE:
        raise ReferenceError(f"a page is between 1 and {MAX_PAGE} subjects")
    if offset < 0:
        raise ReferenceError("an offset cannot be negative")

    filters: list[ColumnElement[bool]] = []
    if not include_archived:
        filters.append(ReferenceSubject.archived.is_(False))
    if kind is not None:
        filters.append(ReferenceSubject.kind == parse_kind(kind).value)
    if search and search.strip():
        pattern = f"%{search.strip().casefold()}%"
        filters.append(func.lower(ReferenceSubject.name).like(pattern))

    total = session.scalar(select(func.count()).select_from(ReferenceSubject).where(*filters))
    rows = session.scalars(
        select(ReferenceSubject)
        .where(*filters)
        # Favourites first as an organisation aid, then most recently touched.
        .order_by(
            ReferenceSubject.favorite.desc(),
            ReferenceSubject.updated_at.desc(),
            ReferenceSubject.id,
        )
        .limit(limit)
        .offset(offset)
    ).all()
    return list(rows), int(total or 0)


def deletion_impact(session: Session, subject: ReferenceSubject) -> DeletionImpact:
    """What deleting this subject would destroy, computed before anything is.

    Only artifacts nobody else references count as exclusive. A photograph
    showing two subjects belongs to both, and removing one of them is not
    permission to delete the picture.
    """

    asset_rows = session.scalars(
        select(ReferenceAsset).where(ReferenceAsset.reference_subject_id == subject.id)
    ).all()
    artifact_ids = [row.artifact_id for row in asset_rows]

    exclusive: list[str] = []
    for artifact_id in artifact_ids:
        others = session.scalar(
            select(func.count())
            .select_from(ReferenceAsset)
            .where(
                ReferenceAsset.artifact_id == artifact_id,
                ReferenceAsset.reference_subject_id != subject.id,
            )
        )
        if not others:
            exclusive.append(artifact_id)

    return DeletionImpact(
        reference_subject_id=subject.id,
        name=subject.name,
        asset_count=len(asset_rows),
        exclusive_artifact_ids=tuple(dict.fromkeys(exclusive)),
    )
