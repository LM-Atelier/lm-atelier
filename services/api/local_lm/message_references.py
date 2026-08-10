"""Bind a turn's reference requests to subjects, and record what it used.

This module is the only writer of `message_references`. Keeping that in one
place is what makes the row trustworthy as history: a second writer that filled
in the snapshot slightly differently would produce records that disagree about
what a turn referred to, and nothing downstream could tell which was right.

Resolution refuses rather than drops. A turn that names a subject which no
longer exists is a turn whose meaning has changed, and quietly generating
without it would produce an image the request did not ask for while reporting
success.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import MessageReference, ReferenceAsset, ReferenceSubject
from .references import MentionSource, ReferenceError, ReferenceRequest


@dataclass(frozen=True)
class ResolvedReference:
    """One Reference a turn used, frozen as the subject stood at that moment.

    Everything identifying is copied rather than referenced. The subject can be
    renamed or deleted afterwards and this stays true, which is the whole
    reason the record exists.
    """

    reference_subject_id: str
    mention_slug: str
    subject_name: str
    subject_kind: str
    reference_asset_ids: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    role: str | None = None
    strength: float | None = None
    source: MentionSource = MentionSource.MENTION


def resolve_reference_requests(
    session: Session, requests: Sequence[ReferenceRequest]
) -> tuple[ResolvedReference, ...]:
    """Bind each request to a subject that exists, or refuse the whole turn.

    An unknown subject id is an error and never a reason to continue with one
    fewer reference. The caller asked for something specific; silently dropping
    it produces a result that looks like success and is not.

    Selected assets are checked against the subject that was named. An asset id
    belonging to a different subject is refused rather than ignored, because
    accepting it would let a turn pull an image from a subject it never named.
    """

    resolved: list[ResolvedReference] = []
    for request in requests:
        subject = session.get(ReferenceSubject, request.reference_subject_id)
        if subject is None:
            raise ReferenceError(
                f"reference {request.reference_subject_id!r} no longer exists; "
                "remove it from the turn or create it again"
            )

        artifact_ids: tuple[str, ...] = ()
        if request.selected_asset_ids:
            rows = session.scalars(
                select(ReferenceAsset).where(
                    ReferenceAsset.id.in_(request.selected_asset_ids),
                    ReferenceAsset.reference_subject_id == subject.id,
                )
            ).all()
            found = {row.id: row for row in rows}
            missing = [one for one in request.selected_asset_ids if one not in found]
            if missing:
                raise ReferenceError(
                    f"{subject.name} does not hold image(s) {', '.join(sorted(missing))}"
                )
            # Selection order, not database order: the caller chose a sequence.
            artifact_ids = tuple(found[one].artifact_id for one in request.selected_asset_ids)

        resolved.append(
            ResolvedReference(
                reference_subject_id=subject.id,
                mention_slug=subject.mention_slug,
                subject_name=subject.name,
                subject_kind=subject.kind,
                # Empty means the turn pinned no particular images, not that it
                # wanted none. Which images a generation actually conditions on
                # is a decision made later and recorded by whoever makes it.
                reference_asset_ids=tuple(request.selected_asset_ids),
                artifact_ids=artifact_ids,
                role=request.role,
                strength=request.strength,
                source=request.source,
            )
        )
    return tuple(resolved)


def record_message_references(
    session: Session, message_id: str, resolved: Sequence[ResolvedReference]
) -> tuple[MessageReference, ...]:
    """Write what a turn referred to, once.

    Adds rows to the caller's transaction and does not commit. The turn that
    creates the message owns whether any of it survives, so a rollback there
    must take these with it rather than leaving a record of a turn that never
    happened.

    Re-recording a message is refused. These rows are the immutable answer to
    what a past turn used, and rewriting them is how that answer becomes a
    guess.
    """

    if not resolved:
        return ()
    existing = session.scalars(
        select(MessageReference).where(MessageReference.message_id == message_id)
    ).first()
    if existing is not None:
        raise ReferenceError(f"message {message_id} already records what it referred to")

    rows = [
        MessageReference(
            message_id=message_id,
            position=position,
            reference_subject_id=one.reference_subject_id,
            mention_slug=one.mention_slug,
            subject_name=one.subject_name,
            subject_kind=one.subject_kind,
            role=one.role,
            strength=one.strength,
            source=one.source.value,
            reference_asset_ids_json=list(one.reference_asset_ids),
            artifact_ids_json=list(one.artifact_ids),
        )
        for position, one in enumerate(resolved)
    ]
    session.add_all(rows)
    return tuple(rows)


def message_references(session: Session, message_id: str) -> tuple[ResolvedReference, ...]:
    """Read back what a turn referred to, from the snapshot rather than a join."""

    rows = session.scalars(
        select(MessageReference)
        .where(MessageReference.message_id == message_id)
        .order_by(MessageReference.position)
    ).all()
    return tuple(
        ResolvedReference(
            reference_subject_id=row.reference_subject_id,
            mention_slug=row.mention_slug,
            subject_name=row.subject_name,
            subject_kind=row.subject_kind,
            reference_asset_ids=tuple(row.reference_asset_ids_json),
            artifact_ids=tuple(row.artifact_ids_json),
            role=row.role,
            strength=row.strength,
            source=MentionSource(row.source),
        )
        for row in rows
    )


def carry_message_references(
    session: Session, *, source_message_id: str, target_message_id: str
) -> tuple[MessageReference, ...]:
    """Carry a turn's references onto a regeneration, verbatim.

    Deliberately a copy and not a re-resolution. Regenerating an image must use
    what the original turn used, so a subject renamed or deleted in between
    cannot change - or refuse - a repeat of something that already ran.
    """

    return record_message_references(
        session, target_message_id, message_references(session, source_message_id)
    )
