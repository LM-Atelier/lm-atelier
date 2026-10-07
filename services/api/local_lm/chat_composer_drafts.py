"""Keep each chat's unsent composer draft in the workspace database.

A draft is saved often and read rarely, so the rules that matter are about
writes. Every write names the revision it was based on, and a write based on
anything older is refused rather than merged: two windows typing into the same
chat would otherwise take turns silently undoing each other. The refusal names
the current revision so the writer can read it and decide.

A revision is never reused. Discarding a draft empties it and moves it to the
next revision rather than deleting the row, so a window still holding an older
revision is refused even after the draft was discarded and written again.

Files attached to a draft are rows with a foreign key to the stored file, so a
file stays while any draft holds it and retention sees the hold. Nothing in a
draft's JSON may name a stored file: the only settings key that can, an edit
mask, is refused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .domain import ArtifactKind, RoutingMode, utcnow
from .models import Artifact, ChatComposerDraft, ChatComposerDraftAttachment
from .schemas import (
    MAX_DRAFT_TEMPLATE_SETTINGS_BYTES,
    ChatComposerDraftAttachmentIn,
    ChatComposerDraftIn,
    ChatComposerDraftMentionIn,
    ChatComposerDraftOut,
    ChatComposerDraftTemplateIn,
    PromptComposerSourceIn,
)

#: What a draft may attach: an upload, or a picture or video the workspace made.
ATTACHABLE_KINDS = frozenset(
    {ArtifactKind.INPUT.value, ArtifactKind.IMAGE.value, ArtifactKind.VIDEO.value}
)


@dataclass(frozen=True)
class DraftRevisionStale(Exception):
    """The draft changed after the writer read it."""

    current_revision: int


class DraftAttachmentUnavailable(Exception):
    """An attachment names a file that does not exist or cannot be attached."""


class DraftSettingsUnsupported(Exception):
    """The template settings hold something a draft cannot keep."""


def read_draft(session: Session, chat_id: str) -> ChatComposerDraftOut:
    """The chat's draft, or an empty one at revision 0 when it has none."""

    row = session.get(ChatComposerDraft, chat_id)
    if row is None:
        return ChatComposerDraftOut(chat_id=chat_id, revision=0)
    return ChatComposerDraftOut(
        chat_id=chat_id,
        revision=row.revision,
        updated_at=row.updated_at,
        text=row.text,
        prompt_source=(
            PromptComposerSourceIn.model_validate(row.prompt_source_json)
            if row.prompt_source_json
            else None
        ),
        mode=row.mode,
        output_count=row.output_count,
        attachments=[
            ChatComposerDraftAttachmentIn(
                artifact_id=attachment.artifact_id, kind=attachment.kind, origin=attachment.origin
            )
            for attachment in row.attachments
        ],
        mentions=[ChatComposerDraftMentionIn.model_validate(item) for item in row.mentions_json],
        template_settings=(
            ChatComposerDraftTemplateIn.model_validate(row.template_settings_json)
            if row.template_settings_json
            else None
        ),
    )


def has_content(row: ChatComposerDraft | None) -> bool:
    """Whether a stored draft holds anything somebody put there."""

    if row is None:
        return False
    return bool(
        row.text.strip()
        or row.prompt_source_json
        or row.attachments
        or row.mentions_json
        or row.template_settings_json
    )


def write_draft(
    session: Session, chat_id: str, expected_revision: int, draft: ChatComposerDraftIn
) -> ChatComposerDraftOut:
    """Replace the chat's draft if it is still at `expected_revision`.

    The caller has already established that the chat exists and may be written.
    Commits nothing: the route commits, so a refusal leaves the transaction to be
    rolled back whole.
    """

    template = _stored_template(draft.template_settings)
    values = {
        "text": draft.text,
        "prompt_source_json": (
            draft.prompt_source.model_dump(mode="json") if draft.prompt_source else None
        ),
        "mode": draft.mode.value,
        "output_count": draft.output_count,
        "mentions_json": [mention.model_dump(mode="json") for mention in draft.mentions],
        "template_settings_json": template,
        "updated_at": utcnow(),
    }
    if expected_revision == 0:
        # Creating, not replacing: a draft that already exists means another
        # writer got there first. Refused by the statement itself rather than a
        # read before it, and without a savepoint, whose release on SQLite would
        # commit a draft the checks below may still refuse.
        result = session.execute(
            sqlite_insert(ChatComposerDraft)
            .values(chat_id=chat_id, revision=1, created_at=values["updated_at"], **values)
            .on_conflict_do_nothing(index_elements=["chat_id"])
        )
        if getattr(result, "rowcount", 0) != 1:
            raise DraftRevisionStale(_current_revision(session, chat_id))
    else:
        # The revision check and the write are one statement, so two writers
        # that read the same revision cannot both succeed.
        result = session.execute(
            update(ChatComposerDraft)
            .where(
                ChatComposerDraft.chat_id == chat_id,
                ChatComposerDraft.revision == expected_revision,
            )
            .values(revision=expected_revision + 1, **values)
            .execution_options(synchronize_session=False)
        )
        if getattr(result, "rowcount", 0) != 1:
            raise DraftRevisionStale(_current_revision(session, chat_id))
    # Checked after the draft row is written, which takes the database's
    # writer reservation: a deletion either finished before this read or
    # waits behind this transaction, and then sees the attachment below.
    _check_attachments(session, draft.attachments)
    session.execute(
        delete(ChatComposerDraftAttachment).where(ChatComposerDraftAttachment.chat_id == chat_id)
    )
    try:
        for position, attachment in enumerate(draft.attachments):
            session.execute(
                insert(ChatComposerDraftAttachment).values(
                    chat_id=chat_id,
                    position=position,
                    artifact_id=attachment.artifact_id,
                    kind=attachment.kind,
                    origin=attachment.origin,
                )
            )
    except IntegrityError:
        raise DraftAttachmentUnavailable from None
    session.expire_all()
    return read_draft(session, chat_id)


def discard_draft(session: Session, chat_id: str, expected_revision: int) -> None:
    """Empty the chat's draft, and let go of its files, if it is still at that revision.

    The row stays and its revision advances, so the emptied draft is a revision
    of its own: no later draft can be given a revision an old window still holds.
    """

    result = session.execute(
        update(ChatComposerDraft)
        .where(
            ChatComposerDraft.chat_id == chat_id,
            ChatComposerDraft.revision == expected_revision,
        )
        .values(
            revision=expected_revision + 1,
            text="",
            prompt_source_json=None,
            mode=RoutingMode.AUTO.value,
            output_count=1,
            mentions_json=[],
            template_settings_json=None,
            updated_at=utcnow(),
        )
        .execution_options(synchronize_session=False)
    )
    if getattr(result, "rowcount", 0) != 1:
        raise DraftRevisionStale(_current_revision(session, chat_id))
    session.execute(
        delete(ChatComposerDraftAttachment).where(ChatComposerDraftAttachment.chat_id == chat_id)
    )
    session.expire_all()


def _current_revision(session: Session, chat_id: str) -> int:
    session.expire_all()
    return (
        session.scalar(
            select(ChatComposerDraft.revision).where(ChatComposerDraft.chat_id == chat_id)
        )
        or 0
    )


def _check_attachments(session: Session, attachments: list[ChatComposerDraftAttachmentIn]) -> None:
    ids = [attachment.artifact_id for attachment in attachments]
    if len(set(ids)) != len(ids):
        raise DraftAttachmentUnavailable
    if not ids:
        return
    kinds: dict[str, str] = {
        artifact_id: kind
        for artifact_id, kind in session.execute(
            select(Artifact.id, Artifact.kind).where(Artifact.id.in_(ids))
        )
    }
    if any(kinds.get(artifact_id, "") not in ATTACHABLE_KINDS for artifact_id in ids):
        raise DraftAttachmentUnavailable


def _stored_template(template: ChatComposerDraftTemplateIn | None) -> dict[str, object] | None:
    if template is None:
        return None
    # A mask names a stored file, and a draft holds files only through its
    # attachment rows. Masks belong to Studio, not to a one-click template.
    if "mask" in template.settings:
        raise DraftSettingsUnsupported
    stored = template.model_dump(mode="json")
    encoded = json.dumps(stored, separators=(",", ":"), ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_DRAFT_TEMPLATE_SETTINGS_BYTES:
        raise DraftSettingsUnsupported
    return stored
