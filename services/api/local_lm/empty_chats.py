"""Which chats are empty, and which of those are safe to offer for deletion.

Emptiness is settled by enumeration rather than by judgement. Every table with a
foreign key to `chats.id` is listed in `WORK_PRODUCERS`, a chat with no real row
in any of them has no transcript and no work, and that is the entire test. A
structural test reads the list back off the schema, so a table added later that
references a chat fails a test instead of silently widening what this tool is
willing to offer.

`TurnCreationClaim` is the one a title heuristic would never find. A turn can be
claimed and in flight with nothing written yet, so a chat can look blank in
every visible way and be seconds from having a transcript. `active_head_message_id`
is deliberately not part of the test either: it is a pointer that can be stale in
both directions.

Four references carry no foreign key and are checked by hand. A setup
verification names the chat it ran in, and an edit verification job names its
chat in its payload; both mean the chat did work. A fork records the chat it
was forked from, and a studio session the chat it was opened from, each in its
origin; either means another chat still points at this one.

This module only reads. Nothing here deletes a chat.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sqlalchemy import ColumnElement, and_, exists, func, or_, select
from sqlalchemy.orm import InstrumentedAttribute, Session

from .domain import JobKind, RoutingMode
from .models import (
    Chat,
    ChatItemRemovalReceipt,
    ChatWorkflowSelection,
    Job,
    Message,
    PromptExpansionBatch,
    Run,
    SetupVerification,
    TurnCreationClaim,
    WorkPlan,
)
from .profile_service import AUTO_PROFILE_ID
from .prompt_helpers import STANDARD_CHAT_SCOPE
from .schemas import new_chat_vision_settings


class EmptyChatClass(StrEnum):
    """Why a chat is being offered, or why it is not."""

    STRICT_BLANK = "strict_blank"
    CONFIGURED_BLANK = "configured_blank"
    INCONSISTENT = "inconsistent"


#: The reasons a chat is `configured_blank`, as stable slugs.
#:
#: Reported instead of the values themselves. A maintenance list that shows
#: titles and drafts is a list of what somebody wrote, and the reason is what the
#: decision actually needs.
CONFIGURED_REASONS = (
    "custom_title",
    "in_project",
    "pinned",
    "archived",
    "has_draft",
    "routing_chosen",
    "confirmation_changed",
    "profile_chosen",
    "web_access_granted",
    "forked",
    "settings_overridden",
    "linked_chat",
)

#: The reasons a chat is `inconsistent`.
INCONSISTENT_REASONS = ("stale_head",)


@dataclass(frozen=True)
class EmptyChat:
    chat_id: str
    classification: EmptyChatClass
    reasons: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


#: Every table with a foreign key to `chats.id`. Enforced structurally by
#: `test_every_chat_reference_is_an_audited_producer`.
WORK_PRODUCERS = (
    Message,
    WorkPlan,
    Run,
    TurnCreationClaim,
    ChatItemRemovalReceipt,
    PromptExpansionBatch,
    ChatWorkflowSelection,
)

#: Old enough that a chat opened and abandoned minutes ago is not offered while
#: the person who opened it may still be looking at it. That is a choice about
#: the product, so the surface applies it and this function does not: a reader
#: that silently withheld rows nobody excluded would make every caller's page
#: depend on a default it never passed.
DEFAULT_MINIMUM_AGE_HOURS = 24.0

#: A page is bounded so one request cannot ask the database for every chat.
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200

#: How many raw rows one page may classify while filling itself. Filtering by
#: classification means walking past ineligible rows, and this is the bound on
#: that walk - a short page with a cursor is honest, an unbounded scan is not.
MAX_ROWS_EXAMINED = 2000


#: Rows a work table holds for *every* chat because creating a chat writes them,
#: and which therefore say nothing about whether anybody used it.
#:
#: Creating a chat mirrors four `ChatWorkflowSelection` rows - chat, vision,
#: image and video - each `automatic` with no family. Counting those as work
#: would make every chat that has ever existed busy, so the page could never
#: offer a single real one. A row that records an actual choice still counts.
_DEFAULT_ROWS: Mapping[object, Callable[[], ColumnElement[bool]]] = {
    ChatWorkflowSelection: lambda: and_(
        ChatWorkflowSelection.mode == "automatic",
        ChatWorkflowSelection.workflow_family_id.is_(None),
    ),
}


def _has_work(chat_id: InstrumentedAttribute[str]) -> ColumnElement[bool]:
    """True when any work table holds a real row for this chat, or a verification names it.

    "Real" excludes rows that creating a chat writes by itself. Every table is
    still consulted - the enumeration is the whole test - but a table populated
    unconditionally has to be asked whether its row means anything, or the test
    answers "busy" for a chat nobody has touched.
    """

    clauses: list[ColumnElement[bool]] = []
    for model in WORK_PRODUCERS:
        condition = model.chat_id == chat_id
        default = _DEFAULT_ROWS.get(model)
        if default is not None:
            condition = and_(condition, ~default())
        clauses.append(exists().where(condition))
    clauses.append(exists().where(SetupVerification.chat_id == chat_id))
    # Any status: a finished verification still belongs to the chat it checked,
    # and one that is queued is work about to happen in it.
    clauses.append(
        exists().where(
            and_(
                Job.kind == JobKind.EDIT_VERIFY.value,
                Job.payload_json["chat_id"].as_string() == chat_id,
            )
        )
    )
    return or_(*clauses)


@dataclass(frozen=True)
class EmptyChatPage:
    """One bounded page, and where the next one starts.

    A page is a snapshot of one moment and nothing more. A chat listed here can
    acquire work before the next page is read, and a chat excluded here can lose
    its last message and belong on a later page - the ordering is stable but the
    contents are not frozen. Nothing downstream may treat a page as a guarantee.
    """

    entries: tuple[EmptyChat, ...]
    next_cursor: str | None


def empty_chat_page(
    session: Session,
    *,
    limit: int = DEFAULT_PAGE_SIZE,
    after_id: str | None = None,
    minimum_age_hours: float = 0.0,
    include_archived: bool = False,
    include_configured: bool = False,
    now: datetime | None = None,
) -> EmptyChatPage:
    """Standard-scope chats with no work in any audited table, one page.

    Keyset on `Chat.id` rather than an offset. An offset re-reads rows that
    moved and skips rows that shifted under it, which on a list whose whole
    purpose is deletion means silently never offering some chats and offering
    others twice.

    Studio sessions, prompt helpers and setup verification chats are excluded
    here rather than in the caller: a scope filtered in one reader and not
    another is a scope that leaks the first time somebody adds a second reader.

    **The page is filled, not truncated.** `configured` cannot be expressed in
    SQL without writing a second definition of it beside `configured_reasons`,
    and the one that governs a decision has to be the only one. So ineligible
    rows are found by classifying them, and they must not consume the caller's
    limit on the way: a window that happened to be all configured would
    otherwise return an empty page with a cursor, and the surface reading it
    would say there is nothing to clean up while an eligible chat sat one
    cursor away.

    Filling is bounded by `MAX_ROWS_EXAMINED` rather than by the number of rows
    it might have to walk. Exhausting that budget returns a short page with a
    cursor, which is honest - the caller can continue - rather than an unbounded
    scan disguised as a page.
    """

    if limit < 1:
        raise ValueError("A page must contain at least one chat.")
    if minimum_age_hours < 0:
        raise ValueError("A minimum age cannot be negative.")
    bounded = min(limit, MAX_PAGE_SIZE)

    query = select(Chat).where(and_(Chat.scope == STANDARD_CHAT_SCOPE, ~_has_work(Chat.id)))
    # Age is a filter on the row rather than on the page, so a young chat is
    # never counted against the limit and then dropped.
    if minimum_age_hours > 0:
        moment = now or datetime.now(UTC)
        query = query.where(Chat.created_at <= moment - timedelta(hours=minimum_age_hours))
    # Archived is shell state, so an archived chat is `configured_blank` too.
    # It gets its own switch because archiving is the one piece of that state
    # somebody chose as a way of saying "not now" rather than "not mine".
    if not include_archived:
        query = query.where(Chat.archived.is_(False))

    # One more eligible entry than asked for, so "is there a next page" is
    # answered by fact rather than guessed from a full page.
    target = bounded + 1
    entries: list[EmptyChat] = []
    cursor = after_id
    examined = 0
    exhausted = False
    while len(entries) < target and examined < MAX_ROWS_EXAMINED:
        window = query
        if cursor is not None:
            window = window.where(Chat.id > cursor)
        # Ask for what is still wanted, never fewer than one, and never more
        # than the examination budget allows.
        span = min(max(target - len(entries), 1), MAX_ROWS_EXAMINED - examined)
        rows = session.scalars(window.order_by(Chat.id).limit(span)).all()
        if not rows:
            exhausted = True
            break
        for row in rows:
            examined += 1
            cursor = row.id
            entry = classify(session, row)
            if not include_configured and entry.classification is EmptyChatClass.CONFIGURED_BLANK:
                continue
            entries.append(entry)
            if len(entries) == target:
                break
        if len(rows) < span:
            exhausted = True
            break

    if len(entries) > bounded:
        # The extra one proves a next page exists; the caller resumes from the
        # last entry it was actually given.
        entries = entries[:bounded]
        next_cursor: str | None = entries[-1].chat_id
    elif exhausted:
        next_cursor = None
    else:
        # Stopped on the examination budget without proving a successor. A
        # cursor is still owed: the caller has not been told the set is done.
        next_cursor = cursor
    return EmptyChatPage(entries=tuple(entries), next_cursor=next_cursor)


def configured_reasons(session: Session, chat: Chat) -> tuple[str, ...]:
    """Every piece of shell state that makes an empty chat someone's decision."""

    found: list[str] = []
    if chat.title != "New chat":
        found.append("custom_title")
    if chat.project_id:
        found.append("in_project")
    if chat.pinned:
        found.append("pinned")
    if chat.archived:
        found.append("archived")
    if chat.draft_prompt.strip():
        found.append("has_draft")
    if chat.routing_mode != RoutingMode.AUTO.value:
        found.append("routing_chosen")
    if not chat.confirm_uncertain_media:
        found.append("confirmation_changed")
    # Measured against what creating a chat already writes, not against null.
    # Creation sets all four profile pointers to the automatic sentinel and
    # seeds the vision settings, so treating any present value as a decision
    # would make every real chat configured and empty the default list of the
    # exact chats it exists to offer.
    if any(
        value is not None and value != AUTO_PROFILE_ID
        for value in (
            chat.active_chat_profile_id,
            chat.active_vision_profile_id,
            chat.active_image_profile_id,
            chat.active_video_profile_id,
        )
    ):
        found.append("profile_chosen")
    if chat.web_settings_json:
        found.append("web_access_granted")
    if chat.origin_json:
        found.append("forked")
    if (
        chat.generation_settings_json
        or chat.generation_preset_ids_json
        or _vision_settings_differ(chat.vision_settings_json)
    ):
        found.append("settings_overridden")
    if _is_named_as_a_source(session, chat.id):
        found.append("linked_chat")
    return tuple(found)


def _is_named_as_a_source(session: Session, chat_id: str) -> bool:
    """Whether another chat still records this one as where it came from.

    A fork records the chat it was forked from, and a studio session the chat it
    was opened from. Deleting that source would leave the other record naming a
    chat that no longer exists, so a chat something else points back to is a
    decision rather than a blank - whatever that other chat's scope.
    """

    forked = func.json_extract(Chat.origin_json, "$.forked_from_chat_id")
    opened = func.json_extract(Chat.origin_json, "$.source_chat_id")
    return (
        session.scalar(
            select(Chat.id)
            .where(and_(Chat.id != chat_id, or_(forked == chat_id, opened == chat_id)))
            .limit(1)
        )
        is not None
    )


def _vision_settings_differ(value: object) -> bool:
    """True only when vision settings say something the baseline does not.

    A new chat is seeded with the baseline vision settings, so comparing against
    emptiness reports every chat as overridden. Comparing against the baseline
    keeps a real change visible while a freshly created chat stays strict.
    """

    if not isinstance(value, Mapping):
        return bool(value)
    if not value:
        # The column default. Nothing was ever written, so nothing was chosen.
        return False
    return dict(value) != new_chat_vision_settings().model_dump(mode="json")


def classify(session: Session, chat: Chat) -> EmptyChat:
    """Classify one chat that is already known to be empty.

    `inconsistent` is not a worse kind of empty, it is a chat whose state
    contradicts itself: a head pointer naming a message that is not there.
    Nothing may offer to delete those, because a tool that removes what it just
    called inconsistent is removing the evidence.
    """

    if chat.active_head_message_id is not None:
        head = session.get(Message, chat.active_head_message_id)
        if head is None or head.chat_id != chat.id:
            return EmptyChat(
                chat.id,
                EmptyChatClass.INCONSISTENT,
                ("stale_head",),
                chat.created_at,
                chat.updated_at,
            )

    reasons = configured_reasons(session, chat)
    classification = EmptyChatClass.CONFIGURED_BLANK if reasons else EmptyChatClass.STRICT_BLANK
    return EmptyChat(chat.id, classification, reasons, chat.created_at, chat.updated_at)
