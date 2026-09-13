"""Which chats are empty, and which of those are safe to offer for deletion.

Emptiness is settled by enumeration rather than by judgement. Every table with a
foreign key to `chats.id` is listed in `WORK_PRODUCERS` or `CONFIGURATION_TABLES`.
A chat with no real row in the first has no transcript and no work, and that is
the entire test; a row in the second is somebody's decision about the chat and
makes it configured rather than busy. A structural test reads both lists back
off the schema, so a table added later that references a chat fails a test
instead of silently widening what this tool is willing to offer.

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

Reading is most of this module. Deleting happens in one place,
`execute_deletion`, and only for a selection a preview bound moments earlier and
that still classifies exactly as it did then.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Literal, get_args

from sqlalchemy import ColumnElement, and_, delete, exists, func, or_, select
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy.orm import InstrumentedAttribute, Session

from .artifact_library import begin_artifact_write_fence
from .chat_composer_drafts import has_content as composer_draft_has_content
from .domain import JobKind, RoutingMode
from .models import (
    Chat,
    ChatComposerDraft,
    ChatItemRemovalReceipt,
    ChatWorkflowSelection,
    EmptyChatDeletion,
    EmptyChatPreviewRecord,
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


#: Every table with a foreign key to `chats.id` whose rows are work, alongside
#: `CONFIGURATION_TABLES`. Enforced structurally by
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

#: Tables with a foreign key to `chats.id` whose rows are not work: nothing was
#: sent or ran. An unsent composer draft is text and attachments somebody left
#: there, so it makes an otherwise empty chat configured (`has_draft`), which is
#: listed but never selected for deletion without acknowledgement.
CONFIGURATION_TABLES = (ChatComposerDraft,)

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
    # Either kind of unsent text: a prompt helper's draft, or whatever is waiting
    # in the composer, including only an attachment.
    if chat.draft_prompt.strip() or composer_draft_has_content(
        session.get(ChatComposerDraft, chat.id)
    ):
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


# ---- binding a selection, and deleting it once --------------------------------------


#: How long a preview may be spent. Short, because the whole point of the digest
#: is that the world moves underneath it.
PREVIEW_LIFETIME = timedelta(minutes=10)

#: How long an expired issuance is kept before it is swept. Long enough that a
#: caller who lapsed is told their preview expired rather than that it never
#: existed, which are different problems with different fixes.
PREVIEW_RETENTION = timedelta(days=1)


@dataclass(frozen=True)
class EmptyChatSelectionFilters:
    """The filter state a selection was chosen under, bound into its digest."""

    minimum_age_hours: float
    include_archived: bool
    include_configured: bool


#: Why a chosen chat is no longer part of the selection, as stable slugs.
ConflictReason = Literal[
    "missing",
    "out_of_scope",
    "not_empty",
    "too_young",
    "archived_excluded",
    "inconsistent",
    "filtered_out",
]
CONFLICT_REASONS: tuple[ConflictReason, ...] = get_args(ConflictReason)


@dataclass(frozen=True)
class EmptyChatConflictEntry:
    chat_id: str
    reason: ConflictReason


@dataclass(frozen=True)
class EmptyChatPreview:
    digest: str
    strict_count: int
    configured_count: int
    conflicts: tuple[EmptyChatConflictEntry, ...]
    #: Each bound chat's state fingerprint, by id. Kept with the issuance on the
    #: server and never sent to the caller; see `chat_state_fingerprint`.
    chat_states: Mapping[str, str]


def _as_utc(value: datetime) -> datetime:
    """SQLite keeps these naive and they are UTC, as the serializers assume."""

    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _jsonable(value: object) -> object:
    if isinstance(value, datetime):
        return _as_utc(value).isoformat()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"no canonical form for {type(value).__name__}")


def chat_state_fingerprint(session: Session, row: Chat) -> str:
    """A hash of everything a chat currently is, so any change to it is noticed.

    The classification and its reasons say what KIND of chat a preview bound,
    and a chat can change without changing kind: a configured chat whose title is
    edited or whose draft is rewritten is still configured, for the same reasons,
    and a deletion confirmed against the old version would remove something
    nobody was shown. So every column of the chat is covered, not a chosen list
    of them, which also keeps a column added later from being silently left out.
    The chats that name this one as their source are covered by id, because
    they live on other rows.

    Nothing readable leaves the server. The fingerprint is kept only with the
    issuance it belongs to - never in the caller's digest, and never in the
    record of a completed deletion - so a deleted chat's title or draft cannot
    be recovered by guessing at a stored hash after the chat is gone.
    """

    state = {
        attribute.key: getattr(row, attribute.key)
        for attribute in sqlalchemy_inspect(Chat).column_attrs
    }
    forked = func.json_extract(Chat.origin_json, "$.forked_from_chat_id")
    opened = func.json_extract(Chat.origin_json, "$.source_chat_id")
    state["named_as_source_by"] = list(
        session.scalars(
            select(Chat.id)
            .where(and_(Chat.id != row.id, or_(forked == row.id, opened == row.id)))
            .order_by(Chat.id)
        )
    )
    # The unsent draft lives on its own row. All of it is covered, not just its
    # revision, so the fingerprint does not depend on how revisions are counted.
    draft = session.get(ChatComposerDraft, row.id)
    state["composer_draft"] = (
        None
        if draft is None
        else {
            **{
                attribute.key: getattr(draft, attribute.key)
                for attribute in sqlalchemy_inspect(ChatComposerDraft).column_attrs
            },
            "attachments": [attachment.artifact_id for attachment in draft.attachments],
        }
    )
    canonical = json.dumps(
        state, default=_jsonable, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def selection_digest(entries: Iterable[EmptyChat], filters: EmptyChatSelectionFilters) -> str:
    """A digest over what the answer was computed from, not over the ids alone.

    Binding ids alone would still match after a chat changed kind - somebody
    opens it in another window, types a draft, picks a workflow - and the delete
    would proceed on a classification nobody saw. Covering each classification
    and its reasons means that drift is refused instead of confirmed.

    The filter state is bound too, because the same ids chosen with configured
    chats included describe a different decision from the same ids chosen
    without them.

    A change that keeps a chat's kind is caught by its state fingerprint, which
    is bound to the issuance on the server rather than folded in here: this
    digest goes back to the caller and into the record of a deletion.
    """

    payload = {
        "filters": {
            "minimum_age_hours": filters.minimum_age_hours,
            "include_archived": filters.include_archived,
            "include_configured": filters.include_configured,
        },
        "entries": [
            {
                "id": entry.chat_id,
                "classification": entry.classification.value,
                "reasons": sorted(entry.reasons),
            }
            for entry in sorted(entries, key=lambda item: item.chat_id)
        ],
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def preview_selection(
    session: Session,
    *,
    chat_ids: Iterable[str],
    filters: EmptyChatSelectionFilters,
    now: datetime | None = None,
) -> EmptyChatPreview:
    """Bind a selection, and name every member that no longer belongs to it.

    **Every predicate the page applied is applied again here.** Binding a filter
    value into a digest does not establish that the rows satisfy it: the digest
    proves what the answer was computed from, and this is what computes it. So a
    chat younger than the floor, or archived with archived excluded, is a
    conflict here even though the same filter state is hashed.

    The age cutoff is taken from one evaluation instant, so two members of the
    same selection are never judged against different moments.
    """

    moment = now or datetime.now(UTC)
    cutoff = moment - timedelta(hours=filters.minimum_age_hours)
    wanted = sorted(set(chat_ids))
    rows = {row.id: row for row in session.scalars(select(Chat).where(Chat.id.in_(wanted)))}
    empty = set(
        session.scalars(select(Chat.id).where(and_(Chat.id.in_(wanted), ~_has_work(Chat.id))))
    )

    conflicts: list[EmptyChatConflictEntry] = []
    bound: list[EmptyChat] = []
    states: dict[str, str] = {}
    for chat_id in wanted:
        row = rows.get(chat_id)
        if row is None:
            conflicts.append(EmptyChatConflictEntry(chat_id, "missing"))
            continue
        if row.scope != STANDARD_CHAT_SCOPE:
            # The page never offers another scope, so one arriving here was not
            # chosen from this surface.
            conflicts.append(EmptyChatConflictEntry(chat_id, "out_of_scope"))
            continue
        if chat_id not in empty:
            conflicts.append(EmptyChatConflictEntry(chat_id, "not_empty"))
            continue
        if filters.minimum_age_hours > 0 and _as_utc(row.created_at) > cutoff:
            conflicts.append(EmptyChatConflictEntry(chat_id, "too_young"))
            continue
        # Judged before the configured classification, because an archived chat
        # is configured too: deciding it by classification would let including
        # configured chats admit one the archived switch excluded.
        if row.archived and not filters.include_archived:
            conflicts.append(EmptyChatConflictEntry(chat_id, "archived_excluded"))
            continue
        entry = classify(session, row)
        if entry.classification is EmptyChatClass.INCONSISTENT:
            conflicts.append(EmptyChatConflictEntry(chat_id, "inconsistent"))
            continue
        if (
            entry.classification is EmptyChatClass.CONFIGURED_BLANK
            and not filters.include_configured
        ):
            conflicts.append(EmptyChatConflictEntry(chat_id, "filtered_out"))
            continue
        bound.append(entry)
        states[chat_id] = chat_state_fingerprint(session, row)

    return EmptyChatPreview(
        digest=selection_digest(bound, filters),
        strict_count=sum(
            1 for entry in bound if entry.classification is EmptyChatClass.STRICT_BLANK
        ),
        configured_count=sum(
            1 for entry in bound if entry.classification is EmptyChatClass.CONFIGURED_BLANK
        ),
        conflicts=tuple(conflicts),
        chat_states=states,
    )


def issue_preview(
    session: Session, *, digest: str, chat_states: Mapping[str, str], now: datetime
) -> tuple[str, datetime]:
    """Record one issuance, and return the id and deadline that belong to it.

    The bound chats' state fingerprints are recorded with it, so a deletion can
    tell whether any of them changed since, without the caller ever holding them.

    A new row every time, never an update: a second preview of the same selection
    must not move the first one's deadline. Long-expired rows are swept on the
    way past, well after expiry, so a preview that has only just lapsed still
    refuses as expired rather than as one that never existed.
    """

    expires_at = now + PREVIEW_LIFETIME
    record = EmptyChatPreviewRecord(
        digest=digest,
        chat_states_json=dict(chat_states),
        evaluated_at=now,
        expires_at=expires_at,
    )
    session.add(record)
    session.execute(
        delete(EmptyChatPreviewRecord)
        .where(EmptyChatPreviewRecord.expires_at < now - PREVIEW_RETENTION)
        # Decided by the database alone. Evaluating the condition against rows
        # already loaded in this session would compare SQLite's naive stored
        # instants with an aware one and fail; a swept row is long expired and
        # nothing in the session has any use for it.
        .execution_options(synchronize_session=False)
    )
    session.flush()
    return record.id, expires_at


class EmptyChatExecuteRefusal(StrEnum):
    """Every way a deletion is refused, as the stable code the surface returns."""

    #: The preview's deadline has passed.
    PREVIEW_EXPIRED = "empty-chat-preview-expired"
    #: No preview was issued with this id and digest. Distinct from expiry on
    #: purpose: "your preview is too old" and "you did not preview this" call for
    #: different things from the caller.
    PREVIEW_UNKNOWN = "empty-chat-preview-unknown"
    SELECTION_DRIFTED = "empty-chat-selection-drifted"
    COUNT_MISMATCH = "empty-chat-count-mismatch"
    CONFIGURED_NOT_ACKNOWLEDGED = "empty-chat-configured-not-acknowledged"


class EmptyChatExecuteError(Exception):
    """A refusal raised before anything is deleted, naming ids and never content."""

    def __init__(self, refusal: EmptyChatExecuteRefusal, chat_ids: Iterable[str] = ()) -> None:
        self.refusal = refusal
        self.chat_ids = tuple(sorted(chat_ids))
        super().__init__(refusal.value)


@dataclass(frozen=True)
class EmptyChatDeletionResult:
    operation_id: str
    deleted_ids: tuple[str, ...]
    deleted_at: datetime
    replayed: bool


def recorded_deletion(session: Session, operation_id: str) -> EmptyChatDeletionResult | None:
    """The result an operation id already settled, returned exactly as it was."""

    existing = session.scalar(
        select(EmptyChatDeletion).where(EmptyChatDeletion.operation_id == operation_id)
    )
    if existing is None:
        return None
    return EmptyChatDeletionResult(
        operation_id=existing.operation_id,
        deleted_ids=tuple(existing.deleted_ids_json),
        # Normalized, because SQLite returns it naive and a replay must report
        # the same instant the first call returned.
        deleted_at=_as_utc(existing.deleted_at),
        replayed=True,
    )


def execute_deletion(
    session: Session,
    *,
    operation_id: str,
    preview_id: str,
    digest: str,
    acknowledged_count: int,
    acknowledged_configured: bool,
    chat_ids: Iterable[str],
    filters: EmptyChatSelectionFilters,
    now: datetime | None = None,
) -> EmptyChatDeletionResult:
    """Delete a previewed selection once, however many times it is asked for.

    The caller holds every selected chat's lifecycle guard, so no turn can be
    created in one of them while this runs. This takes the database's writer
    reservation before its first read, so the revalidation and the deletion
    observe one state and a competing writer waits rather than interleaving.

    Every refusal happens before anything is deleted, and nothing is ever
    half-deleted. The order of the checks is chosen for what the caller should
    hear: a replay first, then an unknown or expired preview, then drift, and
    only then the acknowledgements - a caller whose selection drifted should be
    told that, not that its count is wrong, because the count was right for what
    it was shown.

    **The retry guard is a unique constraint, not a check.** A repeated
    operation id returns the recorded result unchanged, including when the
    world has moved on since: what it reports is what that operation did.
    """

    moment = now or datetime.now(UTC)
    begin_artifact_write_fence(session)
    replay = recorded_deletion(session, operation_id)
    if replay is not None:
        return replay

    issued = session.scalar(
        select(EmptyChatPreviewRecord).where(
            EmptyChatPreviewRecord.id == preview_id,
            # Both must agree. The id names an issuance and the digest says what
            # was selected; a pair that disagrees is not a preview of anything.
            EmptyChatPreviewRecord.digest == digest,
        )
    )
    if issued is None:
        raise EmptyChatExecuteError(EmptyChatExecuteRefusal.PREVIEW_UNKNOWN)
    if moment > _as_utc(issued.expires_at):
        raise EmptyChatExecuteError(EmptyChatExecuteRefusal.PREVIEW_EXPIRED)

    wanted = sorted(set(chat_ids))
    current = preview_selection(session, chat_ids=wanted, filters=filters, now=moment)
    issued_states = issued.chat_states_json if isinstance(issued.chat_states_json, dict) else {}
    # A chat that kept its kind but not its state: an edited title, a rewritten
    # draft. Named by id like any other drift, and never by what changed.
    changed = {
        chat_id
        for chat_id in wanted
        if chat_id in current.chat_states
        and current.chat_states[chat_id] != issued_states.get(chat_id)
    }
    if current.digest != digest or current.conflicts or changed:
        raise EmptyChatExecuteError(
            EmptyChatExecuteRefusal.SELECTION_DRIFTED,
            changed | {entry.chat_id for entry in current.conflicts},
        )
    if acknowledged_count != current.strict_count + current.configured_count:
        raise EmptyChatExecuteError(EmptyChatExecuteRefusal.COUNT_MISMATCH)
    if current.configured_count > 0 and not acknowledged_configured:
        raise EmptyChatExecuteError(EmptyChatExecuteRefusal.CONFIGURED_NOT_ACKNOWLEDGED)

    for row in session.scalars(select(Chat).where(Chat.id.in_(wanted)).order_by(Chat.id)):
        session.delete(row)
    session.flush()
    session.add(
        EmptyChatDeletion(
            operation_id=operation_id,
            digest=digest,
            deleted_ids_json=list(wanted),
            deleted_count=len(wanted),
            deleted_at=moment,
        )
    )
    session.flush()
    return EmptyChatDeletionResult(
        operation_id=operation_id,
        deleted_ids=tuple(wanted),
        deleted_at=moment,
        replayed=False,
    )
