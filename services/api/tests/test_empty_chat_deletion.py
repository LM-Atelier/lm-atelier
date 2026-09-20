"""Binding a selection of empty chats, and deleting it once or not at all.

The listing's own tests cover what counts as empty. These cover the two calls
that end in a deletion: the preview that binds a choice to a digest and a
deadline, and the execute that spends it. Almost every case is about a refusal,
because the only interesting property of a delete button is what it will not do.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from local_lm import empty_chats
from local_lm.db import Base
from local_lm.empty_chats import (
    CONFLICT_REASONS,
    PREVIEW_LIFETIME,
    PREVIEW_RETENTION,
    EmptyChat,
    EmptyChatDeletionResult,
    EmptyChatExecuteError,
    EmptyChatExecuteRefusal,
    EmptyChatSelectionFilters,
    execute_deletion,
    issue_preview,
    preview_selection,
)
from local_lm.models import (
    Chat,
    ChatActivityEvent,
    ChatWorkflowSelection,
    EmptyChatDeletion,
    EmptyChatPreviewRecord,
    Message,
    MessageRole,
)
from local_lm.schemas import EmptyChatConflictOut

OLD = timedelta(hours=48)


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def chat(session: Session, **values: object) -> Chat:
    row = Chat(**values)
    session.add(row)
    session.flush()
    return row


def _filters(
    minimum_age_hours: float = 24.0,
    *,
    include_archived: bool = False,
    include_configured: bool = False,
) -> EmptyChatSelectionFilters:
    return EmptyChatSelectionFilters(
        minimum_age_hours=minimum_age_hours,
        include_archived=include_archived,
        include_configured=include_configured,
    )


def _spend(
    session: Session,
    rows: list[Chat],
    *,
    operation_id: str = "op-1",
    filters: EmptyChatSelectionFilters | None = None,
    acknowledged_count: int | None = None,
    acknowledged_configured: bool = False,
    now: datetime | None = None,
) -> EmptyChatDeletionResult:
    """Preview and issue exactly as the surface does, then spend the preview."""

    chosen = filters or _filters(0)
    ids = [row.id for row in rows]
    moment = datetime.now(UTC)
    preview = preview_selection(session, chat_ids=ids, filters=chosen, now=moment)
    preview_id, _ = issue_preview(
        session, digest=preview.digest, chat_states=preview.chat_states, now=moment
    )
    return execute_deletion(
        session,
        operation_id=operation_id,
        preview_id=preview_id,
        digest=preview.digest,
        acknowledged_count=(
            preview.strict_count + preview.configured_count
            if acknowledged_count is None
            else acknowledged_count
        ),
        acknowledged_configured=acknowledged_configured,
        chat_ids=ids,
        filters=chosen,
        now=now,
    )


def _remaining(session: Session, rows: list[Chat]) -> set[str]:
    ids = [row.id for row in rows]
    return set(session.scalars(select(Chat.id).where(Chat.id.in_(ids))))


# ---- the preview ----------------------------------------------------------------------


def test_the_digest_covers_the_classification_and_not_only_the_ids(session: Session) -> None:
    """The reason the digest exists at all.

    Both previews use the same ids and the same filters, so the only thing that
    can move the digest is the classification. A draft typed somewhere else
    has to invalidate a decision made before it.
    """

    blank = chat(session)
    wide = _filters(0, include_configured=True)
    before = preview_selection(session, chat_ids=[blank.id], filters=wide).digest

    blank.draft_prompt = "typed somewhere else"
    session.flush()
    after = preview_selection(session, chat_ids=[blank.id], filters=wide).digest

    assert before != after


def test_the_digest_covers_the_filter_state(session: Session) -> None:
    blank = chat(session)

    plain = preview_selection(session, chat_ids=[blank.id], filters=_filters(0)).digest
    widened = preview_selection(
        session, chat_ids=[blank.id], filters=_filters(0, include_configured=True)
    ).digest

    assert plain != widened


def test_the_digest_does_not_depend_on_the_order_ids_arrive_in(session: Session) -> None:
    first, second = chat(session), chat(session)

    one = preview_selection(session, chat_ids=[first.id, second.id], filters=_filters(0))
    other = preview_selection(session, chat_ids=[second.id, first.id], filters=_filters(0))

    assert one.digest == other.digest


def test_a_member_that_no_longer_belongs_is_named_rather_than_dropped(session: Session) -> None:
    """Silently shrinking the selection would confirm a decision nobody made."""

    blank, busy, titled = chat(session), chat(session), chat(session, title="Mine")
    helper = chat(session, scope="prompt_helper")
    broken = chat(session, active_head_message_id="msg_absent")
    session.add(Message(chat_id=busy.id, role=MessageRole.USER.value))
    session.flush()

    preview = preview_selection(
        session,
        chat_ids=[blank.id, busy.id, titled.id, helper.id, broken.id, "chat_never_existed"],
        filters=_filters(0),
    )

    assert {(entry.chat_id, entry.reason) for entry in preview.conflicts} == {
        (busy.id, "not_empty"),
        (titled.id, "filtered_out"),
        (helper.id, "out_of_scope"),
        (broken.id, "inconsistent"),
        ("chat_never_existed", "missing"),
    }
    assert (preview.strict_count, preview.configured_count) == (1, 0)


def test_the_preview_enforces_the_age_floor_it_binds(session: Session) -> None:
    """Hashing a filter value is not the same as applying it."""

    moment = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    young = chat(session, created_at=moment - timedelta(minutes=5))
    exact = chat(session, created_at=moment - timedelta(hours=24))
    inside = chat(session, created_at=moment - timedelta(hours=24) + timedelta(seconds=1))

    preview = preview_selection(
        session, chat_ids=[young.id, exact.id, inside.id], filters=_filters(), now=moment
    )

    assert {(entry.chat_id, entry.reason) for entry in preview.conflicts} == {
        (young.id, "too_young"),
        (inside.id, "too_young"),
    }
    assert preview.strict_count == 1


def test_archived_is_judged_before_the_configured_classification(session: Session) -> None:
    """An archived chat is configured too, so the configured switch must not admit it."""

    filed = chat(session, archived=True, title="Filed away", created_at=datetime.now(UTC) - OLD)

    def reasons(**switches: bool) -> set[str]:
        preview = preview_selection(session, chat_ids=[filed.id], filters=_filters(**switches))
        return {entry.reason for entry in preview.conflicts}

    assert reasons() == {"archived_excluded"}
    assert reasons(include_configured=True) == {"archived_excluded"}
    assert reasons(include_archived=True) == {"filtered_out"}
    assert reasons(include_archived=True, include_configured=True) == set()


def test_every_conflict_reason_is_a_value_the_api_can_carry() -> None:
    """The two vocabularies are declared apart, because one module cannot import the other."""

    field = EmptyChatConflictOut.model_fields["reason"].annotation
    assert set(CONFLICT_REASONS) == set(field.__args__)


def test_a_second_preview_does_not_move_the_first_ones_deadline(session: Session) -> None:
    row = chat(session, created_at=datetime.now(UTC) - OLD)
    digest = preview_selection(session, chat_ids=[row.id], filters=_filters(0)).digest
    start = datetime.now(UTC)

    first_id, first_deadline = issue_preview(session, digest=digest, chat_states={}, now=start)
    second_id, second_deadline = issue_preview(
        session, digest=digest, chat_states={}, now=start + timedelta(minutes=5)
    )

    assert first_id != second_id
    assert first_deadline == start + PREVIEW_LIFETIME
    assert second_deadline == start + timedelta(minutes=5) + PREVIEW_LIFETIME
    record = session.get(EmptyChatPreviewRecord, first_id)
    assert record is not None
    assert record.expires_at.replace(tzinfo=UTC) == first_deadline


def test_long_expired_previews_are_swept_and_recent_ones_are_kept(session: Session) -> None:
    now = datetime.now(UTC)
    ancient, _ = issue_preview(
        session, digest="a" * 64, chat_states={}, now=now - PREVIEW_RETENTION - timedelta(hours=1)
    )
    lapsed, _ = issue_preview(
        session, digest="b" * 64, chat_states={}, now=now - timedelta(hours=1)
    )

    issue_preview(session, digest="c" * 64, chat_states={}, now=now)

    assert session.get(EmptyChatPreviewRecord, ancient) is None
    assert session.get(EmptyChatPreviewRecord, lapsed) is not None


# ---- spending it ------------------------------------------------------------------------


def test_a_bound_selection_is_deleted_with_the_rows_every_chat_starts_with(
    session: Session,
) -> None:
    first, second = chat(session), chat(session)
    for capability in ("chat", "vision", "image", "video"):
        session.add(
            ChatWorkflowSelection(
                chat_id=first.id, selector_capability=capability, mode="automatic"
            )
        )
    session.flush()

    result = _spend(session, [first, second])

    assert (result.deleted_ids, result.replayed) == (tuple(sorted([first.id, second.id])), False)
    assert _remaining(session, [first, second]) == set()
    record = session.scalar(select(EmptyChatDeletion))
    assert record is not None
    assert (record.operation_id, record.deleted_count) == ("op-1", 2)


def test_a_repeated_operation_returns_the_first_result_rather_than_deleting_again(
    session: Session,
) -> None:
    """A client cannot tell a lost response from a lost request.

    The replay must not even look at the rest of the request: the chats are gone
    and the preview is nonsense, and the first result still comes back unchanged.
    """

    first, second = chat(session), chat(session)
    result = _spend(session, [first, second])

    again = execute_deletion(
        session,
        operation_id="op-1",
        preview_id="not-a-preview",
        digest="0" * 64,
        acknowledged_count=999,
        acknowledged_configured=False,
        chat_ids=["chat_unrelated"],
        filters=_filters(0),
    )

    assert again.replayed is True
    assert again.deleted_ids == result.deleted_ids
    assert again.deleted_at == result.deleted_at


def test_a_chat_that_changed_kind_since_the_preview_refuses_the_whole_selection(
    session: Session,
) -> None:
    first, second = chat(session), chat(session)
    ids = [first.id, second.id]
    moment = datetime.now(UTC)
    stale = preview_selection(session, chat_ids=ids, filters=_filters(0), now=moment)
    preview_id, _ = issue_preview(
        session, digest=stale.digest, chat_states=stale.chat_states, now=moment
    )
    second.title = "Named after the preview"
    session.flush()

    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-2",
            preview_id=preview_id,
            digest=stale.digest,
            acknowledged_count=2,
            acknowledged_configured=False,
            chat_ids=ids,
            filters=_filters(0),
        )

    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert refused.value.chat_ids == (second.id,)
    assert _remaining(session, [first, second]) == {first.id, second.id}


@pytest.mark.parametrize("work", ["message", "activity"])
def test_a_chat_that_gained_work_since_the_preview_refuses_the_whole_selection(
    session: Session,
    work: str,
) -> None:
    first, second = chat(session), chat(session)
    ids = [first.id, second.id]
    moment = datetime.now(UTC)
    stale = preview_selection(session, chat_ids=ids, filters=_filters(0), now=moment)
    preview_id, _ = issue_preview(
        session, digest=stale.digest, chat_states=stale.chat_states, now=moment
    )
    if work == "activity":
        session.add(
            ChatActivityEvent(
                id="act_completed",
                chat_id=first.id,
                message_id="msg_absent",
                response_revision_id="rev_absent",
                job_id="job_completed",
                attempt=1,
                kind="output",
                occurred_at=moment,
            )
        )
    else:
        session.add(Message(chat_id=first.id, role=MessageRole.USER.value))
    session.flush()

    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-busy",
            preview_id=preview_id,
            digest=stale.digest,
            acknowledged_count=2,
            acknowledged_configured=False,
            chat_ids=ids,
            filters=_filters(0),
        )

    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert _remaining(session, [first, second]) == {first.id, second.id}


#: Edits that leave a configured chat configured, for exactly the same reasons.
SAME_KIND_EDITS: dict[str, tuple[dict[str, object], dict[str, object]]] = {
    "title": ({"title": "Harbor study"}, {"title": "Orchard study"}),
    "draft": ({"draft_prompt": "A quiet harbor"}, {"draft_prompt": "A quiet orchard at dusk"}),
    "settings": (
        {"generation_settings_json": {"image": {"steps": 20}}},
        {"generation_settings_json": {"image": {"steps": 30}}},
    ),
    "web access": (
        {"web_settings_json": {"enabled": True}},
        {"web_settings_json": {"enabled": True, "max_results": 3}},
    ),
    # A column the classification never reads still counts as a change.
    "last update": ({}, {"updated_at": datetime(2026, 9, 1, tzinfo=UTC)}),
}


def _issued(
    session: Session, rows: list[Chat], filters: EmptyChatSelectionFilters
) -> tuple[str, str, int]:
    moment = datetime.now(UTC)
    preview = preview_selection(
        session, chat_ids=[row.id for row in rows], filters=filters, now=moment
    )
    preview_id, _ = issue_preview(
        session, digest=preview.digest, chat_states=preview.chat_states, now=moment
    )
    return preview_id, preview.digest, preview.strict_count + preview.configured_count


@pytest.mark.parametrize("edit", sorted(SAME_KIND_EDITS))
def test_a_chat_changed_without_changing_kind_is_named_and_kept(
    session: Session, edit: str
) -> None:
    """The kind and reasons match, so only the chat's state can tell."""

    before, after = SAME_KIND_EDITS[edit]
    untouched = chat(session, title="Untouched study")
    edited = chat(session, **{"title": "Kept study", "pinned": True, **before})
    rows = [untouched, edited]
    filters = _filters(0, include_configured=True)
    preview_id, digest, count = _issued(session, rows, filters)
    reasons_before = empty_chats.classify(session, edited).reasons
    for key, value in after.items():
        setattr(edited, key, value)
    session.flush()
    assert empty_chats.classify(session, edited).reasons == reasons_before

    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-edited",
            preview_id=preview_id,
            digest=digest,
            acknowledged_count=count,
            acknowledged_configured=True,
            chat_ids=[row.id for row in rows],
            filters=filters,
        )

    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert refused.value.chat_ids == (edited.id,)
    assert _remaining(session, rows) == {untouched.id, edited.id}
    assert session.scalar(select(EmptyChatDeletion.id)) is None


def test_another_chat_newly_naming_this_one_as_its_source_is_a_change(
    session: Session,
) -> None:
    """Already linked, so the reason stays; a second link is still a new fact."""

    source = chat(session, title="Source study")
    chat(session, origin_json={"forked_from_chat_id": source.id})
    filters = _filters(0, include_configured=True)
    preview_id, digest, count = _issued(session, [source], filters)
    chat(session, origin_json={"source_chat_id": source.id})

    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-linked",
            preview_id=preview_id,
            digest=digest,
            acknowledged_count=count,
            acknowledged_configured=True,
            chat_ids=[source.id],
            filters=filters,
        )

    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert refused.value.chat_ids == (source.id,)


def test_an_unchanged_configured_chat_is_deleted_with_its_acknowledgement(
    session: Session,
) -> None:
    configured = chat(session, title="Harbor study", draft_prompt="A quiet harbor")

    result = _spend(
        session,
        [configured],
        filters=_filters(0, include_configured=True),
        acknowledged_configured=True,
    )

    assert result.deleted_ids == (configured.id,)
    assert _remaining(session, [configured]) == set()


def test_a_fresh_preview_after_an_edit_can_be_spent(session: Session) -> None:
    edited = chat(session, title="Harbor study")
    filters = _filters(0, include_configured=True)
    stale_id, stale_digest, count = _issued(session, [edited], filters)
    edited.title = "Orchard study"
    session.flush()
    with pytest.raises(EmptyChatExecuteError):
        execute_deletion(
            session,
            operation_id="op-stale",
            preview_id=stale_id,
            digest=stale_digest,
            acknowledged_count=count,
            acknowledged_configured=True,
            chat_ids=[edited.id],
            filters=filters,
        )

    result = _spend(
        session, [edited], operation_id="op-fresh", filters=filters, acknowledged_configured=True
    )

    assert result.deleted_ids == (edited.id,)


def test_what_a_chat_says_reaches_neither_the_digest_nor_the_deletion_record(
    session: Session,
) -> None:
    """The caller's digest and the durable record hold nothing a title could be guessed from."""

    first = chat(session, title="Harbor study", draft_prompt="A quiet harbor")
    filters = _filters(0, include_configured=True)
    digest = preview_selection(session, chat_ids=[first.id], filters=filters).digest
    first.title, first.draft_prompt = "Orchard study", "A quiet orchard at dusk"
    session.flush()
    assert preview_selection(session, chat_ids=[first.id], filters=filters).digest == digest

    _spend(session, [first], filters=filters, acknowledged_configured=True)

    record = session.scalars(select(EmptyChatDeletion)).one()
    assert set(EmptyChatDeletion.__table__.columns.keys()) == {
        "id",
        "operation_id",
        "digest",
        "deleted_ids_json",
        "deleted_count",
        "deleted_at",
        "created_at",
        "updated_at",
    }
    assert record.digest == digest


def test_a_chosen_chat_the_preview_already_excluded_is_never_deleted(session: Session) -> None:
    """A preview binds only the members that belonged; the rest cannot ride along.

    The young chat is empty and would delete cleanly, which is exactly why it is
    the case to check: its preview named it too young, so spending that preview
    with the same ids must refuse rather than delete it beside the eligible one.
    """

    moment = datetime.now(UTC)
    old = chat(session, created_at=moment - OLD)
    young = chat(session, created_at=moment - timedelta(minutes=1))
    ids = [old.id, young.id]
    preview = preview_selection(session, chat_ids=ids, filters=_filters(), now=moment)
    assert [entry.reason for entry in preview.conflicts] == ["too_young"]
    preview_id, _ = issue_preview(
        session, digest=preview.digest, chat_states=preview.chat_states, now=moment
    )

    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-young",
            preview_id=preview_id,
            digest=preview.digest,
            acknowledged_count=1,
            acknowledged_configured=False,
            chat_ids=ids,
            filters=_filters(),
            now=moment,
        )

    assert refused.value.refusal is EmptyChatExecuteRefusal.SELECTION_DRIFTED
    assert _remaining(session, [old, young]) == {old.id, young.id}


def test_the_count_must_match_what_the_preview_bound(session: Session) -> None:
    blank = chat(session)

    with pytest.raises(EmptyChatExecuteError) as refused:
        _spend(session, [blank], acknowledged_count=2)

    assert refused.value.refusal is EmptyChatExecuteRefusal.COUNT_MISMATCH
    assert _remaining(session, [blank]) == {blank.id}


def test_configured_chats_need_their_own_acknowledgement(session: Session) -> None:
    """The count catches a selection that grew; this catches one that changed kind."""

    titled = chat(session, title="Somebody named this")
    wide = _filters(0, include_configured=True)

    with pytest.raises(EmptyChatExecuteError) as refused:
        _spend(session, [titled], filters=wide, acknowledged_configured=False)
    assert refused.value.refusal is EmptyChatExecuteRefusal.CONFIGURED_NOT_ACKNOWLEDGED
    assert _remaining(session, [titled]) == {titled.id}

    assert _spend(session, [titled], filters=wide, acknowledged_configured=True).deleted_ids == (
        titled.id,
    )


def test_a_preview_is_spendable_at_its_deadline_and_not_a_tick_after(session: Session) -> None:
    """Both sides of the boundary, so neither is asserted alone."""

    late, on_time = chat(session), chat(session)
    moment = datetime.now(UTC)

    def issued(row: Chat) -> tuple[str, str, datetime]:
        preview = preview_selection(session, chat_ids=[row.id], filters=_filters(0), now=moment)
        preview_id, deadline = issue_preview(
            session, digest=preview.digest, chat_states=preview.chat_states, now=moment
        )
        return preview_id, preview.digest, deadline

    late_id, late_digest, deadline = issued(late)
    with pytest.raises(EmptyChatExecuteError) as refused:
        execute_deletion(
            session,
            operation_id="op-late",
            preview_id=late_id,
            digest=late_digest,
            acknowledged_count=1,
            acknowledged_configured=False,
            chat_ids=[late.id],
            filters=_filters(0),
            now=deadline + timedelta(microseconds=1),
        )
    assert refused.value.refusal is EmptyChatExecuteRefusal.PREVIEW_EXPIRED
    assert _remaining(session, [late]) == {late.id}

    on_time_id, on_time_digest, deadline = issued(on_time)
    result = execute_deletion(
        session,
        operation_id="op-on-time",
        preview_id=on_time_id,
        digest=on_time_digest,
        acknowledged_count=1,
        acknowledged_configured=False,
        chat_ids=[on_time.id],
        filters=_filters(0),
        now=deadline,
    )
    assert result.deleted_ids == (on_time.id,)


def test_a_digest_the_server_never_issued_is_not_a_preview(session: Session) -> None:
    """Neither a made-up id nor a real id paired with another selection's digest."""

    first, second = chat(session), chat(session)
    moment = datetime.now(UTC)
    one = preview_selection(session, chat_ids=[first.id], filters=_filters(0), now=moment)
    other = preview_selection(session, chat_ids=[second.id], filters=_filters(0), now=moment)
    one_id, _ = issue_preview(session, digest=one.digest, chat_states=one.chat_states, now=moment)

    for preview_id, digest in ((one_id, other.digest), ("emptyprev_invented", other.digest)):
        with pytest.raises(EmptyChatExecuteError) as refused:
            execute_deletion(
                session,
                operation_id=f"op-{preview_id}",
                preview_id=preview_id,
                digest=digest,
                acknowledged_count=1,
                acknowledged_configured=False,
                chat_ids=[second.id],
                filters=_filters(0),
            )
        assert refused.value.refusal is EmptyChatExecuteRefusal.PREVIEW_UNKNOWN

    assert _remaining(session, [first, second]) == {first.id, second.id}


# ---- against a second connection -----------------------------------------------------------


def _file_engine(path: Path, *, timeout: float) -> object:
    return create_engine(f"sqlite:///{path.as_posix()}", connect_args={"timeout": timeout})


def test_a_second_connection_cannot_add_work_between_the_check_and_the_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race a single-session test cannot reach.

    Waiting until the function returns would prove only that a DELETE takes a
    write lock, which it does with or without the reservation taken first. The
    window that matters is between revalidating and deleting, so the competing
    writer runs from inside the revalidation - and must be refused there.
    """

    path = tmp_path / "race.db"
    engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"timeout": 0})
    Base.metadata.create_all(engine)
    with engine.connect() as setup:
        setup.exec_driver_sql("PRAGMA journal_mode=WAL")
    try:
        with Session(engine, expire_on_commit=False) as seed:
            target = Chat()
            seed.add(target)
            seed.flush()
            moment = datetime.now(UTC)
            preview = preview_selection(seed, chat_ids=[target.id], filters=_filters(0), now=moment)
            preview_id, _ = issue_preview(
                seed, digest=preview.digest, chat_states=preview.chat_states, now=moment
            )
            seed.commit()
            chat_id = target.id

        deleter = Session(engine, expire_on_commit=False)
        writer = Session(engine, expire_on_commit=False)
        outcome: list[str] = []
        original = empty_chats.classify

        def intrude(session: Session, row: Chat) -> EmptyChat:
            if not outcome:
                try:
                    writer.add(Message(chat_id=row.id, role=MessageRole.USER.value))
                    writer.commit()
                    outcome.append("wrote")
                except OperationalError:
                    writer.rollback()
                    outcome.append("refused")
            return original(session, row)

        monkeypatch.setattr(empty_chats, "classify", intrude)
        try:
            result = execute_deletion(
                deleter,
                operation_id="op-race",
                preview_id=preview_id,
                digest=preview.digest,
                acknowledged_count=1,
                acknowledged_configured=False,
                chat_ids=[chat_id],
                filters=_filters(0),
            )
            deleter.commit()

            assert outcome == ["refused"]
            assert result.deleted_ids == (chat_id,)
            assert deleter.get(Chat, chat_id) is None
        finally:
            deleter.close()
            writer.close()
    finally:
        engine.dispose()


def test_the_same_operation_from_two_connections_deletes_once_and_replays_once(
    tmp_path: Path,
) -> None:
    """Two genuinely concurrent identical requests, on two real connections and threads.

    On one shared connection each session would see the other's commit and
    the test would pass without the writer reservation at all.
    """

    path = tmp_path / "twice.db"
    setup_engine = create_engine(f"sqlite:///{path.as_posix()}")
    Base.metadata.create_all(setup_engine)
    with Session(setup_engine, expire_on_commit=False) as setup:
        row = Chat(created_at=datetime.now(UTC) - OLD)
        setup.add(row)
        setup.flush()
        moment = datetime.now(UTC)
        bound = preview_selection(setup, chat_ids=[row.id], filters=_filters(0), now=moment)
        preview_id, _ = issue_preview(
            setup, digest=bound.digest, chat_states=bound.chat_states, now=moment
        )
        setup.commit()
        chat_id, digest = row.id, bound.digest
    setup_engine.dispose()

    both_ready = threading.Barrier(2)
    outcomes: dict[str, object] = {}

    def attempt(tag: str) -> None:
        engine = create_engine(f"sqlite:///{path.as_posix()}", connect_args={"timeout": 30})
        try:
            with Session(engine, expire_on_commit=False) as session:
                both_ready.wait()
                try:
                    result = execute_deletion(
                        session,
                        operation_id="op-twice",
                        preview_id=preview_id,
                        digest=digest,
                        acknowledged_count=1,
                        acknowledged_configured=False,
                        chat_ids=[chat_id],
                        filters=_filters(0),
                    )
                    session.commit()
                    outcomes[tag] = result.replayed
                except Exception as failure:
                    outcomes[tag] = failure
        finally:
            engine.dispose()

    threads = [threading.Thread(target=attempt, args=(tag,)) for tag in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert sorted(outcomes.values(), key=repr) == [False, True], outcomes
