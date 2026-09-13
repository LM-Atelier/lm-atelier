"""What counts as an empty chat, and what makes one unsafe to offer.

The feature this serves ends in a delete button, so the interesting question is
what the classifier refuses to call empty. These tests are mostly about chats
that look empty and are not.

`test_a_claimed_turn_is_work_before_it_is_a_message` is the one to read. It is
the case no title heuristic and no glance at the transcript can see, and it is
why emptiness is settled by enumerating every table that references a chat
rather than by judgement.
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import JobKind, Operation
from local_lm.empty_chats import (
    CONFIGURED_REASONS,
    MAX_PAGE_SIZE,
    WORK_PRODUCERS,
    EmptyChatClass,
    classify,
    empty_chat_page,
)
from local_lm.models import (
    Artifact,
    Chat,
    ChatItemRemovalReceipt,
    ChatWorkflowSelection,
    Job,
    Message,
    MessageRole,
    PromptExpansionBatch,
    Run,
    SetupVerification,
    TurnCreationClaim,
    WorkflowFamily,
    WorkPlan,
)


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


def studio(session: Session, **origin: str) -> Chat:
    """A studio session, which needs a real artifact to have been opened on."""

    digest = "f" * 64
    if session.get(Artifact, f"sha256:{digest}") is None:
        session.add(
            Artifact(
                id=f"sha256:{digest}",
                sha256=digest,
                media_type="image/png",
                size_bytes=1,
                relative_path=f"ff/ff/{digest}",
            )
        )
        session.flush()
    return chat(
        session,
        scope="studio",
        archived=True,
        origin_json={"source_artifact_id": f"sha256:{digest}", **origin},
    )


def candidates(session: Session) -> list[str]:
    return [entry.chat_id for entry in empty_chat_page(session, limit=MAX_PAGE_SIZE).entries]


def shown(session: Session, moment: datetime, **switches: bool) -> set[str]:
    page = empty_chat_page(
        session,
        limit=MAX_PAGE_SIZE,
        minimum_age_hours=24,
        now=moment,
        include_archived=switches.get("include_archived", False),
        include_configured=switches.get("include_configured", False),
    )
    return {entry.chat_id for entry in page.entries}


def test_an_untouched_chat_is_a_strict_blank(session: Session) -> None:
    blank = chat(session)

    assert candidates(session) == [blank.id]
    assert classify(session, blank).classification is EmptyChatClass.STRICT_BLANK


def test_a_claimed_turn_is_work_before_it_is_a_message(session: Session) -> None:
    """The case that makes enumeration necessary rather than tidy.

    A turn can be claimed and in flight with nothing written yet. The chat has
    no messages, no runs and no plans; its title is still the default and its
    head pointer is still null. Every visible signal says empty, and it is
    seconds from having a transcript.
    """

    claimed = chat(session)
    session.add(TurnCreationClaim(chat_id=claimed.id, idempotency_key="k", owner_token="t"))
    session.flush()

    assert candidates(session) == []


def _add_work(session: Session, busy: Chat, table: str) -> None:
    """One row in exactly one work table, and nothing else.

    The run and the removal receipt name messages that were never written. This
    in-memory schema does not enforce foreign keys, and that is what lets each
    case prove its own table alone disqualifies: with real messages beside them,
    a classifier that ignored runs or receipts would still pass on the messages.
    """

    if table == "message":
        session.add(Message(chat_id=busy.id, role=MessageRole.USER.value))
    elif table == "run":
        session.add(
            Run(
                chat_id=busy.id,
                user_message_id="msg_absent_user",
                assistant_message_id="msg_absent_assistant",
                operation=Operation.TEXT.value,
            )
        )
    elif table == "removal":
        session.add(
            ChatItemRemovalReceipt(
                chat_id=busy.id,
                operation_key="remove-one",
                message_id="msg_absent",
                request_sha256="a" * 64,
                message_revision_id="b" * 64,
                content_removed_at=datetime(2026, 9, 1, tzinfo=UTC),
            )
        )
    elif table == "plan":
        session.add(WorkPlan(chat_id=busy.id, transcript_sequence=1))
    elif table == "batch":
        session.add(
            PromptExpansionBatch(
                chat_id=busy.id,
                idempotency_key="batch-one",
                prompt_template_id="ptdef_neutral",
                prompt_template_revision_id="ptrev_neutral",
                contract_sha256="c" * 64,
                request_json='{"item_count": 1}',
                model_snapshot_json='{"version": 1}',
                original_plan_sha256="d" * 64,
                plan_sha256="d" * 64,
            )
        )
    elif table == "claim":
        session.add(TurnCreationClaim(chat_id=busy.id, idempotency_key="k", owner_token="t"))
    else:
        # An explicit choice, not the automatic row that creating a chat writes
        # for all four capabilities. Those have their own test below.
        family = WorkflowFamily(name="Chosen")
        session.add(family)
        session.flush()
        session.add(
            ChatWorkflowSelection(
                chat_id=busy.id,
                selector_capability="image",
                mode="family",
                workflow_family_id=family.id,
            )
        )
    session.flush()


_WORK_KINDS = ["message", "plan", "run", "claim", "removal", "batch", "workflow"]


@pytest.mark.parametrize("table", _WORK_KINDS)
def test_any_work_table_is_enough_to_be_busy(session: Session, table: str) -> None:
    """Each one alone disqualifies. A case per table, because one `or` typo hides the rest."""

    _add_work(session, chat(session), table)

    assert candidates(session) == []


def test_every_audited_table_has_its_own_busy_case() -> None:
    """The parametrized case above has to grow with the audit, not lag behind it."""

    assert len(_WORK_KINDS) == len(WORK_PRODUCERS)


def test_a_setup_verification_that_ran_in_a_chat_is_work(session: Session) -> None:
    """No foreign key points here, so the enumeration alone cannot see it."""

    verified = chat(session)
    session.add(
        SetupVerification(
            role="chat",
            evidence_key="neutral-evidence",
            model_install_id="install_neutral",
            profile_id="profile_neutral",
            chat_id=verified.id,
        )
    )
    session.flush()

    assert candidates(session) == []


@pytest.mark.parametrize("status", ["queued", "completed"])
def test_an_edit_verification_that_names_a_chat_is_work(session: Session, status: str) -> None:
    """Named in a job payload rather than by a foreign key, and work in either state."""

    verified = chat(session)
    session.add(
        Job(
            kind=JobKind.EDIT_VERIFY.value,
            status=status,
            payload_json={"chat_id": verified.id, "source_run_id": "run_absent"},
        )
    )
    session.flush()

    assert candidates(session) == []


def test_an_edit_verification_for_another_chat_does_not_make_this_one_busy(
    session: Session,
) -> None:
    blank = chat(session)
    session.add(
        Job(
            kind=JobKind.EDIT_VERIFY.value,
            status="completed",
            payload_json={"chat_id": "chat_somewhere_else"},
        )
    )
    session.flush()

    assert candidates(session) == [blank.id]


def test_a_chat_a_studio_session_came_from_is_somebody_s_decision(session: Session) -> None:
    """Listed, never a strict blank: another session still names it as its source."""

    source = chat(session)
    studio(session, source_chat_id=source.id)

    result = classify(session, source)

    assert result.classification is EmptyChatClass.CONFIGURED_BLANK
    assert result.reasons == ("linked_chat",)


def test_a_chat_a_fork_came_from_is_somebody_s_decision(session: Session) -> None:
    """A fork's origin is the only record the fork happened, so its source is kept too."""

    source = chat(session)
    chat(
        session,
        origin_json={"forked_from_chat_id": source.id, "forked_from_message_id": "msg_x"},
    )

    result = classify(session, source)

    assert result.classification is EmptyChatClass.CONFIGURED_BLANK
    assert result.reasons == ("linked_chat",)


def test_a_chat_is_not_linked_by_its_own_origin(session: Session) -> None:
    """Only ANOTHER chat pointing back makes a link; its own origin is already `forked`."""

    lonely = chat(session)
    lonely.origin_json = {"source_chat_id": lonely.id}
    session.flush()

    assert "linked_chat" not in classify(session, lonely).reasons


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ({"title": "Tax questions"}, "custom_title"),
        ({"pinned": True}, "pinned"),
        ({"archived": True}, "archived"),
        ({"draft_prompt": "half a thought"}, "has_draft"),
        ({"routing_mode": "image"}, "routing_chosen"),
        ({"confirm_uncertain_media": False}, "confirmation_changed"),
        ({"web_settings_json": {"enabled": True}}, "web_access_granted"),
        ({"origin_json": {"forked_from_chat_id": "chat_x"}}, "forked"),
        ({"active_chat_profile_id": "profile_1"}, "profile_chosen"),
        ({"active_vision_profile_id": "profile_2"}, "profile_chosen"),
        ({"generation_settings_json": {"steps": 20}}, "settings_overridden"),
        ({"vision_settings_json": {"compile_visual_prompts": False}}, "settings_overridden"),
    ],
)
def test_shell_state_makes_an_empty_chat_somebody_s_decision(
    session: Session, values: dict[str, object], reason: str
) -> None:
    """Configured blanks are listed and never safely preselected.

    `web_access_granted` and `forked` are why this class exists rather than
    being folded into strict. Internet access for one conversation is a grant
    somebody had to make deliberately and is never copied into a new chat, so
    deleting it silently discards something that takes thought to recreate. A
    fork's parentage is the only record the fork happened at all.
    """

    configured = chat(session, **values)
    result = classify(session, configured)

    assert result.classification is EmptyChatClass.CONFIGURED_BLANK
    assert reason in result.reasons
    assert set(result.reasons) <= set(CONFIGURED_REASONS)


def test_a_head_pointing_nowhere_is_inconsistent_rather_than_empty(session: Session) -> None:
    """Reported, not offered.

    The chat has no messages and a head naming one, which cannot both be true.
    Offering it for deletion would remove the evidence of whatever produced that
    state.
    """

    broken = chat(session, active_head_message_id="msg_missing")
    result = classify(session, broken)

    assert result.classification is EmptyChatClass.INCONSISTENT
    assert result.reasons == ("stale_head",)


@pytest.mark.parametrize("scope", ["studio", "prompt_helper", "setup_verification"])
def test_other_scopes_never_appear(session: Session, scope: str) -> None:
    """Excluded at the query, not by the caller.

    A studio session is empty by every test here and is not a conversation
    anybody chose to keep. Filtering it in the caller would mean the next reader
    of this query has to remember to filter it too, which is how a scope leaks.
    """

    if scope == "studio":
        studio(session)
    else:
        chat(session, scope=scope)

    assert candidates(session) == []


def test_every_chat_reference_is_an_audited_producer() -> None:
    """The claim that the audit is complete, enforced rather than asserted.

    The classification rests on `WORK_PRODUCERS` being every table that can hold
    work for a chat. That was true when it was written by reading the models,
    and reading the models is not a thing that keeps being true: a table added
    later would make this tool willing to offer chats that have work in it,
    silently, with every other test still green.

    So the invariant is read off `Base.metadata`. Adding a table that references
    `chats.id` fails here, and whoever adds it has to decide whether it means
    work.
    """

    audited = {model.__tablename__ for model in WORK_PRODUCERS}
    referencing = {
        table.name
        for table in Base.metadata.tables.values()
        for column in table.columns
        for foreign_key in column.foreign_keys
        if foreign_key.column.table.name == "chats"
    }

    assert referencing == audited, (
        "tables referencing chats.id that empty_chats does not audit: "
        f"{sorted(referencing - audited)}; audited but no longer referencing: "
        f"{sorted(audited - referencing)}"
    )


def test_the_selection_rows_creating_a_chat_writes_are_not_work(session: Session) -> None:
    """The rows every chat has, which must not make every chat busy.

    Creating a chat mirrors four automatic `ChatWorkflowSelection` rows with no
    family. They record no decision, and counting them would make every chat
    that has ever existed busy.
    """

    blank = chat(session)
    for capability in ("chat", "vision", "image", "video"):
        session.add(
            ChatWorkflowSelection(
                chat_id=blank.id,
                selector_capability=capability,
                mode="automatic",
            )
        )
    session.flush()

    assert candidates(session) == [blank.id]
    assert classify(session, blank).classification is EmptyChatClass.STRICT_BLANK


def test_a_page_is_bounded_and_walks_by_keyset(session: Session) -> None:
    """One request cannot ask for every chat, and paging cannot skip one."""

    made = sorted(chat(session).id for _ in range(5))

    first = empty_chat_page(session, limit=2)
    assert [entry.chat_id for entry in first.entries] == made[:2]
    assert first.next_cursor == made[1]

    second = empty_chat_page(session, limit=2, after_id=first.next_cursor)
    assert [entry.chat_id for entry in second.entries] == made[2:4]

    last = empty_chat_page(session, limit=2, after_id=second.next_cursor)
    assert [entry.chat_id for entry in last.entries] == made[4:]
    # No cursor rather than an empty page after it: the caller stops on fact.
    assert last.next_cursor is None


def test_a_page_refuses_a_nonsense_size_and_caps_a_greedy_one(session: Session) -> None:
    chat(session)
    with pytest.raises(ValueError):
        empty_chat_page(session, limit=0)
    with pytest.raises(ValueError):
        empty_chat_page(session, minimum_age_hours=-1)
    # Capped rather than refused: asking for more than the maximum is a caller
    # wanting everything, and the cursor already tells them there is more.
    assert empty_chat_page(session, limit=MAX_PAGE_SIZE * 10).next_cursor is None


def test_a_chat_younger_than_the_floor_is_not_offered(session: Session) -> None:
    """Filtered in the query, so dropping young rows never shortens a page."""

    moment = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    old = chat(session, created_at=moment - timedelta(hours=48))
    chat(session, created_at=moment - timedelta(minutes=30))

    assert shown(session, moment) == {old.id}


def test_the_age_floor_can_be_turned_off_entirely(session: Session) -> None:
    moment = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    fresh = chat(session, created_at=moment - timedelta(minutes=1))

    page = empty_chat_page(session, limit=MAX_PAGE_SIZE, minimum_age_hours=0, now=moment)

    assert [entry.chat_id for entry in page.entries] == [fresh.id]


def test_archived_and_configured_are_separate_switches(session: Session) -> None:
    """Archiving is 'not now'; the rest of the shell state is 'this is mine'.

    An archived chat is configured too, so including configured without
    including archived must still leave it out - otherwise the narrower switch
    would silently widen the selection.
    """

    moment = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    old = moment - timedelta(hours=48)
    plain = chat(session, created_at=old)
    titled = chat(session, title="Something", created_at=old)
    filed = chat(session, archived=True, created_at=old)

    assert shown(session, moment) == {plain.id}
    assert shown(session, moment, include_configured=True) == {plain.id, titled.id}
    assert shown(session, moment, include_archived=True) == {plain.id}
    assert shown(session, moment, include_archived=True, include_configured=True) == {
        plain.id,
        titled.id,
        filed.id,
    }


def test_a_configured_window_does_not_consume_the_page(session: Session) -> None:
    """The defect an all-of-each-kind fixture cannot reach.

    Filtering after the SQL limit would let ineligible rows spend the caller's
    page: a first window that happened to be entirely configured would return
    nothing with a cursor while an eligible chat sat one cursor away.
    """

    chat(session, id="chat_001", title="Configured one")
    chat(session, id="chat_002", title="Configured two")
    strict = chat(session, id="chat_003")

    page = empty_chat_page(session, limit=2)

    assert [entry.chat_id for entry in page.entries] == [strict.id]


def test_filling_walks_past_many_ineligible_rows(session: Session) -> None:
    for index in range(12):
        chat(session, id=f"chat_{index:03d}", title=f"Configured {index}")
    first = chat(session, id="chat_900")
    second = chat(session, id="chat_901")

    page = empty_chat_page(session, limit=2)

    assert [entry.chat_id for entry in page.entries] == [first.id, second.id]
    # Nothing eligible is left, so no cursor is offered.
    assert page.next_cursor is None


def test_a_full_page_still_offers_a_cursor(session: Session) -> None:
    for index in range(4):
        chat(session, id=f"chat_{index:03d}")

    page = empty_chat_page(session, limit=2)

    assert len(page.entries) == 2
    assert page.next_cursor == "chat_001"
    following = empty_chat_page(session, limit=2, after_id=page.next_cursor)
    assert [entry.chat_id for entry in following.entries] == ["chat_002", "chat_003"]
    assert following.next_cursor is None
