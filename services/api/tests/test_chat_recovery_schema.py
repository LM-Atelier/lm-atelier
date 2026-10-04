"""A recoverable chat refuses writes even when they bypass the ORM."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm.chat_recovery_schema import (
    CHAT_OWNER_SQL,
    CREATE_CHAT_RECOVERY_TRIGGER_SQL,
    INDEPENDENT_REFERENCES,
    ROW_COLUMNS,
    SET_NULL_LINKS,
)
from local_lm.config import Settings
from local_lm.db import Base, create_database_engine
from local_lm.models import (
    Chat,
    ChatComposerDraft,
    Job,
    Message,
    MessagePart,
    Project,
    RecoveryItem,
    Run,
    TurnCreationClaim,
    WorkPlan,
    WorkStep,
)


@pytest.fixture
def recovery_session(tmp_path: Path) -> Iterator[Session]:
    settings = Settings(data_dir=tmp_path / "data", chat_engine="mock", media_engine="mock")
    settings.prepare()
    engine = create_database_engine(settings)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Project(id="project-garden", name="Garden"))
        session.flush()
        session.add_all(
            (
                Chat(id="chat-garden", title="Garden notes", project_id="project-garden"),
                Chat(id="chat-other", title="Other notes"),
            )
        )
        session.flush()
        session.add_all(
            (
                ChatComposerDraft(chat_id="chat-garden", text="Keep the path wide"),
                Message(id="message-garden", chat_id="chat-garden"),
                Message(id="message-other", chat_id="chat-other"),
            )
        )
        session.flush()
        session.add(MessagePart(id="part-garden", message_id="message-garden", position=0))
        session.flush()
        session.add(
            RecoveryItem(
                deletion_id="recovery-garden",
                kind="chat",
                subject_id="chat-garden",
                display_label="Garden notes",
                original_project_id="project-garden",
                original_project_label="Garden",
                deleted_at=datetime.now(UTC),
                purge_after=datetime.now(UTC) + timedelta(days=30),
                state="recoverable",
                subject_revision="a" * 64,
            )
        )
        session.commit()
        yield session
    engine.dispose()


def test_every_conversation_attachment_is_classified() -> None:
    guarded = set(CHAT_OWNER_SQL)
    referenced = {
        table.name
        for table in Base.metadata.tables.values()
        if any(foreign.column.table.name in guarded for foreign in table.foreign_keys)
    }
    assert referenced <= guarded | INDEPENDENT_REFERENCES
    assert not guarded & INDEPENDENT_REFERENCES
    for table, columns in ROW_COLUMNS.items():
        assert columns == tuple(Base.metadata.tables[table].columns.keys())
    actual_links = {
        table: tuple(
            (foreign.parent.name, foreign.column.table.name, foreign.column.name)
            for foreign in sorted(
                Base.metadata.tables[table].foreign_keys, key=lambda item: item.parent.name
            )
            if foreign.ondelete == "SET NULL"
        )
        for table in guarded
    }
    assert {table: links for table, links in actual_links.items() if links} == SET_NULL_LINKS


@pytest.mark.parametrize("operation", ["insert", "update", "delete", "raw-update"])
def test_bulk_composer_writes_leave_the_recovery_draft_unchanged(
    recovery_session: Session, operation: str
) -> None:
    session = recovery_session
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        if operation == "insert":
            session.execute(insert(ChatComposerDraft).values(chat_id="chat-garden", text="Changed"))
        elif operation == "update":
            session.execute(
                update(ChatComposerDraft)
                .where(ChatComposerDraft.chat_id == "chat-garden")
                .values(text="Changed")
            )
        elif operation == "delete":
            session.execute(
                delete(ChatComposerDraft).where(ChatComposerDraft.chat_id == "chat-garden")
            )
        else:
            session.execute(
                text(
                    "UPDATE chat_composer_drafts SET text = 'Changed' WHERE chat_id = 'chat-garden'"
                )
            )
    session.rollback()
    draft = session.get(ChatComposerDraft, "chat-garden")
    assert draft is not None and draft.text == "Keep the path wide"


@pytest.mark.parametrize("operation", ["insert", "update", "delete", "move"])
def test_message_and_part_writes_cannot_enter_or_leave_a_deleted_chat(
    recovery_session: Session, operation: str
) -> None:
    session = recovery_session
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        if operation == "insert":
            session.add(Message(id="message-new", chat_id="chat-garden"))
            session.flush()
        elif operation == "update":
            session.execute(
                update(MessagePart).where(MessagePart.id == "part-garden").values(text="Changed")
            )
        elif operation == "delete":
            session.execute(delete(MessagePart).where(MessagePart.id == "part-garden"))
        else:
            session.execute(
                update(MessagePart)
                .where(MessagePart.id == "part-garden")
                .values(message_id="message-other")
            )
    session.rollback()
    part = session.get(MessagePart, "part-garden")
    assert part is not None and part.message_id == "message-garden" and part.text is None
    assert session.get(Message, "message-new") is None


@pytest.mark.parametrize("state", ["recoverable", "restoring", "blocked", "purging", "purged"])
def test_no_recovery_state_accepts_a_new_runless_job(recovery_session: Session, state: str) -> None:
    session = recovery_session
    item = session.get(RecoveryItem, "recovery-garden")
    assert item is not None
    item.state = state
    session.commit()
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        session.add(Job(id="job-new", kind="edit_verify", payload_json={"chat_id": "chat-garden"}))
        session.flush()
    session.rollback()
    assert session.get(Job, "job-new") is None


def test_a_new_turn_claim_cannot_enter_a_deleted_chat(recovery_session: Session) -> None:
    session = recovery_session
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        session.add(
            TurnCreationClaim(
                chat_id="chat-garden", idempotency_key="new-turn", owner_token="b" * 64
            )
        )
        session.flush()
    session.rollback()
    assert session.scalar(select(TurnCreationClaim.id)) is None


def test_permanent_deletion_requires_the_purging_state(recovery_session: Session) -> None:
    session = recovery_session
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        session.execute(delete(Chat).where(Chat.id == "chat-garden"))
    session.rollback()
    assert session.get(Chat, "chat-garden") is not None
    item = session.get(RecoveryItem, "recovery-garden")
    assert item is not None
    item.state = "purging"
    session.flush()
    session.execute(delete(Chat).where(Chat.id == "chat-garden"))
    session.commit()
    assert session.get(Chat, "chat-garden") is None
    assert session.get(ChatComposerDraft, "chat-garden") is None
    assert session.get(MessagePart, "part-garden") is None
    assert session.get(Chat, "chat-other") is not None


def test_deleting_a_project_keeps_the_original_recovery_location(recovery_session: Session) -> None:
    session = recovery_session
    session.execute(delete(Project).where(Project.id == "project-garden"))
    session.commit()
    chat = session.get(Chat, "chat-garden")
    item = session.get(RecoveryItem, "recovery-garden")
    assert chat is not None and chat.project_id is None and chat.title == "Garden notes"
    assert item is not None and item.original_project_id == "project-garden"
    assert item.original_project_label == "Garden"


def test_clearing_a_live_parent_is_not_a_cascade_exception(recovery_session: Session) -> None:
    session = recovery_session
    with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
        session.execute(update(Chat).where(Chat.id == "chat-garden").values(project_id=None))
    session.rollback()
    chat = session.get(Chat, "chat-garden")
    assert chat is not None and chat.project_id == "project-garden"


def test_removing_the_membership_restores_writes_to_the_same_chat(
    recovery_session: Session,
) -> None:
    session = recovery_session
    session.execute(delete(RecoveryItem).where(RecoveryItem.deletion_id == "recovery-garden"))
    session.execute(
        update(ChatComposerDraft)
        .where(ChatComposerDraft.chat_id == "chat-garden")
        .values(text="Restored")
    )
    session.commit()
    draft = session.get(ChatComposerDraft, "chat-garden")
    assert draft is not None and draft.text == "Restored"
    chat = session.get(Chat, "chat-garden")
    assert chat is not None and chat.title == "Garden notes"


def test_the_installed_guards_have_the_canonical_bodies(recovery_session: Session) -> None:
    installed = {
        name: statement
        for name, statement in recovery_session.execute(
            text(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'chat_recovery_%'"
            )
        ).tuples()
    }
    expected = {statement.split()[2]: statement for statement in CREATE_CHAT_RECOVERY_TRIGGER_SQL}
    assert installed == expected


@pytest.mark.parametrize("target", ["composer", "message"])
def test_a_writer_that_read_before_trash_cannot_commit_after_it(
    recovery_session: Session, target: str
) -> None:
    session = recovery_session
    item = session.get(RecoveryItem, "recovery-garden")
    assert item is not None
    record = {column.name: getattr(item, column.name) for column in RecoveryItem.__table__.columns}
    session.delete(item)
    session.commit()
    with Session(session.get_bind()) as writer:
        if target == "composer":
            draft = writer.get(ChatComposerDraft, "chat-garden")
            assert draft is not None and draft.text == "Keep the path wide"
        else:
            message = writer.get(Message, "message-garden")
            assert message is not None and message.status == "complete"
        assert writer.scalar(select(RecoveryItem.deletion_id)) is None
        session.add(RecoveryItem(**record))
        session.commit()
        if target == "composer":
            assert draft is not None
            draft.text = "Changed after deletion"
        else:
            assert message is not None
            message.status = "pending"
        with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
            writer.commit()
        writer.rollback()
    session.expire_all()
    retained = session.get(ChatComposerDraft, "chat-garden")
    retained_message = session.get(Message, "message-garden")
    assert retained is not None and retained.text == "Keep the path wide"
    assert retained_message is not None and retained_message.status == "complete"
    assert session.get(RecoveryItem, "recovery-garden") is not None


def test_purging_a_complete_graph_removes_its_jobs_before_their_links_are_cleared(
    recovery_session: Session,
) -> None:
    session = recovery_session
    item = session.get(RecoveryItem, "recovery-garden")
    assert item is not None
    record = {column.name: getattr(item, column.name) for column in RecoveryItem.__table__.columns}
    session.delete(item)
    session.commit()
    session.add(Message(id="assistant-garden", chat_id="chat-garden", parent_id="message-garden"))
    session.add(
        WorkPlan(
            id="plan-garden",
            chat_id="chat-garden",
            context_head_message_id="message-garden",
            transcript_sequence=1,
        )
    )
    session.flush()
    step = WorkStep(id="step-garden", plan_id="plan-garden", ordinal=1, operation="text")
    session.add(step)
    session.flush()
    run = Run(
        id="run-garden",
        chat_id="chat-garden",
        user_message_id="message-garden",
        assistant_message_id="assistant-garden",
        work_plan_id="plan-garden",
        work_step_id="step-garden",
    )
    session.add(run)
    session.flush()
    step.run_id = run.id
    session.add_all(
        [
            Job(
                id="job-garden",
                kind="chat",
                status="complete",
                run_id=run.id,
                work_plan_id="plan-garden",
                work_step_id=step.id,
                payload_json={"chat_id": "chat-garden"},
            ),
            Job(
                id="verification-garden",
                kind="edit_verify",
                status="complete",
                payload_json={"source_run_id": run.id},
            ),
            Job(
                id="job-other",
                kind="chat",
                status="complete",
                payload_json={"chat_id": "chat-other"},
            ),
        ]
    )
    session.flush()
    session.add(RecoveryItem(**record))
    session.commit()
    item = session.get(RecoveryItem, "recovery-garden")
    assert item is not None
    item.state = "purging"
    session.flush()
    ownership = CHAT_OWNER_SQL["jobs"].format(row="j", chat=":chat_id")
    session.execute(text(f"DELETE FROM jobs AS j WHERE ({ownership})"), {"chat_id": "chat-garden"})
    session.execute(delete(Chat).where(Chat.id == "chat-garden"))
    session.commit()
    session.expire_all()
    for model, identity in [
        (Chat, "chat-garden"),
        (Message, "message-garden"),
        (Message, "assistant-garden"),
        (Run, "run-garden"),
        (WorkPlan, "plan-garden"),
        (WorkStep, "step-garden"),
        (Job, "job-garden"),
        (Job, "verification-garden"),
    ]:
        assert session.get(model, identity) is None
    assert session.get(Chat, "chat-other") is not None
    assert session.get(Message, "message-other") is not None
    assert session.get(Job, "job-other") is not None
    assert session.get(RecoveryItem, "recovery-garden") is not None
