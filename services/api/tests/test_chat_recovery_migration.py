"""Upgrading adds recovery guards without rewriting a conversation's history."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_migrations import _is_the_recorded_uniqueness_allowance

from local_lm.artifact_library_schema import ENTRY_DELETE_TRIGGER
from local_lm.chat_recovery_schema import CREATE_CHAT_RECOVERY_TRIGGER_SQL
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base, create_database_engine
from local_lm.media_recovery_schema import CREATE_MEDIA_RECOVERY_TRIGGER_SQL
from local_lm.models import Chat, ChatComposerDraft, Message, MessagePart, RecoveryItem
from local_lm.project_recovery_schema import CREATE_PROJECT_RECOVERY_TRIGGER_SQL

PREVIOUS = "fbc390fda45f"
RECOVERY = "4b6e18d7c092"
TABLES = {"recovery_items", "recovery_operations", "recovery_previews"}


def _previous_database(tmp_path: Path):
    settings = Settings(data_dir=tmp_path / "data", chat_engine="mock", media_engine="mock")
    settings.prepare()
    command.upgrade(alembic_config(settings), PREVIOUS)
    return settings, create_database_engine(settings)


def test_upgrade_keeps_history_and_installs_the_canonical_bulk_write_guards(tmp_path: Path) -> None:
    settings, engine = _previous_database(tmp_path)
    with Session(engine) as session:
        session.add(Chat(id="chat-garden", title="Garden notes"))
        session.flush()
        session.add_all(
            [
                Message(id="message-garden", chat_id="chat-garden"),
                ChatComposerDraft(chat_id="chat-garden", text="Keep the garden path wide"),
            ]
        )
        session.flush()
        session.add(
            MessagePart(
                id="part-garden",
                message_id="message-garden",
                position=0,
                text="Space the garden beds evenly",
            )
        )
        session.commit()
        before = tuple(session.execute(text("SELECT * FROM message_parts")).all())
    command.upgrade(alembic_config(settings), RECOVERY)
    command.upgrade(alembic_config(settings), RECOVERY)
    with Session(engine) as session:
        assert tuple(session.execute(text("SELECT * FROM message_parts")).all()) == before
        assert session.scalar(select(Chat.title)) == "Garden notes"
        triggers = dict(
            session.execute(
                text("SELECT name, sql FROM sqlite_master WHERE type = 'trigger'")
            ).all()
        )
        for statement in CREATE_CHAT_RECOVERY_TRIGGER_SQL:
            assert triggers[statement.split()[2]].strip() == statement.strip()
        for statement in CREATE_MEDIA_RECOVERY_TRIGGER_SQL[1:]:
            assert triggers[statement.split()[2]].strip() == statement.strip()
        for statement in CREATE_PROJECT_RECOVERY_TRIGGER_SQL:
            assert triggers[statement.split()[2]].strip() == statement.strip()
        session.add(
            RecoveryItem(
                kind="chat",
                subject_id="chat-garden",
                display_label="Garden notes",
                purge_after=datetime.now(UTC) + timedelta(days=30),
                subject_revision="a" * 64,
            )
        )
        session.commit()
        item = session.scalar(select(RecoveryItem))
        assert item is not None and len(item.deletion_id) <= 40
        with pytest.raises(IntegrityError, match="chat-recovery-write-refused"):
            session.execute(update(ChatComposerDraft).values(text="Rewrite after deletion"))
        session.rollback()
        assert session.scalar(select(ChatComposerDraft.text)) == "Keep the garden path wide"
    engine.dispose()


def test_upgraded_recovery_tables_match_the_current_models(tmp_path: Path) -> None:
    settings, engine = _previous_database(tmp_path)
    command.upgrade(alembic_config(settings), "head")
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        differences = compare_metadata(context, Base.metadata)
        assert [
            item for item in differences if not _is_the_recorded_uniqueness_allowance(item)
        ] == []
    engine.dispose()


def test_an_empty_recovery_schema_can_return_to_the_previous_version(tmp_path: Path) -> None:
    settings, engine = _previous_database(tmp_path)
    command.upgrade(alembic_config(settings), RECOVERY)
    command.downgrade(alembic_config(settings), PREVIOUS)
    with engine.connect() as connection:
        tables = set(
            connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).scalars()
        )
        assert not tables & TABLES
        assert (
            connection.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one()
            == PREVIOUS
        )
        assert not connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'chat_recovery_%'"
        ).all()
        assert not connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'media_recovery_%'"
        ).all()
        assert not connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'project_recovery_%'"
        ).all()
        assert (
            connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE name='artifact_library_entry_delete_guard'"
            )
            .scalar_one()
            .strip()
            == ENTRY_DELETE_TRIGGER.strip()
        )
    command.upgrade(alembic_config(settings), RECOVERY)
    engine.dispose()


def test_a_recoverable_membership_refuses_a_downgrade_that_would_make_it_live(
    tmp_path: Path,
) -> None:
    settings, engine = _previous_database(tmp_path)
    command.upgrade(alembic_config(settings), RECOVERY)
    with Session(engine) as session:
        session.add(Chat(id="chat-garden", title="Garden notes"))
        session.flush()
        session.add(
            RecoveryItem(
                deletion_id="recovery-garden",
                kind="chat",
                subject_id="chat-garden",
                display_label="Garden notes",
                purge_after=datetime.now(UTC) + timedelta(days=30),
                subject_revision="a" * 64,
            )
        )
        session.commit()
    with pytest.raises(RuntimeError, match="Restore or purge recoverable items"):
        command.downgrade(alembic_config(settings), PREVIOUS)
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            RECOVERY,
        )
        assert connection.execute("SELECT deletion_id, state FROM recovery_items").fetchall() == [
            ("recovery-garden", "recoverable")
        ]
    engine.dispose()
