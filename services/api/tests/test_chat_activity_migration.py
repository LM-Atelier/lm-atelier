from __future__ import annotations

import sqlite3

import pytest
from alembic import command
from fastapi import FastAPI
from httpx2 import AsyncClient
from test_chat_activity_lifecycle import seed_running

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import SessionLocal
from local_lm.models import Message, ResponseRevision
from local_lm.scheduler import JobClaim

PARENT = "f6c3a8d91b20"
REVISION = "c8a4f72e91bd"


def test_activity_migration_preserves_existing_responses_and_triggers(settings: Settings) -> None:
    _, message_id, run_id, _ = seed_running()
    with SessionLocal() as session:
        message = session.get(Message, message_id)
        assert message is not None
        revision = ResponseRevision(
            message_id=message_id, run_id=run_id, sequence=1, status="pending"
        )
        session.add(revision)
        session.commit()
    config = alembic_config(settings)
    command.downgrade(config, PARENT)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info(response_revisions)")]
        assert "activity_json" not in columns
        rows = connection.execute("SELECT * FROM response_revisions").fetchall()
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT " + ", ".join(columns) + " FROM response_revisions"
            ).fetchall()
            == rows
        )
        assert connection.execute("SELECT activity_json FROM response_revisions").fetchall() == [
            (None,)
        ]
        assert connection.execute("SELECT COUNT(*) FROM chat_activity_events").fetchone() == (0,)
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
            ).fetchall()
            == triggers
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        table_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name='chat_activity_events'"
        ).fetchone()[0]
        assert "AUTOINCREMENT" in table_sql
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            REVISION,
        )


async def test_downgrade_refuses_to_discard_activity(
    client: AsyncClient, app: FastAPI, settings: Settings
) -> None:
    _, _, run_id, job_id = seed_running()
    await app.state.services.orchestrator._fail(
        job_id, run_id, "Neutral failed attempt", claim=JobClaim(token="current-attempt", attempt=2)
    )
    with pytest.raises(RuntimeError, match="Chat activity requires"):
        command.downgrade(alembic_config(settings), PARENT)
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            REVISION,
        )
        assert connection.execute("SELECT COUNT(*) FROM chat_activity_events").fetchone() == (1,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
