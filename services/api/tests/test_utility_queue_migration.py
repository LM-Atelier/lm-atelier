"""Video utility jobs and their lane require a version that knows them."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from test_workflow_source_offer_migration import _populated_database

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

PARENT = "9c4e7a2d1830"
REVISION = "2a07d89a3b46"


def prepare(tmp_path: Path) -> tuple[Settings, Path]:
    settings, database = _populated_database(tmp_path)
    command.upgrade(alembic_config(settings), PARENT)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO generation_queue_policies VALUES ('transfer', 'paused', 1)")
    return settings, database


def snapshot(database: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(database) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in ("generation_queue_policies", "generation_queue_receipts", "jobs")
        }


def version(database: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(database) as connection:
        return connection.execute("SELECT version_num FROM alembic_version").fetchall()


def test_the_upgrade_changes_no_stored_work_and_reverts_before_use(tmp_path: Path) -> None:
    settings, database = prepare(tmp_path)
    before = snapshot(database)
    config = alembic_config(settings)

    command.upgrade(config, REVISION)
    assert snapshot(database) == before
    assert version(database) == [(REVISION,)]

    command.downgrade(config, PARENT)
    assert snapshot(database) == before
    assert version(database) == [(PARENT,)]


@pytest.mark.parametrize("state", ["policy", "receipt", "queued-job", "finished-job"])
def test_the_downgrade_refuses_once_the_lane_or_a_utility_job_exists(
    tmp_path: Path, state: str
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        if state == "policy":
            connection.execute(
                "INSERT INTO generation_queue_policies VALUES ('utility', 'paused', 1)"
            )
        elif state == "receipt":
            connection.execute(
                "INSERT INTO generation_queue_receipts VALUES "
                "('utility', 'retained-command', 'pause_after_current', 0, '{}')"
            )
        else:
            connection.execute(
                "UPDATE jobs SET kind = 'media_utility', status = ? WHERE id = 'job_migration'",
                ("queued" if state == "queued-job" else "complete",),
            )
    before = snapshot(database)

    with pytest.raises(RuntimeError, match="Video utility jobs require"):
        command.downgrade(config, PARENT)

    assert snapshot(database) == before
    assert version(database) == [(REVISION,)]
