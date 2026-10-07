"""Adding workflow completion jobs preserves accepted downloads and enforces their link."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from test_workflow_source_offer_migration import _populated_database

from local_lm.database_migrations import alembic_config

PARENT = "a4c6e82b910f"
REVISION = "e9a5c7d31b62"


def test_completion_job_migration_preserves_existing_offers_jobs_links_and_triggers(
    tmp_path: Path,
) -> None:
    settings, database = _populated_database(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        columns = [
            row[1] for row in connection.execute("PRAGMA table_info(workflow_install_offers)")
        ]
        offers = connection.execute("SELECT * FROM workflow_install_offers").fetchall()
        jobs = connection.execute("SELECT * FROM jobs").fetchall()
        links = connection.execute("SELECT * FROM workflow_install_offer_downloads").fetchall()
        triggers = connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
        ).fetchall()
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT " + ", ".join(columns) + " FROM workflow_install_offers"
            ).fetchall()
            == offers
        )
        assert connection.execute(
            "SELECT completion_job_id FROM workflow_install_offers"
        ).fetchall() == [(None,)]
        assert connection.execute("SELECT * FROM jobs").fetchall() == jobs
        assert (
            connection.execute("SELECT * FROM workflow_install_offer_downloads").fetchall() == links
        )
        assert (
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='trigger' ORDER BY name"
            ).fetchall()
            == triggers
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM workflow_install_offers").fetchall() == offers
        assert connection.execute("SELECT * FROM jobs").fetchall() == jobs
        assert (
            connection.execute("SELECT * FROM workflow_install_offer_downloads").fetchall() == links
        )


def test_completion_job_link_restricts_deletion_and_loss_on_downgrade(tmp_path: Path) -> None:
    settings, database = _populated_database(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database, isolation_level=None) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("UPDATE workflow_install_offers SET completion_job_id='missing'")
        connection.execute("UPDATE workflow_install_offers SET completion_job_id='job_migration'")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute("DELETE FROM jobs WHERE id='job_migration'")
    with pytest.raises(RuntimeError, match="Workflow installation jobs"):
        command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT completion_job_id FROM workflow_install_offers"
        ).fetchall() == [("job_migration",)]
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
