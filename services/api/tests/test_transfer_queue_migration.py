"""Transfer policy revisions keep older schedulers from bypassing a pause."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command

from local_lm import database_migrations
from local_lm.config import Settings
from local_lm.database_migrations import DatabaseVersionError, alembic_config, upgrade_database

PARENT = "e9a5c7d31b62"
REVISION = "f6c3a8d91b20"
MIGRATION = REVISION + "_transfer_queue_policy.py"


def prepare(tmp_path: Path) -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "queue-migration")
    settings.prepare()
    command.upgrade(alembic_config(settings), PARENT)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO generation_queue_policies VALUES ('generation', 'paused', 1)"
        )
        response = {
            "lane": "generation",
            "dispatch_state": "paused",
            "revision": 1,
            "running_jobs": 0,
            "allowed_actions": ["resume"],
        }
        connection.execute(
            "INSERT INTO generation_queue_receipts VALUES "
            "('generation', 'kept-command', 'pause_after_current', 0, ?)",
            (json.dumps(response),),
        )
    return settings, database


def rows(database: Path) -> tuple[list[tuple[object, ...]], list[tuple[object, ...]]]:
    with sqlite3.connect(database) as connection:
        return (
            connection.execute("SELECT * FROM generation_queue_policies ORDER BY lane").fetchall(),
            connection.execute(
                "SELECT * FROM generation_queue_receipts ORDER BY lane, command_key"
            ).fetchall(),
        )


def test_transfer_policy_upgrade_preserves_generation_and_can_revert_before_use(
    tmp_path: Path,
) -> None:
    settings, database = prepare(tmp_path)
    before = rows(database)
    config = alembic_config(settings)
    command.upgrade(config, "head")
    assert rows(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]
    command.downgrade(config, PARENT)
    assert rows(database) == before


@pytest.mark.parametrize("state", ["open", "draining", "paused"])
def test_transfer_policy_downgrade_refuses_to_discard_supported_dispatch_state(
    tmp_path: Path, state: str
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO generation_queue_policies VALUES ('transfer', ?, 2)", (state,)
        )
    before = rows(database)
    with pytest.raises(RuntimeError, match="Transfer queue controls require"):
        command.downgrade(config, PARENT)
    assert rows(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]


def test_previous_migration_set_refuses_transfer_database_without_changing_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO generation_queue_policies VALUES ('transfer', 'paused', 1)")
    before = rows(database)
    current_scripts = Path(database_migrations.__file__).parent / "migrations"
    previous_scripts = tmp_path / "previous-migrations"
    shutil.copytree(
        current_scripts,
        previous_scripts,
        ignore=shutil.ignore_patterns(MIGRATION, "__pycache__"),
    )
    config.set_main_option("script_location", str(previous_scripts))
    monkeypatch.setattr(database_migrations, "alembic_config", lambda _settings: config)
    with pytest.raises(DatabaseVersionError, match="does not recognize"):
        upgrade_database(settings)
    assert rows(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]
