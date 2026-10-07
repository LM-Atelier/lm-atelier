"""Installation dispatch state requires a scheduler that understands its ownership."""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from test_workflow_source_offer_migration import _populated_database

from local_lm import database_migrations
from local_lm.config import Settings
from local_lm.database_migrations import DatabaseVersionError, alembic_config, upgrade_database

PARENT = "ffccb898c9c6"
REVISION = "a64c0d289e71"


def prepare(tmp_path: Path) -> tuple[Settings, Path]:
    settings, database = _populated_database(tmp_path)
    command.upgrade(alembic_config(settings), PARENT)
    with sqlite3.connect(database) as connection:
        for lane in ("generation", "transfer"):
            connection.execute(
                "INSERT INTO generation_queue_policies VALUES (?, 'paused', 1)", (lane,)
            )
            connection.execute(
                "INSERT INTO generation_queue_receipts VALUES "
                "(?, 'retained-command', 'pause_after_current', 0, ?)",
                (lane, json.dumps({"lane": lane, "revision": 1, "dispatch_state": "paused"})),
            )
    return settings, database


def snapshot(database: Path) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(database) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in (
                "generation_queue_policies",
                "generation_queue_receipts",
                "jobs",
                "workflow_install_offers",
                "workflow_install_offer_downloads",
            )
        }


def test_install_policy_upgrade_preserves_accepted_work_and_can_revert_before_use(
    tmp_path: Path,
) -> None:
    settings, database = prepare(tmp_path)
    before = snapshot(database)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    assert snapshot(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]
    command.downgrade(config, PARENT)
    assert snapshot(database) == before


@pytest.mark.parametrize("state", ["open", "draining", "paused", "receipt", "legacy-job"])
def test_install_policy_downgrade_retains_policy_receipts_and_legacy_completion(
    tmp_path: Path, state: str
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        if state == "legacy-job":
            connection.execute(
                "UPDATE workflow_install_offers SET completion_job_id = 'job_migration'"
            )
        elif state == "receipt":
            connection.execute(
                "INSERT INTO generation_queue_receipts VALUES "
                "('install', 'retained-command', 'pause_after_current', 0, '{}')"
            )
        else:
            connection.execute(
                "INSERT INTO generation_queue_policies VALUES ('install', ?, 2)", (state,)
            )
    before = snapshot(database)
    with pytest.raises(RuntimeError, match="Installation queue controls require"):
        command.downgrade(config, PARENT)
    assert snapshot(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]


@pytest.mark.parametrize("kind", ["activate", "registry_prepare", "workflow_install"])
@pytest.mark.parametrize("status", ["queued", "paused", "running", "complete", "failed"])
def test_install_policy_downgrade_retains_unfinished_or_claimed_installations(
    tmp_path: Path, kind: str, status: str
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE jobs SET kind = ?, status = ?, claim_owner = ? WHERE id = 'job_migration'",
            (kind, status, "retained-claim" if status in {"complete", "failed"} else None),
        )
    before = snapshot(database)
    with pytest.raises(RuntimeError, match="Installation queue controls require"):
        command.downgrade(config, PARENT)
    assert snapshot(database) == before


def test_previous_migration_set_refuses_installation_state_without_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        connection.execute("INSERT INTO generation_queue_policies VALUES ('install', 'paused', 1)")
    before = snapshot(database)
    current_scripts = Path(database_migrations.__file__).parent / "migrations"
    previous_scripts = tmp_path / "previous-migrations"
    previous_names = {
        Path(revision.path).name
        for revision in ScriptDirectory.from_config(config).walk_revisions(base="base", head=PARENT)
    }
    newer_names = {
        path.name
        for path in (current_scripts / "versions").glob("*.py")
        if path.name != "__init__.py" and path.name not in previous_names
    }
    shutil.copytree(
        current_scripts,
        previous_scripts,
        ignore=shutil.ignore_patterns(*newer_names, "__pycache__"),
    )
    config.set_main_option("script_location", str(previous_scripts))
    monkeypatch.setattr(database_migrations, "alembic_config", lambda _settings: config)
    with pytest.raises(DatabaseVersionError, match="does not recognize"):
        upgrade_database(settings)
    assert snapshot(database) == before
