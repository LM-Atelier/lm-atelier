"""Persist manual dispatch order without changing accepted work or forgetting receipts."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.script import ScriptDirectory
from sqlalchemy.orm import sessionmaker
from test_install_queue_migration import prepare, snapshot
from test_queue_manual_order import move_before, seed

from local_lm import database_migrations
from local_lm.database_migrations import DatabaseVersionError, alembic_config, upgrade_database
from local_lm.db import create_database_engine
from local_lm.queue_order import change_queue_order, read_queue_order
from local_lm.queue_order_v1 import QueueOrderCommand

PARENT = "a64c0d289e71"
REVISION = "b7a9d3c14620"


def test_order_upgrade_preserves_accepted_work_and_can_revert_before_use(tmp_path: Path) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, PARENT)
    before = snapshot(database)
    command.upgrade(config, REVISION)
    assert snapshot(database) == before
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM queue_order_entries").fetchall() == []
        assert connection.execute("SELECT * FROM queue_order_receipts").fetchall() == []
        assert connection.execute("SELECT version_num FROM alembic_version").fetchall() == [
            (REVISION,)
        ]
    command.downgrade(config, PARENT)
    assert snapshot(database) == before
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master")}
        assert "queue_order_entries" not in tables and "queue_order_receipts" not in tables


@pytest.mark.parametrize("retained", ["entry", "receipt"])
def test_order_downgrade_refuses_to_discard_saved_positions_or_receipts(
    tmp_path: Path, retained: str
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    with sqlite3.connect(database) as connection:
        if retained == "entry":
            connection.execute(
                "INSERT INTO queue_order_entries VALUES "
                "('transfer', 'job', 'download-example', 'network', 'network', 0, 0)"
            )
        else:
            connection.execute(
                "INSERT INTO queue_order_receipts VALUES ('transfer', 'retained-move', '{}', '{}')"
            )
        before = {
            table: connection.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("queue_order_entries", "queue_order_receipts", "alembic_version")
        }
    with pytest.raises(RuntimeError, match="Manual queue ordering requires"):
        command.downgrade(config, PARENT)
    with sqlite3.connect(database) as connection:
        assert {
            table: connection.execute(f"SELECT * FROM {table}").fetchall() for table in before
        } == before


def test_migrated_order_and_exact_move_receipt_survive_reopening_the_database(
    tmp_path: Path,
) -> None:
    settings, database = prepare(tmp_path)
    command.upgrade(alembic_config(settings), "head")
    engine = create_database_engine(settings)
    sessions = sessionmaker(engine)
    try:
        seed(sessions)
        with sessions() as session:
            page = read_queue_order(session, "transfer")
        move = QueueOrderCommand.model_validate(move_before(page.model_dump(mode="json"), 2, 0))
        with sessions() as session:
            result = change_queue_order(session, "transfer", move)
    finally:
        engine.dispose()
    reopened = create_database_engine(settings)
    try:
        with sessionmaker(reopened)() as session:
            latest = read_queue_order(session, "transfer")
        assert [item.owner.id for item in latest.items] == [
            "download-2",
            "download-0",
            "download-1",
            "job_migration",
        ]
        with sessionmaker(reopened)() as session:
            assert change_queue_order(session, "transfer", move) == result
        with sqlite3.connect(database) as connection:
            assert connection.execute("SELECT COUNT(*) FROM queue_order_receipts").fetchone() == (
                1,
            )
    finally:
        reopened.dispose()


def test_previous_build_refuses_the_ordering_database_before_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, database = prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, "head")
    before = database.read_bytes()
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
    assert database.read_bytes() == before


def test_migrated_order_entries_reject_negative_positions(tmp_path: Path) -> None:
    settings, database = prepare(tmp_path)
    command.upgrade(alembic_config(settings), "head")
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="ck_queue_order_position"):
            connection.execute(
                "INSERT INTO queue_order_entries VALUES "
                "('transfer', 'job', 'download-example', 'network', 'network', 0, -1)"
            )
        assert connection.execute("SELECT * FROM queue_order_entries").fetchall() == []
