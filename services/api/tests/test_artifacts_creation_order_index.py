"""The creation-order index the retention sweep reads through is added and removed safely.

SQLite commits a CREATE INDEX on its own, so an upgrade that fails after it
leaves the index behind a version row that says it was never made. The
migration finishes such a replay, refuses a same-name index of another shape
rather than adopting it, and on downgrade removes only the index it makes.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

REVISION = "af4bf615e063"
PRIOR = "d41c7a9e2b58"
INDEX = "ix_artifacts_created_at_id"


def _migrated_to(tmp_path: Path, revision: str) -> tuple[Config, Path]:
    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, revision)
    return config, settings.state_dir / "local-lm.sqlite3"


def _version_and_columns(database: Path) -> tuple[tuple[str, ...] | None, list[str]]:
    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        columns = [row[2] for row in connection.execute(f"PRAGMA index_info('{INDEX}')")]
    return version, columns


def test_the_upgrade_adds_the_index_and_the_downgrade_removes_it(tmp_path: Path) -> None:
    config, database = _migrated_to(tmp_path, REVISION)
    assert _version_and_columns(database) == ((REVISION,), ["created_at", "id"])

    command.downgrade(config, PRIOR)

    assert _version_and_columns(database) == ((PRIOR,), [])


def test_an_upgrade_that_stopped_after_the_index_finishes_on_the_next_start(
    tmp_path: Path,
) -> None:
    config, database = _migrated_to(tmp_path, PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"CREATE INDEX {INDEX} ON artifacts (created_at, id)")
        connection.commit()

    command.upgrade(config, REVISION)

    assert _version_and_columns(database) == ((REVISION,), ["created_at", "id"])


def test_a_same_name_index_of_another_shape_is_refused_not_adopted(tmp_path: Path) -> None:
    config, database = _migrated_to(tmp_path, PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"CREATE INDEX {INDEX} ON artifacts (created_at)")
        connection.commit()

    with pytest.raises(Exception, match="already exists with a different definition"):
        command.upgrade(config, REVISION)

    assert _version_and_columns(database) == ((PRIOR,), ["created_at"])


def test_the_downgrade_keeps_a_same_name_index_of_another_shape(tmp_path: Path) -> None:
    config, database = _migrated_to(tmp_path, REVISION)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(f"DROP INDEX {INDEX}")
        connection.execute(f"CREATE INDEX {INDEX} ON artifacts (id, created_at)")
        connection.commit()

    command.downgrade(config, PRIOR)

    assert _version_and_columns(database) == ((PRIOR,), ["id", "created_at"])
