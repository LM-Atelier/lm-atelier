"""Workflow recovery guards have the same definitions in new and upgraded databases."""

from pathlib import Path

from alembic import command
from httpx2 import AsyncClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base, SessionLocal, create_database_engine
from local_lm.workflow_recovery_schema import (
    CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL,
    DIRECT_WORKFLOW_COLUMNS,
    OWNED_WORKFLOW_SQL,
)


def _installed(session: Session) -> dict[str, str]:
    return {
        name: statement
        for name, statement in session.execute(
            text(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE 'workflow_recovery_%'"
            )
        ).tuples()
    }


async def test_workflow_guards_match_after_migration_and_create(
    client: AsyncClient, tmp_path: Path
) -> None:
    expected = {
        statement.split()[2]: statement for statement in CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL
    }
    with SessionLocal() as migrated:
        assert _installed(migrated) == expected
    settings = Settings(data_dir=tmp_path / "fresh", chat_engine="mock", media_engine="mock")
    settings.prepare()
    engine = create_database_engine(settings)
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as fresh:
            assert _installed(fresh) == expected
    finally:
        engine.dispose()


def test_workflow_guards_return_after_reapplying_an_empty_recovery_migration(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "upgrade", chat_engine="mock", media_engine="mock")
    settings.prepare()
    configuration = alembic_config(settings)
    command.upgrade(configuration, "4b6e18d7c092")
    engine = create_database_engine(settings)
    expected = {
        statement.split()[2]: statement for statement in CREATE_WORKFLOW_RECOVERY_TRIGGER_SQL
    }
    try:
        with Session(engine) as upgraded:
            assert _installed(upgraded) == expected
        command.downgrade(configuration, "fbc390fda45f")
        with Session(engine) as removed:
            assert _installed(removed) == {}
        command.upgrade(configuration, "4b6e18d7c092")
        with Session(engine) as reapplied:
            assert _installed(reapplied) == expected
    finally:
        engine.dispose()


def test_workflow_identity_foreign_keys_are_classified_for_recovery() -> None:
    targets = {"workflow_families", "workflow_definitions", "workflow_revisions"}
    actual = {
        name
        for name, table in Base.metadata.tables.items()
        if name not in targets
        and any(key.column.table.name in targets for key in table.foreign_keys)
    }
    assert actual <= set(OWNED_WORKFLOW_SQL) | set(DIRECT_WORKFLOW_COLUMNS)
    for name, columns in DIRECT_WORKFLOW_COLUMNS.items():
        assert all(column in Base.metadata.tables[name].c for column in columns)
