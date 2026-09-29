"""Preserve saved recipes and choices across database version boundaries."""

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy.orm import Session

from local_lm import models
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base, create_database_engine
from local_lm.models import Chat, GenerationPreset, Project

PARENT = "b7a9d3c14620"
REVISION = "c3b6d91e8a20"
TABLES = (
    "workflow_use_case_presets",
    "project_workflow_use_case_selections",
    "chat_workflow_use_case_selections",
)


def _prepare(tmp_path: Path) -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "data", dev=True)
    settings.prepare()
    command.upgrade(alembic_config(settings), PARENT)
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session:
            session.add_all(
                [
                    Project(id="project", name="Example"),
                    Chat(id="chat"),
                    GenerationPreset(
                        id="legacy", name="Legacy", role="image", settings_json={"seed": 9}
                    ),
                ]
            )
            session.commit()
    finally:
        engine.dispose()
    return settings, settings.state_dir / "local-lm.sqlite3"


def _snapshot(database: Path, tables: tuple[str, ...]) -> dict[str, list[tuple[object, ...]]]:
    with sqlite3.connect(database) as connection:
        return {
            table: connection.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in tables
        }


def test_upgrade_preserves_existing_role_settings_and_empty_tables_can_downgrade(
    tmp_path: Path,
) -> None:
    settings, database = _prepare(tmp_path)
    previous = ("generation_presets", "chats", "projects")
    before = _snapshot(database, previous)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    assert _snapshot(database, previous) == before
    assert _snapshot(database, TABLES) == dict.fromkeys(TABLES, [])
    assert _snapshot(database, ("alembic_version",)) == {"alembic_version": [(REVISION,)]}
    command.downgrade(config, PARENT)
    assert _snapshot(database, previous) == before
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert not tables.intersection(TABLES)


@pytest.mark.parametrize("retained", ["preset", "project-automatic", "chat-automatic"])
def test_downgrade_refuses_before_discarding_any_recipe_or_automatic_choice(
    tmp_path: Path, retained: str
) -> None:
    settings, database = _prepare(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, REVISION)
    engine = create_database_engine(settings)
    try:
        with Session(engine) as session:
            row: Base
            if retained == "preset":
                recipe_type = models.WorkflowUseCasePreset
                row = recipe_type(id="saved", name="Small", use_case="image_generation")
            elif retained == "project-automatic":
                project_type = models.ProjectWorkflowUseCaseSelection
                row = project_type(
                    project_id="project", use_case="image_generation", preset_id=None
                )
            else:
                chat_type = models.ChatWorkflowUseCaseSelection
                row = chat_type(chat_id="chat", use_case="image_generation", preset_id=None)
            session.add(row)
            session.commit()
    finally:
        engine.dispose()
    tables = (*TABLES, "alembic_version", "generation_presets", "chats", "projects")
    before = _snapshot(database, tables)
    with pytest.raises(RuntimeError, match="Saved workflow use-case choices require"):
        command.downgrade(config, PARENT)
    assert _snapshot(database, tables) == before
