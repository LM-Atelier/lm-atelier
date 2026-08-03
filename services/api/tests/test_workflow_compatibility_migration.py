from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

_BASE = "c8f2d7a91e64"
_REVISION = "f1c8a2d47b90"
_TIMESTAMP = "2026-08-03 00:00:00"


def _database(tmp_path: Path) -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "workflow-compatibility-migration")
    settings.prepare()
    return settings, settings.state_dir / "local-lm.sqlite3"


def test_workflow_compatibility_migration_is_additive_and_reversible_when_empty(
    tmp_path: Path,
) -> None:
    settings, database = _database(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, _BASE)
    command.upgrade(config, _REVISION)

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "workflow_profile_compatibility",
            "chat_workflow_selections",
            "project_workflow_selections",
        } <= tables
        chat_foreign_keys = {
            (row[3], row[2]): row[6]
            for row in connection.execute(
                "PRAGMA foreign_key_list(chat_workflow_selections)"
            ).fetchall()
        }
        assert chat_foreign_keys[("chat_id", "chats")] == "CASCADE"
        assert chat_foreign_keys[("workflow_family_id", "workflow_families")] == "RESTRICT"
        project_foreign_keys = {
            (row[3], row[2]): row[6]
            for row in connection.execute(
                "PRAGMA foreign_key_list(project_workflow_selections)"
            ).fetchall()
        }
        assert project_foreign_keys[("project_id", "projects")] == "CASCADE"
        assert project_foreign_keys[("workflow_revision_id", "workflow_revisions")] == ("RESTRICT")

    command.downgrade(config, _BASE)
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "workflow_profile_compatibility" not in tables
        assert "chat_workflow_selections" not in tables
        assert "project_workflow_selections" not in tables


def test_workflow_compatibility_migration_refuses_populated_downgrade(
    tmp_path: Path,
) -> None:
    settings, database = _database(tmp_path)
    config = alembic_config(settings)
    command.upgrade(config, _REVISION)

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO model_profiles (
                id, model_install_id, name, use_case, role, engine,
                load_settings_json, request_settings_json, is_default,
                created_at, updated_at
            ) VALUES (?, NULL, ?, '', 'chat', 'llama.cpp', '{}', '{}', 1, ?, ?)
            """,
            ("profile_legacy", "Legacy", _TIMESTAMP, _TIMESTAMP),
        )
        connection.execute(
            """
            INSERT INTO workflow_families (
                id, name, description, use_case, tags_json, enabled, archived,
                created_at, updated_at
            ) VALUES (?, ?, '', '', '[]', 1, 0, ?, ?)
            """,
            ("wffamily_legacy", "Legacy", _TIMESTAMP, _TIMESTAMP),
        )
        connection.execute(
            """
            INSERT INTO workflow_profile_compatibility (
                model_profile_id, workflow_family_id, source_fingerprint_sha256,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                "profile_legacy",
                "wffamily_legacy",
                "a" * 64,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="mirrored selections exist"):
        command.downgrade(config, _BASE)
