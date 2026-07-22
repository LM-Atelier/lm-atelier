from __future__ import annotations

import sqlite3

from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config, upgrade_database


def test_migrations_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "migrations")
    settings.prepare()
    config = alembic_config(settings)
    upgrade_database(settings)
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type = 'table'"
            ).fetchall()
        }
        chat_columns = {row[1] for row in connection.execute("PRAGMA table_info(chats)").fetchall()}
        project_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(projects)").fetchall()
        }
    assert {"projects", "messages", "generation_presets", "custom_node_installs"} <= tables
    assert "active_head_message_id" in chat_columns
    assert {"image_workflow_revision_id", "video_workflow_revision_id"} <= project_columns

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
