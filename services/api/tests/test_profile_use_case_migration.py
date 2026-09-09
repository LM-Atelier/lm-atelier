from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config


def test_use_case_provenance_upgrade_preserves_existing_text_and_round_trips(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "profile-use-case-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c3e1d7a94f20")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        for identifier, use_case in [
            ("neutral-written", "My own description"),
            ("neutral-cleared", ""),
        ]:
            connection.execute(
                "INSERT INTO model_profiles (id, name, use_case, role, engine, load_settings_json, "
                "request_settings_json, is_default, created_at, updated_at) "
                "VALUES (?, 'Neutral profile', ?, 'chat', 'llama.cpp', '{}', '{}', 0, ?, ?)",
                (identifier, use_case, "2026-09-09 00:00:00", "2026-09-09 00:00:00"),
            )
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_profiles)")}
        assert "use_case_derived" in columns
        assert connection.execute(
            "SELECT id, use_case, use_case_derived FROM model_profiles "
            "WHERE id LIKE 'neutral-%' ORDER BY id"
        ).fetchall() == [("neutral-cleared", "", 0), ("neutral-written", "My own description", 0)]
    command.downgrade(config, "c3e1d7a94f20")
    with sqlite3.connect(database) as connection:
        assert "use_case_derived" not in {
            row[1] for row in connection.execute("PRAGMA table_info(model_profiles)")
        }
        assert connection.execute(
            "SELECT id, use_case FROM model_profiles WHERE id LIKE 'neutral-%' ORDER BY id"
        ).fetchall() == [("neutral-cleared", ""), ("neutral-written", "My own description")]
