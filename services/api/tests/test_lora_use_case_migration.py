from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config


def test_lora_provenance_migration_preserves_manual_and_cleared_use_cases(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "lora-use-case-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c3b6d91e8a20")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        for identifier, use_case in [("written", "Watercolor landscapes"), ("cleared", "")]:
            connection.execute(
                "INSERT INTO model_asset_installs "
                "(id, name, kind, local_path, size_bytes, manifest_json, active, use_case, "
                "auto_apply, default_model_strength, default_clip_strength, typed_trigger_words, "
                "created_at, updated_at) VALUES (?, 'Neutral LoRA', 'lora', 'neutral', 1, '{}', "
                "0, ?, 0, 1.0, 1.0, '[]', '2026-09-29 00:00:00', '2026-09-29 00:00:00')",
                (identifier, use_case),
            )
    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_asset_installs)")}
        assert "use_case_derived" in columns
        assert connection.execute(
            "SELECT id, use_case, use_case_derived FROM model_asset_installs ORDER BY id"
        ).fetchall() == [("cleared", "", 0), ("written", "Watercolor landscapes", 0)]
    command.downgrade(config, "c3b6d91e8a20")
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(model_asset_installs)")}
        assert "use_case_derived" not in columns
        assert connection.execute(
            "SELECT id, use_case FROM model_asset_installs ORDER BY id"
        ).fetchall() == [("cleared", ""), ("written", "Watercolor landscapes")]
