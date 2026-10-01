from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

PREVIOUS = "a7d4e9b62c10"
TABLES = {"generation_experiments", "generation_experiment_arms", "generation_experiment_trials"}


def _tables(database: Path) -> set[str]:
    with sqlite3.connect(database) as connection:
        return {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }


def _migrated(tmp_path: Path) -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "generation-experiment-migration")
    settings.prepare()
    command.upgrade(alembic_config(settings), PREVIOUS)
    database = settings.state_dir / "local-lm.sqlite3"
    assert not TABLES & _tables(database)
    command.upgrade(alembic_config(settings), "head")
    assert _tables(database) >= TABLES
    return settings, database


def test_an_empty_database_returns_to_the_previous_version(tmp_path: Path) -> None:
    settings, database = _migrated(tmp_path)
    command.downgrade(alembic_config(settings), PREVIOUS)
    assert not TABLES & _tables(database)


def test_a_database_holding_a_comparison_refuses_to_go_back(tmp_path: Path) -> None:
    settings, database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO generation_experiments (id, name, state, operation, contract_version, "
            "app_version, seed_policy, seed_equivalence, common_json, estimate_json, "
            "preflight_sha256, snapshot_sha256, idempotency_key, request_sha256, created_at, "
            "updated_at) VALUES ('gexp_kept', 'Kept', 'ready', 'text_to_image', 1, '0.0.0', "
            "'random_per_trial', 'none', '{}', '[]', ?, ?, 'kept', ?, "
            "'2026-10-01 00:00:00', '2026-10-01 00:00:00')",
            ("a" * 64, "b" * 64, "c" * 64),
        )
    with pytest.raises(RuntimeError, match="require the current database version"):
        command.downgrade(alembic_config(settings), PREVIOUS)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT id FROM generation_experiments").fetchall() == [
            ("gexp_kept",)
        ]


def test_started_comparisons_refuse_to_go_back_and_ready_ones_keep_their_rows(
    tmp_path: Path,
) -> None:
    settings, database = _migrated(tmp_path)
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(generation_experiments)")
        }
        assert {"work_plan_id", "start_idempotency_key", "start_request_sha256", "started_at"} <= (
            columns
        )
        trial_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(generation_experiment_trials)")
        }
        assert {"work_step_id", "run_id"} <= trial_columns
        connection.execute(
            "INSERT INTO generation_experiments (id, name, state, operation, contract_version, "
            "app_version, seed_policy, seed_equivalence, common_json, estimate_json, "
            "preflight_sha256, snapshot_sha256, idempotency_key, request_sha256, created_at, "
            "updated_at) VALUES ('gexp_started', 'Started', 'started', 'text_to_image', 1, "
            "'0.0.0', 'random_per_trial', 'none', '{}', '[]', ?, ?, 'started', ?, "
            "'2026-10-01 00:00:00', '2026-10-01 00:00:00')",
            ("a" * 64, "b" * 64, "c" * 64),
        )
    with pytest.raises(RuntimeError, match="Started generation comparisons"):
        command.downgrade(alembic_config(settings), "b55b291b1bfb")
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE generation_experiments SET state = 'ready'")
    command.downgrade(alembic_config(settings), "b55b291b1bfb")
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(generation_experiments)")
        }
        assert "work_plan_id" not in columns
        assert connection.execute("SELECT id, state FROM generation_experiments").fetchall() == [
            ("gexp_started", "ready")
        ]
