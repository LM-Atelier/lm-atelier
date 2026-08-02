from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.util.exc import CommandError

from local_lm.config import Settings
from local_lm.database_migrations import (
    DatabaseVersionError,
    alembic_config,
    upgrade_database,
)


def test_unknown_database_revision_has_actionable_error(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "newer-database")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    unknown_revision = "newer_lm_atelier_revision"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY)")
        connection.execute(
            "INSERT INTO alembic_version VALUES (?)",
            (unknown_revision,),
        )

    with pytest.raises(
        DatabaseVersionError,
        match="database schema revision that this build does not recognize",
    ) as captured:
        upgrade_database(settings)

    assert isinstance(captured.value.__cause__, CommandError)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            unknown_revision,
        )


def test_unrelated_alembic_command_error_is_not_reclassified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "migration-failure")
    expected = CommandError("unrelated migration failure")

    def fail_upgrade(*_args: object, **_kwargs: object) -> None:
        raise expected

    monkeypatch.setattr(command, "upgrade", fail_upgrade)

    with pytest.raises(CommandError) as captured:
        upgrade_database(settings)

    assert captured.value is expected


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
        profile_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(model_profiles)").fetchall()
        }
        asset_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(model_asset_installs)").fetchall()
        }
        registry_install_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
        }
        unique_run_indexes = {
            tuple(
                column[2]
                for column in connection.execute(f'PRAGMA index_info("{index[1]}")').fetchall()
            )
            for index in connection.execute("PRAGMA index_list(runs)").fetchall()
            if index[2]
        }
    assert {
        "projects",
        "messages",
        "generation_presets",
        "custom_node_installs",
        "comfy_registry_installs",
        "model_asset_installs",
        "turn_creation_claims",
    } <= tables
    assert {
        "active_head_message_id",
        "active_vision_profile_id",
        "generation_settings_json",
        "generation_preset_ids_json",
        "vision_settings_json",
    } <= chat_columns
    assert {
        "image_workflow_revision_id",
        "video_workflow_revision_id",
        "generation_settings_json",
        "generation_preset_ids_json",
    } <= project_columns
    assert "use_case" in profile_columns
    assert {
        "source_id",
        "name",
        "kind",
        "family",
        "local_path",
        "size_bytes",
        "manifest_json",
        "active",
        "verified_at",
    } <= asset_columns
    assert {
        "package_id",
        "package_version",
        "registry_record_id",
        "archive_sha256",
        "manifest_sha256",
        "installed_path",
        "node_types_json",
        "pip_dependencies_json",
        "review_json",
        "trusted",
        "active",
    } <= registry_install_columns
    assert ("chat_id", "idempotency_key") in unique_run_indexes
    assert ("idempotency_key",) not in unique_run_indexes

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_generation_defaults_migration_backfills_existing_records(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "existing-generation-defaults")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f7a2c9e51b40")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-07-25 00:00:00"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO projects (
                id, name, description, instructions, archived, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("project-existing", "Existing", "", "", 0, timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO chats (
                id, project_id, title, archived, routing_mode, confirm_uncertain_media,
                active_chat_profile_id, active_image_profile_id, active_video_profile_id,
                active_head_message_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "chat-existing",
                "project-existing",
                "Existing",
                0,
                "auto",
                1,
                None,
                None,
                None,
                None,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        rows = [
            connection.execute(
                """
                SELECT generation_settings_json, generation_preset_ids_json
                FROM projects
                WHERE id = ?
                """,
                ("project-existing",),
            ).fetchone(),
            connection.execute(
                """
                SELECT generation_settings_json, generation_preset_ids_json
                FROM chats
                WHERE id = ?
                """,
                ("chat-existing",),
            ).fetchone(),
        ]
        for row in rows:
            assert row is not None
            assert json.loads(row[0]) == {}
            assert json.loads(row[1]) == {}


def test_sanitized_v017_database_upgrades_to_head_without_data_loss(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "v017-upgrade")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    fixture = Path(__file__).parent / "fixtures" / "v0.1.7-sanitized.sql"
    with sqlite3.connect(database) as connection:
        connection.executescript(fixture.read_text(encoding="utf-8"))

    upgrade_database(settings)

    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        project = connection.execute(
            """
            SELECT name, instructions, generation_settings_json, generation_preset_ids_json
            FROM projects
            WHERE id = 'project_v017'
            """
        ).fetchone()
        chat = connection.execute(
            """
            SELECT title, active_head_message_id,
                   generation_settings_json, generation_preset_ids_json,
                   active_vision_profile_id, vision_settings_json
            FROM chats
            WHERE id = 'chat_v017'
            """
        ).fetchone()
        run = connection.execute(
            """
            SELECT chat_id, idempotency_key, settings_json, provenance_json
            FROM runs
            WHERE id = 'run_v017'
            """
        ).fetchone()
        messages = connection.execute(
            """
            SELECT messages.id, message_parts.text
            FROM messages
            JOIN message_parts ON message_parts.message_id = messages.id
            WHERE messages.chat_id = 'chat_v017'
            ORDER BY messages.created_at
            """
        ).fetchall()
        profile = connection.execute(
            "SELECT use_case FROM model_profiles WHERE id = 'profile_v017'"
        ).fetchone()
        workflow = connection.execute(
            "SELECT trusted FROM workflow_revisions WHERE id = 'revision_v017'"
        ).fetchone()

    assert project is not None
    assert project["name"] == "Synthetic project"
    assert project["instructions"] == "Keep answers concise"
    assert json.loads(project["generation_settings_json"]) == {}
    assert json.loads(project["generation_preset_ids_json"]) == {}
    assert chat is not None
    assert chat["title"] == "Synthetic chat"
    assert chat["active_head_message_id"] == "assistant_v017"
    assert json.loads(chat["generation_settings_json"]) == {}
    assert json.loads(chat["generation_preset_ids_json"]) == {}
    assert chat["active_vision_profile_id"] == "__auto__"
    assert json.loads(chat["vision_settings_json"]) == {
        "max_images": 4,
        "max_video_frames": 6,
        "include_prior_visual": True,
    }
    assert run is not None
    assert run["chat_id"] == "chat_v017"
    assert run["idempotency_key"] == "legacy-idempotency-key"
    assert json.loads(run["settings_json"]) == {"max_tokens": 256}
    assert json.loads(run["provenance_json"]) == {
        "input_artifact_ids": ["artifact_v017"],
        "synthetic": True,
    }
    assert [tuple(row) for row in messages] == [
        ("user_v017", "Synthetic prompt"),
        ("assistant_v017", "Synthetic response"),
    ]
    assert profile is not None and profile["use_case"] == "writing"
    assert workflow is not None and workflow["trusted"] == 1


def test_idempotency_migration_downgrade_preserves_cross_chat_runs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    settings = Settings(data_dir=tmp_path / "idempotency-downgrade")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    fixture = Path(__file__).parent / "fixtures" / "v0.1.7-sanitized.sql"
    with sqlite3.connect(database) as connection:
        connection.executescript(fixture.read_text(encoding="utf-8"))
    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        connection.execute(
            """
                INSERT INTO chats (
                    id, project_id, title, archived, routing_mode, confirm_uncertain_media,
                    active_chat_profile_id, active_vision_profile_id,
                    active_image_profile_id, active_video_profile_id,
                    created_at, updated_at, active_head_message_id,
                    generation_settings_json, generation_preset_ids_json,
                    vision_settings_json
                )
                SELECT
                    'chat_v017_other', project_id, 'Other synthetic chat', archived,
                    routing_mode, confirm_uncertain_media, active_chat_profile_id,
                    active_vision_profile_id, active_image_profile_id, active_video_profile_id,
                    created_at, updated_at, 'assistant_v017_other',
                    generation_settings_json, generation_preset_ids_json,
                    vision_settings_json
            FROM chats
            WHERE id = 'chat_v017'
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, chat_id, parent_id, role, status, created_at, updated_at
            ) VALUES (
                'user_v017_other', 'chat_v017_other', NULL, 'user', 'complete',
                '2026-07-25 00:00:00', '2026-07-25 00:00:00'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO messages (
                id, chat_id, parent_id, role, status, created_at, updated_at
            ) VALUES (
                'assistant_v017_other', 'chat_v017_other', 'user_v017_other',
                'assistant', 'complete',
                '2026-07-25 00:00:01', '2026-07-25 00:00:01'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO runs (
                id, idempotency_key, chat_id, user_message_id, assistant_message_id,
                operation, status, standalone_prompt, profile_id, workflow_revision_id,
                settings_json, provenance_json, error, started_at, completed_at,
                duration_ms, created_at, updated_at
            )
            SELECT
                'run_v017_other', idempotency_key, 'chat_v017_other',
                'user_v017_other', 'assistant_v017_other', operation, status,
                standalone_prompt, profile_id, workflow_revision_id, settings_json,
                provenance_json, error, started_at, completed_at, duration_ms,
                created_at, updated_at
            FROM runs
            WHERE id = 'run_v017'
            """
        )
        connection.commit()

    command.downgrade(config, "a4d7c2e91b63")
    with sqlite3.connect(database) as connection:
        keys = connection.execute(
            """
            SELECT id, idempotency_key
            FROM runs
            WHERE id IN ('run_v017', 'run_v017_other')
            ORDER BY id
            """
        ).fetchall()
        assert len(keys) == 2
        assert keys[0][1] != keys[1][1]
        assert all(key for _run_id, key in keys)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM runs
            WHERE id IN ('run_v017', 'run_v017_other')
            """
        ).fetchone() == (2,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_artifact_hash_backfills_existing_revisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """An existing installation keeps its evidence rather than re-proving models.

    The backfill reads stored JSON only, so it needs no live runtime, and it must
    agree with what the application computes for the same revision afterwards.
    """
    from alembic import command

    from local_lm.model_planner import workflow_artifact_contract

    settings = Settings(data_dir=tmp_path / "backfill")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c6e9b2f41d30")

    api_graph = {"3": {"class_type": "KSampler"}}
    input_schema = {"type": "object"}
    dependencies = {"template_id": "example", "model_install_ids": ["local_only"]}
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "insert into workflow_definitions (id, name, description, operation, created_at,"
            " updated_at) values ('wf_1', 'Example', '', 'text_to_image', '2026-01-01',"
            " '2026-01-01')"
        )
        connection.execute(
            "insert into workflow_revisions (id, workflow_id, version, engine, ui_graph_json,"
            " api_graph_json, input_schema_json, dependencies_json, trusted, created_at,"
            " updated_at) values ('rev_1', 'wf_1', 1, 'comfyui', ?, ?, ?, ?, 1, '2026-01-01',"
            " '2026-01-01')",
            (
                json.dumps({"nodes": []}),
                json.dumps(api_graph),
                json.dumps(input_schema),
                json.dumps(dependencies),
            ),
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        digest = connection.execute(
            "select artifact_sha256 from workflow_revisions where id = 'rev_1'"
        ).fetchone()[0]

    assert digest == workflow_artifact_contract(
        operation="text_to_image",
        engine="comfyui",
        api_graph=api_graph,
        input_schema=input_schema,
        dependencies=dependencies,
    )
