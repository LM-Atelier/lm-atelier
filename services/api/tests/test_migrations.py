from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from threading import Event

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from alembic.util.exc import CommandError
from alembic_head import EXPECTED_ALEMBIC_HEAD
from sqlalchemy import UniqueConstraint, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError as SAIntegrityError
from sqlalchemy.orm import Session

from local_lm import db, models  # noqa: F401 - importing registers every table to compare
from local_lm.artifact_library_schema import CREATE_TRIGGER_SQL
from local_lm.backups import BackupManager
from local_lm.chat_item_removal_schema import (
    CREATE_CHAT_ITEM_REMOVAL_TRIGGER_SQL,
    DROP_CHAT_ITEM_REMOVAL_TRIGGER_SQL,
)
from local_lm.config import Settings
from local_lm.database_migrations import (
    DatabaseUpgradeError,
    DatabaseVersionError,
    alembic_config,
    upgrade_database,
)
from local_lm.db import Base, configure_database
from local_lm.migrations.versions import (
    c5a8e1d72f40_chat_item_content_tombstones as chat_item_tombstone_migration,
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


def test_chat_item_content_tombstone_migration_round_trips(tmp_path: Path) -> None:
    assert chat_item_tombstone_migration._CREATE_TRIGGER_SQL == CREATE_CHAT_ITEM_REMOVAL_TRIGGER_SQL
    assert chat_item_tombstone_migration._DROP_TRIGGER_SQL == DROP_CHAT_ITEM_REMOVAL_TRIGGER_SQL
    settings = Settings(data_dir=tmp_path / "chat-item-content-tombstone")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f2a7c9d41e63")
    database = settings.state_dir / "local-lm.sqlite3"

    with sqlite3.connect(database) as connection:
        before = {row[1]: row for row in connection.execute("PRAGMA table_info(messages)")}
        assert "content_removed_at" not in before

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        after = {row[1]: row for row in connection.execute("PRAGMA table_info(messages)")}
        assert after["content_removed_at"][3] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name = 'ix_messages_content_removed_at'"
        ).fetchone() == ("ix_messages_content_removed_at",)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            EXPECTED_ALEMBIC_HEAD,
        )

    engine = create_engine(f"sqlite:///{database}")
    with Session(engine) as session:
        chat = models.Chat(id="chat_tombstone_guard", title="Tombstone guard")
        empty = models.Message(
            id="msg_tombstone_empty",
            chat_id=chat.id,
            role="user",
            status="complete",
        )
        carrying = models.Message(
            id="msg_tombstone_payload",
            chat_id=chat.id,
            role="user",
            status="complete",
            parts=[
                models.MessagePart(
                    position=0,
                    type="text",
                    text="must detach first",
                )
            ],
        )
        revision_source = models.Message(
            id="msg_tombstone_revision_source",
            chat_id=chat.id,
            role="assistant",
            status="complete",
        )
        revision = models.ResponseRevision(
            id="revision_tombstone_payload",
            message_id=revision_source.id,
            sequence=1,
            status="complete",
            parts=[
                models.ResponseRevisionPart(
                    position=0,
                    type="text",
                    text="must not cross the tombstone boundary",
                )
            ],
        )
        session.add_all([chat, empty, carrying, revision_source, revision])
        session.commit()
        empty.content_removed_at = empty.updated_at
        session.commit()

    with Session(engine) as session:
        loaded_empty = session.get(models.Message, "msg_tombstone_empty")
        assert loaded_empty is not None
        loaded_empty.parts.append(
            models.MessagePart(position=0, type="text", text="must stay absent")
        )
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive message parts",
        ):
            session.flush()

    with Session(engine) as session:
        loaded_revision = session.get(
            models.ResponseRevision,
            "revision_tombstone_payload",
        )
        assert loaded_revision is not None
        loaded_revision.message_id = "msg_tombstone_empty"
        with pytest.raises(
            SAIntegrityError,
            match="removed chat item cannot receive a populated revision",
        ):
            session.flush()

    with Session(engine) as session:
        loaded_carrying = session.get(models.Message, "msg_tombstone_payload")
        assert loaded_carrying is not None
        loaded_carrying.content_removed_at = loaded_carrying.updated_at
        with pytest.raises(
            SAIntegrityError,
            match="chat item payload must be detached before tombstoning",
        ):
            session.flush()

    with Session(engine) as session:
        reloaded_empty = session.get(models.Message, "msg_tombstone_empty")
        assert reloaded_empty is not None
        reloaded_empty.content_removed_at = None
        with pytest.raises(
            SAIntegrityError,
            match="chat item content tombstone is immutable",
        ):
            session.flush()
    engine.dispose()

    command.downgrade(config, "f2a7c9d41e63")
    with sqlite3.connect(database) as connection:
        restored = {row[1]: row for row in connection.execute("PRAGMA table_info(messages)")}
        assert "content_removed_at" not in restored
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f2a7c9d41e63",
        )


def test_chat_item_removal_receipt_migration_preserves_replay_authority(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "chat-item-removal-receipts")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "d6a8f1c42b90")
    database = settings.state_dir / "local-lm.sqlite3"

    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'chat_item_removal_receipts'"
            ).fetchone()
            is None
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chat_item_removal_receipts)")
        }
        assert columns == {
            "id",
            "chat_id",
            "operation_key",
            "message_id",
            "request_sha256",
            "message_revision_id",
            "content_removed_at",
            "created_at",
            "updated_at",
        }

    engine = create_engine(f"sqlite:///{database}")
    with Session(engine) as session:
        chat = models.Chat(id="chat_receipt_migration", title="Constructed")
        message = models.Message(
            id="msg_receipt_migration",
            chat=chat,
            role="user",
            status="complete",
        )
        session.add_all([chat, message])
        session.flush()
        session.add(
            models.ChatItemRemovalReceipt(
                id="rmreceipt_migration",
                chat_id=chat.id,
                operation_key="remove-once",
                message_id=message.id,
                request_sha256="1" * 64,
                message_revision_id="2" * 64,
                content_removed_at=datetime.now(UTC),
            )
        )
        session.commit()

    with pytest.raises(RuntimeError, match="durable replay receipts exist"):
        command.downgrade(config, "d6a8f1c42b90")

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM chat_item_removal_receipts")
        connection.commit()
    command.downgrade(config, "d6a8f1c42b90")
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'chat_item_removal_receipts'"
            ).fetchone()
            is None
        )
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "d6a8f1c42b90",
        )


@contextmanager
def _application_sqlite(database: Path) -> Iterator[sqlite3.Connection]:
    """Use the application's default-closed SQLite functions, without its ORM guard."""
    engine = create_engine(f"sqlite:///{database}")
    try:
        with engine.connect() as connection:
            driver = connection.connection.driver_connection
            assert isinstance(driver, sqlite3.Connection)
            with driver:
                yield driver
    finally:
        engine.dispose()


def test_artifact_library_entry_migration_backfills_once_and_seals_membership(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "library-entry-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c7e1d4a83b56")
    database = settings.state_dir / "local-lm.sqlite3"
    stamp = "2026-08-12 12:00:00"
    rows = [
        ("legacy-image", "1" * 64, "image", "image/png", "portrait.png", 1),
        ("legacy-video", "2" * 64, "video", "video/mp4", None, 0),
        ("legacy-input", "3" * 64, "input", "image/png", "upload.png", 1),
    ]
    with sqlite3.connect(database) as connection:
        connection.executemany(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES (?, ?, ?, ?, 7, ?, ?, '{}', ?, ?, ?)
            """,
            [
                (artifact_id, digest, kind, media_type, digest, name, favorite, stamp, stamp)
                for artifact_id, digest, kind, media_type, name, favorite in rows
            ],
        )
    command.upgrade(config, "head")

    with _application_sqlite(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        entries = connection.execute(
            """
            SELECT id, artifact_id, display_name, favorite, state, version,
                   created_at, updated_at
            FROM artifact_library_entries ORDER BY artifact_id
            """
        ).fetchall()
        assert entries == [
            (
                "libentry:sha256:" + "1" * 64,
                "legacy-image",
                "portrait.png",
                1,
                "visible",
                1,
                stamp,
                stamp,
            ),
            (
                "libentry:sha256:" + "2" * 64,
                "legacy-video",
                "2" * 64,
                0,
                "visible",
                1,
                stamp,
                stamp,
            ),
        ]
        connection.execute(
            """
            INSERT INTO artifact_library_entries
              (id, artifact_id, display_name, favorite, state, deleted_at, recovery_id,
               version, created_at, updated_at)
            SELECT 'libentry:sha256:' || sha256, id,
                   COALESCE(NULLIF(trim(original_name), ''), sha256), favorite,
                   'visible', NULL, NULL, 1, created_at, updated_at
            FROM artifacts WHERE kind IN ('image', 'video')
            ON CONFLICT(artifact_id) DO NOTHING
            """
        )
        assert connection.execute("SELECT count(*) FROM artifact_library_entries").fetchone() == (
            2,
        )
        expected_triggers = {statement.split()[2] for statement in CREATE_TRIGGER_SQL}
        migrated_triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        assert expected_triggers <= migrated_triggers
        with pytest.raises(sqlite3.IntegrityError, match="version is stale"):
            connection.execute(
                "UPDATE artifact_library_entries SET display_name = 'changed' "
                "WHERE artifact_id = 'legacy-image'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            connection.execute(
                "UPDATE artifact_library_entries SET artifact_id = 'legacy-video', "
                "version = 2 WHERE artifact_id = 'legacy-image'"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM artifacts WHERE id = 'legacy-image'")
        with pytest.raises(sqlite3.IntegrityError, match="library artifact identity"):
            connection.execute("UPDATE artifacts SET kind = 'other' WHERE id = 'legacy-image'")
        with pytest.raises(sqlite3.IntegrityError, match="deletion is not authorized"):
            connection.execute(
                "DELETE FROM artifact_library_entries WHERE artifact_id = 'legacy-image'"
            )
        connection.execute(
            "UPDATE artifact_library_entries SET state = 'trashed', deleted_at = ?, "
            "recovery_id = 'recover-image', version = 2 WHERE artifact_id = 'legacy-image'",
            (stamp,),
        )
        connection.execute(
            "UPDATE artifact_library_entries SET state = 'trashed', deleted_at = ?, "
            "recovery_id = 'recover-video', version = 2 WHERE artifact_id = 'legacy-video'",
            (stamp,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            connection.execute(
                "UPDATE artifact_library_entries SET recovery_id = 'recover-image', version = 3 "
                "WHERE artifact_id = 'legacy-video'"
            )
        assert connection.execute(
            "SELECT kind, original_name, favorite FROM artifacts ORDER BY id"
        ).fetchall() == [
            ("image", "portrait.png", 1),
            ("input", "upload.png", 1),
            ("video", None, 0),
        ]


@pytest.mark.parametrize(
    "payload",
    (
        '{"artifact_ids":"not-a-list"}',
        '{"unterminated":',
        '{"artifact_id":"sha256:' + "f" * 64 + '"}',
    ),
)
def test_artifact_library_migration_refuses_preexisting_corrupt_reference_json(
    tmp_path: Path,
    payload: str,
) -> None:
    settings = Settings(data_dir=tmp_path / "library-entry-corrupt-reference")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c7e1d4a83b56")
    database = settings.state_dir / "local-lm.sqlite3"
    stamp = "2026-08-12 12:00:00"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO jobs
              (id, kind, status, progress, phase, payload_json, result_json,
               progress_json, queue_priority, attempt, cancellable, created_at, updated_at)
            VALUES ('corrupt-before-library', 'chat', 'queued', 0, 'queued',
                    ?, '{}', '{}', 0, 0, 1, ?, ?)
            """,
            (payload, stamp, stamp),
        )

    with pytest.raises(SAIntegrityError, match="artifact JSON reference is invalid"):
        command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c7e1d4a83b56",
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'artifact_library_entries'"
            ).fetchone()
            is None
        )
        assert connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' "
            "AND name LIKE '%artifact_reference%_guard'"
        ).fetchone() == (0,)
        assert connection.execute(
            "SELECT payload_json FROM jobs WHERE id = 'corrupt-before-library'"
        ).fetchone() == (payload,)


@pytest.mark.parametrize("original_name", ("x" * 501, "bad\0name"))
def test_artifact_library_migration_refuses_legacy_names_before_ddl(
    tmp_path: Path,
    original_name: str,
) -> None:
    settings = Settings(data_dir=tmp_path / "library-entry-invalid-name")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c7e1d4a83b56")
    database = settings.state_dir / "local-lm.sqlite3"
    stamp = "2026-08-12 12:00:00"
    digest = "a" * 64
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES ('legacy-invalid-name', ?, 'image', 'image/png', 7, ?, ?, '{}', 0, ?, ?)
            """,
            (digest, digest, original_name, stamp, stamp),
        )

    with pytest.raises(SAIntegrityError, match="artifact JSON reference is invalid"):
        command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c7e1d4a83b56",
        )
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name = 'artifact_library_entries'"
            ).fetchone()
            is None
        )


def test_artifact_library_migration_fence_blocks_concurrent_dangling_writer(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "library-entry-migration-race")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "c7e1d4a83b56")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")

    audit_started = Event()
    writer_started = Event()
    writer_done = Event()
    writer_was_blocked: list[bool] = []

    def trace(statement: str) -> None:
        if "SELECT CASE" not in statement or audit_started.is_set():
            return
        audit_started.set()
        if writer_started.wait(5):
            writer_was_blocked.append(not writer_done.wait(0.2))

    def register_trace(dbapi_connection: object, _record: object) -> None:
        dbapi_connection.set_trace_callback(trace)  # type: ignore[attr-defined]

    def upgrade() -> None:
        command.upgrade(config, "head")

    def write_dangling_reference() -> str:
        assert audit_started.wait(5)
        writer_started.set()
        try:
            with sqlite3.connect(database, timeout=5) as connection:
                connection.execute(
                    """
                    INSERT INTO jobs
                      (id, kind, status, progress, phase, payload_json, result_json,
                       progress_json, queue_priority, attempt, cancellable, created_at, updated_at)
                    VALUES ('migration-race-writer', 'chat', 'queued', 0, 'queued',
                            ?, '{}', '{}', 0, 0, 1, ?, ?)
                    """,
                    (
                        '{"artifact_id":"sha256:' + "f" * 64 + '"}',
                        "2026-08-12 12:00:00",
                        "2026-08-12 12:00:00",
                    ),
                )
        except sqlite3.IntegrityError as error:
            return str(error)
        finally:
            writer_done.set()
        return "dangling reference committed"

    event.listen(Engine, "connect", register_trace)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            upgrade_result = executor.submit(upgrade)
            writer_result = executor.submit(write_dangling_reference)
            upgrade_result.result(timeout=15)
            outcome = writer_result.result(timeout=15)
    finally:
        event.remove(Engine, "connect", register_trace)

    assert writer_was_blocked == [True]
    assert outcome == "artifact JSON reference is invalid"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM jobs WHERE id = 'migration-race-writer'"
        ).fetchone() == (0,)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            EXPECTED_ALEMBIC_HEAD,
        )
        membership_schema = connection.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'media_collection_memberships'"
        ).fetchone()
        assert membership_schema is not None
        assert "note = trim(note)" in membership_schema[0]


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
        workflow_definition_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
        }
        workflow_revision_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
        }
        workflow_family_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(workflow_families)").fetchall()
        }
        workflow_preference_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_preferences)").fetchall()
        }
        workflow_preference_indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(workflow_preferences)").fetchall()
        }
        workflow_install_offer_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(workflow_install_offers)").fetchall()
        }
        workflow_install_offer_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_install_offers)"
            ).fetchall()
        }
        workflow_install_offer_indexes = {
            row[1]
            for row in connection.execute("PRAGMA index_list(workflow_install_offers)").fetchall()
        }
        workflow_default_index_sql = connection.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'uq_workflow_preferences_default_selector'
            """
        ).fetchone()
        workflow_definition_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_definitions)"
            ).fetchall()
        }
        workflow_preference_foreign_keys = {
            (row[3], row[2], row[4], row[6])
            for row in connection.execute(
                "PRAGMA foreign_key_list(workflow_preferences)"
            ).fetchall()
        }
        workflow_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
            if row[1] == "id"
        )
        workflow_revision_definition_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
            if row[1] == "workflow_id"
        )
        registry_install_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
            if row[1] == "id"
        )
        component_manifest_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_component_manifests)").fetchall()
            if row[1] == "id"
        )
        capability_evidence_id_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_capability_evidence)").fetchall()
            if row[1] == "id"
        )
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
        "workflow_families",
        "workflow_preferences",
        "workflow_dependency_slots",
        "workflow_activations",
        "workflow_dependency_bindings",
        "workflow_profile_compatibility",
        "chat_workflow_selections",
        "project_workflow_selections",
        "workflow_install_offers",
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
        "wheel_closure_sha256",
        "wheel_environment_sha256",
        "wheel_environment_path",
        "trusted",
        "active",
    } <= registry_install_columns
    assert {"family_id", "variant_key"} <= workflow_definition_columns
    assert {"capabilities_json", "dependency_contract_sha256"} <= workflow_revision_columns
    assert {
        "id",
        "name",
        "description",
        "use_case",
        "tags_json",
        "enabled",
        "archived",
    } <= workflow_family_columns
    assert {
        "id",
        "workflow_family_id",
        "selector_capability",
        "enabled",
        "is_default",
        "sort_order",
    } <= workflow_preference_columns
    assert {
        "ix_workflow_preferences_selector_order",
        "uq_workflow_preferences_default_selector",
    } <= workflow_preference_indexes
    assert {
        "id",
        "workflow_revision_id",
        "workflow_artifact_sha256",
        "dependency_contract_sha256",
        "binding_plan_sha256",
        "offer_sha256",
        "selections_json",
        "assets_json",
        "plan_count",
        "total_bytes",
        "status",
        "queued_at",
        "completed_at",
        "invalidated_at",
        "invalidation_code",
        "invalidation_reason",
    } <= workflow_install_offer_columns
    assert {
        "ix_workflow_install_offers_workflow_revision_id",
        "ix_workflow_install_offers_offer_sha256",
        "ix_workflow_install_offers_status",
        "ix_workflow_install_offer_revision_status",
    } <= workflow_install_offer_indexes
    assert workflow_default_index_sql is not None
    assert "WHERE is_default = 1" in workflow_default_index_sql[0]
    assert (
        "family_id",
        "workflow_families",
        "id",
        "SET NULL",
    ) in workflow_definition_foreign_keys
    assert (
        "workflow_family_id",
        "workflow_families",
        "id",
        "CASCADE",
    ) in workflow_preference_foreign_keys
    assert (
        "workflow_revision_id",
        "workflow_revisions",
        "id",
        "CASCADE",
    ) in workflow_install_offer_foreign_keys
    assert ("chat_id", "idempotency_key") in unique_run_indexes
    assert ("idempotency_key",) not in unique_run_indexes
    assert workflow_id_type == "VARCHAR(64)"
    assert workflow_revision_definition_type == "VARCHAR(64)"
    assert registry_install_id_type == "VARCHAR(64)"
    assert component_manifest_id_type == "VARCHAR(64)"
    assert capability_evidence_id_type == "VARCHAR(64)"

    command.downgrade(config, "base")
    command.upgrade(config, "head")
    with sqlite3.connect(settings.state_dir / "local-lm.sqlite3") as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_workflow_family_migration_preserves_existing_revisions(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "workflow-family-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "d8b3e6f92a41")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-03 00:00:00"
    definition = (
        "workflow_existing",
        "Existing workflow",
        "image_to_image",
        "Existing description",
        "revision_existing",
        timestamp,
        timestamp,
    )
    second_definition = (
        "workflow_existing_alternative",
        "Existing alternative",
        "image_to_image",
        "Another workflow with the same legacy operation",
        None,
        timestamp,
        timestamp,
    )
    revision = (
        "revision_existing",
        "workflow_existing",
        3,
        "comfyui",
        "0.3.50",
        '{"nodes":[{"id":1}]}',
        '{"1":{"class_type":"KSampler"}}',
        '{"type":"object"}',
        '{"model_install_ids":["model_existing"]}',
        "c" * 64,
        1,
        timestamp,
        timestamp,
    )
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, current_revision_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            definition,
        )
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, current_revision_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            second_definition,
        )
        connection.execute(
            """
            INSERT INTO workflow_revisions (
                id, workflow_id, version, engine, engine_version,
                ui_graph_json, api_graph_json, input_schema_json,
                dependencies_json, artifact_sha256, trusted,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            revision,
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        migrated_definition = connection.execute(
            """
            SELECT id, name, operation, description, current_revision_id,
                   created_at, updated_at, family_id, variant_key
            FROM workflow_definitions
            WHERE id = 'workflow_existing'
            """
        ).fetchone()
        migrated_revision = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   capabilities_json, dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at
            FROM workflow_revisions
            WHERE id = 'revision_existing'
            """
        ).fetchone()
        assert migrated_definition == (*definition, None, None)
        assert migrated_revision == (*revision[:8], "[]", *revision[8:])
        assert connection.execute(
            """
            SELECT COUNT(*)
            FROM workflow_definitions
            WHERE operation = 'image_to_image'
              AND family_id IS NULL
              AND variant_key IS NULL
            """
        ).fetchone() == (2,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_families").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_preferences").fetchone() == (0,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []

        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO workflow_families (
                id, name, description, use_case, tags_json, enabled, archived,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wffamily_delete_probe",
                "Delete probe",
                "",
                "",
                "[]",
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            UPDATE workflow_definitions
            SET family_id = 'wffamily_delete_probe', variant_key = 'edit'
            WHERE id = 'workflow_existing'
            """
        )
        connection.execute(
            """
            INSERT INTO workflow_preferences (
                id, workflow_family_id, selector_capability, enabled,
                is_default, sort_order, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wfpref_delete_probe",
                "wffamily_delete_probe",
                "image",
                1,
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute("DELETE FROM workflow_families WHERE id = 'wffamily_delete_probe'")
        assert connection.execute(
            "SELECT family_id, variant_key FROM workflow_definitions WHERE id = 'workflow_existing'"
        ).fetchone() == (None, "edit")
        assert connection.execute("SELECT COUNT(*) FROM workflow_preferences").fetchone() == (0,)
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_revisions WHERE id = 'revision_existing'"
        ).fetchone() == (1,)
        connection.commit()

    command.downgrade(config, "d8b3e6f92a41")

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        restored_definition = connection.execute(
            """
            SELECT id, name, operation, description, current_revision_id,
                   created_at, updated_at
            FROM workflow_definitions
            WHERE id = 'workflow_existing'
            """
        ).fetchone()
        restored_revision = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at
            FROM workflow_revisions
            WHERE id = 'revision_existing'
            """
        ).fetchone()
        assert "workflow_families" not in tables
        assert "workflow_preferences" not in tables
        assert restored_definition == definition
        assert restored_revision == revision
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_definitions WHERE operation = 'image_to_image'"
        ).fetchone() == (2,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_workflow_family_migration_refuses_lossy_downgrade(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "workflow-family-downgrade")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-03 00:00:00"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO workflow_families (
                id, name, description, use_case, tags_json, enabled, archived,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wffamily_keep",
                "Keep this family",
                "",
                "",
                "[]",
                1,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    with pytest.raises(RuntimeError, match="Cannot downgrade workflow families"):
        command.downgrade(config, "d8b3e6f92a41")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "b6a1e4d92c70",
        )
        assert connection.execute(
            "SELECT name FROM workflow_families WHERE id = 'wffamily_keep'"
        ).fetchone() == ("Keep this family",)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_generated_identifier_width_migration_preserves_existing_rows(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "generated-identifier-widths")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "e6c42a9b13fd")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-02 00:00:00"
    workflow_id = f"workflow_{'a' * 32}"
    revision_id = f"wfrev_{'b' * 32}"
    registry_id = f"registry_{'c' * 32}"
    model_id = f"model_{'d' * 32}"
    component_id = f"component_{'e' * 32}"
    evidence_id = f"evidence_{'f' * 32}"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO model_installs (
                id, name, role, engine, local_path, size_bytes, compatibility,
                manifest_json, active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                "Existing model",
                "image",
                "comfyui",
                "models/existing.safetensors",
                1,
                "ready",
                "{}",
                1,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_component_manifests (
                id, model_install_id, kind, relative_path, target_folder,
                required, metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                component_id,
                model_id,
                "checkpoint",
                "existing.safetensors",
                "checkpoints",
                1,
                "{}",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO model_capability_evidence (
                id, model_install_id, evidence_key, result,
                component_hashes_json, runtime_build, adapter_contract_version,
                launch_contract_version, hardware_class, probe_version,
                details_json, probed_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                model_id,
                "existing-evidence",
                "passed",
                "{}",
                "test-runtime",
                1,
                "test-contract",
                "test-hardware",
                "test-probe",
                "{}",
                timestamp,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_definitions (
                id, name, operation, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                "Existing workflow",
                "text_to_image",
                "",
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_revisions (
                id, workflow_id, version, engine, ui_graph_json, api_graph_json,
                input_schema_json, dependencies_json, trusted, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                revision_id,
                workflow_id,
                1,
                "comfyui",
                "{}",
                "{}",
                "{}",
                "{}",
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.execute(
            """
            INSERT INTO comfy_registry_installs (
                id, package_id, package_version, registry_record_id, repository_url,
                download_url, archive_sha256, manifest_sha256, installed_path,
                node_types_json, pip_dependencies_json, review_json, trusted, active,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                registry_id,
                "example-node",
                "1.2.3",
                "record-123",
                "https://github.com/example/node",
                "https://api.comfy.org/example.zip",
                "d" * 64,
                "e" * 64,
                "example-node",
                "[]",
                "[]",
                "{}",
                0,
                0,
                timestamp,
                timestamp,
            ),
        )
        connection.commit()

    command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        workflow_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_definitions)").fetchall()
            if row[1] == "id"
        )
        revision_workflow_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(workflow_revisions)").fetchall()
            if row[1] == "workflow_id"
        )
        registry_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(comfy_registry_installs)").fetchall()
            if row[1] == "id"
        )
        component_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_component_manifests)").fetchall()
            if row[1] == "id"
        )
        evidence_type = next(
            row[2]
            for row in connection.execute("PRAGMA table_info(model_capability_evidence)").fetchall()
            if row[1] == "id"
        )
        assert workflow_type == "VARCHAR(64)"
        assert revision_workflow_type == "VARCHAR(64)"
        assert registry_type == "VARCHAR(64)"
        assert component_type == "VARCHAR(64)"
        assert evidence_type == "VARCHAR(64)"
        assert connection.execute(
            "SELECT id FROM workflow_definitions WHERE id = ?",
            (workflow_id,),
        ).fetchone() == (workflow_id,)
        assert connection.execute(
            "SELECT workflow_id FROM workflow_revisions WHERE id = ?",
            (revision_id,),
        ).fetchone() == (workflow_id,)
        assert connection.execute(
            "SELECT id FROM comfy_registry_installs WHERE id = ?",
            (registry_id,),
        ).fetchone() == (registry_id,)
        assert connection.execute(
            "SELECT id FROM model_component_manifests WHERE id = ?",
            (component_id,),
        ).fetchone() == (component_id,)
        assert connection.execute(
            "SELECT id FROM model_capability_evidence WHERE id = ?",
            (evidence_id,),
        ).fetchone() == (evidence_id,)
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
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


def _parent_revision(settings: Settings) -> str:
    script = ScriptDirectory.from_config(alembic_config(settings))
    head = script.get_revision("head")
    assert head.down_revision, "expected the migration chain to have more than one revision"
    return str(head.down_revision)


def test_expected_alembic_head_matches_script_directory(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "expected-alembic-head")
    settings.prepare()
    script = ScriptDirectory.from_config(alembic_config(settings))

    assert script.get_current_head() == EXPECTED_ALEMBIC_HEAD


def test_a_failed_upgrade_gives_the_data_back_on_the_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this guards is not hypothetical: SQLite's driver leaves DDL
    standing when the transaction around it fails, while the revision marker
    rolls back. So the failure is injected after real DDL has been applied."""

    settings = Settings(data_dir=tmp_path / "interrupted-upgrade")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    upgrade_database(settings)
    _run(
        database,
        ("CREATE TABLE canary (id TEXT PRIMARY KEY)", ()),
        ("INSERT INTO canary VALUES ('irreplaceable')", ()),
        ("UPDATE alembic_version SET version_num = ?", (_parent_revision(settings),)),
    )

    def half_apply_then_die(*_args: object, **_kwargs: object) -> None:
        _run(database, ("CREATE TABLE added_by_the_migration (id TEXT PRIMARY KEY)", ()))
        raise RuntimeError("interrupted partway")

    monkeypatch.setattr(command, "upgrade", half_apply_then_die)

    with pytest.raises(DatabaseUpgradeError, match="Restart LM Atelier"):
        upgrade_database(settings)

    # Without the restore the install is finished: the schema holds half a
    # migration the data says was never applied.
    assert _table_exists(database, "added_by_the_migration")

    assert BackupManager(settings).apply_pending_restore()

    assert not _table_exists(database, "added_by_the_migration")
    assert _query(database, "SELECT id FROM canary") == [("irreplaceable",)]


def test_data_from_a_newer_build_is_never_restored_over(tmp_path: Path) -> None:
    """Nothing was applied, so there is nothing to give back, and replacing the
    data would destroy what a newer build wrote."""

    settings = Settings(data_dir=tmp_path / "newer-build")
    settings.prepare()
    _run(
        settings.state_dir / "local-lm.sqlite3",
        ("CREATE TABLE alembic_version (version_num TEXT PRIMARY KEY)", ()),
        ("INSERT INTO alembic_version VALUES ('from_a_newer_build')", ()),
    )

    with pytest.raises(DatabaseVersionError):
        upgrade_database(settings)

    assert not (settings.state_dir / "restore-on-next-start.json").exists()


def test_an_upgrade_with_nothing_to_do_does_not_copy_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_dir=tmp_path / "already-current")
    settings.prepare()
    upgrade_database(settings)
    before = {path.name for path in settings.backup_dir.glob("*.sqlite3")}

    upgrade_database(settings)

    assert {path.name for path in settings.backup_dir.glob("*.sqlite3")} == before


def _run(database: Path, *statements: tuple[str, tuple[object, ...]]) -> None:
    """Windows will not let a restore replace a file that is still open."""

    with closing(sqlite3.connect(database)) as connection:
        for sql, parameters in statements:
            connection.execute(sql, parameters)
        connection.commit()


def _query(database: Path, sql: str) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(sql).fetchall()


def _table_exists(database: Path, name: str) -> bool:
    return bool(
        _query(database, f"SELECT name FROM sqlite_master WHERE type='table' AND name = '{name}'")
    )


def test_the_restore_is_armed_before_the_upgrade_not_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A process that is killed never reaches an exception handler, so the
    request has to already be on disk while the upgrade is in flight."""

    settings = Settings(data_dir=tmp_path / "armed-first")
    settings.prepare()
    database = settings.state_dir / "local-lm.sqlite3"
    upgrade_database(settings)
    _run(
        database,
        ("UPDATE alembic_version SET version_num = ?", (_parent_revision(settings),)),
    )
    marker = settings.state_dir / "restore-on-next-start.json"
    armed_during_upgrade: list[bool] = []

    def observe(*_args: object, **_kwargs: object) -> None:
        armed_during_upgrade.append(marker.is_file())

    monkeypatch.setattr(command, "upgrade", observe)

    upgrade_database(settings)

    assert armed_during_upgrade == [True]
    # And withdrawn once the upgrade survived, so an ordinary start does not
    # roll itself back.
    assert not marker.exists()


def test_a_brand_new_database_is_not_snapshotted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """There is nothing to restore to, and trying produced a traceback on the
    first launch of every new install - describing a loss that cannot happen."""

    settings = Settings(data_dir=tmp_path / "first-launch")
    settings.prepare()

    with caplog.at_level("WARNING"):
        upgrade_database(settings)

    assert not list(settings.backup_dir.glob("*.sqlite3"))
    assert not (settings.state_dir / "restore-on-next-start.json").exists()
    assert "Could not snapshot" not in caplog.text
    # And it really did upgrade.
    assert _recorded_revisions_for(settings)


def _recorded_revisions_for(settings: Settings) -> set[str]:
    with closing(sqlite3.connect(settings.state_dir / "local-lm.sqlite3")) as connection:
        return {row[0] for row in connection.execute("SELECT version_num FROM alembic_version")}


def _trigger_definitions(database: Path) -> dict[str, str]:
    with sqlite3.connect(database) as connection:
        return dict(
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
            ).fetchall()
        )


def test_metadata_bootstrap_preserves_migrated_triggers(tmp_path: Path) -> None:
    """``create_all`` over a migrated database preserves its exact guards."""

    settings = Settings(data_dir=tmp_path / "migrated-trigger-bootstrap", dev=True)
    settings.prepare()
    upgrade_database(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    before = _trigger_definitions(database)

    engine = create_engine(f"sqlite:///{database}")
    try:
        Base.metadata.create_all(engine)
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    with sqlite3.connect(database) as connection:
        assert _trigger_definitions(database) == before
        with pytest.raises(sqlite3.IntegrityError, match="requires media"):
            connection.execute(
                """
                INSERT INTO artifact_library_entries (
                  id, artifact_id, display_name, favorite, state, version,
                  created_at, updated_at
                ) VALUES (
                  'libentry:sha256:missing', 'missing', 'Missing', 0,
                  'visible', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
                )
                """
            )


def test_metadata_bootstrap_trigger_definitions_match_fresh_and_migrated_schema(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "migrated-trigger-parity", dev=True)
    settings.prepare()
    upgrade_database(settings)
    migrated = settings.state_dir / "local-lm.sqlite3"

    fresh = tmp_path / "fresh.sqlite3"
    engine = create_engine(f"sqlite:///{fresh}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    assert _trigger_definitions(fresh) == _trigger_definitions(migrated)


def test_metadata_bootstrap_refuses_a_same_name_trigger_with_the_wrong_body(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "tampered-trigger", dev=True)
    settings.prepare()
    upgrade_database(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    trigger_name = "artifact_library_entry_insert_guard"
    with sqlite3.connect(database) as connection:
        connection.execute(f"DROP TRIGGER {trigger_name}")
        connection.execute(
            f"""
            CREATE TRIGGER {trigger_name}
            BEFORE INSERT ON artifact_library_entries
            BEGIN
              SELECT 1;
            END
            """
        )

    engine = create_engine(f"sqlite:///{database}")
    try:
        with pytest.raises(RuntimeError, match=trigger_name):
            Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def test_the_migrated_schema_matches_the_models(tmp_path: Path) -> None:
    """The two ways this schema gets built have to produce the same thing.

    Migrations build it on real installs. `Base.metadata` builds it wherever
    something calls `create_all`, which several suites do. Nothing compared
    them, and they had drifted: four indexes existed under one name in a
    migrated database and a different name in a created one, so which names a
    database had depended on how it came to exist.

    That is worth a test rather than a one-time fix because the failure is
    silent and arrives late. Autogenerate compares the models against the
    database, so a drifted name makes the next generated migration propose
    dropping and recreating indexes on real data - a rename nobody asked for,
    inside a migration somebody would reasonably trust because a tool wrote it.

    One blind spot, stated rather than silenced: `runs` and `work_steps` refer
    to each other, and the comparison skips foreign keys on tables it cannot
    order, which is what the warning this emits is saying. Everything else is
    compared; foreign keys between those two are not.
    """

    settings = Settings(data_dir=tmp_path / "parity", dev=True)
    settings.prepare()
    configure_database(settings)
    upgrade_database(settings)
    db.engine.dispose()

    engine = create_engine(f"sqlite:///{settings.state_dir / 'local-lm.sqlite3'}")
    try:
        with engine.connect() as connection:
            difference = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()

    assert [one for one in difference if not _is_the_recorded_uniqueness_allowance(one)] == []


def _is_the_recorded_uniqueness_allowance(entry: object) -> bool:
    """SQLite enforces uniqueness one way, so these two spellings are one thing.

    `runs.work_step_id` is declared as a unique constraint on the model and was
    migrated as a unique index. Both produce the same guarantee, and alembic
    reports the pair as a difference for as long as they coexist. Matched by
    table and column rather than by operation name, so a genuinely new
    uniqueness mismatch anywhere else still fails the test above.
    """

    if not isinstance(entry, tuple) or len(entry) != 2:
        return False
    operation, obj = entry
    table = getattr(getattr(obj, "table", None), "name", None)
    if table != "runs":
        return False
    columns = [column.name for column in getattr(obj, "columns", [])]
    if columns != ["work_step_id"]:
        return False
    return (
        operation == "remove_index" and getattr(obj, "name", None) == "uq_runs_work_step_id"
    ) or (operation == "add_constraint" and isinstance(obj, UniqueConstraint))


def _fk_usable_index(connection: sqlite3.Connection, table: str, column: str) -> str | None:
    """The name of an index SQLite can use for this table's foreign-key lookup.

    Three properties, all necessary, and none of them visible from a probe query.
    An index is usable for the equality lookup that `ON DELETE SET NULL` performs
    only when it is NOT partial, `column` is its LEADING term, and that term
    carries the column's own collation.

    Checking the definition is what makes this sound. An earlier version asked
    the query planner instead, with a bound literal, and a partial index whose
    predicate happened to match that literal planned as SEARCH while the real
    foreign-key lookup still scanned. The probe answered a different question
    from the one being asked.
    """

    for _seq, name, _unique, _origin, partial in connection.execute(
        f"PRAGMA index_list('{table}')"
    ):
        if partial:
            continue
        terms = list(connection.execute(f"PRAGMA index_xinfo('{name}')"))
        if not terms:
            continue
        _seqno, _cid, term_name, _desc, collation, _key = terms[0]
        if term_name == column and (collation or "").upper() == "BINARY":
            return str(name)
    return None


def test_every_artifact_foreign_key_has_an_index_the_delete_can_use(tmp_path: Path) -> None:
    """Deleting an artifact must not scan a table to find its referrers.

    SQLite applies `ON DELETE SET NULL` by finding every row referring to the
    dying artifact. Without a usable index on the referring column that is a full
    scan, once per deleted artifact, per table, and the retention sweep deletes in
    thousands.

    "Usable" is checked against the index DEFINITION, because two shapes this
    codebase uses idiomatically are indistinguishable from a real index by any
    cheaper means:

        a PARTIAL index - `sqlite_where=...` appears on several models and one
        already sits on artifact_library_entries, which carries a foreign key to
        artifacts.id;

        an index whose leading term carries a different COLLATE than the column.

    Neither can serve the foreign-key lookup. `PRAGMA index_info` reports the bare
    column name for both, and asking the query planner is not a way out either: a
    partial index whose predicate matches the probe's bound value plans as SEARCH
    while the real delete still scans. That was measured, not assumed.

    The assertion is universal, so a foreign key added later without a usable
    index fails here rather than quietly costing a scan per deletion.
    """

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    upgrade_database(settings)

    database = next(iter(sorted((tmp_path / "data").rglob("*.sqlite3"))), None)
    assert database is not None, "the migrations produced no database to inspect"

    unusable: list[str] = []
    examined: list[str] = []
    with closing(sqlite3.connect(database)) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            referring = [
                row[3]
                for row in connection.execute(f"PRAGMA foreign_key_list('{table}')")
                if row[2] == "artifacts"
            ]
            if not referring:
                continue
            # The BINARY expectation above is only right while no referring
            # column declares its own collation. Assert that rather than assume
            # it, so a future COLLATE on one of these columns fails loudly here
            # instead of silently making every index look wrong.
            declaration = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
                (table,),
            ).fetchone()
            for column in referring:
                examined.append(f"{table}.{column}")
                assert "COLLATE" not in (declaration[0] or "").upper(), (
                    f"{table} declares a column collation, so the BINARY "
                    f"expectation in _fk_usable_index no longer holds"
                )
                if _fk_usable_index(connection, table, column) is None:
                    unusable.append(f"{table}.{column}")

    # A floor on what was examined. Without it the final assertion is satisfied
    # by a discovery loop that found nothing, which is the classic vacuous pass.
    assert "message_parts.artifact_id" in examined, (
        f"the discovery loop did not find a foreign key it should have; it "
        f"examined {sorted(examined)}, so this test is measuring nothing"
    )

    assert unusable == [], (
        f"these foreign keys to artifacts.id have no index the delete can use, "
        f"so every artifact deletion scans their tables: {sorted(unusable)}"
    )


_ARTIFACT_INDEX_REVISION = "b41e7c0a92d5"
_ARTIFACT_INDEX_PRIOR = "c9e1d4a70b82"


def _migrated_to(tmp_path: Path, revision: str) -> tuple[Settings, object, Path]:
    """Bring a fresh database to an exact revision and hand back its handles."""

    settings = Settings(data_dir=tmp_path / "data")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, revision)
    database = next(iter(sorted((tmp_path / "data").rglob("*.sqlite3"))))
    return settings, config, database


def test_the_artifact_index_guard_accepts_an_exact_replay(tmp_path: Path) -> None:
    """A partially applied migration must be able to finish on the next start.

    SQLite commits each CREATE INDEX, but a failure part-way rolls
    alembic_version back, so the schema can hold indexes the version row says
    were never created. Without a guard the replay dies on "already exists" and
    the application never starts again - which was reproduced before the guard
    existed. This pins the recovery as a repository test rather than a
    throwaway script.
    """

    _settings, config, database = _migrated_to(tmp_path, _ARTIFACT_INDEX_PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE INDEX ix_message_parts_artifact_id ON message_parts (artifact_id)"
        )
        connection.commit()

    command.upgrade(config, _ARTIFACT_INDEX_REVISION)

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version == (_ARTIFACT_INDEX_REVISION,), (
        "the replay did not advance the revision, so the next start replays again"
    )


def test_the_artifact_index_guard_refuses_a_same_name_index_of_another_shape(
    tmp_path: Path,
) -> None:
    """Advancing over an index this migration did not create is the failure mode.

    `CREATE INDEX IF NOT EXISTS` returns success against a same-name index on a
    different column and leaves the wrong one in place, so the revision advances
    without the index it claims to create. Refusing loudly is the point: the
    guard compares the exact definition, not the name.
    """

    _settings, config, database = _migrated_to(tmp_path, _ARTIFACT_INDEX_PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("CREATE INDEX ix_message_parts_artifact_id ON message_parts (id)")
        connection.commit()

    with pytest.raises(Exception, match="already exists with a different definition"):
        command.upgrade(config, "head")

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        columns = [
            row[2]
            for row in connection.execute("PRAGMA index_info('ix_message_parts_artifact_id')")
        ]
    assert version == (_ARTIFACT_INDEX_PRIOR,), (
        "the revision advanced without creating the index it claims to create"
    )
    assert columns == ["id"], "the pre-existing index was altered or removed"


def test_the_artifact_index_downgrade_preserves_a_differently_shaped_namesake(
    tmp_path: Path,
) -> None:
    """Downgrade must not delete an index of another shape that shares the name.

    The guarantee is specifically about a DIFFERENT definition. An index of the
    exact same shape is indistinguishable from one this revision created -
    SQLite records only that it came from an explicit CREATE INDEX, not which
    migration ran it - so that case is deliberately not claimed.
    """

    _settings, config, database = _migrated_to(tmp_path, "head")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DROP INDEX ix_reference_subjects_cover_artifact_id")
        connection.execute(
            "CREATE INDEX ix_reference_subjects_cover_artifact_id ON reference_subjects (id)"
        )
        connection.commit()

    command.downgrade(config, _ARTIFACT_INDEX_PRIOR)

    with closing(sqlite3.connect(database)) as connection:
        columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info('ix_reference_subjects_cover_artifact_id')"
            )
        ]
    assert columns == ["id"], (
        "downgrade deleted a same-name index of a different shape, which this "
        "revision did not create and has no authority to remove"
    )


_STAMP = "2026-09-03 12:00:00"

_INSERT_ARTIFACT = """
INSERT INTO artifacts
  (id, sha256, kind, media_type, size_bytes, relative_path, metadata_json,
   created_at, updated_at)
VALUES (?, ?, 'image', 'image/png', 1, ?, '{}', ?, ?)
"""

_INSERT_VERIFICATION = """
INSERT INTO setup_verifications
  (id, role, evidence_key, state, model_install_id, profile_id,
   input_artifact_id, created_at, updated_at)
VALUES (?, 'image', ?, ?, 'install-1', 'profile-1', ?, ?, ?)
"""


def _setup_verification_migrated_to(tmp_path: Path, name: str, revision: str) -> Path:
    settings = Settings(data_dir=tmp_path / name)
    settings.prepare()
    command.upgrade(alembic_config(settings), revision)
    return settings.state_dir / "local-lm.sqlite3"


def test_deleting_an_artifact_nulls_the_setup_verification_that_named_it(
    tmp_path: Path,
) -> None:
    """The database clears the reference, with no ORM deletion guard involved.

    A delete that reaches the database directly must leave the column null
    rather than a dangling identifier, and must leave the verification row
    itself intact.

    Rows are inserted with foreign keys off and the pragma is enabled only for
    the delete. SQLite applies the constraint at statement time, so this
    exercises exactly that contract without fabricating unrelated model install
    and profile parents whose own schemas have nothing to do with it.
    """

    database = _setup_verification_migrated_to(tmp_path, "fk-nulls", "head")
    with _application_sqlite(database) as connection:
        connection.execute(_INSERT_ARTIFACT, ("art-1", "a" * 64, "one.png", _STAMP, _STAMP))
        connection.execute(
            _INSERT_VERIFICATION, ("verify-1", "key-1", "running", "art-1", _STAMP, _STAMP)
        )
        connection.commit()

        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("DELETE FROM artifacts WHERE id = 'art-1'")
        connection.commit()

        assert connection.execute(
            "SELECT input_artifact_id FROM setup_verifications WHERE id = 'verify-1'"
        ).fetchone() == (None,), "the deleted artifact left a dangling identifier behind"
        assert connection.execute(
            "SELECT COUNT(*) FROM setup_verifications WHERE id = 'verify-1'"
        ).fetchone() == (1,), "clearing the reference must not remove the verification"


def test_the_enforcement_migration_clears_only_identifiers_that_no_longer_resolve(
    tmp_path: Path,
) -> None:
    """Dangling values are nulled; resolving ones are left exactly as they are.

    The distinction is the whole of the migration's data behaviour. A dangling
    identifier cannot be preserved - its target is already gone, and neither a
    foreign key nor an invented artifact row would restore the relationship -
    but a resolving one is a real reference and nulling it would discard what a
    verification was run against.
    """

    database = _setup_verification_migrated_to(tmp_path, "fk-migration", "b41e7c0a92d5")
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(_INSERT_ARTIFACT, ("art-live", "b" * 64, "live.png", _STAMP, _STAMP))
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-resolving", "key-resolving", "running", "art-live", _STAMP, _STAMP),
        )
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-dangling", "key-dangling", "running", "art-gone", _STAMP, _STAMP),
        )
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-empty", "key-empty", "failed", None, _STAMP, _STAMP),
        )
        connection.commit()

    settings = Settings(data_dir=tmp_path / "fk-migration")
    command.upgrade(alembic_config(settings), "head")

    with closing(sqlite3.connect(database)) as connection:
        rows = {
            row[0]: row[1:]
            for row in connection.execute(
                "SELECT id, input_artifact_id, state, evidence_key, model_install_id, role "
                "FROM setup_verifications ORDER BY id"
            )
        }

    # Adding the constraint rebuilds the table by copying it, so every column of
    # every row has to survive that copy, not only the column the constraint is
    # on.
    assert rows == {
        "verify-resolving": ("art-live", "running", "key-resolving", "install-1", "image"),
        "verify-dangling": (None, "running", "key-dangling", "install-1", "image"),
        "verify-empty": (None, "failed", "key-empty", "install-1", "image"),
    }, "the migration must clear only the identifiers whose artifact is gone"

    # PRAGMA foreign_key_check walks declared keys across stored rows, and the
    # backup verifier fails a backup on any violation it reports. Alembic runs
    # with foreign keys off and SQLite never revalidates stored rows, so this is
    # the property the migration's clearing exists to hold.
    with closing(sqlite3.connect(database)) as connection:
        violations = [
            row for row in connection.execute("PRAGMA foreign_key_check") if row[2] == "artifacts"
        ]
    # Scoped to the artifacts relationship. The fixture inserts with foreign keys
    # off and creates no model_installs or model_profiles parents, so an unscoped
    # check would report those absent parents rather than this key.
    assert violations == [], "the migrated database declares an artifact key it already violates"


_SETUP_FK_REVISION = "e4b7d1c5a960"
_SETUP_FK_PRIOR = "b41e7c0a92d5"


def test_the_setup_verification_index_guard_accepts_an_exact_replay(tmp_path: Path) -> None:
    """A replay must finish rather than dying on "already exists".

    The rebuild that adds the constraint reflects and recreates this table's
    indexes, so on a replay the index is already present by the time the guard
    runs. Accepting an exact match is what lets a partially applied revision
    complete on the next start instead of stopping the application for good.
    """

    _settings, config, database = _migrated_to(tmp_path, _SETUP_FK_PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE INDEX ix_setup_verifications_input_artifact_id "
            "ON setup_verifications (input_artifact_id)"
        )
        connection.commit()

    command.upgrade(config, _SETUP_FK_REVISION)

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version == (_SETUP_FK_REVISION,), (
        "the replay did not advance the revision, so the next start replays again"
    )


def test_the_setup_verification_index_guard_refuses_a_same_name_index_of_another_shape(
    tmp_path: Path,
) -> None:
    """Advancing over an index this revision did not create is the failure mode.

    A same-name index on another column would leave the foreign-key lookup
    unindexed while the revision advanced as though it had been made cheap.
    The guard compares the exact definition rather than the name, and refusing
    loudly is the point.
    """

    _settings, config, database = _migrated_to(tmp_path, _SETUP_FK_PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE INDEX ix_setup_verifications_input_artifact_id ON setup_verifications (state)"
        )
        connection.commit()

    with pytest.raises(Exception, match="already exists with a different definition"):
        command.upgrade(config, _SETUP_FK_REVISION)

    with closing(sqlite3.connect(database)) as connection:
        version = connection.execute("SELECT version_num FROM alembic_version").fetchone()
        columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info('ix_setup_verifications_input_artifact_id')"
            )
        ]

    assert version == (_SETUP_FK_PRIOR,), (
        "the revision advanced without the index it claims to make"
    )
    assert columns == ["state"], (
        "the guard adopted or replaced an index this revision did not create"
    )


def test_the_enforcement_migration_records_the_distribution_it_claims_to_log(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The counts must reach the log, not merely be computed.

    Once the migration has run, the difference between a dangling identifier and
    one that was never set is gone for good and cannot be recovered from the
    result, so the recorded distribution is the only lasting account of what the
    store held beforehand.

    The application configures the root logger at INFO and installs an INFO
    handler around the startup stage that runs migrations, which is how these
    records reach api.log; this covers the migration's half of that.
    """

    _settings, config, database = _migrated_to(tmp_path, _SETUP_FK_PRIOR)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(_INSERT_ARTIFACT, ("art-live", "c" * 64, "kept.png", _STAMP, _STAMP))
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-resolving", "key-r", "running", "art-live", _STAMP, _STAMP),
        )
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-dangling", "key-d", "running", "art-gone", _STAMP, _STAMP),
        )
        # Two states, because the record is per verification state: one state
        # produces a single group, which a whole-table aggregate is
        # indistinguishable from.
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-done", "key-f", "failed", "art-live", _STAMP, _STAMP),
        )
        connection.commit()

    with caplog.at_level(logging.INFO):
        command.upgrade(config, _SETUP_FK_REVISION)

    assert "setup verification input artifacts before enforcement" in caplog.text, (
        "the distribution was computed and never recorded"
    )
    # Whole records, including `resolving`, which is the only derived value in
    # the line: asserting a substring that appears in the input leaves the one
    # computed field free to be wrong.
    assert "state=running rows=2 named=2 dangling=1 resolving=1" in caplog.text
    assert "state=failed rows=1 named=1 dangling=0 resolving=1" in caplog.text
    assert "cleared 1 setup verification input artifact identifier" in caplog.text


def test_the_enforcement_migration_downgrades_back_to_an_unconstrained_column(
    tmp_path: Path,
) -> None:
    """The downgrade must remove exactly what the upgrade added, and keep the rows.

    A downgrade that leaves the constraint behind is worse than one that fails:
    the schema then disagrees with the revision the database claims, and the
    next upgrade replays onto a table that already has the foreign key.
    """

    _settings, config, database = _migrated_to(tmp_path, _SETUP_FK_REVISION)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(_INSERT_ARTIFACT, ("art-keep", "d" * 64, "keep.png", _STAMP, _STAMP))
        connection.execute(
            _INSERT_VERIFICATION,
            ("verify-keep", "key-keep", "running", "art-keep", _STAMP, _STAMP),
        )
        connection.commit()

    command.downgrade(config, _SETUP_FK_PRIOR)

    with closing(sqlite3.connect(database)) as connection:
        targets = [
            row[2] for row in connection.execute("PRAGMA foreign_key_list('setup_verifications')")
        ]
        indexes = [row[1] for row in connection.execute("PRAGMA index_list('setup_verifications')")]
        column = next(
            row
            for row in connection.execute("PRAGMA table_info('setup_verifications')")
            if row[1] == "input_artifact_id"
        )
        surviving = connection.execute(
            "SELECT id, input_artifact_id FROM setup_verifications"
        ).fetchall()

    assert "artifacts" not in targets, "the downgrade left the foreign key in place"
    assert "ix_setup_verifications_input_artifact_id" not in indexes
    assert column[2] == "VARCHAR(40)", f"the column was not narrowed back, it is {column[2]}"
    assert surviving == [("verify-keep", "art-keep")], "the downgrade lost or altered a row"

    # The three constraints this table already had must outlive both rebuilds.
    assert sorted(targets) == ["model_installs", "model_profiles", "workflow_revisions"], (
        "a rebuild dropped a foreign key that had nothing to do with this revision"
    )


def test_the_enforcement_downgrade_preserves_a_differently_shaped_namesake(
    tmp_path: Path,
) -> None:
    """The downgrade drops only the index this revision made.

    An index carrying the same name but a different shape belongs to something
    else, and this revision has no authority to remove it. The guard is
    symmetric with the upgrade's, which refuses such an index rather than
    adopting it.
    """

    _settings, config, database = _migrated_to(tmp_path, _SETUP_FK_REVISION)
    with closing(sqlite3.connect(database)) as connection:
        connection.execute("DROP INDEX ix_setup_verifications_input_artifact_id")
        connection.execute(
            "CREATE INDEX ix_setup_verifications_input_artifact_id ON setup_verifications (state)"
        )
        connection.commit()

    command.downgrade(config, _SETUP_FK_PRIOR)

    with closing(sqlite3.connect(database)) as connection:
        columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info('ix_setup_verifications_input_artifact_id')"
            )
        ]
        targets = [
            row[2] for row in connection.execute("PRAGMA foreign_key_list('setup_verifications')")
        ]

    assert columns == ["state"], (
        "the downgrade dropped a same-name index of a different shape, which this "
        "revision did not create and has no authority to remove"
    )
    assert "artifacts" not in targets, "the downgrade left the foreign key in place"


def test_artifact_deletion_authority_migration_round_trips(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "deletion-authority")
    settings.prepare()
    config = alembic_config(settings)
    database = settings.state_dir / "local-lm.sqlite3"
    trigger_query = "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' ORDER BY name"
    command.upgrade(config, "e4b7d1c5a960")
    with sqlite3.connect(database) as connection:
        before = dict(connection.execute(trigger_query).fetchall())
    original = before["artifact_json_reference_delete_guard"]
    assert "artifact_deletion_authorized" not in original

    command.upgrade(config, "f5c2a8d91e40")
    with sqlite3.connect(database) as connection:
        after = dict(connection.execute(trigger_query).fetchall())
    assert set(after) == set(before)
    assert "artifact_deletion_authorized" in after["artifact_json_reference_delete_guard"]
    assert {k: v for k, v in after.items() if k != "artifact_json_reference_delete_guard"} == {
        k: v for k, v in before.items() if k != "artifact_json_reference_delete_guard"
    }

    command.downgrade(config, "e4b7d1c5a960")
    with sqlite3.connect(database) as connection:
        assert dict(connection.execute(trigger_query).fetchall()) == before
    command.upgrade(config, "f5c2a8d91e40")
    with sqlite3.connect(database) as connection:
        assert dict(connection.execute(trigger_query).fetchall()) == after
