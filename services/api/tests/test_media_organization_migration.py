"""Organization upgrades preserve records and refuse inconsistent legacy names."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

PREVIOUS = "7d39a4c81e20"
REVISION = "9c4e7a2d1830"
ALBUM = "collection_" + "a" * 32
TAG = "mediatag_" + "b" * 32


def _legacy(tmp_path: Path, *, slug: str = "study-label") -> tuple[Settings, Path]:
    settings = Settings(data_dir=tmp_path / "organization-upgrade", dev=True)
    settings.prepare()
    command.upgrade(alembic_config(settings), PREVIOUS)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT INTO media_collections (id, kind, name, description, version, created_at, "
            "updated_at) VALUES (?, 'manual', 'Studies', 'Kept album', 1, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (ALBUM,),
        )
        connection.execute(
            "INSERT INTO media_tags (id, slug, label, color, version, created_at, updated_at) "
            "VALUES (?, ?, 'Study   label', NULL, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (TAG, slug),
        )
    return settings, database


def test_upgrade_preserves_existing_records_and_checks_normalized_tag_renames(
    tmp_path: Path,
) -> None:
    settings, database = _legacy(tmp_path)
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE media_collections SET name='Changed', version=version+1 WHERE id=?",
                (ALBUM,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE media_tags SET label='Landscape', slug='landscape', "
                "version=version+1 WHERE id=?",
                (TAG,),
            )
    command.upgrade(alembic_config(settings), REVISION)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT id, name, version FROM media_collections").fetchall() == [
            (ALBUM, "Studies", 1)
        ]
        assert connection.execute("SELECT id, slug, label, version FROM media_tags").fetchall() == [
            (TAG, "study-label", "Study   label", 1)
        ]
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(media_organization_impacts)")
        }
        assert {"recovery_json", "operation_key", "response_json"} <= columns
        connection.execute(
            "UPDATE media_tags SET label='Landscape   study', slug='landscape-study', "
            "version=version+1 WHERE id=?",
            (TAG,),
        )
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="normalized name"):
            connection.execute(
                "UPDATE media_tags SET slug='wrong-name', version=version+1 WHERE id=?", (TAG,)
            )
        assert connection.execute("SELECT slug, label, version FROM media_tags").fetchone() == (
            "landscape-study",
            "Landscape   study",
            2,
        )


def test_new_tag_inserts_enforce_normalized_names_after_upgrade(tmp_path: Path) -> None:
    settings, database = _legacy(tmp_path)
    command.upgrade(alembic_config(settings), REVISION)
    statement = (
        "INSERT INTO media_tags (id, slug, label, color, version, created_at, updated_at) "
        "VALUES (?, ?, 'Landscape   study', NULL, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
    )
    with sqlite3.connect(database) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="normalized name"):
            connection.execute(statement, ("mediatag_" + "c" * 32, "wrong-name"))
        connection.execute(statement, ("mediatag_" + "c" * 32, "landscape-study"))
        assert connection.execute("SELECT slug FROM media_tags ORDER BY slug").fetchall() == [
            ("landscape-study",),
            ("study-label",),
        ]


def test_inconsistent_legacy_tags_refuse_upgrade_before_any_schema_change(
    tmp_path: Path,
) -> None:
    settings, database = _legacy(tmp_path, slug="unrelated-name")
    with sqlite3.connect(database) as connection:
        before = connection.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
    with pytest.raises(RuntimeError, match="Media tag names must be consistent"):
        command.upgrade(alembic_config(settings), REVISION)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
            ).fetchall()
            == before
        )
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            PREVIOUS,
        )
        assert connection.execute("SELECT slug, label, version FROM media_tags").fetchone() == (
            "unrelated-name",
            "Study   label",
            1,
        )


def test_downgrade_restores_metadata_immutability_and_can_upgrade_again(
    tmp_path: Path,
) -> None:
    settings, database = _legacy(tmp_path)
    command.upgrade(alembic_config(settings), REVISION)
    command.downgrade(alembic_config(settings), PREVIOUS)
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='media_organization_impacts'"
            ).fetchall()
            == []
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE media_collections SET name='Changed', version=version+1 WHERE id=?",
                (ALBUM,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE media_tags SET label='Landscape', slug='landscape', "
                "version=version+1 WHERE id=?",
                (TAG,),
            )
        connection.execute("UPDATE media_tags SET version=version+1 WHERE id=?", (TAG,))
        connection.execute("UPDATE media_collections SET version=version+1 WHERE id=?", (ALBUM,))
    command.upgrade(alembic_config(settings), REVISION)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT name, version FROM media_collections").fetchone() == (
            "Studies",
            2,
        )
        assert connection.execute("SELECT label, version FROM media_tags").fetchone() == (
            "Study   label",
            2,
        )
