from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from alembic import command
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.reference_review_schema import (
    CREATE_REFERENCE_REVIEW_TRIGGER_SQL,
    DROP_REFERENCE_REVIEW_TRIGGER_SQL,
)


def test_reference_review_migration_preserves_unchecked_assets_and_installs_guards(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "reference-review-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f9b7a1c42d60")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-13 12:00:00"
    digest = "a" * 64
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES (?, ?, 'input', 'image/png', 1, ?, 'legacy.png', '{}', 0, ?, ?)
            """,
            (f"sha256:{digest}", digest, f"aa/aa/{digest}", timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_subjects
              (id, name, mention_slug, kind, description, aliases_json, tags_json,
               cover_artifact_id, favorite, archived, created_at, updated_at)
            VALUES
              ('refsubject_legacy', 'Legacy', 'legacy', 'person', NULL, '[]', '[]',
               NULL, 0, 0, ?, ?)
            """,
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_assets
              (id, reference_subject_id, artifact_id, caption, purpose, view_label,
               sort_order, validation_state, validation_reasons_json, width, height,
               created_at, updated_at)
            VALUES
              ('refasset_legacy', 'refsubject_legacy', ?, NULL, 'identity', NULL,
               0, 'unchecked', '[]', NULL, NULL, ?, ?)
            """,
            (f"sha256:{digest}", timestamp, timestamp),
        )

    command.upgrade(config, "head")
    expected_triggers = {statement.split()[2] for statement in CREATE_REFERENCE_REVIEW_TRIGGER_SQL}
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT validation_state, validation_reasons_json, width, height, review_version "
            "FROM reference_assets WHERE id = 'refasset_legacy'"
        ).fetchone() == ("unchecked", "[]", None, None, 1)
        assert connection.execute(
            "SELECT count(*) FROM reference_asset_review_events"
        ).fetchone() == (0,)
        migrated_triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).fetchall()
        }
        assert expected_triggers <= migrated_triggers

    command.downgrade(config, "f9b7a1c42d60")
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reference_assets)").fetchall()
        }
        assert "review_version" not in columns
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name = 'reference_asset_review_events'"
            ).fetchone()
            is None
        )


def test_a_pre_settled_row_without_its_event_refuses_the_upgrade(
    tmp_path: Path,
) -> None:
    """Authority without a human review event must stop the migration cold.

    validation_state and validation_reasons_json are NOT NULL at the parent
    revision (verified by PRAGMA, so the preflight's != cannot be blinded by
    NULL - the R1159 class does not apply here).
    """
    settings = Settings(data_dir=tmp_path / "reference-review-refusal")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f9b7a1c42d60")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-13 12:00:00"
    digest = "b" * 64
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES (?, ?, 'input', 'image/png', 1, ?, 'settled.png', '{}', 0, ?, ?)
            """,
            (f"sha256:{digest}", digest, f"bb/bb/{digest}", timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_subjects
              (id, name, mention_slug, kind, description, aliases_json, tags_json,
               cover_artifact_id, favorite, archived, created_at, updated_at)
            VALUES
              ('refsubject_settled', 'Settled', 'settled', 'person', NULL, '[]',
               '[]', NULL, 0, 0, ?, ?)
            """,
            (timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_assets
              (id, reference_subject_id, artifact_id, caption, purpose, view_label,
               sort_order, validation_state, validation_reasons_json, width, height,
               created_at, updated_at)
            VALUES
              ('refasset_settled', 'refsubject_settled', ?, NULL, 'identity', NULL,
               0, 'usable', '["looks fine"]', 512, 512, ?, ?)
            """,
            (f"sha256:{digest}", timestamp, timestamp),
        )

    with pytest.raises(RuntimeError, match="without a human review event"):
        command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f9b7a1c42d60",
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reference_assets)").fetchall()
        }
        assert "review_version" not in columns


def test_a_tampered_reference_trigger_refuses_the_bootstrap(
    tmp_path: Path,
) -> None:
    """A same-name reference trigger with a weakened body must fail closed.

    This locks the reference wiring specifically: the historical delta
    registered these six through raw DDL, which recreated blindly; routing
    them through _install_sqlite_trigger is the whole point of this child
    (grok/R1179 reject, codex/R1684 acceptance).
    """
    settings = Settings(data_dir=tmp_path / "reference-review-tamper", dev=True)
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER reference_asset_review_insert_guard")
        connection.execute(
            """
            CREATE TRIGGER reference_asset_review_insert_guard
            BEFORE INSERT ON reference_assets
            BEGIN
              SELECT 1;
            END
            """
        )

    engine = create_engine(f"sqlite:///{database}")
    try:
        with pytest.raises(RuntimeError, match="reference_asset_review_insert_guard"):
            Base.metadata.create_all(engine)
    finally:
        engine.dispose()


def test_a9_write_fence_blocks_a_concurrent_settled_writer(tmp_path: Path) -> None:
    """A writer racing the migration must not slip a settled row in after the
    preflight read. The fence takes the write lock BEFORE the preflight, so the
    racer blocks until commit - and then the new insert guard aborts it."""
    settings = Settings(data_dir=tmp_path / "reference-review-fence-race")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "f9b7a1c42d60")
    database = settings.state_dir / "local-lm.sqlite3"
    timestamp = "2026-08-13 12:00:00"
    digest = "c" * 64
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            INSERT INTO artifacts
              (id, sha256, kind, media_type, size_bytes, relative_path,
               original_name, metadata_json, favorite, created_at, updated_at)
            VALUES (?, ?, 'input', 'image/png', 1, ?, 'race.png', '{}', 0, ?, ?)
            """,
            (f"sha256:{digest}", digest, f"cc/cc/{digest}", timestamp, timestamp),
        )
        connection.execute(
            """
            INSERT INTO reference_subjects
              (id, name, mention_slug, kind, description, aliases_json, tags_json,
               cover_artifact_id, favorite, archived, created_at, updated_at)
            VALUES
              ('refsubject_race', 'Race', 'race', 'person', NULL, '[]', '[]',
               NULL, 0, 0, ?, ?)
            """,
            (timestamp, timestamp),
        )

    preflight_seen = Event()
    writer_started = Event()
    writer_done = Event()
    writer_was_blocked: list[bool] = []

    def trace(statement: str) -> None:
        if preflight_seen.is_set():
            return
        if "validation_state != 'unchecked'" not in statement:
            return
        if "NEW." in statement:
            return
        preflight_seen.set()
        if writer_started.wait(5):
            writer_was_blocked.append(not writer_done.wait(0.2))

    def register_trace(dbapi_connection: object, _record: object) -> None:
        dbapi_connection.set_trace_callback(trace)  # type: ignore[attr-defined]

    def upgrade() -> None:
        command.upgrade(config, "head")

    def write_settled_row() -> str:
        assert preflight_seen.wait(5)
        writer_started.set()
        try:
            with sqlite3.connect(database, timeout=5) as connection:
                connection.execute(
                    """
                    INSERT INTO reference_assets
                      (id, reference_subject_id, artifact_id, caption, purpose,
                       view_label, sort_order, validation_state,
                       validation_reasons_json, width, height, created_at,
                       updated_at)
                    VALUES
                      ('refasset_race', 'refsubject_race', ?, NULL, 'identity',
                       NULL, 0, 'usable', '["raced in"]', 512, 512, ?, ?)
                    """,
                    (f"sha256:{digest}", timestamp, timestamp),
                )
        except sqlite3.IntegrityError as error:
            return str(error)
        finally:
            writer_done.set()
        return "settled row committed"

    event.listen(Engine, "connect", register_trace)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            upgrade_result = executor.submit(upgrade)
            writer_result = executor.submit(write_settled_row)
            upgrade_result.result(timeout=15)
            outcome = writer_result.result(timeout=15)
    finally:
        event.remove(Engine, "connect", register_trace)

    assert writer_was_blocked == [True]
    assert outcome == "reference asset must begin unchecked"
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT count(*) FROM reference_assets WHERE id = 'refasset_race'"
        ).fetchone() == (0,)
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c8e2f4a71d90",
        )


# Pinned independently of the production tuples: an install test that derives
# its expectations from the same mutable tuple shrinks with the bug it is
# meant to catch (codex/R1690 item 3, R1691).
EXPECTED_TRIGGER_NAMES = frozenset(
    {
        "reference_asset_review_insert_guard",
        "reference_asset_review_update_guard",
        "reference_asset_review_identity_guard",
        "reference_asset_review_event_insert_guard",
        "reference_asset_review_event_update_guard",
        "reference_asset_review_event_delete_guard",
    }
)


def _migrated_database(tmp_path: Path, name: str) -> Path:
    settings = Settings(data_dir=tmp_path / name)
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    return settings.state_dir / "local-lm.sqlite3"


def _seed_unchecked(connection: sqlite3.Connection, digest: str) -> None:
    timestamp = "2026-08-13 12:00:00"
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute(
        """
        INSERT INTO artifacts
          (id, sha256, kind, media_type, size_bytes, relative_path,
           original_name, metadata_json, favorite, created_at, updated_at)
        VALUES (?, ?, 'input', 'image/png', 1, ?, 'seed.png', '{}', 0, ?, ?)
        """,
        (f"sha256:{digest}", digest, f"dd/dd/{digest}", timestamp, timestamp),
    )
    connection.execute(
        """
        INSERT INTO reference_subjects
          (id, name, mention_slug, kind, description, aliases_json, tags_json,
           cover_artifact_id, favorite, archived, created_at, updated_at)
        VALUES ('refsubject_t', 'T', 't', 'person', NULL, '[]', '[]', NULL,
                0, 0, ?, ?)
        """,
        (timestamp, timestamp),
    )
    connection.execute(
        """
        INSERT INTO reference_assets
          (id, reference_subject_id, artifact_id, caption, purpose, view_label,
           sort_order, validation_state, validation_reasons_json, width, height,
           created_at, updated_at)
        VALUES ('refasset_t', 'refsubject_t', ?, NULL, 'identity', NULL, 0,
                'unchecked', '[]', NULL, NULL, ?, ?)
        """,
        (f"sha256:{digest}", timestamp, timestamp),
    )


def _settle(connection: sqlite3.Connection, **overrides: object) -> None:
    values: dict[str, object] = {
        "validation_state": "usable",
        "validation_reasons_json": '["sharp"]',
        "width": 512,
        "height": 512,
        "review_version": 2,
    }
    values.update(overrides)
    connection.execute(
        """
        UPDATE reference_assets
        SET validation_state = :validation_state,
            validation_reasons_json = :validation_reasons_json,
            width = :width, height = :height, review_version = :review_version
        WHERE id = 'refasset_t'
        """,
        values,
    )


def _insert_event(
    connection: sqlite3.Connection,
    digest: str,
    or_replace: bool = False,
    **overrides: object,
) -> None:
    decision = "e" * 64
    values: dict[str, object] = {
        "id": f"refreview:sha256:{decision}",
        "reference_asset_id": "refasset_t",
        "artifact_id": f"sha256:{digest}",
        "artifact_sha256": digest,
        "reviewer_kind": "local-human",
        "decision": "usable",
        "expected_state": "unchecked",
        "expected_version": 1,
        "result_version": 2,
        "width": 512,
        "height": 512,
        "reasons_json": '["sharp"]',
        "decision_sha256": decision,
        "reviewed_at": "2026-08-13 12:00:00",
    }
    values.update(overrides)
    connection.execute(
        ("INSERT OR REPLACE INTO" if or_replace else "INSERT INTO")
        + """ reference_asset_review_events
          (id, reference_asset_id, artifact_id, artifact_sha256, reviewer_kind,
           decision, expected_state, expected_version, result_version, width,
           height, reasons_json, decision_sha256, reviewed_at)
        VALUES (:id, :reference_asset_id, :artifact_id, :artifact_sha256,
                :reviewer_kind, :decision, :expected_state, :expected_version,
                :result_version, :width, :height, :reasons_json,
                :decision_sha256, :reviewed_at)
        """,
        values,
    )


def test_the_six_trigger_names_are_pinned_in_both_tuples() -> None:
    created = {statement.split()[2] for statement in CREATE_REFERENCE_REVIEW_TRIGGER_SQL}
    dropped = {statement.split()[2] for statement in DROP_REFERENCE_REVIEW_TRIGGER_SQL}
    assert created == EXPECTED_TRIGGER_NAMES
    assert dropped == EXPECTED_TRIGGER_NAMES


def test_each_guard_is_proved_by_the_operation_it_protects(tmp_path: Path) -> None:
    """One behavioral assertion per trigger, so a paired CREATE+DROP removal
    of any guard fails HERE, not merely at teardown (codex/R1690 item 3,
    R1691: five of six could previously vanish with the suite still green)."""
    digest = "d" * 64
    database = _migrated_database(tmp_path, "behavioral")
    with sqlite3.connect(database) as connection:
        _seed_unchecked(connection, digest)

        with pytest.raises(sqlite3.IntegrityError, match="must begin unchecked"):
            connection.execute(
                """
                INSERT INTO reference_assets
                  (id, reference_subject_id, artifact_id, caption, purpose,
                   view_label, sort_order, validation_state,
                   validation_reasons_json, width, height, created_at,
                   updated_at)
                VALUES ('refasset_born', 'refsubject_t', ?, NULL, 'identity',
                        NULL, 0, 'usable', '["x"]', 1, 1, ?, ?)
                """,
                (
                    f"sha256:{digest}",
                    "2026-08-13 12:00:00",
                    "2026-08-13 12:00:00",
                ),
            )

        with pytest.raises(sqlite3.IntegrityError, match="version is stale"):
            _settle(connection, review_version=1)
        with pytest.raises(sqlite3.IntegrityError, match="dimensions are invalid"):
            _settle(connection, width=1.5, height=2.5)

        _settle(connection)
        _insert_event(connection, digest)

        with pytest.raises(sqlite3.IntegrityError, match="already settled"):
            _settle(connection, review_version=3)

        with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
            connection.execute(
                "UPDATE reference_assets SET artifact_id = 'sha256:' || ? WHERE id = 'refasset_t'",
                ("f" * 64,),
            )

        with pytest.raises(sqlite3.IntegrityError, match="review event is invalid"):
            _insert_event(
                connection,
                digest,
                id="refreview:sha256:" + "a" * 64,
                decision_sha256="a" * 64,
                width=1.5,
                height=2.5,
                expected_version=2,
                result_version=3,
            )
        with pytest.raises(sqlite3.IntegrityError, match="does not match settled"):
            _insert_event(
                connection,
                digest,
                id="refreview:sha256:" + "b" * 64,
                decision_sha256="b" * 64,
                width=511,
                expected_version=2,
                result_version=3,
            )

        with pytest.raises(sqlite3.IntegrityError, match="events are immutable"):
            connection.execute("UPDATE reference_asset_review_events SET decision = 'weak'")
        with pytest.raises(sqlite3.IntegrityError, match="events are immutable"):
            connection.execute("DELETE FROM reference_asset_review_events")


def test_downgrade_refuses_while_review_authority_exists(tmp_path: Path) -> None:
    """The old downgrade silently destroyed review history and left a database
    that could never re-upgrade (codex/R1690 item 1)."""
    digest = "d" * 64
    settings = Settings(data_dir=tmp_path / "downgrade-refusal")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        _seed_unchecked(connection, digest)
        _settle(connection)
        _insert_event(connection, digest)

    with pytest.raises(RuntimeError, match="destroy authoritative review history"):
        command.downgrade(config, "f9b7a1c42d60")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "a9c4e7d21b60",
        )
        assert connection.execute(
            "SELECT validation_state, review_version, width, height "
            "FROM reference_assets WHERE id = 'refasset_t'"
        ).fetchone() == ("usable", 2, 512, 512)
        assert connection.execute(
            "SELECT count(*) FROM reference_asset_review_events"
        ).fetchone() == (1,)


def test_fresh_and_migrated_reference_tables_have_identical_pragmas(
    tmp_path: Path,
) -> None:
    """Column-level parity including server defaults - the trigger parity test
    compares only triggers (codex/R1690 item 2)."""
    migrated = _migrated_database(tmp_path, "pragma-migrated")
    fresh = tmp_path / "pragma-fresh.sqlite3"
    engine = create_engine(f"sqlite:///{fresh}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    def by_name(rows: list[tuple]) -> dict[str, tuple]:
        # Column ORDER legitimately differs (batch add_column appends; fresh
        # metadata declares inline); type, nullability, default, and pk must
        # not. The default mismatch was the R1690 item-2 defect.
        return {row[1]: row[2:] for row in rows}

    for table in ("reference_assets", "reference_asset_review_events"):
        with sqlite3.connect(migrated) as a, sqlite3.connect(fresh) as b:
            left = a.execute(f"PRAGMA table_info({table})").fetchall()
            right = b.execute(f"PRAGMA table_info({table})").fetchall()
        assert by_name(left) == by_name(right), table

    for label, database in (("migrated", migrated), ("fresh", fresh)):
        with sqlite3.connect(database) as connection:
            digest = "c" * 64 if label == "migrated" else "b" * 64
            _seed_unchecked(connection, digest)
            assert connection.execute(
                "SELECT review_version FROM reference_assets WHERE id = 'refasset_t'"
            ).fetchone() == (1,), label


def test_downgrade_fence_blocks_a_concurrent_settle_writer(tmp_path: Path) -> None:
    """The downgrade preflight is fenced like the upgrade: a racer cannot slip
    review authority in after the emptiness read (codex/R1690 item 1)."""
    digest = "d" * 64
    settings = Settings(data_dir=tmp_path / "downgrade-race")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        _seed_unchecked(connection, digest)

    preflight_seen = Event()
    writer_started = Event()
    writer_done = Event()
    writer_was_blocked: list[bool] = []

    def trace(statement: str) -> None:
        if preflight_seen.is_set():
            return
        if "reference_asset_review_events LIMIT 1" not in statement:
            return
        preflight_seen.set()
        if writer_started.wait(5):
            writer_was_blocked.append(not writer_done.wait(0.2))

    def register_trace(dbapi_connection: object, _record: object) -> None:
        dbapi_connection.set_trace_callback(trace)  # type: ignore[attr-defined]

    def downgrade() -> None:
        command.downgrade(config, "f9b7a1c42d60")

    def write_racing_settle() -> str:
        assert preflight_seen.wait(5)
        writer_started.set()
        try:
            with sqlite3.connect(database, timeout=5) as connection:
                _settle(connection)
        except sqlite3.OperationalError as error:
            return str(error)
        finally:
            writer_done.set()
        return "settle committed"

    event.listen(Engine, "connect", register_trace)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            downgrade_result = executor.submit(downgrade)
            writer_result = executor.submit(write_racing_settle)
            downgrade_result.result(timeout=15)
            outcome = writer_result.result(timeout=15)
    finally:
        event.remove(Engine, "connect", register_trace)

    assert writer_was_blocked == [True]
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f9b7a1c42d60",
        )
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(reference_assets)").fetchall()
        }
        assert "review_version" not in columns
    assert outcome != "settle committed" or "no such column" in outcome


def test_insert_or_replace_cannot_rewrite_a_review_event(tmp_path: Path) -> None:
    """OR REPLACE conflict resolution deletes the losing row WITHOUT firing
    the DELETE guard while PRAGMA recursive_triggers is off - SQLite's
    default, asserted here rather than assumed - so immutability must not
    hang on that trigger alone. The insert guard refuses while ANY
    conflicting row exists, closing the rewrite at the schema boundary where
    a connection-local PRAGMA cannot help (codex/R1696). Both conflict forms
    are exercised on both construction paths, and the original event must
    survive byte for byte under its original rowid."""
    digest = "d" * 64
    for label in ("migrated", "fresh"):
        if label == "migrated":
            database = _migrated_database(tmp_path, "or-replace-migrated")
        else:
            database = tmp_path / "or-replace-fresh.sqlite3"
            engine = create_engine(f"sqlite:///{database}")
            try:
                Base.metadata.create_all(engine)
            finally:
                engine.dispose()
        with sqlite3.connect(database) as connection:
            assert connection.execute("PRAGMA recursive_triggers").fetchone() == (0,), label
            _seed_unchecked(connection, digest)
            _settle(connection)
            _insert_event(connection, digest)
            original = connection.execute(
                "SELECT rowid, * FROM reference_asset_review_events"
            ).fetchall()

            with pytest.raises(sqlite3.IntegrityError, match="already exists"):
                _insert_event(
                    connection,
                    digest,
                    or_replace=True,
                    id="refreview:sha256:" + "f" * 64,
                    decision_sha256="f" * 64,
                    reviewed_at="2099-01-01 00:00:00",
                )
            with pytest.raises(sqlite3.IntegrityError, match="already exists"):
                _insert_event(
                    connection,
                    digest,
                    or_replace=True,
                    reviewed_at="2100-01-01 00:00:00",
                )

            surviving = connection.execute(
                "SELECT rowid, * FROM reference_asset_review_events"
            ).fetchall()
            assert surviving == original, label


def test_downgrade_refuses_a_non_pristine_asset_even_without_events(
    tmp_path: Path,
) -> None:
    """The downgrade preflight previously trusted events, state, and version
    alone, so a weakened same-name update trigger could leave an
    unchecked-but-dirty row that downgraded successfully - and the immediate
    re-upgrade then refused, stranding the database at the parent revision
    (codex/R1697, completing R1690 item 1). The preflight must be symmetric
    with the upgrade preflight: reasons validity and emptiness and NULL
    dimensions as well as state and version. The pristine control proves
    downgrade and immediate re-upgrade both still succeed."""
    digest = "d" * 64
    settings = Settings(data_dir=tmp_path / "downgrade-non-pristine")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        _seed_unchecked(connection, digest)
        connection.execute("DROP TRIGGER reference_asset_review_update_guard")
        connection.execute(
            """
            CREATE TRIGGER reference_asset_review_update_guard
            BEFORE UPDATE ON reference_assets
            BEGIN
              SELECT 1;
            END
            """
        )
        _settle(
            connection,
            validation_state="unchecked",
            validation_reasons_json='["never reviewed"]',
            width=10,
            height=11,
            review_version=1,
        )

    with pytest.raises(RuntimeError, match="destroy authoritative review history"):
        command.downgrade(config, "f9b7a1c42d60")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "a9c4e7d21b60",
        )
        assert connection.execute(
            "SELECT validation_state, validation_reasons_json, width, height, review_version "
            "FROM reference_assets WHERE id = 'refasset_t'"
        ).fetchone() == ("unchecked", '["never reviewed"]', 10, 11, 1)

    pristine = Settings(data_dir=tmp_path / "downgrade-pristine")
    pristine.prepare()
    pristine_config = alembic_config(pristine)
    command.upgrade(pristine_config, "head")
    pristine_database = pristine.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(pristine_database) as connection:
        _seed_unchecked(connection, digest)
    command.downgrade(pristine_config, "f9b7a1c42d60")
    command.upgrade(pristine_config, "head")
    with sqlite3.connect(pristine_database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "c8e2f4a71d90",
        )
        assert connection.execute(
            "SELECT validation_state, validation_reasons_json, width, height, review_version "
            "FROM reference_assets WHERE id = 'refasset_t'"
        ).fetchone() == ("unchecked", "[]", None, None, 1)
