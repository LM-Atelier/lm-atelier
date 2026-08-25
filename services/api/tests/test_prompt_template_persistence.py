from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from local_lm import models  # noqa: F401 - registers metadata
from local_lm.config import Settings
from local_lm.database_migrations import alembic_config
from local_lm.db import Base
from local_lm.models import (
    PromptTemplateDefinition,
    PromptTemplateRevision,
)
from local_lm.prompt_templates import (
    parse_prompt_template_contract,
    prompt_template_contract_payload,
    prompt_template_contract_sha256,
)

_LEGACY_NULL_UNSAFE_REVISION_INSERT_TRIGGER = """
CREATE TRIGGER prompt_template_revision_insert_guard
BEFORE INSERT ON prompt_template_revisions
BEGIN
  SELECT CASE WHEN NEW.version != COALESCE((
    SELECT max(existing.version) + 1
    FROM prompt_template_revisions AS existing
    WHERE existing.prompt_template_id = NEW.prompt_template_id
  ), 1)
    THEN RAISE(ABORT, 'prompt template revision version is not append-only') END;
  SELECT CASE WHEN NOT json_valid(NEW.contract_json)
                        OR json_type(NEW.contract_json) != 'object'
                        OR json_extract(NEW.contract_json, '$.schema_version')
                           != NEW.schema_version
    THEN RAISE(ABORT, 'prompt template revision contract is invalid') END;
END
"""


def _contract_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "operation": "text_to_image",
        "body": "A {{subject}} in {{setting}}.",
        "slots": [
            {"name": "subject", "mode": "input", "variation_scope": "item"},
            {
                "name": "setting",
                "mode": "choice",
                "variation_scope": "batch",
                "choices": ["studio", "forest"],
            },
        ],
        "resource_policy": {"mode": "inherited"},
    }


def _canonical_contract() -> tuple[dict[str, object], str]:
    contract = parse_prompt_template_contract(_contract_payload())
    return (
        prompt_template_contract_payload(contract),
        prompt_template_contract_sha256(contract),
    )


def _insert_definition(
    connection: sqlite3.Connection,
    *,
    definition_id: str = "ptdef_test",
    name: str = "Portrait variations",
) -> None:
    connection.execute(
        """
        INSERT INTO prompt_template_definitions
          (id, name, description, archived, current_revision_id,
           created_at, updated_at)
        VALUES (?, ?, 'Synthetic description', 0, NULL,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (definition_id, name),
    )


def _insert_revision(
    connection: sqlite3.Connection,
    *,
    revision_id: str,
    definition_id: str = "ptdef_test",
    version: int,
    payload: dict[str, object] | None = None,
    digest: str | None = None,
) -> None:
    canonical, canonical_digest = _canonical_contract()
    connection.execute(
        """
        INSERT INTO prompt_template_revisions
          (id, prompt_template_id, version, schema_version,
           contract_json, contract_sha256, created_at, updated_at)
        VALUES (?, ?, ?, 1, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """,
        (
            revision_id,
            definition_id,
            version,
            json.dumps(
                payload if payload is not None else canonical,
                sort_keys=True,
                separators=(",", ":"),
            ),
            digest or canonical_digest,
        ),
    )


def test_migration_upgrade_downgrade_and_trigger_inventory(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-template-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "head")
    database = settings.state_dir / "local-lm.sqlite3"

    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "prompt_template_definitions",
            "prompt_template_revisions",
            "prompt_template_import_winners",
        } <= tables
        triggers = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'trigger' AND name LIKE 'prompt_template_%'"
            )
        }
        assert triggers == {
            "prompt_template_definition_insert_guard",
            "prompt_template_definition_update_guard",
            "prompt_template_definition_delete_guard",
            "prompt_template_revision_insert_guard",
            "prompt_template_revision_update_guard",
            "prompt_template_revision_delete_guard",
            "prompt_template_import_winner_insert_guard",
            "prompt_template_import_winner_update_guard",
            "prompt_template_import_winner_delete_guard",
        }
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f2a7c9d41e63",
        )

    command.downgrade(config, "a9c4e7d21b60")
    with sqlite3.connect(database) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE 'prompt_template_%'"
            ).fetchall()
            == []
        )

    command.upgrade(config, "head")
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "f2a7c9d41e63",
        )


def test_revisions_are_append_only_and_definitions_archive_instead_of_delete(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-template-immutability")
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    database = settings.state_dir / "local-lm.sqlite3"

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_definition(connection)
        _insert_revision(connection, revision_id="ptrev_1", version=1)
        connection.execute(
            "UPDATE prompt_template_definitions "
            "SET current_revision_id = 'ptrev_1' WHERE id = 'ptdef_test'"
        )
        # Restoring old content is a new version with the same canonical digest.
        _insert_revision(connection, revision_id="ptrev_2", version=2)
        connection.execute(
            "UPDATE prompt_template_definitions "
            "SET current_revision_id = 'ptrev_2' WHERE id = 'ptdef_test'"
        )

        with pytest.raises(sqlite3.IntegrityError, match="revision version is not append-only"):
            _insert_revision(connection, revision_id="ptrev_4", version=4)
        with pytest.raises(sqlite3.IntegrityError, match="revisions are immutable"):
            connection.execute(
                "UPDATE prompt_template_revisions SET version = 3 WHERE id = 'ptrev_2'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="revisions are immutable"):
            connection.execute("DELETE FROM prompt_template_revisions WHERE id = 'ptrev_1'")
        with pytest.raises(sqlite3.IntegrityError, match="definitions are archived, not deleted"):
            connection.execute("DELETE FROM prompt_template_definitions WHERE id = 'ptdef_test'")
        with pytest.raises(sqlite3.IntegrityError, match="current revision cannot be cleared"):
            connection.execute(
                "UPDATE prompt_template_definitions SET current_revision_id = NULL "
                "WHERE id = 'ptdef_test'"
            )
        with pytest.raises(sqlite3.IntegrityError, match="current revision is invalid"):
            connection.execute(
                "UPDATE prompt_template_definitions "
                "SET current_revision_id = 'ptrev_1' WHERE id = 'ptdef_test'"
            )

        connection.execute(
            "UPDATE prompt_template_definitions SET archived = 1 WHERE id = 'ptdef_test'"
        )
        assert connection.execute(
            "SELECT archived, current_revision_id FROM prompt_template_definitions"
        ).fetchone() == (1, "ptrev_2")
        assert connection.execute(
            "SELECT version, contract_sha256 FROM prompt_template_revisions ORDER BY version"
        ).fetchall() == [
            (1, _canonical_contract()[1]),
            (2, _canonical_contract()[1]),
        ]


def test_current_revision_must_be_the_latest_revision_of_that_definition(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-template-current-owner")
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    database = settings.state_dir / "local-lm.sqlite3"

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_definition(connection)
        _insert_definition(
            connection,
            definition_id="ptdef_other",
            name="Other template",
        )
        _insert_revision(connection, revision_id="ptrev_1", version=1)
        _insert_revision(
            connection,
            revision_id="ptrev_other_1",
            definition_id="ptdef_other",
            version=1,
        )
        with pytest.raises(sqlite3.IntegrityError, match="current revision is invalid"):
            connection.execute(
                "UPDATE prompt_template_definitions "
                "SET current_revision_id = 'ptrev_other_1' "
                "WHERE id = 'ptdef_test'"
            )
        connection.execute(
            "UPDATE prompt_template_definitions "
            "SET current_revision_id = 'ptrev_1' WHERE id = 'ptdef_test'"
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("version", 0),
        ("schema_version", 2),
        ("contract_sha256", "A" * 64),
        ("contract_sha256", "x" * 64),
    ],
)
def test_revision_database_constraints_fail_closed(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    settings = Settings(data_dir=tmp_path / f"prompt-template-invalid-{column}")
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    database = settings.state_dir / "local-lm.sqlite3"
    payload, digest = _canonical_contract()
    row: dict[str, object] = {
        "version": 1,
        "schema_version": 1,
        "contract_sha256": digest,
    }
    row[column] = value
    with sqlite3.connect(database) as connection:
        _insert_definition(connection)
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO prompt_template_revisions
                  (id, prompt_template_id, version, schema_version,
                   contract_json, contract_sha256, created_at, updated_at)
                VALUES ('ptrev_bad', 'ptdef_test', ?, ?, ?, ?,
                        CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    row["version"],
                    row["schema_version"],
                    json.dumps(payload),
                    row["contract_sha256"],
                ),
            )


@pytest.mark.parametrize(
    "payload",
    (
        {"schema_version": 2},
        {},
        {"schema_version": None},
        {"schema_version": True},
        {"schema_version": 1.0},
    ),
)
def test_revision_insert_guard_binds_json_schema_version(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-template-json-version")
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    database = settings.state_dir / "local-lm.sqlite3"
    _, digest = _canonical_contract()

    with sqlite3.connect(database) as connection:
        _insert_definition(connection)
        with pytest.raises(sqlite3.IntegrityError, match="revision contract is invalid"):
            _insert_revision(
                connection,
                revision_id="ptrev_bad",
                version=1,
                payload=payload,
                digest=digest,
            )


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"schema_version": None},
        {"schema_version": True},
        {"schema_version": 1.0},
    ),
)
def test_guard_upgrade_refuses_preexisting_invalid_contract(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-template-invalid-upgrade")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, "d2f8c1a94e70")
    database = settings.state_dir / "local-lm.sqlite3"
    _, digest = _canonical_contract()

    with sqlite3.connect(database) as connection:
        connection.execute("DROP TRIGGER prompt_template_revision_insert_guard")
        connection.execute(_LEGACY_NULL_UNSAFE_REVISION_INSERT_TRIGGER)
        _insert_definition(connection)
        _insert_revision(
            connection,
            revision_id="ptrev_bad",
            version=1,
            payload=payload,
            digest=digest,
        )

    with pytest.raises(
        RuntimeError,
        match="stored revision contract is invalid",
    ):
        command.upgrade(config, "head")

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            "d2f8c1a94e70",
        )


def test_historical_d2_trigger_is_identical_on_fresh_upgrade_and_downgrade(
    tmp_path: Path,
) -> None:
    def revision_insert_trigger(database: Path) -> str:
        with sqlite3.connect(database) as connection:
            row = connection.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'prompt_template_revision_insert_guard'"
            ).fetchone()
        assert row is not None
        return str(row[0])

    fresh = Settings(data_dir=tmp_path / "prompt-template-fresh-d2")
    fresh.prepare()
    fresh_config = alembic_config(fresh)
    command.upgrade(fresh_config, "d2f8c1a94e70")
    fresh_sql = revision_insert_trigger(fresh.state_dir / "local-lm.sqlite3")

    downgraded = Settings(data_dir=tmp_path / "prompt-template-downgraded-d2")
    downgraded.prepare()
    downgraded_config = alembic_config(downgraded)
    command.upgrade(downgraded_config, "head")
    command.downgrade(downgraded_config, "d2f8c1a94e70")
    downgraded_sql = revision_insert_trigger(downgraded.state_dir / "local-lm.sqlite3")

    assert fresh_sql == downgraded_sql
    assert " != NEW.schema_version" in fresh_sql
    assert " IS NOT NEW.schema_version" not in fresh_sql


def test_fresh_metadata_schema_matches_orm_relationships_and_guards(
    tmp_path: Path,
) -> None:
    database = tmp_path / "fresh-prompt-templates.sqlite3"
    engine = create_engine(f"sqlite:///{database}")
    try:
        Base.metadata.create_all(engine)
        with Session(engine) as session:
            definition = PromptTemplateDefinition(
                name="Fresh template",
                description="Synthetic",
            )
            session.add(definition)
            session.flush()
            assert len(definition.id) <= 40

            payload, digest = _canonical_contract()
            revision = PromptTemplateRevision(
                prompt_template_id=definition.id,
                version=1,
                schema_version=1,
                contract_json=payload,
                contract_sha256=digest,
            )
            session.add(revision)
            session.flush()
            assert len(revision.id) <= 40
            definition.current_revision_id = revision.id
            session.commit()

        with Session(engine) as session:
            loaded = session.scalars(
                select(PromptTemplateDefinition).where(
                    PromptTemplateDefinition.name == "Fresh template"
                )
            ).one()
            assert loaded.current_revision_id is not None
            assert [revision.version for revision in loaded.revisions] == [1]

        with (
            engine.begin() as connection,
            pytest.raises(IntegrityError, match="revisions are immutable"),
        ):
            connection.exec_driver_sql("UPDATE prompt_template_revisions SET version = 2")
    finally:
        engine.dispose()


def _insert_import_winner(
    connection: sqlite3.Connection,
    *,
    key: str = "import-winner",
    definition_id: str = "ptdef_test",
    revision_id: str = "ptrev_1",
    contract_sha256: str | None = None,
) -> None:
    digest = contract_sha256 or _canonical_contract()[1]
    connection.execute(
        """
        INSERT INTO prompt_template_import_winners
          (idempotency_key, request_sha256, bundle_sha256, authority_rule,
           prompt_template_id, prompt_template_revision_id, contract_sha256,
           created_at)
        VALUES (?, ?, ?, 'prompt-template-import-authority-v1',
                ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (key, "a" * 64, "b" * 64, definition_id, revision_id, digest),
    )


def test_import_winner_guard_binds_exact_initial_revision_and_is_immutable(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "prompt-import-winner-guard")
    settings.prepare()
    command.upgrade(alembic_config(settings), "head")
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        _insert_definition(connection)
        _insert_revision(connection, revision_id="ptrev_1", version=1)
        with pytest.raises(sqlite3.IntegrityError, match="winner is invalid"):
            _insert_import_winner(connection)
        connection.execute(
            "UPDATE prompt_template_definitions "
            "SET current_revision_id = 'ptrev_1' WHERE id = 'ptdef_test'"
        )
        with pytest.raises(sqlite3.IntegrityError, match="winner is invalid"):
            _insert_import_winner(connection, contract_sha256="c" * 64)
        _insert_import_winner(connection)
        with pytest.raises(sqlite3.IntegrityError, match="winners are immutable"):
            connection.execute(
                "UPDATE prompt_template_import_winners SET bundle_sha256 = ?",
                ("d" * 64,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="winners are immutable"):
            connection.execute("DELETE FROM prompt_template_import_winners")


def test_import_winner_triggers_are_identical_after_migration_and_fresh_metadata(
    tmp_path: Path,
) -> None:
    migrated = Settings(data_dir=tmp_path / "prompt-import-winner-migrated")
    migrated.prepare()
    command.upgrade(alembic_config(migrated), "head")
    migrated_database = migrated.state_dir / "local-lm.sqlite3"
    fresh_database = tmp_path / "prompt-import-winner-fresh.sqlite3"
    engine = create_engine(f"sqlite:///{fresh_database}")
    try:
        Base.metadata.create_all(engine)
    finally:
        engine.dispose()

    def trigger_sql(database: Path) -> dict[str, str]:
        with sqlite3.connect(database) as connection:
            return {
                str(name): str(sql)
                for name, sql in connection.execute(
                    "SELECT name, sql FROM sqlite_master "
                    "WHERE type = 'trigger' "
                    "AND name LIKE 'prompt_template_import_winner_%' "
                    "ORDER BY name"
                )
            }

    assert trigger_sql(migrated_database) == trigger_sql(fresh_database)
