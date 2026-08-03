from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command

from local_lm.config import Settings
from local_lm.database_migrations import alembic_config

_BASE = "b6a1e4d92c70"
_REVISION = "c8f2d7a91e64"
_TIMESTAMP = "2026-08-03 00:00:00"


def _insert_definition_and_revision(
    connection: sqlite3.Connection, suffix: str
) -> tuple[object, ...]:
    definition_id = f"workflow_{suffix}"
    revision_id = f"wfrev_{suffix}"
    connection.execute(
        """
        INSERT INTO workflow_definitions (
            id, name, operation, description, current_revision_id,
            family_id, variant_key, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            definition_id,
            f"Workflow {suffix}",
            "text_to_image",
            "Legacy definition",
            revision_id,
            None,
            None,
            _TIMESTAMP,
            _TIMESTAMP,
        ),
    )
    revision = (
        revision_id,
        definition_id,
        1,
        "comfyui",
        "0.3.50",
        '{"ui":true}',
        '{"1":{"class_type":"SaveImage"}}',
        '{"type":"object"}',
        '["image"]',
        '{"model_install_ids":["model_legacy"]}',
        "a" * 64,
        1,
        _TIMESTAMP,
        _TIMESTAMP,
    )
    connection.execute(
        """
        INSERT INTO workflow_revisions (
            id, workflow_id, version, engine, engine_version,
            ui_graph_json, api_graph_json, input_schema_json,
            capabilities_json, dependencies_json, artifact_sha256, trusted,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        revision,
    )
    return revision


def test_workflow_dependency_migration_preserves_legacy_rows_and_defaults(
    tmp_path: Path,
) -> None:
    settings = Settings(data_dir=tmp_path / "workflow-dependency-migration")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, _BASE)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        kept_revision = _insert_definition_and_revision(connection, "keep")
        _insert_definition_and_revision(connection, "delete")
        connection.commit()

    command.upgrade(config, _REVISION)

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        migrated = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   capabilities_json, dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at, dependency_contract_sha256
            FROM workflow_revisions WHERE id = 'wfrev_keep'
            """
        ).fetchone()
        assert migrated == (*kept_revision, None)
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "workflow_dependency_slots",
            "workflow_activations",
            "workflow_dependency_bindings",
        } <= tables

        foreign_key_rows = connection.execute(
            "PRAGMA foreign_key_list(workflow_dependency_bindings)"
        ).fetchall()
        foreign_keys = {(row[3], row[2]): row[6] for row in foreign_key_rows}
        assert foreign_keys[("workflow_activation_id", "workflow_activations")] == "CASCADE"
        assert foreign_keys[("workflow_dependency_slot_id", "workflow_dependency_slots")] == (
            "NO ACTION"
        )
        assert foreign_keys[("model_profile_id", "model_profiles")] == "RESTRICT"
        assert foreign_keys[("model_install_id", "model_installs")] == "RESTRICT"
        assert foreign_keys[("model_asset_install_id", "model_asset_installs")] == "RESTRICT"
        assert foreign_keys[("custom_node_install_id", "custom_node_installs")] == "RESTRICT"
        assert foreign_keys[("comfy_registry_install_id", "comfy_registry_installs")] == (
            "RESTRICT"
        )
        assert {
            (row[3], row[4], row[6]) for row in foreign_key_rows if row[2] == "workflow_activations"
        } == {
            ("workflow_activation_id", "id", "CASCADE"),
            ("workflow_revision_id", "workflow_revision_id", "CASCADE"),
        }
        assert {
            (row[3], row[4], row[6])
            for row in foreign_key_rows
            if row[2] == "workflow_dependency_slots"
        } == {
            ("workflow_dependency_slot_id", "id", "NO ACTION"),
            ("workflow_revision_id", "workflow_revision_id", "NO ACTION"),
        }
        partial_index = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'index' AND name = 'uq_workflow_activation_active_revision'
            """
        ).fetchone()
        assert partial_index is not None
        assert "WHERE is_active = 1" in partial_index[0]

        connection.execute(
            """
            INSERT INTO workflow_dependency_slots (
                id, workflow_revision_id, name, resource_kind,
                contract_sha256, ordinal, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wfslot_delete",
                "wfrev_delete",
                "primary",
                "runtime",
                "b" * 64,
                0,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_activations (
                id, workflow_revision_id, resolver_version,
                dependency_contract_sha256, binding_sha256,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wfact_delete",
                "wfrev_delete",
                "resolver-v1",
                "c" * 64,
                "d" * 64,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )
        connection.execute(
            """
            INSERT INTO workflow_dependency_bindings (
                id, workflow_revision_id, workflow_activation_id,
                workflow_dependency_slot_id,
                requirement_key, runtime_key, resource_identity_sha256,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "wfbind_delete",
                "wfrev_delete",
                "wfact_delete",
                "wfslot_delete",
                "default",
                "comfyui-v1",
                "e" * 64,
                _TIMESTAMP,
                _TIMESTAMP,
            ),
        )
        assert connection.execute(
            """
            SELECT required, satisfaction, requirements_json
            FROM workflow_dependency_slots WHERE id = 'wfslot_delete'
            """
        ).fetchone() == (1, "all_of", "[]")
        assert connection.execute(
            """
            SELECT state, is_active, details_json
            FROM workflow_activations WHERE id = 'wfact_delete'
            """
        ).fetchone() == ("ready", 0, "{}")
        assert connection.execute(
            """
            SELECT mount_json, resource_identity_json
            FROM workflow_dependency_bindings WHERE id = 'wfbind_delete'
            """
        ).fetchone() == ("{}", "{}")

        invalid_updates = [
            (
                "UPDATE workflow_revisions SET dependency_contract_sha256 = ? "
                "WHERE id = 'wfrev_keep'",
                "A" * 64,
            ),
            (
                "UPDATE workflow_dependency_slots SET contract_sha256 = ? "
                "WHERE id = 'wfslot_delete'",
                "g" * 64,
            ),
            (
                "UPDATE workflow_activations SET dependency_contract_sha256 = ? "
                "WHERE id = 'wfact_delete'",
                "g" * 64,
            ),
            (
                "UPDATE workflow_activations SET binding_sha256 = ? WHERE id = 'wfact_delete'",
                "A" * 64,
            ),
            (
                "UPDATE workflow_dependency_bindings SET resource_identity_sha256 = ? "
                "WHERE id = 'wfbind_delete'",
                "g" * 64,
            ),
        ]
        for statement, value in invalid_updates:
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement, (value,))
        for statement in (
            "UPDATE workflow_dependency_slots SET required = 2 WHERE id = 'wfslot_delete'",
            "UPDATE workflow_activations SET is_active = 2 WHERE id = 'wfact_delete'",
        ):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(statement)

        connection.execute("DELETE FROM workflow_revisions WHERE id = 'wfrev_delete'")
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_dependency_bindings"
        ).fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_activations").fetchone() == (0,)
        assert connection.execute("SELECT COUNT(*) FROM workflow_dependency_slots").fetchone() == (
            0,
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        connection.commit()

    command.downgrade(config, _BASE)

    with sqlite3.connect(database) as connection:
        restored = connection.execute(
            """
            SELECT id, workflow_id, version, engine, engine_version,
                   ui_graph_json, api_graph_json, input_schema_json,
                   capabilities_json, dependencies_json, artifact_sha256, trusted,
                   created_at, updated_at
            FROM workflow_revisions WHERE id = 'wfrev_keep'
            """
        ).fetchone()
        assert restored == kept_revision
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "workflow_dependency_slots" not in tables
        assert "workflow_activations" not in tables
        assert "workflow_dependency_bindings" not in tables
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("kind", ["slot", "activation", "revision_hash"])
def test_workflow_dependency_migration_refuses_lossy_downgrade(tmp_path: Path, kind: str) -> None:
    settings = Settings(data_dir=tmp_path / f"workflow-dependency-downgrade-{kind}")
    settings.prepare()
    config = alembic_config(settings)
    command.upgrade(config, _REVISION)
    database = settings.state_dir / "local-lm.sqlite3"
    with sqlite3.connect(database) as connection:
        _insert_definition_and_revision(connection, "keep")
        if kind == "slot":
            connection.execute(
                """
                INSERT INTO workflow_dependency_slots (
                    id, workflow_revision_id, name, resource_kind, required,
                    satisfaction, requirements_json, contract_sha256, ordinal,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "wfslot_keep",
                    "wfrev_keep",
                    "primary",
                    "runtime",
                    1,
                    "any_of",
                    '[{"key":"default","constraints":{}}]',
                    "b" * 64,
                    0,
                    _TIMESTAMP,
                    _TIMESTAMP,
                ),
            )
        elif kind == "activation":
            connection.execute(
                """
                INSERT INTO workflow_activations (
                    id, workflow_revision_id, resolver_version,
                    dependency_contract_sha256, binding_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "wfact_keep",
                    "wfrev_keep",
                    "resolver-v1",
                    "b" * 64,
                    "c" * 64,
                    _TIMESTAMP,
                    _TIMESTAMP,
                ),
            )
        else:
            connection.execute(
                """
                UPDATE workflow_revisions SET dependency_contract_sha256 = ?
                WHERE id = 'wfrev_keep'
                """,
                ("b" * 64,),
            )
        connection.commit()

    with pytest.raises(RuntimeError, match="Cannot downgrade workflow dependency bindings"):
        command.downgrade(config, _BASE)

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            _REVISION,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM workflow_revisions WHERE id = 'wfrev_keep'"
        ).fetchone() == (1,)
        if kind == "slot":
            assert connection.execute(
                "SELECT COUNT(*) FROM workflow_dependency_slots WHERE id = 'wfslot_keep'"
            ).fetchone() == (1,)
        elif kind == "activation":
            assert connection.execute(
                "SELECT COUNT(*) FROM workflow_activations WHERE id = 'wfact_keep'"
            ).fetchone() == (1,)
        else:
            assert connection.execute(
                "SELECT dependency_contract_sha256 FROM workflow_revisions WHERE id = 'wfrev_keep'"
            ).fetchone() == ("b" * 64,)
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
