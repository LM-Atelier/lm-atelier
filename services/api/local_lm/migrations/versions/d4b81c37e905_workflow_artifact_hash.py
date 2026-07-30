"""Record what each workflow revision executes.

Revision ID: d4b81c37e905
Revises: c6e9b2f41d30
"""

from __future__ import annotations

import hashlib
import json

import sqlalchemy as sa
from alembic import op

revision = "d4b81c37e905"
down_revision = "c6e9b2f41d30"
branch_labels = None
depends_on = None

# Mirrors model_planner.workflow_artifact_contract. Duplicated deliberately: a
# migration must keep producing the same values years from now, even if the
# application's definition moves on.
_CONTRACT_VERSION = 1
_EXECUTION_DEPENDENCY_KEYS = (
    "model_files",
    "custom_nodes",
    "extensions",
)


def _artifact_hash(
    operation: str,
    engine: str,
    api_graph: dict,  # type: ignore[type-arg]
    input_schema: dict,  # type: ignore[type-arg]
    dependencies: dict,  # type: ignore[type-arg]
) -> str:
    payload = {
        "version": _CONTRACT_VERSION,
        "operation": operation,
        "engine": engine,
        "api_graph": api_graph,
        "input_schema": input_schema,
        "dependencies": {
            key: dependencies[key] for key in _EXECUTION_DEPENDENCY_KEYS if key in dependencies
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _loaded(value: object) -> dict:  # type: ignore[type-arg]
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def upgrade() -> None:
    with op.batch_alter_table("workflow_revisions") as batch:
        batch.add_column(sa.Column("artifact_sha256", sa.String(length=64), nullable=True))
    op.create_index(
        "ix_workflow_revisions_artifact_sha256",
        "workflow_revisions",
        ["artifact_sha256"],
    )

    # Backfill from stored JSON. This needs no live runtime, so an existing
    # installation keeps its evidence rather than being asked to re-prove models
    # it has already proven.
    connection = op.get_bind()
    revisions = connection.execute(
        sa.text(
            "SELECT r.id, r.engine, r.api_graph_json, r.input_schema_json,"
            " r.dependencies_json, d.operation"
            " FROM workflow_revisions AS r"
            " JOIN workflow_definitions AS d ON d.id = r.workflow_id"
        )
    ).fetchall()
    for row in revisions:
        digest = _artifact_hash(
            str(row.operation or ""),
            str(row.engine or ""),
            _loaded(row.api_graph_json),
            _loaded(row.input_schema_json),
            _loaded(row.dependencies_json),
        )
        connection.execute(
            sa.text("UPDATE workflow_revisions SET artifact_sha256 = :digest WHERE id = :id"),
            {"digest": digest, "id": row.id},
        )


def downgrade() -> None:
    op.drop_index("ix_workflow_revisions_artifact_sha256", table_name="workflow_revisions")
    with op.batch_alter_table("workflow_revisions") as batch:
        batch.drop_column("artifact_sha256")
