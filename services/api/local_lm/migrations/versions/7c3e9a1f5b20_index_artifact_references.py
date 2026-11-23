"""Index which artifacts each stored document names

Revision ID: 7c3e9a1f5b20
Revises: 2a07d89a3b46
Create Date: 2026-11-22

Adds the artifact reference index, the record of changed rows that keeps it
exact, and the stamp that says when it may be trusted; see
`local_lm.artifact_reference_index`. Only schema is created here. The index is
filled by a rebuild after startup, outside this migration, so upgrading a
large store reads none of its documents and holds the writer only for the DDL.

The write guards and the delete guard are restated with their current text.
A rebuild certifies the index only over guards that read exactly as the
application writes them, and a long-lived store could otherwise hold an
older spelling of one forever.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from local_lm.artifact_library_schema import (
    ARTIFACT_JSON_DELETE_TRIGGER,
    JSON_WRITE_TRIGGER_SQL,
)
from local_lm.artifact_reference_index_schema import (
    CHANGES_TABLE,
    CREATE_INDEX_SQL,
    CREATE_TABLE_SQL,
    CREATE_TRIGGER_SQL,
    EDGES_TABLE,
    SEED_STAMP_SQL,
    STAMP_TABLE,
    trigger_name,
)

revision: str = "7c3e9a1f5b20"
down_revision: str | None = "2a07d89a3b46"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (EDGES_TABLE, CHANGES_TABLE, STAMP_TABLE)
_INDEXES = ("ix_artifact_reference_edges_source", "ix_artifact_reference_changes_audit")


def _begin_write_fence() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num = version_num")


def _installed(kind: str, name: str) -> str | None:
    value = (
        op.get_bind()
        .exec_driver_sql("SELECT sql FROM sqlite_master WHERE type = ? AND name = ?", (kind, name))
        .scalar_one_or_none()
    )
    return None if value is None else str(value)


def _create_exactly(kind: str, name: str, statement: str) -> None:
    """Create an object, accept an identical one, and refuse one defined otherwise.

    A same-name object with another definition may belong to something else;
    adopting it would advance this revision without the object it claims.
    """

    existing = _installed(kind, name)
    if existing is None:
        op.execute(statement)
    elif existing.strip() != statement.strip():
        raise RuntimeError(
            f"{kind} {name} already exists with a different definition; "
            "refusing to advance this revision without the one it creates"
        )


def upgrade() -> None:
    _begin_write_fence()
    for statement in (*JSON_WRITE_TRIGGER_SQL, ARTIFACT_JSON_DELETE_TRIGGER):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name(statement)}")
        op.execute(statement)
    for name, statement in zip(_TABLES, CREATE_TABLE_SQL, strict=True):
        _create_exactly("table", name, statement)
    for name, statement in zip(_INDEXES, CREATE_INDEX_SQL, strict=True):
        _create_exactly("index", name, statement)
    # Before the stamp's authority triggers exist, which refuse this insert.
    op.execute(SEED_STAMP_SQL)
    for statement in CREATE_TRIGGER_SQL:
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name(statement)}")
        op.execute(statement)


def downgrade() -> None:
    _begin_write_fence()
    for statement in reversed(CREATE_TRIGGER_SQL):
        op.execute(f"DROP TRIGGER IF EXISTS {trigger_name(statement)}")
    # The index is derived from the source rows, so dropping it loses nothing.
    # Only objects exactly as this revision creates them are dropped.
    for name, statement in zip(_INDEXES, CREATE_INDEX_SQL, strict=True):
        if (_installed("index", name) or "").strip() == statement.strip():
            op.execute(f"DROP INDEX {name}")
    for name, statement in zip(_TABLES, CREATE_TABLE_SQL, strict=True):
        if (_installed("table", name) or "").strip() == statement.strip():
            op.execute(f"DROP TABLE {name}")
