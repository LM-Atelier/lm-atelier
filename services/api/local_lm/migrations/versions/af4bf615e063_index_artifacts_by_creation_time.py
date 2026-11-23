"""Index artifacts in creation order, the order a retention pass reads them in

Revision ID: af4bf615e063
Revises: d41c7a9e2b58
Create Date: 2026-11-23

Each batch of the startup retention sweep goes on from the artifact where the
batch before it stopped. Without an index on that order, finding the next
artifacts sorted the whole table, reading every artifact's metadata on the way,
while the batch held the database writer.

The schema change also moves the reference index's stamp, so the first start
after this upgrade rebuilds that index before the sweep can prove a deletion,
as any migration does: about 47 s for 153,606 rows on a neutral store of
99,426 artifacts.
"""

from __future__ import annotations

from alembic import op
from sqlalchemy.engine import Connection

revision: str = "af4bf615e063"
down_revision: str | None = "d41c7a9e2b58"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_NAME = "ix_artifacts_created_at_id"
_TABLE = "artifacts"
_COLUMNS = ("created_at", "id")

# One key term as PRAGMA index_xinfo reports it: the column name, its collation
# upper-cased, and the descending flag.
_KeyTerm = tuple[str | None, str, int]
# Uniqueness, origin, partiality, and the key terms in order.
_Definition = tuple[bool, str, bool, tuple[_KeyTerm, ...]]
# What `op.create_index(_NAME, _TABLE, list(_COLUMNS))` makes: a plain index from
# CREATE INDEX, neither unique nor partial, ascending on both columns.
_EXPECTED: _Definition = (False, "c", False, tuple((column, "BINARY", 0) for column in _COLUMNS))


def _existing_definition(connection: Connection) -> _Definition | None:
    """Every property of an existing index of this name, or None if there is none."""

    listed = connection.exec_driver_sql(f"PRAGMA index_list('{_TABLE}')").fetchall()
    for _seq, index_name, unique, origin, partial in listed:
        if index_name != _NAME:
            continue
        terms = connection.exec_driver_sql(f"PRAGMA index_xinfo('{_NAME}')").fetchall()
        key_terms = tuple((term[2], (term[4] or "").upper(), term[3]) for term in terms if term[5])
        return (bool(unique), str(origin), bool(partial), key_terms)
    return None


def upgrade() -> None:
    # Guarded as b41e7c0a92d5 is: SQLite commits the CREATE INDEX on its own,
    # so an upgrade that fails after it leaves the index behind a version row
    # that says it was never made, and a bare replay would die on "already
    # exists" at every later start. An exact replay finishes; a same-name index
    # of another shape is refused, never adopted.
    existing = _existing_definition(op.get_bind())
    if existing is None:
        op.create_index(_NAME, _TABLE, list(_COLUMNS))
    elif existing != _EXPECTED:
        raise RuntimeError(
            f"index {_NAME} on {_TABLE} already exists with a different definition "
            f"{existing}, not {_EXPECTED}; refusing to advance this revision without the "
            "index it claims to create, and refusing to adopt one this migration did not make"
        )


def downgrade() -> None:
    # Drops only an exact match, so a same-name index of another shape survives.
    if _existing_definition(op.get_bind()) == _EXPECTED:
        op.drop_index(_NAME, table_name=_TABLE)
