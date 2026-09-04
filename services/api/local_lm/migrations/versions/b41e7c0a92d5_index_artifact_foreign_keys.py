"""Index the artifact foreign keys the delete path scans

Revision ID: b41e7c0a92d5
Revises: c9e1d4a70b82
Create Date: 2026-08-30

Deleting one artifact makes SQLite find every row that refers to it, so that it
can apply `ON DELETE SET NULL`. Without an index on the referring column that is
a full table scan, once per deleted artifact, per table.

How much this saves depends on how many child rows exist, because the cost being
removed is a scan: on a table with more rows the saving is correspondingly
larger. There is no single figure worth quoting for it, and the indexes
themselves build once, in a time too small to weigh against a migration.

It also does not mean an artifact delete stops scanning. The BEFORE DELETE
reference guard still walks six tables looking for JSON references, and those
ids are not foreign keys, so nothing here reaches them. What this removes is the
foreign-key child lookup SQLite performs to apply ON DELETE SET NULL.

Three columns, counted against a MIGRATED database rather than carried over from
a reading of the models:
`artifact_library_entries.artifact_id`, `reference_assets.artifact_id` and
`comfy_registry_source_artifact_reviews.artifact_id` have since gained indexes
of their own, leaving these three.

This is a real but small part of the retention cost, and saying so is the point
rather than modesty: the dominant cost of a deletion is the BEFORE DELETE
reference guard, which parses stored JSON across every referencing table, and
this migration does not touch it. Anyone reading a retention slowdown should
look there first.
"""

from __future__ import annotations

from alembic import op

revision: str = "b41e7c0a92d5"
down_revision: str | None = "c9e1d4a70b82"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("ix_message_parts_artifact_id", "message_parts", "artifact_id"),
    (
        "ix_response_revision_parts_artifact_id",
        "response_revision_parts",
        "artifact_id",
    ),
    (
        "ix_reference_subjects_cover_artifact_id",
        "reference_subjects",
        "cover_artifact_id",
    ),
)


# One key term as PRAGMA index_xinfo reports it: the column name, its
# collation upper-cased, and the descending flag. The name is optional
# because an expression or rowid term has none, and a definition this
# migration did not create may well contain one.
_KeyTerm = tuple[str | None, str, int]
# Uniqueness, origin, partiality, and the key terms in order.
_Definition = tuple[bool, str, bool, tuple[_KeyTerm, ...]]


def _existing_definition(connection: object, table: str, name: str) -> _Definition | None:
    """Every property of an existing index of this name, or None if absent.

    Column names alone are not an identity. A same-name PARTIAL index on the
    expected column reports the same `PRAGMA index_info` as a full one, so a
    columns-only comparison accepts an index that cannot serve the unrestricted
    foreign-key lookup - and would then authorise the downgrade to delete it.
    Uniqueness, collation, sort direction and extra key terms are the same class
    of difference: they may belong to another invariant, and dropping them is not
    this migration's business.

    So the witness carries what `op.create_index(name, table, [column])` actually
    produces: a non-unique, non-partial index created by CREATE INDEX, with one
    ascending key term on the column under its own collation.
    """

    listed = connection.exec_driver_sql(  # type: ignore[attr-defined]
        f"PRAGMA index_list('{table}')"
    ).fetchall()
    for _seq, index_name, unique, origin, partial in listed:
        if index_name != name:
            continue
        terms = connection.exec_driver_sql(  # type: ignore[attr-defined]
            f"PRAGMA index_xinfo('{name}')"
        ).fetchall()
        key_terms = tuple((term[2], (term[4] or "").upper(), term[3]) for term in terms if term[5])
        return (bool(unique), str(origin), bool(partial), key_terms)
    return None


def _expected_definition(column: str) -> _Definition:
    return (False, "c", False, ((column, "BINARY", 0),))


def upgrade() -> None:
    # Guarded because a partial application is otherwise unrecoverable: SQLite
    # commits each CREATE INDEX, but a failure part-way rolls alembic_version
    # back, so the schema holds indexes the version row says were never created.
    # Every later start replays and dies on "already exists", and the application
    # never starts again. Reproduced before this guard existed.
    #
    # NOT "IF NOT EXISTS", which is fail-open on the case that matters: with an
    # unrelated index already carrying one of these names it returns success and
    # leaves the wrong index in place, so the revision advances without the index
    # it claims to create. Measured.
    connection = op.get_bind()
    for name, table, column in _INDEXES:
        existing = _existing_definition(connection, table, name)
        if existing is None:
            op.create_index(name, table, [column])
        elif existing != _expected_definition(column):
            raise RuntimeError(
                f"index {name} on {table} already exists with a different "
                f"definition {existing}, not {_expected_definition(column)}; "
                f"refusing to advance this revision without the index it claims "
                f"to create, and refusing to adopt one this migration did not make"
            )


def downgrade() -> None:
    # Symmetric on purpose: drop only an exact match, so a same-name index of a
    # DIFFERENT shape survives a downgrade of this revision.
    #
    # The narrower claim is the honest one. SQLite records that an index came
    # from an explicit CREATE INDEX, not which migration ran it, so a
    # pre-existing index of exactly this shape is indistinguishable from one
    # this revision created and would be dropped. That case is not claimed. What
    # is guaranteed is the one the guard exists for: an index sharing the name
    # but not the definition is never touched.
    connection = op.get_bind()
    for name, table, column in _INDEXES:
        if _existing_definition(connection, table, name) == _expected_definition(column):
            op.drop_index(name, table_name=table)
