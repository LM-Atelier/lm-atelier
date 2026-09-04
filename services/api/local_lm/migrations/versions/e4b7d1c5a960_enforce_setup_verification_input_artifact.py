"""Enforce the setup verification's artifact reference

Revision ID: e4b7d1c5a960
Revises: b41e7c0a92d5
Create Date: 2026-09-03

`setup_verifications.input_artifact_id` holds an artifact id and is counted as a
strong reference by the retention walk, but it was declared as a plain column.
Deleting a referenced artifact therefore left a DANGLING identifier instead of
clearing it, and a dangling id reads as a live reference until something tries
to follow it. A null reads as "no longer present", which is the truth.

SET NULL rather than RESTRICT, decided from the lifecycle rather than from the
column in isolation. Every writer of this column nulls it, flushes, and only
then deletes its artifact, so an enforced reference never obstructs the normal
path and either choice would be invisible there. The two differ only once the
guard above them has already failed, and there they differ sharply:

- the guard that actually protects live work is in Python, not in the schema.
  The artifact delete refuses when the artifact is still in the reference set,
  and the HTTP layer turns that refusal into a 409;
- exactly one caller gets past it, by handing the delete a reference snapshot it
  computed itself under the write fence: the retention sweep;
- that sweep runs at startup, in a stage that does not catch exceptions, and it
  runs BEFORE the recovery that nulls this column on interrupted verifications.
  So a crashed install reaches the sweep with this column still set.

RESTRICT's only reachable effect is therefore to turn a wrong retention snapshot
into an application that will not start; SET NULL clears a pointer that the next
startup stage clears anyway. The reference is transient machine state belonging
to a disposable self-test, not user data. RESTRICT is reserved in this schema
for durable, user-owned references, where refusing a delete is a meaningful
answer to someone who asked for it.

The column is widened from 40 to 80 characters in the same rebuild, because
that is the width of the key it now references. Every other artifact reference
in this schema omits the type and inherits `artifacts.id`, so this column was
the only one narrower than the ids it stores. SQLite does not enforce a VARCHAR
length, which is why nothing has failed, but a foreign key whose child column is
narrower than its parent key is a defect in the constraint itself.

The index is not decoration. With the constraint in place SQLite has to find
every referring row on each artifact delete in order to apply SET NULL, and
without an index that is a full scan of this table per deleted artifact.

Existing rows are resolved before the constraint is added. `PRAGMA
foreign_key_check` walks every declared foreign key across the rows already
stored, and the backup verifier treats any violation it reports as a failed
backup, so a dangling value left under this key would break every backup taken
afterwards. Alembic's connection runs with foreign keys off and SQLite never
revalidates stored rows, so nothing else will catch it: this migration is the
last point at which such a value can be identified. Only non-resolving
identifiers are cleared. Every verification row is kept, no artifact row is
invented, and a resolving identifier is left exactly as it is - including on a
completed or failed verification, where every writer is supposed to have nulled
it already. That case is reported rather than corrected: it would mean a
synthetic artifact pinned against retention for good, which is worth seeing
rather than silently tidying inside a schema migration.

The distribution is logged by verification state before anything is written,
because this is the last moment at which the pre-migration state can be
observed at all.
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision: str = "e4b7d1c5a960"
down_revision: str | None = "b41e7c0a92d5"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_INDEX = "ix_setup_verifications_input_artifact_id"
_TABLE = "setup_verifications"
_COLUMN = "input_artifact_id"
_CONSTRAINT = "fk_setup_verifications_input_artifact_id"

_logger = logging.getLogger("local_lm.migrations")

_DISTRIBUTION = """
SELECT state,
       COUNT(*),
       SUM(CASE WHEN input_artifact_id IS NOT NULL THEN 1 ELSE 0 END),
       SUM(CASE WHEN input_artifact_id IS NOT NULL
                 AND input_artifact_id NOT IN (SELECT id FROM artifacts)
                THEN 1 ELSE 0 END)
FROM setup_verifications
GROUP BY state
ORDER BY state
"""

_CLEAR_DANGLING = """
UPDATE setup_verifications
SET input_artifact_id = NULL
WHERE input_artifact_id IS NOT NULL
  AND input_artifact_id NOT IN (SELECT id FROM artifacts)
"""


# One key term as PRAGMA index_xinfo reports it: the column name, its collation
# upper-cased, and the descending flag. Uniqueness, origin, partiality, and the
# key terms in order make up the definition.
_KeyTerm = tuple[str | None, str, int]
_Definition = tuple[bool, str, bool, tuple[_KeyTerm, ...]]


def _existing_definition(connection: object) -> _Definition | None:
    """Every property of an existing index of this name, or None if absent.

    Not `IF NOT EXISTS`, and not a presence check either. Both are fail-open on
    the case that matters: an unrelated index already carrying this name leaves
    the wrong index in place while the revision advances as though it had
    created the right one. A same-name PARTIAL index reports the same
    `PRAGMA index_info` as a full one, so columns alone are not an identity
    either - and a partial index cannot serve the unrestricted foreign-key
    lookup this revision exists to make cheap.

    So the witness carries what `op.create_index(name, table, [column])`
    actually produces: a non-unique, non-partial index created by CREATE INDEX,
    with one ascending key term on the column under its own collation.
    """

    listed = connection.exec_driver_sql(  # type: ignore[attr-defined]
        f"PRAGMA index_list('{_TABLE}')"
    ).fetchall()
    for _seq, name, unique, origin, partial in listed:
        if name != _INDEX:
            continue
        terms = connection.exec_driver_sql(  # type: ignore[attr-defined]
            f"PRAGMA index_xinfo('{_INDEX}')"
        ).fetchall()
        key_terms = tuple((term[2], (term[4] or "").upper(), term[3]) for term in terms if term[5])
        return (bool(unique), str(origin), bool(partial), key_terms)
    return None


def _expected_definition() -> _Definition:
    return (False, "c", False, ((_COLUMN, "BINARY", 0),))


def upgrade() -> None:
    connection = op.get_bind()

    for state, rows, named, dangling in connection.exec_driver_sql(_DISTRIBUTION).fetchall():
        _logger.info(
            "setup verification input artifacts before enforcement: "
            "state=%s rows=%s named=%s dangling=%s resolving=%s",
            state,
            rows,
            named,
            dangling,
            named - dangling,
        )

    cleared = connection.exec_driver_sql(_CLEAR_DANGLING).rowcount
    if cleared:
        _logger.info(
            "cleared %s setup verification input artifact identifier(s) that no "
            "longer resolve; every verification row was kept",
            cleared,
        )

    with op.batch_alter_table(_TABLE) as batch:
        batch.alter_column(
            _COLUMN,
            existing_type=sa.String(length=40),
            type_=sa.String(length=80),
            existing_nullable=True,
        )
        batch.create_foreign_key(
            _CONSTRAINT,
            "artifacts",
            [_COLUMN],
            ["id"],
            ondelete="SET NULL",
        )

    # After the rebuild, because the rebuild recreates the table's own indexes
    # and this one is not among them yet.
    existing = _existing_definition(connection)
    if existing is None:
        op.create_index(_INDEX, _TABLE, [_COLUMN])
    elif existing != _expected_definition():
        raise RuntimeError(
            f"index {_INDEX} on {_TABLE} already exists with a different "
            f"definition {existing}, not {_expected_definition()}; refusing to "
            f"advance this revision without the index it claims to create, and "
            f"refusing to adopt one this migration did not make"
        )


def downgrade() -> None:
    connection = op.get_bind()
    # Symmetric with the upgrade: drop only an exact match, so a same-name index
    # of a different shape survives a downgrade of this revision.
    if _existing_definition(connection) == _expected_definition():
        op.drop_index(_INDEX, table_name=_TABLE)
    with op.batch_alter_table(_TABLE) as batch:
        batch.drop_constraint(_CONSTRAINT, type_="foreignkey")
        batch.alter_column(
            _COLUMN,
            existing_type=sa.String(length=80),
            type_=sa.String(length=40),
            existing_nullable=True,
        )
