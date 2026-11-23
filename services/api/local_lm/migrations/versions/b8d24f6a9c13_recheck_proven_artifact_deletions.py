"""Check every proven artifact deletion again inside the database

Revision ID: b8d24f6a9c13
Revises: 7c3e9a1f5b20
Create Date: 2026-11-22

A deletion is proven from the artifact reference index, and the proven delete
used to skip every check in the database. This guard asks the database the
same question the proof asked, at the statement: whether anything still
recorded is undrained, whether a row could not be read, and whether anything
that survives names the artifact. A proof minted wrongly then fails at the
delete instead of removing something still held. Deletions that are not
proven keep the checks they always ran.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

from local_lm.artifact_reference_index_schema import DELETE_GUARD_SQL, DROP_DELETE_GUARD_SQL

revision: str = "b8d24f6a9c13"
down_revision: str | None = "7c3e9a1f5b20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _begin_write_fence() -> None:
    connection = op.get_bind()
    if connection.dialect.name != "sqlite":
        return
    driver = connection.connection.driver_connection
    if not bool(getattr(driver, "in_transaction", False)):
        connection.exec_driver_sql("BEGIN IMMEDIATE")
    else:
        connection.exec_driver_sql("UPDATE alembic_version SET version_num = version_num")


def upgrade() -> None:
    _begin_write_fence()
    op.execute(DROP_DELETE_GUARD_SQL)
    op.execute(DELETE_GUARD_SQL)


def downgrade() -> None:
    _begin_write_fence()
    op.execute(DROP_DELETE_GUARD_SQL)
