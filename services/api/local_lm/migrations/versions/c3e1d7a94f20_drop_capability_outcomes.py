"""Drop the capability-evidence outcome columns nothing could ever set.

`model_capability_evidence` carried `result`, `failure_code` and
`failure_reason` for an outcome the application cannot record. Its only writer
sets result="ready" and neither failure column, so a row that exists is a probe
that passed. The two readers were consistent with that and no more: a query
filter selecting the value its own writer always writes, and a readiness branch
for a non-ready row that cannot exist.

Removing them REFUSES rather than assumes. A database that somehow holds a row
with another outcome - restored from elsewhere, edited by hand, written by a
version that had the writer this one lacks - is exactly the case where dropping
the discriminator would silently promote a failure to a pass. So the upgrade
looks first and stops, leaving the row and the columns intact for someone to
look at, rather than deciding on their behalf.

The downgrade restores the columns and marks every surviving row "ready",
which is what they all were.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "c3e1d7a94f20"
down_revision: str | None = "b8f31d0a6c42"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None


def upgrade() -> None:
    connection = op.get_bind()
    unexpected = connection.execute(
        sa.text(
            "SELECT COUNT(*) FROM model_capability_evidence "
            "WHERE result IS NULL OR result <> 'ready'"
        )
    ).scalar_one()
    if unexpected:
        raise RuntimeError(
            f"{unexpected} capability evidence row(s) record an outcome other than "
            "'ready'. This database has a failure this application cannot write, so "
            "the outcome columns are not being dropped: nothing here can tell whether "
            "those rows should become passes."
        )
    # The index spans (model_install_id, result), so it is rebuilt on the column
    # that survives rather than dropped with the one that does not - the lookup
    # it serves is by install and is still made.
    op.drop_index("ix_model_capability_evidence_install_result", "model_capability_evidence")
    with op.batch_alter_table("model_capability_evidence") as batch:
        batch.drop_column("failure_reason")
        batch.drop_column("failure_code")
        batch.drop_column("result")
    op.create_index(
        "ix_model_capability_evidence_install",
        "model_capability_evidence",
        ["model_install_id"],
    )


def downgrade() -> None:
    with op.batch_alter_table("model_capability_evidence") as batch:
        # The default is only how the existing rows get a value for a column
        # that is NOT NULL; the original carried none. It is dropped again
        # below so the restored table matches what was there before rather
        # than quietly supplying "ready" to a later insert that omits it -
        # which is the very thing this migration refuses to do.
        batch.add_column(sa.Column("result", sa.String(24), nullable=False, server_default="ready"))
        batch.add_column(sa.Column("failure_code", sa.String(80), nullable=True))
        batch.add_column(sa.Column("failure_reason", sa.Text(), nullable=True))
    with op.batch_alter_table("model_capability_evidence") as batch:
        batch.alter_column("result", server_default=None)
    op.drop_index("ix_model_capability_evidence_install", "model_capability_evidence")
    op.create_index(
        "ix_model_capability_evidence_install_result",
        "model_capability_evidence",
        ["model_install_id", "result"],
    )
