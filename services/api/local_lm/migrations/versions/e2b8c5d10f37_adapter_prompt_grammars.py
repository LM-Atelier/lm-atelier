"""record how an installed adapter expects to be prompted

Revision ID: e2b8c5d10f37
Revises: d1a7f3c94e28
Create Date: 2026-08-08
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e2b8c5d10f37"
down_revision: str | None = "d1a7f3c94e28"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_sha256_check(column: str) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = 64 AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "adapter_prompt_grammars",
        sa.Column("id", sa.String(length=48), nullable=False),
        sa.Column("model_asset_install_id", sa.String(length=40), nullable=False),
        # What the grammar was written about. An install row is mutable and its
        # file can be replaced underneath it, so binding to the row alone would
        # let a new file inherit the old file's description of itself.
        sa.Column("asset_sha256", sa.String(length=64), nullable=False),
        sa.Column("source_identity", sa.Text(), nullable=False),
        # Only the digest of the document. The document itself is third-party
        # text on its way to a prompt rewriter, so it stays quarantined outside
        # this row rather than being stored where something might read it.
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("grammar_json", sa.JSON(), nullable=False),
        sa.Column("examples_reviewed", sa.Boolean(), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_asset_install_id"],
            ["model_asset_installs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_asset_install_id", name="uq_adapter_prompt_grammar_install"),
        sa.CheckConstraint(
            _lowercase_sha256_check("asset_sha256"),
            name="ck_adapter_prompt_grammar_asset_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_sha256_check("source_sha256"),
            name="ck_adapter_prompt_grammar_source_sha256",
        ),
    )
    op.create_index(
        "ix_adapter_prompt_grammars_model_asset_install_id",
        "adapter_prompt_grammars",
        ["model_asset_install_id"],
    )
    op.create_index(
        "ix_adapter_prompt_grammars_asset_sha256",
        "adapter_prompt_grammars",
        ["asset_sha256"],
    )


def downgrade() -> None:
    op.drop_index("ix_adapter_prompt_grammars_asset_sha256", table_name="adapter_prompt_grammars")
    op.drop_index(
        "ix_adapter_prompt_grammars_model_asset_install_id",
        table_name="adapter_prompt_grammars",
    )
    op.drop_table("adapter_prompt_grammars")
