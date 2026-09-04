"""record reviewed exact-commit source wheel artifacts

Revision ID: c9e1d4a70b82
Revises: c8e2f4a71d90
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c9e1d4a70b82"
down_revision: str | None = "c8e2f4a71d90"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _lowercase_hex_check(column: str, length: int) -> str:
    remainder = column
    for character in "0123456789abcdef":
        remainder = f"replace({remainder}, '{character}', '')"
    return f"length({column}) = {length} AND lower({column}) = {column} AND {remainder} = ''"


def upgrade() -> None:
    op.create_table(
        "comfy_registry_source_artifact_reviews",
        sa.Column("id", sa.String(length=80), nullable=False),
        sa.Column("source_declaration", sa.Text(), nullable=False),
        sa.Column("source_declaration_sha256", sa.String(length=64), nullable=False),
        sa.Column("repository", sa.String(length=300), nullable=False),
        sa.Column("source_commit", sa.String(length=40), nullable=False),
        sa.Column("artifact_id", sa.String(length=80), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_size_bytes", sa.Integer(), nullable=False),
        sa.Column("wheel_filename", sa.String(length=500), nullable=False),
        sa.Column("wheel_distribution", sa.String(length=200), nullable=False),
        sa.Column("wheel_version", sa.String(length=200), nullable=False),
        sa.Column("evidence_json", sa.JSON(), nullable=False),
        sa.Column("reviewer_kind", sa.String(length=32), nullable=False),
        sa.Column("review_sha256", sa.String(length=64), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifacts.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "source_declaration_sha256",
            name="uq_registry_source_review_declaration",
        ),
        sa.UniqueConstraint(
            "artifact_id",
            name="uq_registry_source_review_artifact",
        ),
        sa.UniqueConstraint(
            "review_sha256",
            name="uq_registry_source_review_digest",
        ),
        sa.CheckConstraint(
            _lowercase_hex_check("source_declaration_sha256", 64),
            name="ck_registry_source_review_declaration_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_hex_check("source_commit", 40),
            name="ck_registry_source_review_commit",
        ),
        sa.CheckConstraint(
            _lowercase_hex_check("artifact_sha256", 64),
            name="ck_registry_source_review_artifact_sha256",
        ),
        sa.CheckConstraint(
            _lowercase_hex_check("review_sha256", 64),
            name="ck_registry_source_review_sha256",
        ),
        sa.CheckConstraint(
            "artifact_size_bytes > 0",
            name="ck_registry_source_review_artifact_size",
        ),
        sa.CheckConstraint(
            "reviewer_kind = 'local-human'",
            name="ck_registry_source_review_reviewer",
        ),
    )
    op.create_index(
        "ix_comfy_registry_source_artifact_reviews_source_declaration_sha256",
        "comfy_registry_source_artifact_reviews",
        ["source_declaration_sha256"],
    )
    op.create_index(
        "ix_comfy_registry_source_artifact_reviews_artifact_id",
        "comfy_registry_source_artifact_reviews",
        ["artifact_id"],
    )
    op.create_index(
        "ix_comfy_registry_source_artifact_reviews_artifact_sha256",
        "comfy_registry_source_artifact_reviews",
        ["artifact_sha256"],
    )
    op.create_index(
        "ix_comfy_registry_source_artifact_reviews_review_sha256",
        "comfy_registry_source_artifact_reviews",
        ["review_sha256"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_comfy_registry_source_artifact_reviews_review_sha256",
        table_name="comfy_registry_source_artifact_reviews",
    )
    op.drop_index(
        "ix_comfy_registry_source_artifact_reviews_artifact_sha256",
        table_name="comfy_registry_source_artifact_reviews",
    )
    op.drop_index(
        "ix_comfy_registry_source_artifact_reviews_artifact_id",
        table_name="comfy_registry_source_artifact_reviews",
    )
    op.drop_index(
        "ix_comfy_registry_source_artifact_reviews_source_declaration_sha256",
        table_name="comfy_registry_source_artifact_reviews",
    )
    op.drop_table("comfy_registry_source_artifact_reviews")
