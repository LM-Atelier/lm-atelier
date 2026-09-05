"""Authorize artifact deletes already proven against the complete reference graph.

Revision ID: f5c2a8d91e40
Revises: e4b7d1c5a960
Create Date: 2026-09-04
"""

from __future__ import annotations

from alembic import op

from local_lm.artifact_library_schema import (
    ARTIFACT_JSON_DELETE_TRIGGER,
    ARTIFACT_JSON_DELETE_TRIGGER_UNPROVEN,
)

revision: str = "f5c2a8d91e40"
down_revision: str | None = "e4b7d1c5a960"
branch_labels: tuple[str, ...] | None = None
depends_on: tuple[str, ...] | None = None

_DROP = "DROP TRIGGER IF EXISTS artifact_json_reference_delete_guard"


def upgrade() -> None:
    op.execute(_DROP)
    op.execute(ARTIFACT_JSON_DELETE_TRIGGER)


def downgrade() -> None:
    op.execute(_DROP)
    op.execute(ARTIFACT_JSON_DELETE_TRIGGER_UNPROVEN)
