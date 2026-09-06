"""Explicit stored workflow review state for tests of downstream behavior."""

from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision


def seed_workflow_trust(revision_id: str) -> None:
    """Seed existing review state without using a public creation request."""
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.trusted = True
        session.commit()
