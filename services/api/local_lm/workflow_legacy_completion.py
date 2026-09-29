"""Queue accepted workflow activation after its downloads settle."""

from __future__ import annotations

from sqlalchemy import text

from .db import SessionLocal
from .models import WorkflowInstallOffer
from .workflow_completion_jobs import stage_workflow_completion_job, workflow_completion_job
from .workflow_offer_completion import (
    WorkflowOfferCompletionError,
    accepted_workflow_offer_downloads,
)

RESOLVABLE_COMPLETION_CODES = frozenset(
    {
        "workflow-review-required",
        "workflow-dependencies-need-selection",
        "download-results-need-binding",
    }
)


def prepare_legacy_completion(offer_id: str) -> str | None:
    """Wait for dependencies before occupying the installation compute queue."""
    with SessionLocal() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        offer = session.get(WorkflowInstallOffer, offer_id)
        if offer is None or offer.status != "queued" or offer.source_plan_id is not None:
            return None
        job = (
            workflow_completion_job(session, offer)
            if offer.completion_job_id is not None
            else stage_workflow_completion_job(session, offer)
        )
        if job.status == "paused" and offer.completion_error_code in RESOLVABLE_COMPLETION_CODES:
            job.status = "queued"
        if job.status != "queued" or job.claim_owner is not None:
            return None
        try:
            downloads = accepted_workflow_offer_downloads(session, offer)
        except ValueError:
            # Let the claimed transaction report invalid approvals or associations.
            session.commit()
            return job.id
        if any(child.status in {"failed", "cancelled"} for child, _, _ in downloads):
            session.commit()
            raise WorkflowOfferCompletionError("workflow-download-failed")
        if any(child.status != "complete" for child, _, _ in downloads):
            job.queue_group = None
            session.commit()
            return None
        session.commit()
        return job.id
