"""Read accepted installation history without granting current workflow readiness."""

from __future__ import annotations

from collections import Counter
from typing import get_args

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Job, WorkflowInstallOffer, WorkflowInstallOfferDownload
from .schemas import (
    WorkflowInstallAttentionCode,
    WorkflowInstallPhase,
    WorkflowInstallProgressOut,
    WorkflowInstallStatus,
)

_ATTENTION_CODES = frozenset(get_args(WorkflowInstallAttentionCode))
_ATTENTION_ADAPTER: TypeAdapter[WorkflowInstallAttentionCode] = TypeAdapter(
    WorkflowInstallAttentionCode
)
_STATUS_ADAPTER: TypeAdapter[WorkflowInstallStatus] = TypeAdapter(WorkflowInstallStatus)


def completion_attention_code(value: object) -> WorkflowInstallAttentionCode:
    """Persist only known content-free codes, never exception text or arbitrary values."""
    if isinstance(value, str):
        if value in _ATTENTION_CODES:
            return _ATTENTION_ADAPTER.validate_python(value)
        if value in {"workflow_review_unavailable", "workflow-revision-needs-attention"}:
            return "workflow-review-required"
    return "workflow-completion-unavailable"


def workflow_install_progress(
    session: Session, offer: WorkflowInstallOffer
) -> WorkflowInstallProgressOut:
    rows = session.execute(
        select(WorkflowInstallOfferDownload.job_id, Job.kind, Job.status)
        .outerjoin(Job, Job.id == WorkflowInstallOfferDownload.job_id)
        .where(WorkflowInstallOfferDownload.offer_id == offer.id)
    ).all()
    counts = Counter(
        status
        if kind == "download"
        and status in {"queued", "running", "paused", "complete", "failed", "cancelled"}
        else "unavailable"
        for _, kind, status in rows
    )
    unavailable = counts["unavailable"] + max(0, offer.plan_count - len(rows))
    code = (
        completion_attention_code(offer.completion_error_code)
        if offer.completion_error_code
        else None
    )
    phase: WorkflowInstallPhase
    if offer.status == "completed":
        # Historical completion is not a claim that an activation is still usable.
        phase = "completed"
        if code != "workflow-media-restore-failed":
            code = None
    elif offer.status in {"invalidated", "expired"}:
        phase = "invalidated" if offer.status == "invalidated" else "expired"
    elif offer.status == "ready":
        phase = "ready"
        unavailable = 0
    elif (
        code == "workflow-extension-review-required"
        and offer.completion_job_id is not None
        and (completion := session.get(Job, offer.completion_job_id)) is not None
        and completion.kind == "workflow_install"
        and completion.status == "paused"
    ):
        phase = "paused"
    elif (
        code
        or unavailable
        or len(rows) != offer.plan_count
        or counts["failed"]
        or counts["cancelled"]
    ):
        phase = "needs_attention"
        if code is None:
            code = (
                "workflow-download-failed"
                if counts["failed"] or counts["cancelled"]
                else "download-acceptance-unavailable"
            )
    elif counts["queued"] or counts["running"]:
        phase = "downloading"
    elif counts["paused"]:
        phase = "paused"
    else:
        phase = "verifying"
    return WorkflowInstallProgressOut(
        id=offer.id,
        workflow_revision_id=offer.workflow_revision_id,
        status=_STATUS_ADAPTER.validate_python(offer.status),
        phase=phase,
        total_downloads=offer.plan_count,
        completed_downloads=counts["complete"],
        failed_downloads=counts["failed"],
        cancelled_downloads=counts["cancelled"],
        paused_downloads=counts["paused"],
        pending_downloads=counts["queued"] + counts["running"],
        unavailable_downloads=unavailable,
        attention_code=code,
    )


def latest_workflow_install_progress(
    session: Session, revision_id: str
) -> WorkflowInstallProgressOut | None:
    offer = session.scalar(
        select(WorkflowInstallOffer)
        .where(
            WorkflowInstallOffer.workflow_revision_id == revision_id,
            WorkflowInstallOffer.queued_at.is_not(None),
        )
        .order_by(WorkflowInstallOffer.created_at.desc(), WorkflowInstallOffer.id.desc())
        .limit(1)
    )
    return workflow_install_progress(session, offer) if offer is not None else None
