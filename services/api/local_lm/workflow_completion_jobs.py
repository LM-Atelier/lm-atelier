"""Keep workflow completion and its runtime result bound to one durable approval."""

from __future__ import annotations

from typing import Literal

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .comfy_registry_activation_batches import (
    deactivate_pending_registry_packages,
    registry_activation_pending,
)
from .domain import JobKind, JobStatus, utcnow
from .models import (
    ComfyRegistryInstall,
    Job,
    WorkflowInstallOffer,
    WorkflowInstallOfferDownload,
    WorkflowInstallOfferPackage,
)
from .progress import update_job_progress
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .schemas import ApiModel
from .workflow_offer_packages import accepted_workflow_offer_packages
from .workflow_package_install_plans import load_stored_workflow_package_install_plan


class WorkflowCompletionJobError(ValueError):
    def __init__(self, code: str = "workflow-completion-unavailable") -> None:
        self.code = code
        super().__init__("The accepted workflow installation cannot continue.")


class WorkflowCompletionResult(ApiModel):
    version: Literal[1] = 1
    approved_runtime_plan_sha256: str
    runtime_plan: RuntimeProvisioningPlan
    activation_id: str | None = None


def _payload(offer: WorkflowInstallOffer) -> dict[str, str | int | None]:
    return {
        "version": 1,
        "workflow_install_offer_id": offer.id,
        "workflow_install_offer_sha256": offer.offer_sha256,
        "source_plan_id": offer.source_plan_id,
    }


def stage_workflow_completion_job(session: Session, offer: WorkflowInstallOffer) -> Job:
    if offer.source_plan_id is None or offer.completion_job_id is not None:
        raise WorkflowCompletionJobError()
    session.flush()
    job = Job(
        kind=JobKind.WORKFLOW_INSTALL.value,
        status=JobStatus.QUEUED.value,
        phase="Waiting for workflow dependencies",
        payload_json=_payload(offer),
        result_json={},
    )
    session.add(job)
    session.flush()
    offer.completion_job_id = job.id
    return job


def workflow_completion_job(session: Session, offer: WorkflowInstallOffer) -> Job:
    job = session.get(Job, offer.completion_job_id) if offer.completion_job_id else None
    if (
        offer.source_plan_id is None
        or job is None
        or job.kind != JobKind.WORKFLOW_INSTALL.value
        or job.payload_json != _payload(offer)
        or job.run_id is not None
        or job.work_plan_id is not None
        or job.work_step_id is not None
        or job.status not in {item.value for item in JobStatus}
    ):
        raise WorkflowCompletionJobError()
    if job.result_json:
        result = WorkflowCompletionResult.model_validate(job.result_json)
        _record, saved = load_stored_workflow_package_install_plan(session, offer.source_plan_id)
        if (
            saved.plan_sha256 != offer.offer_sha256
            or saved.runtime_plan is None
            or result.approved_runtime_plan_sha256 != saved.runtime_plan.plan_sha256
            or result.runtime_plan.engine != "comfyui"
            or result.runtime_plan.operation == "install_managed"
            or result.runtime_plan.download_bytes != 0
            or result.runtime_plan.required_free_bytes != 0
            or (result.activation_id is not None) != (job.status == JobStatus.COMPLETE.value)
        ):
            raise WorkflowCompletionJobError()
    elif job.status == JobStatus.COMPLETE.value:
        raise WorkflowCompletionJobError()
    return job


def running_workflow_completion_job(
    session: Session, offer: WorkflowInstallOffer, attempt: int
) -> Job:
    job = workflow_completion_job(session, offer)
    if job.status != JobStatus.RUNNING.value or job.attempt != attempt or offer.status != "queued":
        raise WorkflowCompletionJobError()
    return job


def begin_workflow_completion(session: Session, offer: WorkflowInstallOffer) -> Job | None:
    job = workflow_completion_job(session, offer)
    if job.status not in {JobStatus.QUEUED.value, JobStatus.RUNNING.value}:
        return None
    job.status = JobStatus.RUNNING.value
    job.attempt += 1
    job.started_at = utcnow()
    job.completed_at = None
    job.error = None
    update_job_progress(job, stage="Preparing workflow runtime", indeterminate=True)
    return job


def defer_workflow_completion(session: Session, offer: WorkflowInstallOffer, attempt: int) -> None:
    job = running_workflow_completion_job(session, offer, attempt)
    job.status = JobStatus.QUEUED.value
    update_job_progress(job, stage="Waiting for the media worker", indeterminate=True)
    offer.completion_error_code = "workflow-review-required"


def record_workflow_runtime(
    session: Session,
    offer: WorkflowInstallOffer,
    attempt: int,
    approved: RuntimeProvisioningPlan,
    installed: RuntimeProvisioningPlan,
) -> None:
    job = running_workflow_completion_job(session, offer, attempt)
    _record, saved = load_stored_workflow_package_install_plan(session, offer.source_plan_id or "")
    if saved.runtime_plan != approved or saved.plan_sha256 != offer.offer_sha256:
        raise WorkflowCompletionJobError()
    result = WorkflowCompletionResult(
        approved_runtime_plan_sha256=approved.plan_sha256,
        runtime_plan=installed,
    )
    if job.result_json and WorkflowCompletionResult.model_validate(job.result_json) != result:
        raise WorkflowCompletionJobError()
    job.result_json = result.model_dump(mode="json")
    workflow_completion_job(session, offer)
    update_job_progress(job, stage="Preparing workflow dependencies", indeterminate=True)


def complete_workflow_job(
    session: Session, offer: WorkflowInstallOffer, attempt: int, activation_id: str
) -> None:
    job = workflow_completion_job(session, offer)
    if (
        job.status != JobStatus.RUNNING.value
        or job.attempt != attempt
        or offer.status != "completed"
    ):
        raise WorkflowCompletionJobError()
    result = WorkflowCompletionResult.model_validate(job.result_json)
    job.result_json = result.model_copy(update={"activation_id": activation_id}).model_dump(
        mode="json"
    )
    job.status = JobStatus.COMPLETE.value
    job.progress = 1.0
    job.completed_at = utcnow()
    update_job_progress(job, stage="Workflow installed", overall_progress=1.0)


def fail_workflow_completion(session: Session, offer: WorkflowInstallOffer, code: str) -> None:
    job = workflow_completion_job(session, offer)
    if job.status == JobStatus.RUNNING.value:
        job.status = JobStatus.FAILED.value
        job.error = "Workflow installation needs attention."
        job.completed_at = utcnow()
        update_job_progress(job, stage="Workflow installation needs attention", indeterminate=True)
        offer.completion_error_code = code


def cancel_workflow_completion(
    session: Session, offer: WorkflowInstallOffer, *, media_worker_stopped: bool = False
) -> Job | None:
    from .workflow_package_acceptance import accepted_workflow_package_jobs

    _source, saved, _downloads = accepted_workflow_package_jobs(session, offer)
    job = workflow_completion_job(session, offer)
    if offer.status != "queued" or job.status not in {
        JobStatus.QUEUED.value,
        JobStatus.RUNNING.value,
        JobStatus.PAUSED.value,
    }:
        return None
    job.status = JobStatus.CANCELLED.value
    job.completed_at = utcnow()
    job.error = None
    update_job_progress(job, stage="Workflow installation cancelled", indeterminate=True)
    offer.completion_error_code = "workflow-completion-unavailable"
    for package in accepted_workflow_offer_packages(session, offer, saved):
        if package.preparation is None and package.job.status in {"queued", "running", "paused"}:
            package.job.status = JobStatus.CANCELLED.value
            package.job.completed_at = utcnow()
            update_job_progress(
                package.job, stage="Extension preparation cancelled", indeterminate=True
            )
    if media_worker_stopped:
        _deactivate_cancelled_packages(session, offer)
    return job


def _deactivate_cancelled_packages(session: Session, offer: WorkflowInstallOffer) -> None:
    identifiers = session.scalars(
        select(WorkflowInstallOfferPackage.registry_install_id).where(
            WorkflowInstallOfferPackage.offer_id == offer.id,
            WorkflowInstallOfferPackage.offer_sha256 == offer.offer_sha256,
            WorkflowInstallOfferPackage.registry_install_id.is_not(None),
        )
    ).all()
    deactivate_pending_registry_packages(
        session, [identifier for identifier in identifiers if identifier is not None]
    )


def cancelled_workflow_activation_packages(session: Session) -> tuple[str, ...]:
    offers = session.scalars(
        select(WorkflowInstallOffer)
        .join(Job, Job.id == WorkflowInstallOffer.completion_job_id)
        .where(
            WorkflowInstallOffer.status == "queued",
            Job.kind == JobKind.WORKFLOW_INSTALL.value,
            Job.status == JobStatus.CANCELLED.value,
        )
    ).all()
    identifiers: set[str] = set()
    for offer in offers:
        try:
            workflow_completion_job(session, offer)
        except ValueError:
            continue
        rows = session.scalars(
            select(ComfyRegistryInstall)
            .join(
                WorkflowInstallOfferPackage,
                WorkflowInstallOfferPackage.registry_install_id == ComfyRegistryInstall.id,
            )
            .where(
                WorkflowInstallOfferPackage.offer_id == offer.id,
                WorkflowInstallOfferPackage.offer_sha256 == offer.offer_sha256,
                ComfyRegistryInstall.active.is_(True),
            )
        ).all()
        identifiers.update(row.id for row in rows if registry_activation_pending(row))
    return tuple(sorted(identifiers))


def deactivate_cancelled_workflow_activations(session: Session) -> None:
    """Clear cancelled pending batches after the caller has stopped the media worker."""
    session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
    session.expire_all()
    deactivate_pending_registry_packages(session, cancelled_workflow_activation_packages(session))


def retry_workflow_completion(
    session: Session, offer: WorkflowInstallOffer
) -> tuple[Job, list[Job]]:
    from .workflow_package_acceptance import accepted_workflow_package_jobs

    _payload, saved, downloads = accepted_workflow_package_jobs(session, offer)
    job = workflow_completion_job(session, offer)
    if offer.status != "queued" or job.status not in {
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.INTERRUPTED.value,
    }:
        raise WorkflowCompletionJobError()
    packages = accepted_workflow_offer_packages(session, offer, saved)
    for child in [*downloads, *(item.job for item in packages), job]:
        if child.status in {
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
            JobStatus.INTERRUPTED.value,
            JobStatus.PAUSED.value,
        }:
            child.status = JobStatus.QUEUED.value
            child.error = None
            child.started_at = None
            child.completed_at = None
            child.claim_owner = None
            child.claim_expires_at = None
            child.heartbeat_at = None
            child.enqueued_at = utcnow()
            update_job_progress(child, stage="Retry queued", indeterminate=True)
    job.attempt += 1
    offer.completion_error_code = None
    return job, downloads


def workflow_completion_offer(session: Session, job_id: str) -> WorkflowInstallOffer:
    offer = session.scalar(
        select(WorkflowInstallOffer).where(WorkflowInstallOffer.completion_job_id == job_id)
    )
    if offer is None:
        raise WorkflowCompletionJobError()
    workflow_completion_job(session, offer)
    return offer


def workflow_download_jobs(session: Session, offer: WorkflowInstallOffer) -> list[Job]:
    return list(
        session.scalars(
            select(Job)
            .join(WorkflowInstallOfferDownload, WorkflowInstallOfferDownload.job_id == Job.id)
            .where(WorkflowInstallOfferDownload.offer_id == offer.id)
        )
    )


def workflow_download_can_cancel(session: Session, offer_id: str, job_id: str) -> bool:
    """Cancel a dependency only while its requesting installation is its last consumer."""
    offer = session.get(WorkflowInstallOffer, offer_id)
    if offer is None or workflow_completion_job(session, offer).status != JobStatus.CANCELLED.value:
        return False
    if job_id not in {job.id for job in workflow_download_jobs(session, offer)}:
        return False
    consumers = session.scalars(
        select(WorkflowInstallOffer)
        .join(
            WorkflowInstallOfferDownload,
            WorkflowInstallOfferDownload.offer_id == WorkflowInstallOffer.id,
        )
        .where(
            WorkflowInstallOfferDownload.job_id == job_id,
            WorkflowInstallOffer.id != offer_id,
            WorkflowInstallOffer.status == "queued",
        )
    ).all()
    for consumer in consumers:
        if consumer.completion_job_id is None:
            return False
        try:
            if workflow_completion_job(session, consumer).status != JobStatus.CANCELLED.value:
                return False
        except ValueError:
            return False
    return True


def recover_workflow_completion_jobs(
    session: Session, *, media_worker_stopped: bool = False
) -> list[tuple[str, str]]:

    session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
    cancellations: list[tuple[str, str]] = []
    offers = session.scalars(
        select(WorkflowInstallOffer).where(
            WorkflowInstallOffer.status == "queued",
            WorkflowInstallOffer.completion_job_id.is_not(None),
        )
    ).all()
    for offer in offers:
        job = None
        try:
            with session.begin_nested():
                job = workflow_completion_job(session, offer)
                if job.status == JobStatus.RUNNING.value:
                    job.status = JobStatus.INTERRUPTED.value
                if job.status == JobStatus.INTERRUPTED.value:
                    retry_workflow_completion(session, offer)
                elif job.status == JobStatus.CANCELLED.value:
                    if media_worker_stopped:
                        _deactivate_cancelled_packages(session, offer)
                    cancellations.extend(
                        (offer.id, child.id)
                        for child in workflow_download_jobs(session, offer)
                        if child.status not in {"complete", "failed", "cancelled"}
                    )
        except ValueError:
            offer.completion_error_code = "workflow-completion-unavailable"
            if job is not None and job.status in {"queued", "running", "interrupted"}:
                job.status = JobStatus.FAILED.value
                job.error = "Workflow installation needs attention."
                job.completed_at = utcnow()
                update_job_progress(
                    job, stage="Workflow installation needs attention", indeterminate=True
                )
    session.flush()
    return [
        (offer_id, child_id)
        for offer_id, child_id in cancellations
        if workflow_download_can_cancel(session, offer_id, child_id)
    ]
