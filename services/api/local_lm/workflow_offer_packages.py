"""Bind accepted extension plans and preparation results in the offer's transaction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_registry_activation_batches import _row
from .comfy_registry_lifecycle import ComfyRegistryPreparation
from .comfy_registry_sources import resolve_comfy_package_source
from .domain import JobKind, utcnow
from .models import ComfyRegistryInstall, Job, WorkflowInstallOffer, WorkflowInstallOfferPackage
from .workflow_package_execution_plan import WorkflowPackageExecutionPlan
from .workflow_package_install_plans import (
    WorkflowPackageInstallPlanOut,
    load_stored_workflow_package_install_plan,
)


class WorkflowOfferPackageError(ValueError):
    def __init__(self) -> None:
        self.code = "workflow-package-preparation-changed"
        super().__init__("The accepted extension preparation changed.")


@dataclass(frozen=True)
class AcceptedWorkflowPackage:
    link: WorkflowInstallOfferPackage
    job: Job
    plan: WorkflowPackageExecutionPlan
    preparation: ComfyRegistryPreparation | None


def _payload(
    offer: WorkflowInstallOffer, package_id: str, plan: WorkflowPackageExecutionPlan
) -> dict[str, str]:
    return {
        "workflow_install_offer_id": offer.id,
        "workflow_install_offer_sha256": offer.offer_sha256,
        "package_id": package_id,
        "execution_plan_sha256": plan.plan_sha256,
    }


def _snapshot(preparation: ComfyRegistryPreparation) -> dict[str, Any]:
    return {"version": 1, "preparation": asdict(preparation)}


def _preparation(value: dict[str, Any]) -> ComfyRegistryPreparation:
    raw = value.get("preparation")
    if (
        set(value) != {"version", "preparation"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or not isinstance(raw, dict)
        or set(raw) != {field.name for field in fields(ComfyRegistryPreparation)}
        or type(raw.get("reused_wheel_environment")) is not bool
        or any(
            not isinstance(item, str)
            for key, item in raw.items()
            if key != "reused_wheel_environment"
        )
    ):
        raise WorkflowOfferPackageError()
    return ComfyRegistryPreparation(**raw)


def _existing(
    session: Session, plan: WorkflowPackageExecutionPlan
) -> ComfyRegistryPreparation | None:
    source = resolve_comfy_package_source(plan.resolution)
    row = session.scalar(
        select(ComfyRegistryInstall).where(
            ComfyRegistryInstall.registry_record_id == source.source_record_id
        )
    )
    if row is None:
        return None
    if (
        not row.wheel_environment_path
        or not row.wheel_closure_sha256
        or not row.wheel_environment_sha256
    ):
        raise WorkflowOfferPackageError()
    prepared = ComfyRegistryPreparation(
        row.id,
        row.installed_path,
        row.wheel_environment_path,
        row.archive_sha256,
        row.manifest_sha256,
        row.wheel_closure_sha256,
        row.wheel_environment_sha256,
        True,
    )
    _row(session, prepared, plan)
    return prepared


def _identity(
    session: Session, offer: WorkflowInstallOffer, saved: WorkflowPackageInstallPlanOut
) -> None:
    _record, stored = load_stored_workflow_package_install_plan(session, saved.id)
    if stored != saved:
        raise WorkflowOfferPackageError()
    if offer.source_plan_id != saved.id or offer.offer_sha256 != saved.plan_sha256:
        raise WorkflowOfferPackageError()
    for package_id, plan in saved.extension_execution.plans.items():
        plan.verify()
        if plan.resolution.package_id != package_id:
            raise WorkflowOfferPackageError()


def stage_workflow_offer_packages(
    session: Session, offer: WorkflowInstallOffer, saved: WorkflowPackageInstallPlanOut
) -> None:
    """Stage requests and any already-prepared exact identities before approval commits."""
    _identity(session, offer, saved)
    session.flush()
    if session.scalar(
        select(WorkflowInstallOfferPackage.id).where(
            WorkflowInstallOfferPackage.offer_id == offer.id
        )
    ):
        raise WorkflowOfferPackageError()
    for package_id, plan in sorted(saved.extension_execution.plans.items()):
        prepared = _existing(session, plan)
        snapshot = _snapshot(prepared) if prepared else {}
        job = Job(
            kind=JobKind.REGISTRY_PREPARE.value,
            status="complete" if prepared else "queued",
            phase="prepared" if prepared else "queued",
            payload_json=_payload(offer, package_id, plan),
            result_json=snapshot,
            progress=1.0 if prepared else 0.0,
            completed_at=utcnow() if prepared else None,
        )
        session.add(job)
        session.flush()
        session.add(
            WorkflowInstallOfferPackage(
                offer_id=offer.id,
                offer_sha256=offer.offer_sha256,
                package_id=package_id,
                execution_plan_sha256=plan.plan_sha256,
                job_id=job.id,
                registry_install_id=prepared.install_id if prepared else None,
                preparation_json=snapshot,
            )
        )
    session.flush()


def accepted_workflow_offer_packages(
    session: Session, offer: WorkflowInstallOffer, saved: WorkflowPackageInstallPlanOut
) -> tuple[AcceptedWorkflowPackage, ...]:
    """Read only the exact jobs and install identities retained by this acceptance."""
    _identity(session, offer, saved)
    plans = saved.extension_execution.plans
    links = list(
        session.scalars(
            select(WorkflowInstallOfferPackage).where(
                WorkflowInstallOfferPackage.offer_id == offer.id
            )
        )
    )
    if len(links) != len(plans) or {link.package_id for link in links} != set(plans):
        raise WorkflowOfferPackageError()
    found = []
    for link in sorted(links, key=lambda item: item.package_id):
        plan = plans[link.package_id]
        job = session.get(Job, link.job_id) if link.job_id else None
        if (
            link.offer_sha256 != offer.offer_sha256
            or link.execution_plan_sha256 != plan.plan_sha256
            or job is None
            or job.kind != JobKind.REGISTRY_PREPARE.value
            or job.payload_json != _payload(offer, link.package_id, plan)
            or job.result_json != link.preparation_json
            or job.status
            not in {"queued", "running", "paused", "failed", "cancelled", "complete", "interrupted"}
        ):
            raise WorkflowOfferPackageError()
        prepared = _preparation(link.preparation_json) if link.preparation_json else None
        if prepared is None:
            if link.registry_install_id is not None or job.status == "complete":
                raise WorkflowOfferPackageError()
        else:
            if link.registry_install_id != prepared.install_id or job.status != "complete":
                raise WorkflowOfferPackageError()
            _row(session, prepared, plan)
        found.append(AcceptedWorkflowPackage(link, job, plan, prepared))
    return tuple(found)


def record_workflow_offer_package(
    session: Session,
    offer: WorkflowInstallOffer,
    saved: WorkflowPackageInstallPlanOut,
    *,
    package_id: str,
    job_id: str,
    preparation: ComfyRegistryPreparation,
) -> None:
    """Stage the exact result in the transaction that persists the inactive package."""
    accepted = {
        item.link.package_id: item
        for item in accepted_workflow_offer_packages(session, offer, saved)
    }
    item = accepted.get(package_id)
    if (
        offer.status != "queued"
        or item is None
        or item.job.id != job_id
        or item.job.status != "running"
        or item.preparation is not None
    ):
        raise WorkflowOfferPackageError()
    row = _row(session, preparation, item.plan)
    if row.trusted or row.active:
        raise WorkflowOfferPackageError()
    snapshot = _snapshot(preparation)
    item.link.registry_install_id = preparation.install_id
    item.link.preparation_json = snapshot
    item.job.result_json = snapshot
    item.job.status = "complete"
    item.job.phase = "prepared"
    item.job.progress = 1.0
    item.job.error = None
    item.job.completed_at = utcnow()
    session.flush()
