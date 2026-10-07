"""Accept a saved source and stage its draft and downloads in one transaction."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .domain import JobKind
from .model_planner import workflow_artifact_contract
from .models import (
    Job,
    WorkflowDefinition,
    WorkflowInstallOffer,
    WorkflowInstallOfferDownload,
    WorkflowRevision,
)
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .workflow_completion_jobs import stage_workflow_completion_job, workflow_completion_job
from .workflow_install_offers import (
    bind_workflow_offer_downloads,
    mark_workflow_install_offer_queued,
)
from .workflow_offer_packages import accepted_workflow_offer_packages, stage_workflow_offer_packages
from .workflow_package_drafts import (
    canonical_package_graph,
    stage_workflow_package_draft,
    workflow_package_draft_dependencies,
    workflow_package_draft_identity,
)
from .workflow_package_install_plans import (
    WorkflowPackageInstallPlanError,
    WorkflowPackageInstallPlanOut,
    WorkflowPackageInstallPlanRequest,
    load_stored_workflow_package_install_plan,
    revalidate_workflow_package_install_plan,
    verify_workflow_package_download_plans,
)
from .workflow_runtime_nodes import WorkflowRuntimeNodeInventory
from .workflow_trust import canonical_graph

if TYPE_CHECKING:
    from .downloads import DownloadManager


def source_workflow_install_offer(session: Session, identifier: str) -> WorkflowInstallOffer | None:
    """Resolve either the accepted offer or the source plan shown before approval."""

    return session.scalar(
        select(WorkflowInstallOffer).where(
            (WorkflowInstallOffer.id == identifier)
            | (WorkflowInstallOffer.source_plan_id == identifier)
        )
    )


def _refuse_changed() -> WorkflowPackageInstallPlanError:
    return WorkflowPackageInstallPlanError(
        "workflow-package-installation-changed", "The accepted workflow installation changed."
    )


def _accepted_jobs(
    session: Session,
    offer: WorkflowInstallOffer,
    saved: WorkflowPackageInstallPlanOut,
    payload: WorkflowPackageInstallPlanRequest,
) -> list[Job]:
    verify_workflow_package_download_plans(session, saved)
    if (
        offer.status not in {"queued", "completed"}
        or offer.queued_at is None
        or offer.source_plan_id != saved.id
        or offer.offer_sha256 != saved.plan_sha256
        or offer.binding_plan_sha256 != saved.asset_binding_sha256
        or offer.assets_json != [item.model_dump(mode="json") for item in saved.assets]
        or offer.selections_json != [item.model_dump(mode="json") for item in payload.selections]
        or offer.plan_count != len(saved.download_requests)
        or offer.total_bytes != saved.total_download_bytes
    ):
        raise _refuse_changed()
    workflow_id, draft_id, digest = workflow_package_draft_identity(
        canonical_package_graph(payload.ui_graph)
    )
    definition = session.get(WorkflowDefinition, workflow_id)
    revision = session.get(WorkflowRevision, offer.workflow_revision_id)
    if (
        definition is None
        or revision is None
        or (offer.status == "queued" and definition.current_revision_id != revision.id)
        or definition.operation != payload.operation.value
        or revision.workflow_id != workflow_id
        or revision.engine != "comfyui"
        or revision.ui_graph_json != payload.ui_graph
        or revision.artifact_sha256 != offer.workflow_artifact_sha256
        or offer.workflow_artifact_sha256
        != workflow_artifact_contract(
            operation=payload.operation.value,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
    ):
        raise _refuse_changed()
    if revision.id == draft_id:
        if (
            offer.status == "completed"
            or revision.trusted
            or revision.api_graph_json != {}
            or revision.input_schema_json != {}
            or revision.dependency_contract_sha256 is not None
            or revision.dependencies_json != workflow_package_draft_dependencies(digest)
            or offer.dependency_contract_sha256 != saved.dependency_contract_sha256
        ):
            raise _refuse_changed()
    else:
        from .revision_dependency_contract import declared_dependency_contract_sha256
        from .workflow_source_dependencies import compiled_workflow_source_dependencies

        dependencies = compiled_workflow_source_dependencies(session, offer, saved, payload)
        if (
            not revision.api_graph_json
            or revision.dependencies_json != dependencies
            or revision.dependency_contract_sha256
            != declared_dependency_contract_sha256(dependencies)
            or offer.dependency_contract_sha256 != revision.dependency_contract_sha256
        ):
            raise _refuse_changed()
    links = list(
        session.scalars(
            select(WorkflowInstallOfferDownload).where(
                WorkflowInstallOfferDownload.offer_id == offer.id
            )
        )
    )
    expected = {
        canonical_graph(item.model_dump(mode="json")): item for item in saved.download_requests
    }
    if len(links) != len(expected) or len(expected) != offer.plan_count:
        raise _refuse_changed()
    jobs: list[Job] = []
    for link in links:
        encoded = canonical_graph(link.request_json)
        job = session.get(Job, link.job_id) if link.job_id is not None else None
        if (
            encoded not in expected
            or link.offer_sha256 != offer.offer_sha256
            or link.request_sha256 != hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            or job is None
            or job.kind != JobKind.DOWNLOAD.value
            or job.payload_json != link.request_json
        ):
            raise _refuse_changed()
        del expected[encoded]
        jobs.append(job)
    accepted_workflow_offer_packages(session, offer, saved)
    return sorted(jobs, key=lambda job: job.id)


def accepted_workflow_package_jobs(
    session: Session, offer: WorkflowInstallOffer
) -> tuple[WorkflowPackageInstallPlanRequest, WorkflowPackageInstallPlanOut, list[Job]]:
    """Verify the accepted source and requests without requiring downloads to remain pending."""

    if offer.source_plan_id is None:
        raise _refuse_changed()
    record, saved = load_stored_workflow_package_install_plan(session, offer.source_plan_id)
    payload = WorkflowPackageInstallPlanRequest.model_validate(record.request_json)
    return payload, saved, _accepted_jobs(session, offer, saved, payload)


def assert_compiled_workflow_package_offer(
    session: Session, offer: WorkflowInstallOffer, revision: WorkflowRevision
) -> None:
    payload, _saved, _jobs = accepted_workflow_package_jobs(session, offer)
    _workflow_id, draft_id, _digest = workflow_package_draft_identity(
        canonical_package_graph(payload.ui_graph)
    )
    if revision.id != offer.workflow_revision_id or revision.id == draft_id:
        raise _refuse_changed()


def accept_workflow_package_install_plan(
    session: Session,
    plan_id: str,
    manager: DownloadManager,
    *,
    available_node_types: Collection[str] | None,
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
    runtime_plan: RuntimeProvisioningPlan | None = None,
    runtime_node_inventory: WorkflowRuntimeNodeInventory | None = None,
) -> tuple[WorkflowInstallOffer, list[Job]]:
    """Stage one accepted plan before the caller commits and starts any workers."""

    # Reserve the SQLite writer before rereading the plan and existing acceptance.
    # No await occurs until the caller has committed or rolled back this transaction.
    session.execute(text("UPDATE workflow_package_install_plans SET id = id WHERE 0"))
    session.expire_all()
    record, saved = load_stored_workflow_package_install_plan(session, plan_id)
    payload = WorkflowPackageInstallPlanRequest.model_validate(record.request_json)
    previous = session.scalar(
        select(WorkflowInstallOffer).where(WorkflowInstallOffer.source_plan_id == plan_id)
    )
    if previous is not None:
        return previous, [
            *_accepted_jobs(session, previous, saved, payload),
            workflow_completion_job(session, previous),
        ]

    saved = revalidate_workflow_package_install_plan(
        session,
        plan_id,
        available_node_types=available_node_types,
        available_asset_filenames=available_asset_filenames,
        installed_package_versions=installed_package_versions,
        extension_execution=saved.extension_execution,
        runtime_plan=runtime_plan,
        runtime_node_inventory=runtime_node_inventory,
    )
    if (
        not saved.can_accept
        or saved.dependency_contract_sha256 is None
        or saved.total_download_bytes is None
    ):
        raise WorkflowPackageInstallPlanError(
            "workflow-package-install-plan-incomplete",
            "Resolve the workflow installation blockers first.",
        )
    _workflow_id, draft_id, _digest = workflow_package_draft_identity(
        canonical_package_graph(payload.ui_graph)
    )
    if (
        session.scalar(
            select(WorkflowInstallOffer.id)
            .where(
                WorkflowInstallOffer.workflow_revision_id == draft_id,
                WorkflowInstallOffer.status.in_(("queued", "completed")),
            )
            .limit(1)
        )
        is not None
    ):
        raise WorkflowPackageInstallPlanError(
            "workflow-package-installation-in-progress",
            "This workflow already has an accepted installation.",
        )
    validated = [manager.validated_request(session, item) for item in saved.download_requests]
    if [item.model_dump(mode="json") for item in validated] != [
        item.model_dump(mode="json") for item in saved.download_requests
    ]:
        raise _refuse_changed()
    definition, revision = stage_workflow_package_draft(session, payload)
    if (
        definition.current_revision_id != revision.id
        or revision.trusted
        or revision.api_graph_json != {}
        or revision.input_schema_json != {}
        or revision.dependency_contract_sha256 is not None
        or revision.artifact_sha256
        != workflow_artifact_contract(
            operation=definition.operation,
            engine="comfyui",
            api_graph={},
            input_schema={},
            dependencies=revision.dependencies_json,
        )
    ):
        raise _refuse_changed()
    # The accepted operation may differ from the earlier preview's initial choice.
    # Only an unaccepted, non-executable source draft can be updated in place.
    definition.operation = payload.operation.value
    revision.artifact_sha256 = workflow_artifact_contract(
        operation=definition.operation,
        engine="comfyui",
        api_graph={},
        input_schema={},
        dependencies=revision.dependencies_json,
    )
    offer = WorkflowInstallOffer(
        source_plan_id=plan_id,
        workflow_revision_id=revision.id,
        workflow_artifact_sha256=revision.artifact_sha256,
        dependency_contract_sha256=saved.dependency_contract_sha256,
        binding_plan_sha256=saved.asset_binding_sha256,
        offer_sha256=saved.plan_sha256,
        selections_json=[item.model_dump(mode="json") for item in payload.selections],
        assets_json=[item.model_dump(mode="json") for item in saved.assets],
        plan_count=len(validated),
        total_bytes=saved.total_download_bytes,
        status="ready",
    )
    session.add(offer)
    jobs = [manager.stage(session, item) for item in validated]
    bind_workflow_offer_downloads(session, offer, jobs)
    stage_workflow_offer_packages(session, offer, saved)
    completion = stage_workflow_completion_job(session, offer)
    mark_workflow_install_offer_queued(offer)
    return offer, [*sorted(jobs, key=lambda job: job.id), completion]
