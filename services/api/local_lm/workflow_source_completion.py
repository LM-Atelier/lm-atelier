"""Compile an accepted source and commit its reviewed executable activation together."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from .adapters.base import MediaAdapter
from .comfy_registry_paths import registry_wheel_environment_root
from .comfy_workflow_compiler import compile_comfyui_ui_graph
from .config import Settings
from .db import SessionLocal
from .models import WorkflowActivation, WorkflowDefinition, WorkflowInstallOffer, WorkflowRevision
from .processes import ProcessSupervisor
from .progress import update_job_progress
from .runtime_provisioning import RuntimeProvisioningError
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .schemas import WorkflowRevisionCreate
from .workflow_activations import (
    materialize_comfy_runtime_dependency,
    revalidate_workflow_activation,
)
from .workflow_completion_jobs import (
    WorkflowCompletionResult,
    begin_workflow_completion,
    complete_workflow_job,
    fail_workflow_completion,
    record_workflow_runtime,
    running_workflow_completion_job,
    workflow_completion_job,
)
from .workflow_offer_completion import (
    WorkflowOfferCompletionError,
    _accepted_results,
    complete_workflow_install_offer,
)
from .workflow_package_acceptance import accepted_workflow_package_jobs
from .workflow_package_drafts import canonical_package_graph, workflow_package_draft_identity
from .workflow_package_inputs import prepare_workflow_package_compilation
from .workflow_package_install_plans import WorkflowPackageInstallPlanRequest
from .workflow_package_runtime import WorkflowPackageRestorationError, workflow_package_runtime
from .workflow_review_runtime import (
    VerifiedReviewedPackages,
    _reviewed_package_inputs,
    _runtime_id,
    review_runtime_object_info,
    verify_reviewed_packages,
)
from .workflow_revision_reviews import ReviewSnapshot, build_review_snapshot, record_review
from .workflow_revision_writes import build_workflow_revision, stage_workflow_revision
from .workflow_source_dependencies import compiled_workflow_source_dependencies
from .workflow_source_launch import require_workflow_source_resource_match
from .workflow_source_runtime import (
    PreparedWorkflowSourceRuntime,
    prepared_workflow_source_runtime,
    require_workflow_source_attempt,
)


@dataclass(frozen=True)
class AcceptedSource:
    payload: WorkflowPackageInstallPlanRequest
    plan_sha256: str
    draft_id: str
    results: list[set[tuple[str, str]]]
    runtime_plan: RuntimeProvisioningPlan | None
    offer_id: str
    job_id: str
    attempt: int


def _accepted_source(session: Session, offer_id: str) -> AcceptedSource | None:
    offer = session.get(WorkflowInstallOffer, offer_id)
    if offer is None or offer.status != "queued" or offer.source_plan_id is None:
        return None
    payload, saved, _jobs = accepted_workflow_package_jobs(session, offer)
    _workflow_id, draft_id, _digest = workflow_package_draft_identity(
        canonical_package_graph(payload.ui_graph)
    )
    if offer.workflow_revision_id != draft_id:
        raise WorkflowOfferCompletionError("workflow-install-offer-changed")
    results = _accepted_results(session, offer)
    if results is None:
        return None
    job = workflow_completion_job(session, offer)
    return AcceptedSource(
        payload,
        saved.plan_sha256,
        draft_id,
        results,
        saved.runtime_plan,
        offer.id,
        job.id,
        job.attempt,
    )


def _require_runtime_plan(processes: ProcessSupervisor, source: AcceptedSource) -> None:
    provisioner = processes.runtimes
    if provisioner is None or source.runtime_plan is None:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-unavailable")
    try:
        current = provisioner.preflight("comfyui")
    except (OSError, RuntimeProvisioningError) as exc:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-unavailable") from exc
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, source.offer_id)
        if offer is None:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        job = running_workflow_completion_job(session, offer, source.attempt)
        result = WorkflowCompletionResult.model_validate(job.result_json)
    if current != result.runtime_plan or current.operation == "install_managed":
        raise WorkflowOfferCompletionError("workflow-runtime-plan-changed")


def _read_source(offer_id: str) -> AcceptedSource | None:
    with SessionLocal() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        offer = session.get(WorkflowInstallOffer, offer_id)
        if offer is None or offer.status != "queued" or offer.source_plan_id is None:
            return None
        job = workflow_completion_job(session, offer)
        if (
            job.status == "paused"
            and offer.completion_error_code == "workflow-extension-review-required"
        ):
            job.status = "queued"
            offer.completion_error_code = None
        if job.status not in {"queued", "running"}:
            return None
        try:
            source = _accepted_source(session, offer_id)
        except WorkflowOfferCompletionError as exc:
            if exc.code == "workflow-download-failed":
                begin_workflow_completion(session, offer)
                fail_workflow_completion(session, offer, exc.code)
                session.commit()
            raise
        if source is None:
            return None
        started = begin_workflow_completion(session, offer)
        if started is None:
            return None
        attempt = started.attempt
        session.commit()
        return replace(source, attempt=attempt)


async def _prepare_runtime(processes: ProcessSupervisor, source: AcceptedSource) -> None:
    await asyncio.to_thread(require_workflow_source_attempt, source.offer_id, source.attempt)
    provisioner = processes.runtimes
    if provisioner is None or source.runtime_plan is None:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-unavailable")
    try:
        await provisioner.provision("comfyui", expected_plan=source.runtime_plan)
        installed = await asyncio.to_thread(provisioner.preflight, "comfyui")
    except (OSError, RuntimeProvisioningError) as exc:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-changed") from exc

    def record() -> None:
        with SessionLocal() as session:
            session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
            if _accepted_source(session, source.offer_id) != source:
                raise WorkflowOfferCompletionError("workflow-install-offer-changed")
            offer = session.get(WorkflowInstallOffer, source.offer_id)
            if offer is None or source.runtime_plan is None:
                raise WorkflowOfferCompletionError("workflow-install-offer-changed")
            record_workflow_runtime(session, offer, source.attempt, source.runtime_plan, installed)
            session.commit()

    await asyncio.to_thread(record)


def _compile(source: AcceptedSource, info: dict[str, Any]) -> WorkflowRevisionCreate:
    payload = source.payload
    prepared = prepare_workflow_package_compilation(payload.ui_graph, info, payload.operation)
    compilation = compile_comfyui_ui_graph(prepared.ui_graph, prepared.object_info)
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, source.offer_id)
        if offer is None or _accepted_source(session, source.offer_id) != source:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        _payload, saved, _jobs = accepted_workflow_package_jobs(session, offer)
        dependencies = compiled_workflow_source_dependencies(session, offer, saved, payload)
    return WorkflowRevisionCreate(
        ui_graph=payload.ui_graph,
        api_graph=prepared.bind(compilation.api_graph),
        input_schema=prepared.input_schema,
        dependencies=dependencies,
    )


def _definition(session: Session, source: AcceptedSource) -> WorkflowDefinition:
    draft = session.get(WorkflowRevision, source.draft_id)
    definition = session.get(WorkflowDefinition, draft.workflow_id) if draft else None
    if definition is None or definition.current_revision_id != source.draft_id:
        raise WorkflowOfferCompletionError("workflow-install-offer-changed")
    return definition


def _finish(
    settings: Settings,
    processes: ProcessSupervisor,
    offer_id: str,
    source: AcceptedSource,
    payload: WorkflowRevisionCreate,
    info: dict[str, Any],
    snapshot: ReviewSnapshot,
    worker_id: int,
    prepared: PreparedWorkflowSourceRuntime,
    registry_verification: VerifiedReviewedPackages | None = None,
) -> str:
    _require_runtime_plan(processes, source)
    batch = prepared.batch.verify_completion(SessionLocal) if prepared.batch is not None else None
    with SessionLocal() as session:
        # Reserve the writer before rereading consent and downloaded resources.
        # Compilation I/O is finished; all staged rows commit with the activation.
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        current = _accepted_source(session, offer_id)
        if (
            current != source
            or _runtime_id(processes) != worker_id
            or processes.launch_scope_sha256("media") != prepared.scope.launch_sha256
        ):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        accepted = session.get(WorkflowInstallOffer, offer_id)
        if accepted is None:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        running_workflow_completion_job(session, accepted, source.attempt)
        definition = _definition(session, source)
        revision = stage_workflow_revision(session, definition, payload)
        fresh = build_review_snapshot(session, definition, revision, object_info=info)
        if fresh != snapshot or fresh.reasons:
            raise WorkflowOfferCompletionError("workflow-review-required")
        if registry_verification is not None:
            registry_verification.require_current(session)
        # Source approval covers this exact compilation after runtime/code checks.
        record_review(session, revision, fresh, approved=True)
        offer = session.get(WorkflowInstallOffer, offer_id)
        if (
            offer is None
            or revision.artifact_sha256 is None
            or revision.dependency_contract_sha256 is None
        ):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        offer.workflow_revision_id = revision.id
        offer.workflow_artifact_sha256 = revision.artifact_sha256
        offer.dependency_contract_sha256 = revision.dependency_contract_sha256
        session.flush()
        provisioner = processes.runtimes
        activation_id = complete_workflow_install_offer(
            session,
            offer_id,
            runtime_materializer=(
                lambda requirement, selection: materialize_comfy_runtime_dependency(
                    provisioner, requirement, selection
                )
            )
            if provisioner is not None
            else None,
            custom_node_root=settings.custom_node_dir,
            registry_environment_root=registry_wheel_environment_root(settings.registry_dir),
        )
        if (
            activation_id is None
            or _runtime_id(processes) != worker_id
            or processes.launch_scope_sha256("media") != prepared.scope.launch_sha256
        ):
            raise WorkflowOfferCompletionError("workflow-completion-unavailable")
        final_scope = revalidate_workflow_activation(
            session,
            activation_id,
            runtime_materializer=(
                lambda requirement, selection: materialize_comfy_runtime_dependency(
                    provisioner, requirement, selection
                )
            )
            if provisioner is not None
            else None,
            custom_node_root=settings.custom_node_dir,
            registry_environment_root=registry_wheel_environment_root(settings.registry_dir),
        )
        require_workflow_source_resource_match(prepared.scope, final_scope)
        if (
            _runtime_id(processes) != worker_id
            or processes.launch_scope_sha256("media") != prepared.scope.launch_sha256
        ):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        complete_workflow_job(session, offer, source.attempt, activation_id)
        if batch is not None:
            batch.complete(session)
        offer.completion_error_code = None
        session.commit()
        return activation_id


def _record_restoration_warning(source: AcceptedSource, activation_id: str) -> None:
    """Retain a cleanup warning only for this attempt's already committed result."""
    with SessionLocal() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        offer = session.get(WorkflowInstallOffer, source.offer_id)
        if offer is None or offer.status != "completed" or offer.offer_sha256 != source.plan_sha256:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        accepted_workflow_package_jobs(session, offer)
        job = workflow_completion_job(session, offer)
        result = WorkflowCompletionResult.model_validate(job.result_json)
        activation = session.get(WorkflowActivation, activation_id)
        if (
            job.id != source.job_id
            or job.attempt != source.attempt
            or job.status != "complete"
            or result.activation_id != activation_id
            or activation is None
            or activation.workflow_revision_id != offer.workflow_revision_id
        ):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        offer.completion_error_code = "workflow-media-restore-failed"
        session.commit()


async def complete_workflow_source(
    settings: Settings,
    processes: ProcessSupervisor | None,
    media: MediaAdapter | None,
    offer_id: str,
) -> str | None:
    """Finish under the caller's primary lease, which outlives cancellation and I/O."""
    source = await asyncio.to_thread(_read_source, offer_id)
    if source is None:
        return None
    if processes is None or media is None:
        raise WorkflowOfferCompletionError("workflow-review-required")
    activation_id: str | None = None
    try:
        async with workflow_package_runtime(processes):
            await _prepare_runtime(processes, source)
            await asyncio.to_thread(_require_runtime_plan, processes, source)
            async with prepared_workflow_source_runtime(
                processes, offer_id, source.attempt
            ) as prepared:
                activation_id = await _complete_running_source(
                    settings, processes, media, offer_id, source, prepared
                )
        return activation_id
    except WorkflowPackageRestorationError:
        if activation_id is None:
            raise
        await asyncio.to_thread(_record_restoration_warning, source, activation_id)
        return activation_id
    except WorkflowOfferCompletionError as exc:
        if exc.code != "workflow-extension-review-required":
            raise
        # Worker restoration has finished before the durable pause is exposed.
        with SessionLocal() as session:
            session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
            offer = session.get(WorkflowInstallOffer, offer_id)
            if offer is None:
                raise WorkflowOfferCompletionError("workflow-install-offer-changed") from exc
            job = running_workflow_completion_job(session, offer, source.attempt)
            job.status = "paused"
            update_job_progress(job, stage="Review extension code to continue", indeterminate=True)
            offer.completion_error_code = exc.code
            session.commit()
        return None


async def _complete_running_source(
    settings: Settings,
    processes: ProcessSupervisor,
    media: MediaAdapter,
    offer_id: str,
    source: AcceptedSource,
    prepared: PreparedWorkflowSourceRuntime,
) -> str:
    worker_id = _runtime_id(processes)
    if (
        worker_id is None
        or prepared.scope.offer_id != offer_id
        or prepared.scope.plan_sha256 != source.plan_sha256
        or processes.launch_scope_sha256("media") != prepared.scope.launch_sha256
    ):
        raise WorkflowOfferCompletionError("workflow-review-required")
    info = await review_runtime_object_info(processes, media)
    if info is None:
        raise WorkflowOfferCompletionError("workflow-review-required")
    payload = _compile(source, info)
    with SessionLocal() as session:
        definition = _definition(session, source)
        preview = build_workflow_revision(definition, payload, version=1, engine="comfyui")
        snapshot = build_review_snapshot(session, definition, preview, object_info=info)
        if snapshot.reasons:
            raise WorkflowOfferCompletionError("workflow-review-required")
        packages = _reviewed_package_inputs(session, snapshot)
    registry_verification = await verify_reviewed_packages(
        settings, None, snapshot, session_factory=SessionLocal, packages=packages
    )
    try:
        errors = await media.validate_workflow(payload.api_graph)
    except (OSError, RuntimeError, ValueError, TimeoutError, httpx.HTTPError):
        raise WorkflowOfferCompletionError("workflow-review-required") from None
    if errors:
        raise WorkflowOfferCompletionError("workflow-review-required")
    refreshed = await review_runtime_object_info(processes, media)
    if (
        refreshed is None
        or _runtime_id(processes) != worker_id
        or processes.launch_scope_sha256("media") != prepared.scope.launch_sha256
        or _compile(source, refreshed) != payload
    ):
        raise WorkflowOfferCompletionError("workflow-review-required")
    if registry_verification is not None:
        registry_verification = await registry_verification.refresh()
    completion = asyncio.create_task(
        asyncio.to_thread(
            _finish,
            settings,
            processes,
            offer_id,
            source,
            payload,
            refreshed,
            snapshot,
            worker_id,
            prepared,
            registry_verification,
        )
    )
    # Once finalization starts, retain its transaction outcome before cleanup.
    # A durable job cancellation is rechecked by the transaction itself.
    while not completion.done():
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            continue
    return completion.result()
