"""Prepare only the extension requests and result identities an installation accepted."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .comfy_registry import ComfyRegistryClient
from .comfy_registry_activation_batches import (
    _deactivate_pending,
    _worker,
    recover_registry_package_batch,
)
from .comfy_registry_closure_driver import ComfyRegistryWheelMetadataClient
from .comfy_registry_downloads import ComfyRegistryArchiveDownloader
from .comfy_registry_interpreter import (
    ComfyRegistryInterpreterError,
    probe_comfy_registry_runtime_target,
)
from .comfy_registry_launch_verification import verify_comfy_registry_launch
from .comfy_registry_lifecycle import ComfyRegistryPreparation
from .comfy_registry_paths import registry_wheel_environment_root
from .comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from .comfy_registry_runtime import (
    ComfyRegistryRuntimeDistribution,
    canonical_comfy_registry_runtime_distributions,
)
from .comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from .comfy_registry_wheel_projects import ComfyRegistryWheelProjectClient
from .domain import JobKind, utcnow
from .models import Job, WorkflowInstallOffer, WorkflowInstallOfferPackage
from .workflow_offer_packages import (
    AcceptedWorkflowPackage,
    WorkflowOfferPackageError,
    accepted_workflow_offer_packages,
    record_workflow_offer_package,
)
from .workflow_package_acceptance import accepted_workflow_package_jobs
from .workflow_package_execution_plan import WorkflowPackageExecutionPlan
from .workflow_package_install_plans import WorkflowPackageInstallPlanOut
from .workflow_package_preparation import (
    InterpreterProbe,
    PreparationContext,
    WorkflowPackagePreparationError,
    prepare_workflow_package,
)

SessionFactory = Callable[[], Session]


@dataclass(frozen=True)
class ExtensionPreparationServices:
    interpreter_probe: InterpreterProbe
    registry: ComfyRegistryClient
    projects: ComfyRegistryWheelProjectClient
    metadata: ComfyRegistryWheelMetadataClient
    archive: ComfyRegistryArchiveDownloader
    wheels: ComfyRegistryWheelDownloader


@dataclass(frozen=True)
class PreparedWorkflowExtensions:
    preparations: tuple[ComfyRegistryPreparation, ...]
    execution_plans: dict[str, WorkflowPackageExecutionPlan]
    reviewed_inputs: ComfyRegistryReviewedInputContext | None = None


def quarantine_workflow_source_extensions(
    session_factory: SessionFactory, offer_id: str, *, media_worker_stopped: bool
) -> None:
    """Leave a refused interrupted installation inactive without changing any package files."""
    if media_worker_stopped is not True:
        return
    with session_factory() as session:
        identifiers = tuple(
            session.scalars(
                select(WorkflowInstallOfferPackage.registry_install_id)
                .join(
                    WorkflowInstallOffer,
                    WorkflowInstallOffer.id == WorkflowInstallOfferPackage.offer_id,
                )
                .where(
                    WorkflowInstallOffer.id == offer_id,
                    WorkflowInstallOffer.status == "queued",
                    WorkflowInstallOfferPackage.offer_sha256 == WorkflowInstallOffer.offer_sha256,
                    WorkflowInstallOfferPackage.registry_install_id.is_not(None),
                )
            )
        )
    _deactivate_pending(
        session_factory, [identifier for identifier in identifiers if identifier is not None]
    )


def _accepted(
    session: Session,
    offer_id: str,
) -> tuple[
    WorkflowInstallOffer, WorkflowPackageInstallPlanOut, tuple[AcceptedWorkflowPackage, ...]
]:
    offer = session.get(WorkflowInstallOffer, offer_id)
    if offer is None or offer.status != "queued":
        raise WorkflowOfferPackageError()
    _source, saved, _jobs = accepted_workflow_package_jobs(session, offer)
    return offer, saved, accepted_workflow_offer_packages(session, offer, saved)


def _failure(
    session_factory: SessionFactory, job_id: str, payload: dict[str, object], *, cancelled: bool
) -> None:
    with session_factory() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        job = session.get(Job, job_id)
        if (
            job is not None
            and job.kind == JobKind.REGISTRY_PREPARE.value
            and job.status == "running"
            and job.payload_json == payload
        ):
            job.status = "cancelled" if cancelled else "failed"
            job.phase = "cancelled" if cancelled else "failed"
            job.error = "The accepted extension preparation did not finish."
            job.completed_at = utcnow()
            session.commit()


async def _prepare_one(
    session_factory: SessionFactory,
    offer_id: str,
    package_id: str,
    context: PreparationContext,
    services: ExtensionPreparationServices,
) -> None:
    with session_factory() as session:
        session.execute(text("UPDATE workflow_install_offers SET status = status WHERE 0"))
        _offer, _saved, packages = _accepted(session, offer_id)
        matches = [item for item in packages if item.link.package_id == package_id]
        if len(matches) != 1:
            raise WorkflowOfferPackageError()
        item = matches[0]
        if item.preparation is not None:
            return
        if item.job.status not in {"queued", "running"}:
            raise WorkflowPackagePreparationError(
                "workflow-package-preparation-paused",
                "Resume the extension preparation before continuing.",
            )
        plan = item.plan.model_copy(deep=True)
        job_id = item.job.id
        job_payload = dict(item.job.payload_json)
        item.job.status = "running"
        item.job.phase = "Preparing extension"
        item.job.attempt += 1
        item.job.started_at = utcnow()
        session.commit()

    def record(session: Session, preparation: ComfyRegistryPreparation) -> None:
        offer, saved, packages = _accepted(session, offer_id)
        accepted = next((item for item in packages if item.job.id == job_id), None)
        if accepted is None or accepted.plan != plan:
            raise WorkflowOfferPackageError()
        record_workflow_offer_package(
            session,
            offer,
            saved,
            package_id=package_id,
            job_id=job_id,
            preparation=preparation,
        )

    try:
        result = await prepare_workflow_package(
            session_factory,
            package_id=package_id,
            version=plan.resolution.declared_version,
            node_types=plan.resolution.node_types,
            context=context,
            media_worker_stopped=True,
            interpreter_probe=services.interpreter_probe,
            registry_client=services.registry,
            project_client=services.projects,
            metadata_client=services.metadata,
            archive_downloader=services.archive,
            wheel_downloader=services.wheels,
            expected_plan=plan,
            record_preparation=record,
        )
        with session_factory() as session:
            _offer, _saved, packages = _accepted(session, offer_id)
            stored = next((item for item in packages if item.job.id == job_id), None)
            if stored is None or stored.preparation != result:
                raise WorkflowOfferPackageError()
    except (Exception, asyncio.CancelledError) as exc:
        _failure(
            session_factory, job_id, job_payload, cancelled=isinstance(exc, asyncio.CancelledError)
        )
        raise


async def prepare_workflow_source_extensions(
    session_factory: SessionFactory,
    offer_id: str,
    *,
    context: PreparationContext,
    media_worker_stopped: bool,
    services: ExtensionPreparationServices | None = None,
) -> PreparedWorkflowExtensions:
    """Run under the caller's primary lease; preparation grants no trust or activation."""
    if media_worker_stopped is not True:
        raise WorkflowPackagePreparationError(
            "media_worker_running", "Stop the media worker before extension setup."
        )
    with session_factory() as session:
        _offer, _saved, initial = _accepted(session, offer_id)
        package_ids = tuple(item.link.package_id for item in initial)
        missing = tuple(item.link.package_id for item in initial if item.preparation is None)
    async with AsyncExitStack() as cleanup:
        if missing and services is None:
            registry = ComfyRegistryClient()
            cleanup.push_async_callback(registry.close)
            projects = ComfyRegistryWheelProjectClient()
            cleanup.push_async_callback(projects.close)
            metadata = ComfyRegistryWheelMetadataClient()
            cleanup.push_async_callback(metadata.close)
            archive = ComfyRegistryArchiveDownloader()
            cleanup.push_async_callback(archive.close)
            wheels = ComfyRegistryWheelDownloader()
            cleanup.push_async_callback(wheels.close)
            services = ExtensionPreparationServices(
                probe_comfy_registry_runtime_target,
                registry,
                projects,
                metadata,
                archive,
                wheels,
            )
        for package_id in missing:
            assert services is not None
            await _prepare_one(session_factory, offer_id, package_id, context, services)

    def snapshot() -> PreparedWorkflowExtensions:
        with session_factory() as session:
            return _prepared_snapshot(session, offer_id, package_ids)

    prepared = await _worker(snapshot)
    reviewed_inputs = None
    runtime: tuple[ComfyRegistryRuntimeDistribution, ...] = ()
    if prepared.preparations and context.source_store is not None:
        probe = services.interpreter_probe if services else probe_comfy_registry_runtime_target
        try:
            target = await probe(context.python_executable)
            if len(target) != 3:
                raise ValueError("Runtime distribution evidence is missing")
            environment, tags, distributions = target
            runtime = canonical_comfy_registry_runtime_distributions(distributions)
            reviewed_inputs = ComfyRegistryReviewedInputContext(
                session_factory, context.source_store, environment, tuple(tags)
            )
        except ComfyRegistryInterpreterError as exc:
            raise WorkflowPackagePreparationError(exc.code, str(exc)) from exc
        except Exception as exc:
            raise WorkflowPackagePreparationError(
                "interpreter_probe_failed",
                "The managed runtime's package target could not be determined.",
            ) from exc
        for plan in prepared.execution_plans.values():
            plan.verify_target(reviewed_inputs.marker_environment, reviewed_inputs.supported_tags)
    await _worker(
        lambda: _recover_and_verify(
            session_factory, offer_id, package_ids, prepared, context, reviewed_inputs, runtime
        )
    )
    return replace(prepared, reviewed_inputs=reviewed_inputs)


def _prepared_snapshot(
    session: Session, offer_id: str, package_ids: tuple[str, ...]
) -> PreparedWorkflowExtensions:
    _offer, _saved, packages = _accepted(session, offer_id)
    if tuple(item.link.package_id for item in packages) != package_ids:
        raise WorkflowOfferPackageError()
    prepared = []
    plans = {}
    for package in packages:
        if package.preparation is None:
            raise WorkflowOfferPackageError()
        prepared.append(package.preparation)
        plans[package.preparation.install_id] = package.plan.model_copy(deep=True)
    return PreparedWorkflowExtensions(tuple(prepared), plans)


def _recover_and_verify(
    session_factory: SessionFactory,
    offer_id: str,
    package_ids: tuple[str, ...],
    prepared: PreparedWorkflowExtensions,
    context: PreparationContext,
    reviewed_inputs: ComfyRegistryReviewedInputContext | None,
    runtime: tuple[ComfyRegistryRuntimeDistribution, ...],
) -> None:
    environment_root = registry_wheel_environment_root(context.state_root)
    recover_registry_package_batch(
        session_factory,
        purpose=offer_id,
        preparations=prepared.preparations,
        execution_plans=prepared.execution_plans,
        custom_node_root=context.custom_node_root,
        environment_root=environment_root,
        media_worker_stopped=True,
        reviewed_inputs=reviewed_inputs,
    )
    verified = (
        verify_comfy_registry_launch(
            session_factory,
            [item.install_id for item in prepared.preparations],
            include_active=True,
            custom_node_root=context.custom_node_root,
            environment_root=environment_root,
            reviewed_inputs=reviewed_inputs,
        )
        if prepared.preparations
        else None
    )
    if (
        verified is not None
        and verified.authorities
        and verified.contract.runtime_distributions != runtime
    ):
        raise WorkflowPackagePreparationError(
            "workflow-package-runtime-changed",
            "The managed runtime's installed packages changed after preparation.",
        )
    with session_factory() as session:
        if verified is not None:
            verified.require_current(session)
        if _prepared_snapshot(session, offer_id, package_ids) != prepared:
            raise WorkflowOfferPackageError()
