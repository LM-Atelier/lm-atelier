"""Prepare accepted extensions and start only their bound workflow runtime."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from .comfy_registry_activation_batches import (
    RegistryActivationBatch,
    activate_registry_package_batch,
)
from .comfy_registry_paths import registry_wheel_environment_root
from .db import SessionLocal
from .models import WorkflowInstallOffer
from .processes import ProcessSupervisor
from .workflow_activations import (
    WorkflowRegistryLaunchBinding,
    WorkflowSourceLaunchScope,
    materialize_comfy_runtime_dependency,
)
from .workflow_completion_jobs import running_workflow_completion_job
from .workflow_offer_completion import WorkflowOfferCompletionError
from .workflow_package_activation import media_worker_stopped
from .workflow_package_preparation import PreparationContext
from .workflow_source_extension_trust import trust_workflow_source_extensions
from .workflow_source_extensions import (
    prepare_workflow_source_extensions,
    quarantine_workflow_source_extensions,
)
from .workflow_source_launch import prepare_workflow_source_launch_scope


@dataclass(frozen=True)
class PreparedWorkflowSourceRuntime:
    scope: WorkflowSourceLaunchScope
    batch: RegistryActivationBatch | None


def require_workflow_source_attempt(offer_id: str, attempt: int) -> None:
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, offer_id)
        if offer is None:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        running_workflow_completion_job(session, offer, attempt)


@asynccontextmanager
async def prepared_workflow_source_runtime(
    processes: ProcessSupervisor, offer_id: str, attempt: int
) -> AsyncIterator[PreparedWorkflowSourceRuntime]:
    """Keep the primary lease and prior-worker restoration outside this scope."""
    context = PreparationContext.from_settings(processes.settings)
    provisioner = processes.runtimes
    if provisioner is None:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-unavailable")
    await asyncio.to_thread(require_workflow_source_attempt, offer_id, attempt)
    try:
        prepared = await prepare_workflow_source_extensions(
            SessionLocal,
            offer_id,
            context=context,
            media_worker_stopped=media_worker_stopped(processes),
        )
    except (ValueError, OSError):
        await asyncio.to_thread(
            quarantine_workflow_source_extensions,
            SessionLocal,
            offer_id,
            media_worker_stopped=media_worker_stopped(processes),
        )
        raise
    await asyncio.to_thread(require_workflow_source_attempt, offer_id, attempt)
    trust = await asyncio.to_thread(
        trust_workflow_source_extensions,
        SessionLocal,
        offer_id,
        context=context,
        media_worker_stopped=media_worker_stopped(processes),
    )
    await asyncio.to_thread(require_workflow_source_attempt, offer_id, attempt)
    if trust.state != "ready":
        raise WorkflowOfferCompletionError("workflow-extension-review-required")

    launch_scope: WorkflowSourceLaunchScope | None = None

    async def start(bindings: tuple[WorkflowRegistryLaunchBinding, ...]) -> None:
        nonlocal launch_scope
        scope = await asyncio.to_thread(
            prepare_workflow_source_launch_scope,
            SessionLocal,
            offer_id,
            context=context,
            runtime_materializer=lambda requirement, selection: (
                materialize_comfy_runtime_dependency(provisioner, requirement, selection)
            ),
        )
        require_workflow_source_attempt(offer_id, attempt)
        if any(binding not in scope.registry_packages for binding in bindings):
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        await processes.start_media(activation_scope=scope)
        await asyncio.to_thread(require_workflow_source_attempt, offer_id, attempt)
        if processes.launch_scope_sha256("media") != scope.launch_sha256:
            raise WorkflowOfferCompletionError("workflow-install-offer-changed")
        launch_scope = scope

    async def stop() -> bool:
        await processes.stop("media")
        return media_worker_stopped(processes)

    if not prepared.preparations:
        await start(())
        if launch_scope is None:
            raise WorkflowOfferCompletionError("workflow-completion-unavailable")
        yield PreparedWorkflowSourceRuntime(launch_scope, None)
        return
    async with activate_registry_package_batch(
        SessionLocal,
        purpose=offer_id,
        preparations=prepared.preparations,
        execution_plans=prepared.execution_plans,
        custom_node_root=context.custom_node_root,
        environment_root=registry_wheel_environment_root(context.state_root),
        media_worker_stopped=media_worker_stopped(processes),
        start_media=start,
        stop_media=stop,
        read_node_inventory=processes.comfy_node_inventory,
    ) as batch:
        if launch_scope is None:
            raise WorkflowOfferCompletionError("workflow-completion-unavailable")
        yield PreparedWorkflowSourceRuntime(launch_scope, batch)
