"""Keep the prior media launch scope across temporary extension setup."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from .comfy_registry_paths import registry_wheel_environment_root
from .db import SessionLocal
from .models import WorkflowActivation, WorkflowRevision
from .processes import ProcessSupervisor
from .workflow_activation_requests import activation_subject
from .workflow_activations import (
    WorkflowActivationLaunchScope,
    materialize_comfy_runtime_dependency,
    revalidate_workflow_activation,
)
from .workflow_package_activation import media_worker_stopped
from .workflow_package_preparation import PreparationContext, WorkflowPackagePreparationError


class WorkflowPackageRestorationError(WorkflowPackagePreparationError):
    """The temporary setup finished, but restoring the previous worker failed."""


@dataclass(frozen=True)
class _RestorationPaths:
    custom_node_root: Path
    state_root: Path


def _scoped_activation(launch_sha256: str) -> str:
    with SessionLocal() as session:
        matches = session.scalars(
            select(WorkflowActivation)
            .where(
                WorkflowActivation.is_active.is_(True),
                WorkflowActivation.state == "ready",
                WorkflowActivation.details_json["launch_sha256"].as_string() == launch_sha256,
            )
            .limit(2)
        ).all()
        if len(matches) != 1:
            raise WorkflowPackagePreparationError(
                "media_scope_unavailable",
                "The current media setup cannot be restored safely.",
            )
        return matches[0].id


def _restore_scope(
    activation_id: str,
    expected_launch_sha256: str,
    context: _RestorationPaths,
    processes: ProcessSupervisor,
) -> WorkflowActivationLaunchScope:
    with SessionLocal() as session:
        session.connection().exec_driver_sql("BEGIN")
        activation = session.get(WorkflowActivation, activation_id)
        if activation is None or not activation.is_active:
            raise WorkflowPackagePreparationError(
                "media_scope_changed", "The previous media setup is no longer selected."
            )
        revision = session.get(WorkflowRevision, activation.workflow_revision_id)
        if revision is None:
            raise WorkflowPackagePreparationError(
                "media_scope_changed", "The previous media workflow is unavailable."
            )
        activation_subject(session, revision.workflow_id, revision.id)
        provisioner = processes.runtimes
        scope = revalidate_workflow_activation(
            session,
            activation,
            runtime_materializer=(
                lambda requirement, selection: materialize_comfy_runtime_dependency(
                    provisioner, requirement, selection
                )
            )
            if provisioner is not None
            else None,
            custom_node_root=context.custom_node_root,
            registry_environment_root=registry_wheel_environment_root(context.state_root),
        )
        if scope.launch_sha256 != expected_launch_sha256:
            raise WorkflowPackagePreparationError(
                "media_scope_changed", "The previous media dependencies changed during setup."
            )
        session.commit()
        return scope


async def _stop(processes: ProcessSupervisor) -> None:
    await processes.stop("media")
    if not media_worker_stopped(processes):
        raise WorkflowPackagePreparationError(
            "media_worker_running", "The media worker did not stop for extension setup."
        )


async def _restore(
    processes: ProcessSupervisor,
    context: _RestorationPaths,
    was_running: bool,
    activation_id: str | None,
    launch_sha256: str | None,
) -> None:
    try:
        await _stop(processes)
        if not was_running:
            return
        try:
            if activation_id is not None and launch_sha256 is not None:
                scope = _restore_scope(activation_id, launch_sha256, context, processes)
                await processes.start_media(activation_scope=scope)
            else:
                await processes.start_media()
        except Exception as exc:
            await _stop(processes)
            raise WorkflowPackagePreparationError(
                "media_restore_failed",
                "Extension setup could not restore the previous media worker.",
            ) from exc
    except Exception as exc:
        raise WorkflowPackageRestorationError(
            exc.code
            if isinstance(exc, WorkflowPackagePreparationError)
            else "media_restore_failed",
            "Extension setup could not restore the previous media worker.",
        ) from exc


@asynccontextmanager
async def workflow_package_runtime(
    processes: ProcessSupervisor, context: PreparationContext | None = None
) -> AsyncIterator[None]:
    """Run inside the primary lease and restore exactly the previous launch scope."""
    paths = _RestorationPaths(
        custom_node_root=context.custom_node_root
        if context is not None
        else processes.settings.custom_node_dir,
        state_root=context.state_root if context is not None else processes.settings.registry_dir,
    )
    was_running = not media_worker_stopped(processes)
    launch_sha256 = processes.launch_scope_sha256("media") if was_running else None
    activation_id = _scoped_activation(launch_sha256) if launch_sha256 else None
    try:
        if was_running:
            await _stop(processes)
        yield
    finally:
        restoration = asyncio.create_task(
            _restore(processes, paths, was_running, activation_id, launch_sha256)
        )
        cancelled = False
        while not restoration.done():
            try:
                await asyncio.shield(restoration)
            except asyncio.CancelledError:
                cancelled = True
        restoration.result()
        if cancelled:
            raise asyncio.CancelledError
