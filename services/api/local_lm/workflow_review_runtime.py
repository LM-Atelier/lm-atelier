"""Revalidate reviewed workflow nodes against the managed media worker and local code."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from sqlalchemy.orm import Session

from .adapters.base import MediaAdapter
from .comfy_registry_installs import _verified_comfy_registry_launch_contract
from .comfy_registry_paths import registry_wheel_environment_root
from .config import Settings
from .custom_nodes import CustomNodeManager
from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    WorkflowDefinition,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from .processes import ProcessSupervisor
from .workflow_package_preparation import PreparationContext
from .workflow_revision_reviews import (
    ReviewSnapshot,
    WorkflowReviewError,
    build_review_snapshot,
    review_is_current,
)


def _runtime_id(processes: ProcessSupervisor) -> int | None:
    for worker in processes.statuses():
        if (
            worker.name == "media"
            and worker.managed
            and worker.running
            and worker.state == "ready"
            and worker.pid is not None
        ):
            return worker.pid
    return None


async def review_runtime_object_info(
    processes: ProcessSupervisor, media: MediaAdapter
) -> dict[str, Any] | None:
    before = _runtime_id(processes)
    describe = getattr(media, "object_info", None)
    if before is None or not callable(describe):
        return None
    invalidate = getattr(media, "invalidate_object_info_cache", None)
    if callable(invalidate):
        invalidate()
    try:
        result = await describe()
    except (OSError, RuntimeError, ValueError, TimeoutError, httpx.HTTPError):
        return None
    if before != _runtime_id(processes) or not isinstance(result, dict):
        return None
    return result


async def verify_reviewed_packages(
    settings: Settings,
    session: Session,
    snapshot: ReviewSnapshot,
    *,
    custom_nodes: CustomNodeManager | None = None,
) -> None:
    seen: set[str] = set()
    registry_installs: list[ComfyRegistryInstall] = []
    for binding in snapshot.node_bindings.values():
        pin = binding.get("pin")
        if not isinstance(pin, dict) or pin["id"] in seen:
            continue
        seen.add(pin["id"])
        if pin["kind"] == "git":
            install = session.get(CustomNodeInstall, pin["id"])
            if install is None:
                raise WorkflowReviewError("workflow_review_node_unavailable")
            if custom_nodes is None:
                custom_nodes = CustomNodeManager(settings)
            await custom_nodes.verify(install)
        else:
            installed = session.get(ComfyRegistryInstall, pin["id"])
            if installed is None:
                raise WorkflowReviewError("workflow_review_node_unavailable")
            registry_installs.append(installed)
    if registry_installs:
        context = PreparationContext.from_settings(settings)
        await asyncio.to_thread(
            _verified_comfy_registry_launch_contract,
            registry_installs,
            custom_node_root=context.custom_node_root,
            environment_root=registry_wheel_environment_root(context.state_root),
        )


async def verify_workflow_review_runtime(
    settings: Settings,
    processes: ProcessSupervisor,
    media: MediaAdapter,
    session: Session,
    revision: WorkflowRevision,
) -> None:
    """Verify before submission while the caller owns the media lifecycle lease."""
    review = session.get(WorkflowRevisionReview, revision.id)
    if review is None:
        return
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or not review_is_current(session, definition, revision):
        raise WorkflowReviewError("workflow_review_changed")
    worker_id = _runtime_id(processes)
    info = await review_runtime_object_info(processes, media)
    snapshot = build_review_snapshot(session, definition, revision, object_info=info)
    if snapshot.reasons or snapshot.subject_sha256 != review.subject_sha256:
        raise WorkflowReviewError("workflow_review_changed")
    await verify_reviewed_packages(settings, session, snapshot)
    current_info = await review_runtime_object_info(processes, media)
    if worker_id is None or worker_id != _runtime_id(processes):
        raise WorkflowReviewError("workflow_review_changed")
    # Package verification awaits I/O. Re-read every durable subject and the
    # live node contract before accepting the original review for submission.
    revision_id = revision.id
    session.expire_all()
    current_revision = session.get(WorkflowRevision, revision_id)
    if current_revision is None:
        raise WorkflowReviewError("workflow_review_changed")
    current_definition = session.get(WorkflowDefinition, current_revision.workflow_id)
    if current_definition is None or not review_is_current(
        session, current_definition, current_revision
    ):
        raise WorkflowReviewError("workflow_review_changed")
    current = build_review_snapshot(
        session, current_definition, current_revision, object_info=current_info
    )
    if (
        current.reasons
        or current.subject_sha256 != snapshot.subject_sha256
        or current.node_bindings != snapshot.node_bindings
    ):
        raise WorkflowReviewError("workflow_review_changed")
