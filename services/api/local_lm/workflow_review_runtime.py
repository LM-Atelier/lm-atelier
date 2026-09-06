"""Revalidate reviewed workflow nodes against the managed media worker and local code."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReviewedPackages:
    git: tuple[CustomNodeInstall, ...]
    registry: tuple[ComfyRegistryInstall, ...]


def _reviewed_package_inputs(session: Session, snapshot: ReviewSnapshot) -> ReviewedPackages:
    """Copy scalar installation records before verification can yield to I/O."""
    seen: set[str] = set()
    git_installs: list[CustomNodeInstall] = []
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
            git_installs.append(
                CustomNodeInstall(
                    **{
                        column.key: deepcopy(getattr(install, column.key))
                        for column in CustomNodeInstall.__table__.columns
                    }
                )
            )
        else:
            installed = session.get(ComfyRegistryInstall, pin["id"])
            if installed is None:
                raise WorkflowReviewError("workflow_review_node_unavailable")
            registry_installs.append(
                ComfyRegistryInstall(
                    **{
                        column.key: deepcopy(getattr(installed, column.key))
                        for column in ComfyRegistryInstall.__table__.columns
                    }
                )
            )
    return ReviewedPackages(tuple(git_installs), tuple(registry_installs))


async def verify_reviewed_packages(
    settings: Settings,
    session: Session | None,
    snapshot: ReviewSnapshot,
    *,
    custom_nodes: CustomNodeManager | None = None,
    packages: ReviewedPackages | None = None,
) -> None:
    if packages is None:
        if session is None:
            raise WorkflowReviewError("workflow_review_node_unavailable")
        packages = _reviewed_package_inputs(session, snapshot)
    for install in packages.git:
        if custom_nodes is None:
            custom_nodes = CustomNodeManager(settings)
        await custom_nodes.verify(install)
    if packages.registry:
        context = PreparationContext.from_settings(settings)
        await asyncio.to_thread(
            _verified_comfy_registry_launch_contract,
            packages.registry,
            custom_node_root=context.custom_node_root,
            environment_root=registry_wheel_environment_root(context.state_root),
        )


@dataclass(frozen=True)
class VerifiedWorkflowReview:
    revision_id: str
    worker_id: int
    snapshot: ReviewSnapshot
    object_info: dict[str, Any] | None


def revalidate_workflow_review_runtime(
    session: Session,
    processes: ProcessSupervisor,
    revision: WorkflowRevision,
    verified: VerifiedWorkflowReview,
) -> None:
    """Match dispatch inputs to the completed verification in its fresh read."""
    if revision.id != verified.revision_id or _runtime_id(processes) != verified.worker_id:
        raise WorkflowReviewError("workflow_review_changed")
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    if definition is None or not review_is_current(session, definition, revision):
        raise WorkflowReviewError("workflow_review_changed")
    current = build_review_snapshot(session, definition, revision, object_info=verified.object_info)
    if (
        current.reasons
        or current.subject_sha256 != verified.snapshot.subject_sha256
        or current.node_bindings != verified.snapshot.node_bindings
    ):
        raise WorkflowReviewError("workflow_review_changed")


async def verify_workflow_review_runtime(
    settings: Settings,
    processes: ProcessSupervisor,
    media: MediaAdapter,
    session_factory: Callable[[], Session],
    revision_id: str,
) -> VerifiedWorkflowReview | None:
    """Verify under the media lease without a database transaction across I/O."""
    with session_factory() as session:
        revision = session.get(WorkflowRevision, revision_id)
        if revision is None:
            raise WorkflowReviewError("workflow_review_changed")
        review = session.get(WorkflowRevisionReview, revision_id)
        if review is None:
            return None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        if definition is None or not review_is_current(session, definition, revision):
            raise WorkflowReviewError("workflow_review_changed")
        reviewed_subject = review.subject_sha256
    worker_id = _runtime_id(processes)
    info = await review_runtime_object_info(processes, media)
    with session_factory() as session:
        revision = session.get(WorkflowRevision, revision_id)
        if revision is None:
            raise WorkflowReviewError("workflow_review_changed")
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        if definition is None or not review_is_current(session, definition, revision):
            raise WorkflowReviewError("workflow_review_changed")
        snapshot = build_review_snapshot(session, definition, revision, object_info=info)
        if snapshot.reasons or snapshot.subject_sha256 != reviewed_subject:
            raise WorkflowReviewError("workflow_review_changed")
        packages = _reviewed_package_inputs(session, snapshot)
    await verify_reviewed_packages(settings, None, snapshot, packages=packages)
    current_info = await review_runtime_object_info(processes, media)
    if worker_id is None or worker_id != _runtime_id(processes):
        raise WorkflowReviewError("workflow_review_changed")
    verified = VerifiedWorkflowReview(revision_id, worker_id, snapshot, deepcopy(current_info))
    with session_factory() as session:
        revision = session.get(WorkflowRevision, revision_id)
        if revision is None:
            raise WorkflowReviewError("workflow_review_changed")
        revalidate_workflow_review_runtime(session, processes, revision, verified)
    return verified
