"""Revalidate reviewed workflow nodes against the managed media worker and local code."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import httpx
from sqlalchemy.orm import Session

from .adapters.base import MediaAdapter
from .artifacts import ArtifactStore
from .comfy_registry_launch_verification import _FIELDS, VerifiedComfyRegistryLaunch
from .comfy_registry_target_verification import ComfyRegistryVerificationTarget
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


def _registry_snapshots(
    rows: tuple[ComfyRegistryInstall, ...] | list[ComfyRegistryInstall],
) -> tuple[str, ...]:
    return tuple(
        json.dumps(
            {name: getattr(row, name) for name in _FIELDS},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in sorted(rows, key=lambda row: row.id)
    )


@dataclass(frozen=True)
class VerifiedReviewedPackages:
    target: ComfyRegistryVerificationTarget
    registry: VerifiedComfyRegistryLaunch

    async def refresh(self) -> VerifiedReviewedPackages:
        """Recheck the same selected packages after awaited consumer work."""

        def unchanged(rows: list[ComfyRegistryInstall]) -> None:
            if _registry_snapshots(rows) != self.registry.snapshots:
                raise WorkflowReviewError("workflow_review_changed")

        try:
            proof = await self.target.verify(
                self.registry.install_ids, include_active=False, prepare_installs=unchanged
            )
        except (ValueError, OSError) as exc:
            raise WorkflowReviewError("workflow_review_node_unavailable") from exc
        return VerifiedReviewedPackages(self.target, proof)

    def require_current(self, session: Session) -> None:
        """Bind a durable decision or final dispatch to current source authority."""
        try:
            self.target.require_configuration()
            self.registry.require_current(session)
        except (ValueError, OSError) as exc:
            raise WorkflowReviewError("workflow_review_changed") from exc


async def verify_reviewed_packages(
    settings: Settings,
    session: Session | None,
    snapshot: ReviewSnapshot,
    *,
    session_factory: Callable[[], Session],
    custom_nodes: CustomNodeManager | None = None,
    packages: ReviewedPackages | None = None,
) -> VerifiedReviewedPackages | None:
    if packages is None:
        if session is None:
            raise WorkflowReviewError("workflow_review_node_unavailable")
        packages = _reviewed_package_inputs(session, snapshot)
    for install in packages.git:
        if custom_nodes is None:
            custom_nodes = CustomNodeManager(settings)
        await custom_nodes.verify(install)
    if not packages.registry:
        return None
    expected = _registry_snapshots(packages.registry)

    def unchanged(rows: list[ComfyRegistryInstall]) -> None:
        if _registry_snapshots(rows) != expected:
            raise WorkflowReviewError("workflow_review_changed")

    context = replace(
        PreparationContext.from_settings(settings), source_store=ArtifactStore(settings)
    )
    target = context.verification_target(session_factory, settings)
    try:
        proof = await target.verify(
            tuple(row.id for row in packages.registry),
            include_active=False,
            prepare_installs=unchanged,
        )
    except (ValueError, OSError) as exc:
        raise WorkflowReviewError("workflow_review_node_unavailable") from exc
    return VerifiedReviewedPackages(target, proof)


@dataclass(frozen=True)
class VerifiedWorkflowReview:
    revision_id: str
    worker_id: int
    snapshot: ReviewSnapshot
    object_info: dict[str, Any] | None
    packages: VerifiedReviewedPackages | None = None


async def refresh_workflow_review_packages(
    verified: VerifiedWorkflowReview,
) -> VerifiedWorkflowReview:
    if verified.packages is None:
        return verified
    return replace(verified, packages=await verified.packages.refresh())


def revalidate_workflow_review_runtime(
    session: Session,
    processes: ProcessSupervisor,
    revision: WorkflowRevision,
    verified: VerifiedWorkflowReview,
) -> None:
    """Match dispatch inputs to the completed verification in its fresh read."""
    if revision.id != verified.revision_id or _runtime_id(processes) != verified.worker_id:
        raise WorkflowReviewError("workflow_review_changed")
    if verified.packages is not None:
        revision_id = revision.id
        verified.packages.require_current(session)
        session.expire_all()
        current_revision = session.get(WorkflowRevision, revision_id)
        if current_revision is None:
            raise WorkflowReviewError("workflow_review_changed")
        revision = current_revision
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
    verified_packages = await verify_reviewed_packages(
        settings, None, snapshot, session_factory=session_factory, packages=packages
    )
    current_info = await review_runtime_object_info(processes, media)
    if worker_id is None or worker_id != _runtime_id(processes):
        raise WorkflowReviewError("workflow_review_changed")
    if verified_packages is not None:
        verified_packages = await verified_packages.refresh()
    verified = VerifiedWorkflowReview(
        revision_id, worker_id, snapshot, deepcopy(current_info), verified_packages
    )
    with session_factory() as session:
        revision = session.get(WorkflowRevision, revision_id)
        if revision is None:
            raise WorkflowReviewError("workflow_review_changed")
        revalidate_workflow_review_runtime(session, processes, revision, verified)
    return verified
