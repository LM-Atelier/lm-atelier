from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from .comfy_workflow_packages import (
    WorkflowPackageError,
    analyze_comfyui_workflow_package,
)
from .domain import new_id, utcnow
from .model_planner import workflow_artifact_contract
from .models import (
    InstallPlan,
    WorkflowDefinition,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowInstallOffer,
    WorkflowRevision,
)
from .schemas import DownloadRequest
from .workflow_asset_aliases import (
    WorkflowAssetAliasError,
    materialize_workflow_asset_aliases,
)
from .workflow_asset_bindings import (
    MAX_WORKFLOW_ASSET_BINDINGS,
    BoundWorkflowAsset,
    WorkflowAssetBindingError,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)
from .workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    compose_workflow_asset_download_requests,
)
from .workflow_dependencies import (
    WorkflowDependencyError,
    parse_workflow_dependency_contract,
    workflow_dependency_contract_sha256,
    workflow_dependency_slot_sha256,
)

WORKFLOW_INSTALL_OFFER_VERSION = 1

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_READY = "ready"


class WorkflowInstallOfferError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ReviewedWorkflowInstallOffer:
    revision: WorkflowRevision
    selections: tuple[dict[str, str], ...]
    assets: tuple[dict[str, str | int], ...]
    binding_plan_sha256: str
    offer_sha256: str
    plan_count: int
    total_bytes: int
    download_requests: tuple[DownloadRequest, ...]


def create_workflow_install_offer(
    session: Session,
    *,
    workflow_id: str,
    revision_id: str,
    selections: Sequence[WorkflowAssetPlanSelection],
    available_node_types: Collection[str],
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
) -> WorkflowInstallOffer:
    """Persist one complete reviewed offer for the exact current revision."""

    reviewed = _review_offer(
        session,
        workflow_id=workflow_id,
        revision_id=revision_id,
        selections=selections,
        available_node_types=available_node_types,
        available_asset_filenames=available_asset_filenames,
        installed_package_versions=installed_package_versions,
    )
    current = list(
        session.scalars(
            select(WorkflowInstallOffer).where(
                WorkflowInstallOffer.workflow_revision_id == revision_id,
                WorkflowInstallOffer.status == _READY,
            )
        ).all()
    )
    for offer in current:
        if _offer_matches_review(offer, reviewed):
            return offer
        invalidate_workflow_install_offer(
            offer,
            code="offer-superseded",
            reason="A newer reviewed install offer replaced this one.",
        )

    revision = reviewed.revision
    offer = WorkflowInstallOffer(
        id=new_id("wfoffer"),
        workflow_revision_id=revision.id,
        workflow_artifact_sha256=str(revision.artifact_sha256),
        dependency_contract_sha256=str(revision.dependency_contract_sha256),
        binding_plan_sha256=reviewed.binding_plan_sha256,
        offer_sha256=reviewed.offer_sha256,
        selections_json=[dict(item) for item in reviewed.selections],
        assets_json=[dict(item) for item in reviewed.assets],
        plan_count=reviewed.plan_count,
        total_bytes=reviewed.total_bytes,
        status=_READY,
    )
    session.add(offer)
    session.flush()
    return offer


def revalidate_workflow_install_offer(
    session: Session,
    offer_id: str,
    *,
    available_node_types: Collection[str],
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
) -> tuple[WorkflowInstallOffer, tuple[DownloadRequest, ...]]:
    """Rebuild an offer from current server records before any job starts."""

    offer = session.get(WorkflowInstallOffer, offer_id)
    if offer is None:
        raise WorkflowInstallOfferError(
            "workflow-install-offer-not-found",
            "Workflow install offer is unavailable.",
        )
    if offer.status != _READY:
        raise WorkflowInstallOfferError(
            "workflow-install-offer-not-actionable",
            "Workflow install offer is no longer actionable.",
        )
    try:
        selections = _stored_selections(offer.selections_json)
        revision = session.get(WorkflowRevision, offer.workflow_revision_id)
        if revision is None:
            raise WorkflowInstallOfferError(
                "workflow-revision-unavailable",
                "The workflow revision no longer exists.",
            )
        reviewed = _review_offer(
            session,
            workflow_id=revision.workflow_id,
            revision_id=revision.id,
            selections=selections,
            available_node_types=available_node_types,
            available_asset_filenames=available_asset_filenames,
            installed_package_versions=installed_package_versions,
        )
        if not _offer_matches_review(offer, reviewed):
            raise WorkflowInstallOfferError(
                "workflow-install-offer-changed",
                "Workflow dependencies or install plans changed; review them again.",
            )
    except WorkflowInstallOfferError as exc:
        invalidate_workflow_install_offer(offer, code=exc.code, reason=str(exc))
        raise
    return offer, reviewed.download_requests


def mark_workflow_install_offer_queued(offer: WorkflowInstallOffer) -> None:
    if offer.status != _READY:
        raise WorkflowInstallOfferError(
            "workflow-install-offer-not-actionable",
            "Workflow install offer is no longer actionable.",
        )
    offer.status = "queued"
    offer.queued_at = utcnow()


def invalidate_workflow_install_offer(
    offer: WorkflowInstallOffer,
    *,
    code: str,
    reason: str,
) -> None:
    if offer.status in {"completed", "invalidated", "expired"}:
        return
    offer.status = "invalidated"
    offer.invalidated_at = utcnow()
    offer.invalidation_code = code[:80]
    offer.invalidation_reason = reason


def _review_offer(
    session: Session,
    *,
    workflow_id: str,
    revision_id: str,
    selections: Sequence[WorkflowAssetPlanSelection],
    available_node_types: Collection[str],
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
) -> ReviewedWorkflowInstallOffer:
    revision = _eligible_revision(session, workflow_id, revision_id)
    try:
        analysis = analyze_comfyui_workflow_package(
            revision.ui_graph_json,
            available_node_types=available_node_types,
            available_asset_filenames=available_asset_filenames,
            installed_package_versions=installed_package_versions,
        )
    except WorkflowPackageError as exc:
        raise WorkflowInstallOfferError(exc.code, str(exc)) from exc

    blocking = tuple(
        issue.code
        for issue in analysis.issues
        if issue.severity == "blocking" and issue.code != "missing_asset"
    )
    if analysis.missing_node_types or blocking:
        raise WorkflowInstallOfferError(
            "workflow-install-offer-incomplete",
            "Resolve non-model workflow dependencies before reviewing downloads.",
        )
    missing = tuple(
        reference
        for reference in analysis.asset_references
        if reference.policy == "supported" and not reference.present_locally
    )
    if not missing:
        raise WorkflowInstallOfferError(
            "workflow-install-not-needed",
            "This workflow has no missing downloadable assets.",
        )
    if len(selections) > MAX_WORKFLOW_ASSET_BINDINGS:
        raise WorkflowInstallOfferError(
            "too-many-workflow-install-selections",
            "Workflow install offer contains too many selections.",
        )

    plan_ids = {selection.install_plan_id for selection in selections}
    plans = {
        plan.id: plan
        for plan in session.scalars(select(InstallPlan).where(InstallPlan.id.in_(plan_ids))).all()
    }
    try:
        materialized, plans = materialize_workflow_asset_aliases(
            session,
            analysis.asset_references,
            selections,
            plans,
        )
        binding = bind_workflow_assets_to_install_plans(
            analysis.asset_references,
            materialized,
            plans,
        )
        downloads = compose_workflow_asset_download_requests(
            binding,
            plans,
            expected_binding_plan_hash=binding.plan_hash,
        )
    except (
        WorkflowAssetAliasError,
        WorkflowAssetBindingError,
        WorkflowAssetDownloadError,
    ) as exc:
        raise WorkflowInstallOfferError(exc.code, str(exc)) from exc

    serialized_selections = tuple(
        {
            "reference_filename": selection.reference_filename,
            "install_plan_id": selection.install_plan_id,
            "artifact_path": selection.artifact_path,
        }
        for selection in sorted(
            materialized,
            key=lambda item: (item.reference_filename.casefold(), item.install_plan_id),
        )
    )
    assets = tuple(asset.as_dict() for asset in binding.assets)
    plan_count = len(downloads)
    total_bytes = _total_download_bytes(binding.assets, plans)
    offer_sha256 = _offer_sha256(
        revision,
        binding_plan_sha256=binding.plan_hash,
        assets=assets,
    )
    return ReviewedWorkflowInstallOffer(
        revision=revision,
        selections=serialized_selections,
        assets=assets,
        binding_plan_sha256=binding.plan_hash,
        offer_sha256=offer_sha256,
        plan_count=plan_count,
        total_bytes=total_bytes,
        download_requests=downloads,
    )


def _eligible_revision(
    session: Session,
    workflow_id: str,
    revision_id: str,
) -> WorkflowRevision:
    revision = session.get(WorkflowRevision, revision_id)
    definition = session.get(WorkflowDefinition, workflow_id)
    if revision is None or definition is None or revision.workflow_id != definition.id:
        raise WorkflowInstallOfferError(
            "workflow-revision-unavailable",
            "Workflow revision is unavailable.",
        )
    if definition.current_revision_id != revision.id:
        raise WorkflowInstallOfferError(
            "workflow-revision-not-current",
            "Only the current workflow revision can receive an install offer.",
        )
    if definition.family_id is not None:
        family = session.get(WorkflowFamily, definition.family_id)
        if family is None or family.archived or not family.enabled:
            raise WorkflowInstallOfferError(
                "workflow-family-unavailable",
                "The workflow family is disabled or archived.",
            )
    if (
        revision.engine != "comfyui"
        or not revision.trusted
        or not isinstance(revision.ui_graph_json, dict)
        or not revision.ui_graph_json
    ):
        raise WorkflowInstallOfferError(
            "workflow-revision-needs-attention",
            "The workflow revision is not trusted ComfyUI content.",
        )
    if (
        not isinstance(revision.artifact_sha256, str)
        or not _DIGEST.fullmatch(revision.artifact_sha256)
        or not isinstance(revision.dependency_contract_sha256, str)
        or not _DIGEST.fullmatch(revision.dependency_contract_sha256)
    ):
        raise WorkflowInstallOfferError(
            "workflow-revision-needs-attention",
            "The workflow revision lacks immutable execution identity.",
        )
    artifact_sha256 = workflow_artifact_contract(
        operation=definition.operation,
        engine=revision.engine,
        api_graph=revision.api_graph_json,
        input_schema=revision.input_schema_json,
        dependencies=revision.dependencies_json,
    )
    if revision.artifact_sha256 != artifact_sha256:
        raise WorkflowInstallOfferError(
            "workflow-artifact-drift",
            "Workflow execution content no longer matches its recorded identity.",
        )
    _assert_dependency_contract(session, revision)
    return revision


def _assert_dependency_contract(session: Session, revision: WorkflowRevision) -> None:
    rows = list(
        session.scalars(
            select(WorkflowDependencySlot)
            .where(WorkflowDependencySlot.workflow_revision_id == revision.id)
            .order_by(WorkflowDependencySlot.ordinal)
        ).all()
    )
    if [row.ordinal for row in rows] != list(range(len(rows))):
        raise WorkflowInstallOfferError(
            "workflow-contract-drift",
            "Workflow dependency slot ordering no longer matches its contract.",
        )
    payload = {
        "version": 1,
        "slots": [
            {
                "name": row.name,
                "resource_kind": row.resource_kind,
                "required": row.required,
                "satisfaction": row.satisfaction,
                "requirements": row.requirements_json,
            }
            for row in rows
        ],
    }
    try:
        contract = parse_workflow_dependency_contract(payload)
    except WorkflowDependencyError as exc:
        raise WorkflowInstallOfferError(exc.code, str(exc)) from exc
    by_name = {slot.name: slot for slot in contract.slots}
    if any(
        row.name not in by_name
        or row.contract_sha256 != workflow_dependency_slot_sha256(by_name[row.name])
        for row in rows
    ) or revision.dependency_contract_sha256 != workflow_dependency_contract_sha256(contract):
        raise WorkflowInstallOfferError(
            "workflow-contract-drift",
            "Workflow dependency contract no longer matches its recorded identity.",
        )


def _stored_selections(value: object) -> tuple[WorkflowAssetPlanSelection, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_WORKFLOW_ASSET_BINDINGS:
        raise WorkflowInstallOfferError(
            "invalid-workflow-install-offer",
            "Stored workflow install selections are invalid.",
        )
    selections: list[WorkflowAssetPlanSelection] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "reference_filename",
            "install_plan_id",
            "artifact_path",
        }:
            raise WorkflowInstallOfferError(
                "invalid-workflow-install-offer",
                "Stored workflow install selections are invalid.",
            )
        if not all(isinstance(item[key], str) for key in item):
            raise WorkflowInstallOfferError(
                "invalid-workflow-install-offer",
                "Stored workflow install selections are invalid.",
            )
        selections.append(
            WorkflowAssetPlanSelection(
                reference_filename=item["reference_filename"],
                install_plan_id=item["install_plan_id"],
                artifact_path=item["artifact_path"],
            )
        )
    return tuple(selections)


def _total_download_bytes(
    assets: Sequence[BoundWorkflowAsset],
    plans: Mapping[str, InstallPlan],
) -> int:
    plan_ids = {asset.install_plan_id for asset in assets}
    total = 0
    for plan_id in sorted(plan_ids):
        plan = plans[plan_id]
        artifacts = plan.artifacts_json
        if not isinstance(artifacts, list):
            raise WorkflowInstallOfferError(
                "invalid-install-plan",
                "Install plan has no bounded artifact list.",
            )
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("required", True) is not True:
                continue
            size = artifact.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise WorkflowInstallOfferError(
                    "unverified-install-artifact",
                    "Every required workflow asset needs an exact size.",
                )
            total += size
    if total <= 0:
        raise WorkflowInstallOfferError(
            "invalid-install-plan",
            "Workflow install offer has no downloadable bytes.",
        )
    return total


def _offer_sha256(
    revision: WorkflowRevision,
    *,
    binding_plan_sha256: str,
    assets: Sequence[dict[str, str | int]],
) -> str:
    payload = {
        "version": WORKFLOW_INSTALL_OFFER_VERSION,
        "workflow_revision_id": revision.id,
        "workflow_artifact_sha256": revision.artifact_sha256,
        "dependency_contract_sha256": revision.dependency_contract_sha256,
        "binding_plan_sha256": binding_plan_sha256,
        "assets": list(assets),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _offer_matches_review(
    offer: WorkflowInstallOffer,
    reviewed: ReviewedWorkflowInstallOffer,
) -> bool:
    revision = reviewed.revision
    return (
        offer.workflow_revision_id == revision.id
        and offer.workflow_artifact_sha256 == revision.artifact_sha256
        and offer.dependency_contract_sha256 == revision.dependency_contract_sha256
        and offer.binding_plan_sha256 == reviewed.binding_plan_sha256
        and offer.offer_sha256 == reviewed.offer_sha256
        and offer.selections_json == [dict(item) for item in reviewed.selections]
        and offer.assets_json == [dict(item) for item in reviewed.assets]
        and offer.plan_count == reviewed.plan_count
        and offer.total_bytes == reviewed.total_bytes
    )
