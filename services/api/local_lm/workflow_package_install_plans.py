"""Preserve raw workflow source and exact download plans before approval."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from typing import Any, Literal

from pydantic import Field
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from .comfy_workflow_packages import (
    WorkflowPackageIssueCode,
    analyze_comfyui_workflow_package,
    validate_bounded_workflow_json,
)
from .domain import Operation
from .models import InstallPlan, WorkflowPackageInstallPlan
from .revision_dependency_contract import declared_dependency_contract
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .schemas import (
    ApiModel,
    BoundWorkflowAssetOut,
    DownloadRequest,
    WorkflowAssetSelectionIn,
    WorkflowPackageDraftRequest,
    WorkflowPackageRequirementOut,
)
from .workflow_asset_aliases import materialize_workflow_asset_aliases
from .workflow_asset_bindings import (
    MAX_WORKFLOW_ASSET_BINDINGS,
    WorkflowAssetPlanSelection,
    bind_workflow_assets_to_install_plans,
)
from .workflow_asset_downloads import compose_workflow_asset_download_requests
from .workflow_bindings import WorkflowBindingError
from .workflow_dependencies import (
    MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS,
    MAX_WORKFLOW_DEPENDENCY_SLOTS,
    WorkflowDependencyError,
    workflow_dependency_contract_payload,
    workflow_dependency_contract_sha256,
)
from .workflow_install_offers import _total_download_bytes
from .workflow_package_extension_preflight import WorkflowPackageExtensionPreflight
from .workflow_runtime_nodes import WorkflowRuntimeNodeInventory
from .workflow_source_dependency_budget import validate_workflow_source_dependency_budget
from .workflow_trust import canonical_graph


class WorkflowPackageInstallPlanRequest(WorkflowPackageDraftRequest):
    dependencies: dict[str, Any] = Field(default_factory=dict)
    selections: list[WorkflowAssetSelectionIn] = Field(
        default_factory=list, max_length=MAX_WORKFLOW_ASSET_BINDINGS
    )


WorkflowPackagePlanBlocker = (
    WorkflowPackageIssueCode
    | Literal[
        "dependency-contract-unknown",
        "dependency-contract-capacity-exceeded",
        "dependency-contract-unrepresentable",
        "node-inventory-unavailable",
        "runtime-nodes-unavailable",
        "extension-plan-unavailable",
        "runtime-plan-unavailable",
        "runtime-installation-unavailable",
    ]
)


class WorkflowPackagePreflight(ApiModel):
    version: Literal[1] = 1
    workflow_name: str
    operation: Operation
    source_sha256: str
    dependency_contract_sha256: str | None
    asset_binding_sha256: str
    install_plan_sha256: dict[str, str]
    assets: list[BoundWorkflowAssetOut]
    download_requests: list[DownloadRequest]
    extensions: list[WorkflowPackageRequirementOut]
    extension_execution: WorkflowPackageExtensionPreflight = Field(
        default_factory=WorkflowPackageExtensionPreflight
    )
    required_node_types: list[str]
    missing_node_types: list[str]
    blockers: list[WorkflowPackagePlanBlocker]
    asset_download_bytes: int
    runtime_plan: RuntimeProvisioningPlan | None = None
    total_download_bytes: int | None
    can_accept: bool


class WorkflowPackageInstallPlanOut(WorkflowPackagePreflight):
    id: str
    plan_sha256: str


class WorkflowPackageInstallPlanError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _digest(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_graph(value).encode("utf-8")).hexdigest()


def _install_plan_fingerprint(plan: InstallPlan) -> str:
    return _digest(
        {
            "provider": plan.provider,
            "remote_id": plan.remote_id,
            "revision": plan.revision,
            "role": plan.role,
            "engine": plan.engine,
            "architecture": plan.architecture,
            "family": plan.family,
            "plan_hash": plan.plan_hash,
            "resolver_version": plan.resolver_version,
            "compatibility": plan.compatibility,
            "artifacts": plan.artifacts_json,
            "runtime_contract": plan.runtime_contract_json,
            "activation_probe": plan.activation_probe_json,
        }
    )


def verify_workflow_package_download_plans(
    session: Session, saved: WorkflowPackageInstallPlanOut
) -> None:
    """Retain exact execution inputs while allowing download status to advance."""

    for identifier, expected in saved.install_plan_sha256.items():
        plan = session.get(InstallPlan, identifier)
        if plan is None or _install_plan_fingerprint(plan) != expected:
            raise WorkflowPackageInstallPlanError(
                "workflow-package-install-plan-changed",
                "The accepted workflow download plan changed.",
            )


def _prepare_plan(
    session: Session,
    payload: WorkflowPackageInstallPlanRequest,
    *,
    available_node_types: Collection[str] | None,
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
    extension_execution: WorkflowPackageExtensionPreflight | None = None,
    runtime_plan: RuntimeProvisioningPlan | None = None,
    runtime_node_inventory: WorkflowRuntimeNodeInventory | None = None,
) -> tuple[dict[str, Any], WorkflowPackagePreflight, str]:
    validate_bounded_workflow_json(payload.ui_graph)
    planned_runtime_nodes = (
        runtime_plan is not None
        and runtime_plan.operation == "install_managed"
        and runtime_node_inventory is not None
        and runtime_node_inventory.runtime_plan_sha256 == runtime_plan.plan_sha256
    )
    if runtime_plan is not None and runtime_plan.operation == "install_managed":
        # An existing worker cannot describe the different runtime being installed.
        # Its measured catalog digest is already bound by the runtime plan's inputs.
        available_node_types = (
            runtime_node_inventory.node_types
            if planned_runtime_nodes and runtime_node_inventory is not None
            else None
        )
    contract = declared_dependency_contract(payload.dependencies)
    if payload.dependencies and contract is None:
        raise WorkflowPackageInstallPlanError(
            "workflow-package-declaration-invalid", "The dependency declaration is not versioned."
        )
    normalized = payload.model_copy(
        update={
            "dependencies": workflow_dependency_contract_payload(contract) if contract else {},
            "selections": sorted(
                payload.selections,
                key=lambda item: (
                    item.reference_filename.casefold(),
                    item.install_plan_id,
                    item.artifact_path,
                ),
            ),
        },
        deep=True,
    )
    request_json = normalized.model_dump(mode="json")
    analysis = analyze_comfyui_workflow_package(
        normalized.ui_graph,
        available_node_types=available_node_types or (),
        available_asset_filenames=available_asset_filenames,
        installed_package_versions=installed_package_versions,
    )
    execution = extension_execution or WorkflowPackageExtensionPreflight()
    requirements = {item.package_id: item for item in analysis.custom_packages}
    if (
        execution.plans.keys() | execution.errors.keys()
    ) - requirements.keys() or execution.plans.keys() & execution.errors.keys():
        raise WorkflowPackageInstallPlanError(
            "workflow-package-extension-plan-mismatch",
            "The extension plans do not match this workflow source.",
        )
    for identifier, plan in execution.plans.items():
        requirement = requirements[identifier]
        if len(requirement.versions) != 1:
            raise WorkflowPackageInstallPlanError(
                "workflow-package-extension-plan-mismatch",
                "The workflow must declare an exact extension version.",
            )
        plan.verify_requested(identifier, requirement.versions[0], requirement.node_types)
    extension_cost_known = execution.plans.keys() == requirements.keys()
    planned_node_types = {
        origin.node_type
        for origin in analysis.node_provenance
        if len(origin.package_ids) == len(origin.package_versions) == 1
        and not origin.core_claimed
        and not origin.unattributed
        and origin.package_ids[0] in execution.plans
        and origin.node_type in execution.plans[origin.package_ids[0]].resolution.node_types
    }
    identifiers = {selection.install_plan_id for selection in normalized.selections}
    plans = {
        plan.id: plan
        for plan in session.scalars(select(InstallPlan).where(InstallPlan.id.in_(identifiers)))
    }
    selections, plans = materialize_workflow_asset_aliases(
        session,
        analysis.asset_references,
        [WorkflowAssetPlanSelection(**item.model_dump()) for item in normalized.selections],
        plans,
    )
    binding = bind_workflow_assets_to_install_plans(analysis.asset_references, selections, plans)
    requests = compose_workflow_asset_download_requests(
        binding, plans, expected_binding_plan_hash=binding.plan_hash
    )
    blockers: set[WorkflowPackagePlanBlocker] = {
        issue.code
        for issue in analysis.issues
        if issue.severity == "blocking" and issue.code != "missing_asset"
    }
    if extension_cost_known:
        # Exact plans cover installation; execution still requires preparation and trust.
        blockers.discard("unresolved_custom_node_package")
    if contract is None:
        blockers.add("dependency-contract-unknown")
    else:
        # Each download can add one resource; extensions and the runtime add one each.
        added = len(requests) + len(requirements) + 1
        if (
            len(contract.slots) + added > MAX_WORKFLOW_DEPENDENCY_SLOTS
            or sum(len(slot.requirements) for slot in contract.slots) + added
            > MAX_WORKFLOW_DEPENDENCY_REQUIREMENTS
        ):
            blockers.add("dependency-contract-capacity-exceeded")
    if runtime_plan is None:
        blockers.add("runtime-plan-unavailable")
    elif (
        runtime_plan.engine != "comfyui"
        or min(runtime_plan.download_bytes, runtime_plan.required_free_bytes) < 0
    ):
        raise WorkflowPackageInstallPlanError(
            "workflow-package-runtime-plan-invalid", "The workflow runtime plan is invalid."
        )
    elif runtime_plan.operation == "install_managed" and not planned_runtime_nodes:
        blockers.add("runtime-installation-unavailable")
    if (
        contract is not None
        and runtime_plan is not None
        and "dependency-contract-capacity-exceeded" not in blockers
    ):
        try:
            validate_workflow_source_dependency_budget(
                contract, requests, plans, execution.plans, runtime_plan.engine
            )
        except (WorkflowDependencyError, WorkflowBindingError) as exc:
            blockers.add(
                "dependency-contract-capacity-exceeded"
                if exc.code
                in {
                    "too_many_workflow_dependencies",
                    "dependency_data_too_large",
                    "dependency_data_too_deep",
                }
                else "dependency-contract-unrepresentable"
            )
    if available_node_types is None:
        blockers.add("node-inventory-unavailable")
    elif set(analysis.missing_node_types) - planned_node_types:
        blockers.add("runtime-nodes-unavailable")
    if not extension_cost_known:
        blockers.add("extension-plan-unavailable")
    asset_bytes = _total_download_bytes(binding.assets, plans) if requests else 0
    preflight = WorkflowPackagePreflight(
        workflow_name=normalized.name,
        operation=normalized.operation,
        source_sha256=_digest(normalized.ui_graph),
        dependency_contract_sha256=(
            workflow_dependency_contract_sha256(contract) if contract else None
        ),
        asset_binding_sha256=binding.plan_hash,
        install_plan_sha256={
            identifier: _install_plan_fingerprint(plans[identifier])
            for identifier in sorted({asset.install_plan_id for asset in binding.assets})
        },
        assets=[BoundWorkflowAssetOut.model_validate(asset.as_dict()) for asset in binding.assets],
        download_requests=list(requests),
        extensions=[
            WorkflowPackageRequirementOut(
                package_id=package.package_id,
                versions=list(package.versions),
                node_types=list(package.node_types),
                locally_resolved=package.locally_resolved,
            )
            for package in analysis.custom_packages
        ],
        extension_execution=execution,
        required_node_types=list(analysis.required_node_types),
        missing_node_types=list(analysis.missing_node_types),
        blockers=sorted(blockers),
        asset_download_bytes=asset_bytes,
        runtime_plan=runtime_plan,
        total_download_bytes=(
            asset_bytes
            + runtime_plan.download_bytes
            + sum(plan.download_bytes for plan in execution.plans.values())
            if extension_cost_known and runtime_plan is not None
            else None
        ),
        can_accept=not blockers,
    )
    digest = _digest({"request": request_json, "preflight": preflight.model_dump(mode="json")})
    return request_json, preflight, digest


def create_workflow_package_install_plan(
    session: Session,
    payload: WorkflowPackageInstallPlanRequest,
    *,
    available_node_types: Collection[str] | None,
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
    extension_execution: WorkflowPackageExtensionPreflight | None = None,
    runtime_plan: RuntimeProvisioningPlan | None = None,
    runtime_node_inventory: WorkflowRuntimeNodeInventory | None = None,
) -> WorkflowPackageInstallPlanOut:
    request_json, preflight, digest = _prepare_plan(
        session,
        payload,
        available_node_types=available_node_types,
        available_asset_filenames=available_asset_filenames,
        installed_package_versions=installed_package_versions,
        extension_execution=extension_execution,
        runtime_plan=runtime_plan,
        runtime_node_inventory=runtime_node_inventory,
    )
    # Identical retries share one immutable record, including concurrent ones.
    # The caller commits the plan and any alias records in the same transaction.
    session.execute(
        insert(WorkflowPackageInstallPlan)
        .values(
            id="wfpplan_" + digest[:32],
            plan_sha256=digest,
            request_json=request_json,
            preflight_json=preflight.model_dump(mode="json"),
        )
        .on_conflict_do_nothing(index_elements=["plan_sha256"])
    )
    record = session.scalar(
        select(WorkflowPackageInstallPlan).where(WorkflowPackageInstallPlan.plan_sha256 == digest)
    )
    if (
        record is None
        or record.id != "wfpplan_" + digest[:32]
        or _digest({"request": record.request_json, "preflight": record.preflight_json}) != digest
    ):
        raise WorkflowPackageInstallPlanError(
            "workflow-package-install-plan-changed",
            "The stored workflow installation plan changed.",
        )
    return WorkflowPackageInstallPlanOut(
        id="wfpplan_" + digest[:32], plan_sha256=digest, **preflight.model_dump()
    )


def load_stored_workflow_package_install_plan(
    session: Session, plan_id: str
) -> tuple[WorkflowPackageInstallPlan, WorkflowPackageInstallPlanOut]:
    """Verify an accepted snapshot without treating installation progress as plan drift."""

    record = session.get(WorkflowPackageInstallPlan, plan_id)
    if record is None:
        raise WorkflowPackageInstallPlanError(
            "workflow-package-install-plan-not-found",
            "The workflow installation plan is unavailable.",
        )
    stored = _digest({"request": record.request_json, "preflight": record.preflight_json})
    if stored != record.plan_sha256 or record.id != "wfpplan_" + stored[:32]:
        raise WorkflowPackageInstallPlanError(
            "workflow-package-install-plan-changed", "The workflow installation plan changed."
        )
    return record, WorkflowPackageInstallPlanOut(
        id=record.id, plan_sha256=stored, **record.preflight_json
    )


def revalidate_workflow_package_install_plan(
    session: Session,
    plan_id: str,
    *,
    available_node_types: Collection[str] | None,
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]],
    extension_execution: WorkflowPackageExtensionPreflight | None = None,
    runtime_plan: RuntimeProvisioningPlan | None = None,
    runtime_node_inventory: WorkflowRuntimeNodeInventory | None = None,
) -> WorkflowPackageInstallPlanOut:
    record, saved = load_stored_workflow_package_install_plan(session, plan_id)
    _, preflight, digest = _prepare_plan(
        session,
        WorkflowPackageInstallPlanRequest.model_validate(record.request_json),
        available_node_types=available_node_types,
        available_asset_filenames=available_asset_filenames,
        installed_package_versions=installed_package_versions,
        extension_execution=extension_execution,
        runtime_plan=runtime_plan,
        runtime_node_inventory=runtime_node_inventory,
    )
    if digest != saved.plan_sha256:
        raise WorkflowPackageInstallPlanError(
            "workflow-package-install-plan-changed",
            "Review the updated workflow installation plan.",
        )
    return WorkflowPackageInstallPlanOut(id=record.id, plan_sha256=digest, **preflight.model_dump())
