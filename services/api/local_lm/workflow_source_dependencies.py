"""Keep every accepted installation resource in the executable dependency contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy.orm import Session

from .models import ModelAssetInstall, ModelInstall, WorkflowInstallOffer
from .workflow_bindings import materialize_model_asset, materialize_model_install
from .workflow_dependencies import (
    parse_workflow_dependency_contract,
)
from .workflow_offer_completion import WorkflowOfferCompletionError, _accepted_results
from .workflow_offer_packages import accepted_workflow_offer_packages
from .workflow_source_dependency_contract import WorkflowSourceDependencyBuilder

if TYPE_CHECKING:
    from .workflow_package_install_plans import (
        WorkflowPackageInstallPlanOut,
        WorkflowPackageInstallPlanRequest,
    )


def compiled_workflow_source_dependencies(
    session: Session,
    offer: WorkflowInstallOffer,
    saved: WorkflowPackageInstallPlanOut,
    payload: WorkflowPackageInstallPlanRequest,
) -> dict[str, Any]:
    """Derive portable requirements from the original approval and its exact results."""
    original = parse_workflow_dependency_contract(payload.dependencies)
    builder = WorkflowSourceDependencyBuilder(original)

    accepted = _accepted_results(session, offer)
    if accepted is None:
        raise WorkflowOfferCompletionError("workflow-downloads-pending")
    for kind, identifier in sorted({item for result in accepted for item in result}):
        if kind == "model_asset":
            asset = session.get(ModelAssetInstall, identifier)
            if asset is None:
                raise WorkflowOfferCompletionError("download-result-unavailable")
            builder.include("model_asset", materialize_model_asset(asset).identity)
        elif kind == "model_install":
            install = session.get(ModelInstall, identifier)
            if install is None:
                raise WorkflowOfferCompletionError("download-result-unavailable")
            builder.include("model_install", materialize_model_install(session, install).identity)
    for package in accepted_workflow_offer_packages(session, offer, saved):
        preparation = package.preparation
        if preparation is None:
            raise WorkflowOfferCompletionError("workflow-extensions-pending")
        builder.include(
            "registry_package",
            {
                "package_id": package.plan.resolution.package_id,
                "package_version": package.plan.resolution.declared_version,
                "archive_sha256": preparation.archive_sha256,
                "manifest_sha256": preparation.manifest_sha256,
                "wheel_closure_sha256": preparation.wheel_closure_sha256,
                "wheel_environment_sha256": preparation.wheel_environment_sha256,
                "node_types": list(package.plan.resolution.node_types),
            },
        )
    if saved.runtime_plan is None:
        raise WorkflowOfferCompletionError("workflow-runtime-plan-unavailable")
    builder.include("runtime", {"engine": saved.runtime_plan.engine})
    return builder.payload()
