"""Declare the codes shared by workflow installation refusals and invalidations."""

from typing import Literal

from .workflow_dependency_error_types import (
    WorkflowAssetAliasErrorCode,
    WorkflowDependencyErrorCode,
)
from .workflow_graph_error_types import WorkflowGraphErrorCode

WorkflowInstallOfferErrorCode = (
    Literal[
        "invalid-install-plan",
        "invalid-workflow-install-offer",
        "too-many-workflow-install-selections",
        "unverified-install-artifact",
        "workflow-artifact-drift",
        "workflow-contract-drift",
        "workflow-family-unavailable",
        "workflow-install-not-needed",
        "workflow-install-offer-changed",
        "workflow-install-offer-incomplete",
        "workflow-install-offer-not-actionable",
        "workflow-install-offer-not-found",
        "workflow-revision-needs-attention",
        "workflow-revision-not-current",
        "workflow-revision-unavailable",
    ]
    | WorkflowGraphErrorCode
    | WorkflowAssetAliasErrorCode
    | WorkflowDependencyErrorCode
)


WorkflowInstallOfferInvalidationCode = (
    WorkflowInstallOfferErrorCode | Literal["asset-download-refused", "offer-superseded"]
)
