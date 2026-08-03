"""The browser's hand-written types must not drift from the API's models.

56 TypeScript interfaces mirror the pydantic models with nothing keeping them
honest, and they have drifted for real: #220 added `started_at`/`completed_at`
to `JobOut` and the TypeScript `Job` never got them, so that PR's phase-duration
data was unreachable from the client until someone noticed by hand.

This compares field *names* per model rather than generating `types.ts`
wholesale. Generation would replace curated unions and comments with mechanical
output across the whole file; what actually failed here was a field silently
appearing on one side only, and that is exactly what this catches. When a model
gains a field, this test names it and the browser type is updated in the same
change - the drift cannot reach main.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[3]
TYPES_FILE = REPOSITORY / "apps" / "web" / "src" / "types.ts"

# TypeScript interface -> OpenAPI component. Only pairs listed here are
# checked; add a pair when a browser type starts mirroring a server model.
# Server-only models (requests, bundles the browser posts blindly) and
# browser-only view models are deliberately absent.
CHECKED_CONTRACTS = {
    "ApplicationInfo": "ApplicationInfo",
    "ArtifactCleanupResult": "ArtifactCleanupResult",
    "ArtifactDeleteResult": "ArtifactDeleteResult",
    "ArtifactLibraryItem": "ArtifactLibraryItem",
    "ArtifactStorageInfo": "ArtifactStorageInfo",
    "BackupInfo": "BackupInfo",
    "CatalogDetail": "CatalogDetail",
    "CatalogModel": "CatalogModel",
    "CatalogPage": "CatalogPage",
    "CatalogPreflight": "CatalogPreflight",
    "ChatDetail": "ChatDetail",
    "CredentialStatus": "CredentialStatus",
    "DraftClassification": "DraftClassification",
    "EditTemplate": "EditTemplateOut",
    "EngineCapabilities": "EngineCapabilities",
    "ExchangeDeletion": "ExchangeDeletionOut",
    "Job": "JobOut",
    "Message": "MessageOut",
    "MessagePart": "MessagePartOut",
    "ModelStorageInfo": "ModelStorageInfo",
    "ModelUpdate": "ModelUpdateOut",
    "PlatformMatrixEntry": "PlatformMatrixEntry",
    "RegistryInstall": "RegistryInstallOut",
    "ResponseRevision": "ResponseRevisionOut",
    "Run": "RunOut",
    "RuntimeStatus": "RuntimeStatus",
    "SettingField": "SettingField",
    "SetupReadinessReport": "SetupReadinessReport",
    "StorageCleanupResult": "StorageCleanupResult",
    "SystemInfo": "SystemInfo",
    "ToolCapabilityProbe": "ToolCapabilityProbe",
    "TurnAccepted": "TurnAccepted",
    "WorkerLogLocation": "WorkerLogLocation",
    "WorkerLogTail": "WorkerLogTail",
    "WorkerResetResult": "WorkerResetResult",
    "WorkerSettings": "WorkerSettings",
    "WorkerStatus": "WorkerStatus",
    "WorkflowAssetReference": "WorkflowAssetReferenceOut",
    "WorkflowDependencyImpact": "WorkflowDependencyImpactOut",
    "WorkflowFamily": "WorkflowFamilyOut",
    "WorkflowFamilyPreferenceUpdate": "WorkflowFamilyPreferenceUpdate",
    "WorkflowFamilyRemovalImpact": "WorkflowFamilyRemovalImpactOut",
    "WorkflowFamilyPreference": "WorkflowFamilyPreferenceOut",
    "WorkflowFamilyVariant": "WorkflowFamilyVariantOut",
    "WorkflowPackageAnalysis": "WorkflowPackageAnalysisOut",
    "WorkflowPackageIssue": "WorkflowPackageIssueOut",
    "WorkflowPackageRequirement": "WorkflowPackageRequirementOut",
    "WorkflowResourceConsumer": "WorkflowResourceConsumerOut",
    "WorkflowResourceConsumers": "WorkflowResourceConsumersOut",
    "WorkflowSelection": "WorkflowSelectionOut",
    "WorkflowSourceCandidate": "WorkflowSourceCandidateOut",
    "WorkPlan": "WorkPlanOut",
    "WorkStep": "WorkStepOut",
}

# Fields the browser deliberately does not mirror, with the reason. Anything
# not listed here is drift, not a decision.
ALLOWED_MISSING = {
    ("Job", "payload_json"): "opaque server payload; the browser reads named fields",
    ("Run", "provenance_json"): "read through helpers, not as a typed shape",
}


def _openapi_schemas() -> dict[str, dict]:
    result = subprocess.run(  # noqa: S603 - fixed argv, repository-local script
        [sys.executable, str(REPOSITORY / "scripts" / "export-openapi.py")],
        capture_output=True,
        check=True,
        cwd=REPOSITORY,
    )
    document = json.loads(result.stdout)
    schemas = document["components"]["schemas"]
    assert isinstance(schemas, dict)
    return schemas


def _typescript_fields(source: str, interface: str) -> set[str]:
    """Field names declared directly on one interface.

    Depth-aware on purpose: several interfaces nest an inline object literal
    (`install_plan: { id: string; ... }`), and those inner names belong to the
    nested shape rather than to the interface being compared.
    """

    match = re.search(
        rf"^export interface {interface}(?: extends ([\w, ]+?))? \{{$",
        source,
        re.MULTILINE,
    )
    if not match:
        raise AssertionError(f"{interface} is not declared in types.ts")
    depth = 1
    fields: set[str] = set()
    # An extending interface has its parent's fields too, exactly as the
    # server model inherits from its base.
    for parent in (match.group(1) or "").replace(" ", "").split(","):
        if parent:
            fields |= _typescript_fields(source, parent)
    for line in source[match.end() :].splitlines()[1:]:
        stripped = line.strip()
        if depth == 1:
            field = re.match(r"([A-Za-z_][\w]*)\??\s*:", stripped)
            if field:
                fields.add(field.group(1))
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break
    return fields


@pytest.fixture(scope="module")
def schemas() -> dict[str, dict]:
    return _openapi_schemas()


@pytest.fixture(scope="module")
def types_source() -> str:
    return TYPES_FILE.read_text(encoding="utf-8")


@pytest.mark.parametrize(("interface", "component"), sorted(CHECKED_CONTRACTS.items()))
def test_browser_type_mirrors_the_api_model(
    interface: str,
    component: str,
    schemas: dict[str, dict],
    types_source: str,
) -> None:
    schema = schemas.get(component)
    assert schema is not None, f"{component} is no longer in the OpenAPI schema"
    server_fields = set(schema.get("properties", {}))
    browser_fields = _typescript_fields(types_source, interface)
    allowed = {name for (owner, name) in ALLOWED_MISSING if owner == interface}
    missing = server_fields - browser_fields - allowed
    assert not missing, (
        f"{interface} in types.ts is missing {sorted(missing)}, which {component} "
        "returns. Add the field, or record it in ALLOWED_MISSING with a reason."
    )
    invented = browser_fields - server_fields
    assert not invented, (
        f"{interface} in types.ts declares {sorted(invented)}, which {component} "
        "does not return. Remove it, or map the interface to the right model."
    )


def test_every_checked_component_still_exists(schemas: dict[str, dict]) -> None:
    """A renamed model must not silently drop out of the comparison."""
    unknown = sorted(set(CHECKED_CONTRACTS.values()) - set(schemas))
    assert not unknown, f"CHECKED_CONTRACTS names components that no longer exist: {unknown}"
