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
    "PromptBatch": "PromptExpansionBatchOut",
    "PromptBatchItem": "PromptExpansionItemOut",
    "PromptTemplateDefinition": "PromptTemplateDefinitionOut",
    "PromptTemplateDetail": "PromptTemplateDetailOut",
    "PromptTemplatePage": "PromptTemplatePageOut",
    "PromptTemplateRevision": "PromptTemplateRevisionOut",
    "PromptTemplateWriteResult": "PromptTemplateWriteOut",
    "ReferenceAsset": "ReferenceAssetOut",
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


def _typescript_field_type(source: str, interface: str, field: str) -> str | None:
    """The declared type of one field, following `extends` as the field check does."""

    match = re.search(
        rf"^export interface {interface}(?: extends ([\w, ]+?))? \{{$",
        source,
        re.MULTILINE,
    )
    if not match:
        return None
    depth = 1
    for line in source[match.end() :].splitlines()[1:]:
        stripped = line.strip()
        if depth == 1:
            declared = re.match(rf"{field}\??\s*:\s*(.+?);\s*$", stripped)
            if declared:
                return declared.group(1)
        depth += line.count("{") - line.count("}")
        if depth <= 0:
            break
    for parent in (match.group(1) or "").replace(" ", "").split(","):
        if parent:
            inherited = _typescript_field_type(source, parent, field)
            if inherited is not None:
                return inherited
    return None


def _declared_literals(source: str, expression: str | None) -> set[str] | None:
    """The string literals a type expression admits, or None if it is not a union.

    Resolves one level of named alias, because the readiness and status unions
    are written that way and comparing against the alias name proves nothing.
    """

    if expression is None:
        return None
    if '"' in expression:
        return set(re.findall(r'"([^"]+)"', expression))
    alias = re.fullmatch(r"(\w+)(\s*\|\s*null)?", expression.strip())
    if alias:
        # Comments are removed first. A union is read up to its semicolon, and
        # a member documented in a sentence containing one would otherwise cut
        # the union short and report the rest as missing - a contract check
        # that punctuation can defeat is worse than none.
        uncommented = re.sub(r"//.*", "", source)
        declaration = re.search(
            rf"^export type {alias.group(1)} =\s*(.+?);", uncommented, re.MULTILINE | re.DOTALL
        )
        if declaration:
            return set(re.findall(r'"([^"]+)"', declaration.group(1)))
    return None


def _admissible_values(spec: dict, schemas: dict[str, dict]) -> list[str] | None:
    """The string values one field admits, following a component reference.

    An inline `enum` is the shape a bare pydantic model produces. OpenAPI does
    not use it for an enum-typed field - it emits a reference instead:

        JobOut.status  ->  {"$ref": "#/components/schemas/JobStatus"}
        JobStatus      ->  {"enum": [...], "type": "string"}

    Reading only the inline form therefore saw NOTHING for every enum field in
    every checked component - eight of them, which is the entire population
    this check was written for. It had never failed because it had never
    looked. Found by removing a member from a browser union and watching the
    suite stay green.

    One level of reference is followed, which is all the generator emits.
    """

    seen = spec
    for _ in range(2):
        values = seen.get("enum") or next(
            (option.get("enum") for option in seen.get("anyOf", []) if option.get("enum")),
            None,
        )
        if values:
            return values
        reference = seen.get("$ref") or next(
            (option.get("$ref") for option in seen.get("allOf", []) if option.get("$ref")),
            None,
        )
        if not reference:
            return None
        seen = schemas.get(reference.rsplit("/", 1)[-1]) or {}
    return None


@pytest.mark.parametrize(("interface", "component"), sorted(CHECKED_CONTRACTS.items()))
def test_browser_can_represent_every_value_the_server_returns(
    interface: str,
    component: str,
    schemas: dict[str, dict],
    types_source: str,
) -> None:
    """Matching field names is not a matching contract.

    A server union can gain a member without any field appearing or vanishing,
    so the existing comparison sees nothing. `review_required` was added to
    workflow readiness and the browser's union kept three members, which meant
    a workflow awaiting review was described to the user as one that cannot run
    on this machine - a true state reported as a different, false one.
    """
    schema = schemas.get(component) or {}
    for field, spec in (schema.get("properties") or {}).items():
        values = _admissible_values(spec, schemas)
        if not values:
            continue
        declared = _declared_literals(
            types_source, _typescript_field_type(types_source, interface, field)
        )
        if declared is None:
            continue
        served = {value for value in values if isinstance(value, str)}
        missing = served - declared
        assert not missing, (
            f"{interface}.{field} in types.ts cannot represent {sorted(missing)}, which "
            f"{component} returns. Add the member, and handle it where the field is read."
        )
        # And the other direction. A member the server cannot produce is not a
        # missed state, so it costs the user nothing directly - but it is still
        # a false statement about the protocol, it makes dead branches look
        # live, and a hand-maintained union drifts BOTH ways. Probing this test
        # with a deliberately invented member showed the one-directional
        # assertion accepted it silently, which is the same blind spot the
        # declaration exists to remove.
        invented = declared - served
        assert not invented, (
            f"{interface}.{field} in types.ts admits {sorted(invented)}, which "
            f"{component} never returns. Remove the member, or correct the server "
            f"if the browser is right about what can happen."
        )


# The last underscore-separated token of a field name that marks it as naming
# a value from a fixed set rather than free text.
VOCABULARY_TOKENS = frozenset(
    {"code", "state", "status", "kind", "mode", "phase", "type", "severity", "category"}
)
# Fields whose name ends in a vocabulary word and whose type is still a plain
# string. This is a RATCHET BASELINE, not an approval list: every entry is
# known debt, the set must never grow, and it must shrink as fields are typed.
#
# The point is that an open string does not fail anywhere - it simply is not
# checked. `test_browser_can_represent_every_value_the_server_returns` skips a
# field whose declared literals are None, which is exactly what a plain string
# produces, so declaring the union is what switches that check on. Until then
# the browser can compare against prose and nothing notices, which is how the
# readiness-wording break reached main.
OPEN_VOCABULARY_BASELINE = frozenset(
    {
        "BoundWorkflowAssetOut.artifact_kind",
        "BoundWorkflowAssetOut.kind",
        "CatalogPreflight.auxiliary_kind",
        "ChatItemRemovalReferenceOut.subject_kind",
        "DeviceInfo.kind",
        "InstallPlanOut.failure_code",
        "InstallPlanOut.status",
        "JobOut.phase",
        "MessageReferenceOut.subject_kind",
        "ModelAssetOut.kind",
        "ModelCapabilityEvidenceOut.failure_code",
        "ModelUpdateOut.kind",
        "ReferenceSubjectCreate.kind",
        "ReferenceSubjectOut.kind",
        "ResponseRevisionOut.status",
        "SetupReadinessCheck.code",
        "SetupVerificationOut.failure_code",
        "StudioToolCapability.kind",
        "WorkPlanOut.status",
        "WorkStepOut.status",
        "WorkflowDependencyImpactOut.resource_kind",
        "WorkflowInstallOfferOut.invalidation_code",
        "WorkflowMissingNodeOut.node_type",
        "WorkflowPackageIssueOut.code",
    }
)

# Names ending in a vocabulary word that are NOT vocabularies. Each reason is
# paired with the check that enforces it, and the classifier consumes these
# rather than repeating the conditions inline. The prose used to sit here
# unreferenced while the same three rules were hardcoded further down, so a
# reader who trusted this list - which its own comment invited - was reading
# something with no authority over the behaviour, and a fourth exclusion added
# in code would have left the list silently wrong.
EXCLUDED_COMPONENTS = frozenset({"ValidationError", "HTTPValidationError"})


def _is_identifier(name: str) -> bool:
    return name == "id" or name.endswith("_id")


def _is_media_type(name: str) -> bool:
    return "mime" in name or "media_type" in name or "content_type" in name


VOCABULARY_NAME_EXCLUSIONS = (
    ("identifiers - anything ending in _id, and a bare id", _is_identifier),
    (
        "media and mime types - a content type is not a closed vocabulary",
        _is_media_type,
    ),
)

VOCABULARY_COMPONENT_EXCLUSION = "FastAPI validation models, which the application does not define"


def _resolve_reference(
    spec: dict, schemas: dict[str, dict], seen: frozenset[str]
) -> tuple[dict, frozenset[str]]:
    """Follow a local component reference to the schema it names.

    Returns the spec unchanged when it is not a reference, when the target is
    not a local component, or when following it would revisit a component
    already on this chain. The cycle guard is not decoration: a component whose
    property refers back to its own container is ordinary in a tree-shaped
    schema, and without the guard the first one recurses until the interpreter
    gives up.
    """

    reference = spec.get("$ref")
    if not isinstance(reference, str):
        return spec, seen
    prefix = "#/components/schemas/"
    if not reference.startswith(prefix):
        return spec, seen
    name = reference[len(prefix) :]
    if name in seen or name not in schemas:
        return spec, seen
    return schemas[name], seen | {name}


def _is_closed_vocabulary(
    spec: dict, schemas: dict[str, dict], seen: frozenset[str] = frozenset()
) -> bool:
    """An enum, a const, or a reference to something that is one.

    The reference is FOLLOWED. An earlier version read the presence of `$ref`
    as proof of closure, which let any field naming an open string component
    past the ratchet - the exact class it exists to stop. A reference that
    cannot be resolved, because the target is unknown or the chain loops, is
    treated as NOT closed, so an unknown target surfaces instead of being waved
    through.
    """

    resolved, seen = _resolve_reference(spec, schemas, seen)
    if resolved is not spec:
        return _is_closed_vocabulary(resolved, schemas, seen)
    if "enum" in spec or "const" in spec:
        return True
    return any(
        _is_closed_vocabulary(sub, schemas, seen)
        for key in ("anyOf", "oneOf", "allOf")
        for sub in spec.get(key, [])
    )


def _declared_types(
    spec: dict, schemas: dict[str, dict], seen: frozenset[str] = frozenset()
) -> set[str]:
    """Every JSON type this field can take, following unions AND references.

    References matter here for the same reason they matter above: a field whose
    schema is a reference to a string component has no `type` of its own, so
    without resolution it reports no types at all and fails the string check
    that decides whether it is a vocabulary worth reporting.
    """

    resolved, seen = _resolve_reference(spec, schemas, seen)
    if resolved is not spec:
        return _declared_types(resolved, schemas, seen)
    found: set[str] = set()
    if "type" in spec:
        found.add(spec["type"])
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in spec.get(key, []):
            found |= _declared_types(sub, schemas, seen)
    return found


def _open_vocabulary_fields(schemas: dict[str, dict]) -> set[str]:
    """Vocabulary-named STRING fields that carry no closed vocabulary.

    The string check is not decoration. Without it an integer exit_code counts
    as an open vocabulary, because an integer is trivially not an enum - and a
    check that cries wolf on a process exit status teaches people to skip the
    whole report.
    """

    found: set[str] = set()
    for component, spec in schemas.items():
        if component in EXCLUDED_COMPONENTS:
            continue
        for field, field_spec in (spec.get("properties") or {}).items():
            if field.split("_")[-1] not in VOCABULARY_TOKENS:
                continue
            lowered = field.lower()
            if any(excluded(lowered) for _, excluded in VOCABULARY_NAME_EXCLUSIONS):
                continue
            if _is_closed_vocabulary(field_spec, schemas):
                continue
            if "string" not in _declared_types(field_spec, schemas):
                continue
            found.add(f"{component}.{field}")
    return found


def test_every_named_exclusion_is_one_the_classifier_actually_honours() -> None:
    """Each reason must be paired with a check the classifier consumes.

    The reasons used to sit in an unreferenced tuple while the same rules were
    hardcoded inside the classifier. Nothing kept the two agreed: a fourth
    exclusion added in code would have left the list silently wrong, and
    editing the list would have changed nothing. This asserts the pairing
    rather than the prose - an entry whose predicate the classifier ignores
    fails here.
    """
    assert VOCABULARY_NAME_EXCLUSIONS, "no named exclusions declared"
    for reason, excluded in VOCABULARY_NAME_EXCLUSIONS:
        assert isinstance(reason, str) and reason.strip(), reason
        assert callable(excluded), reason

    # A field name that every exclusion should let through, so the loop below
    # is measuring the exclusion rather than a name nothing would report.
    assert not any(excluded("status") for _, excluded in VOCABULARY_NAME_EXCLUSIONS)

    schemas = {"Thing": {"properties": {"status": {"type": "string"}}}}
    assert _open_vocabulary_fields(schemas) == {"Thing.status"}

    for reason, excluded in VOCABULARY_NAME_EXCLUSIONS:
        name = next(
            (
                n
                for n in ("chat_id", "content_type", "media_type", "mime_type", "id")
                if excluded(n)
            ),
            None,
        )
        assert name is not None, f"no sample name matches: {reason}"
        schemas = {"Thing": {"properties": {name: {"type": "string"}}}}
        assert _open_vocabulary_fields(schemas) == set(), (
            f"classifier ignores the exclusion: {reason}"
        )


def test_the_excluded_components_are_honoured() -> None:
    """The component exclusion is paired with its reason the same way."""
    assert VOCABULARY_COMPONENT_EXCLUSION.strip()
    assert EXCLUDED_COMPONENTS
    for component in EXCLUDED_COMPONENTS:
        schemas = {component: {"properties": {"status": {"type": "string"}}}}
        assert _open_vocabulary_fields(schemas) == set(), component


def test_no_new_api_field_is_a_vocabulary_typed_as_an_open_string(
    schemas: dict[str, dict],
) -> None:
    """A new open vocabulary has to be argued for, not merely committed."""

    appeared = sorted(_open_vocabulary_fields(schemas) - OPEN_VOCABULARY_BASELINE)
    assert not appeared, (
        "these API fields are a closed vocabulary typed as a plain string, and "
        f"are not in the baseline: {appeared}. Declare the vocabulary, or add "
        "the field to OPEN_VOCABULARY_BASELINE with the reason it cannot be."
    )


def test_the_open_vocabulary_baseline_does_not_rot(schemas: dict[str, dict]) -> None:
    """Every baselined field must still be open, so the list shrinks.

    Without this the baseline only ever holds: a field could be typed properly,
    or renamed away, and its stale entry would keep silently excusing a name
    that no longer exists. A ratchet that cannot tighten is a list.
    """

    fixed = sorted(OPEN_VOCABULARY_BASELINE - _open_vocabulary_fields(schemas))
    assert not fixed, (
        f"these baselined fields are no longer open strings: {fixed}. Remove "
        "them from OPEN_VOCABULARY_BASELINE - the baseline is meant to shrink."
    )


def test_a_referenced_open_vocabulary_is_not_hidden_from_the_ratchet() -> None:
    """A reference must be followed, not taken as proof of closure.

    This is the case the first version let through: it read the presence of
    `$ref` as closure without resolving it, so a field naming a string
    component was absent from the report entirely. Named root schemas and
    generator changes both produce references, so this is ordinary schema
    evolution rather than a speculative edge.
    """

    schemas: dict[str, dict] = {
        "OpenString": {"type": "string"},
        "ClosedEnum": {"type": "string", "enum": ["ready", "failed"]},
        "Widget": {
            "properties": {
                "referenced_status": {"$ref": "#/components/schemas/OpenString"},
                "referenced_kind": {"$ref": "#/components/schemas/ClosedEnum"},
                "nullable_referenced_state": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/OpenString"},
                        {"type": "null"},
                    ]
                },
                "nullable_referenced_mode": {
                    "anyOf": [
                        {"$ref": "#/components/schemas/ClosedEnum"},
                        {"type": "null"},
                    ]
                },
            }
        },
    }

    found = _open_vocabulary_fields(schemas)
    assert "Widget.referenced_status" in found, "a referenced open string must be reported"
    assert "Widget.nullable_referenced_state" in found, "nullable does not close it"
    assert "Widget.referenced_kind" not in found, "a referenced enum is closed"
    assert "Widget.nullable_referenced_mode" not in found, "nullable does not open it"


def test_a_reference_cycle_terminates_instead_of_recursing() -> None:
    """A component chain that loops must stop, not exhaust the stack.

    Both shapes appear in real documents: a component that refers to itself,
    and two that refer to each other.
    """

    schemas: dict[str, dict] = {
        "Loop": {"$ref": "#/components/schemas/Loop"},
        "Ping": {"$ref": "#/components/schemas/Pong"},
        "Pong": {"$ref": "#/components/schemas/Ping"},
        "Widget": {
            "properties": {
                "self_status": {"$ref": "#/components/schemas/Loop"},
                "mutual_kind": {"$ref": "#/components/schemas/Ping"},
            }
        },
    }

    # Terminating at all is the assertion. Neither resolves to a string, so
    # neither is reported, and an unresolvable target counts as NOT closed.
    assert _open_vocabulary_fields(schemas) == set()
    assert not _is_closed_vocabulary({"$ref": "#/components/schemas/Loop"}, schemas)
    assert not _is_closed_vocabulary({"$ref": "#/components/schemas/Missing"}, schemas)
