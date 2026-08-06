from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

MAX_UI_GRAPH_BYTES = 1024 * 1024
MAX_UI_GRAPH_NODES = 4_096
MAX_UI_GRAPH_LINKS = 16_384
MAX_UI_GRAPH_SUBGRAPHS = 256
MAX_UI_GRAPH_VALUES = 100_000
MAX_UI_GRAPH_DEPTH = 32
MAX_UI_GRAPH_KEY_CHARACTERS = 200
MAX_UI_GRAPH_STRING_CHARACTERS = 65_536
MAX_ASSET_REFERENCE_CHARACTERS = 1_000

SUPPORTED_MODEL_SUFFIXES = frozenset({".safetensors", ".gguf", ".json"})
BLOCKED_MODEL_SUFFIXES = frozenset({".bin", ".ckpt", ".pickle", ".pkl", ".pt", ".pth"})
KNOWN_MODEL_SUFFIXES = SUPPORTED_MODEL_SUFFIXES | BLOCKED_MODEL_SUFFIXES | frozenset({".onnx"})

# These definitions live in the ComfyUI frontend rather than /object_info.
#
# The rgthree entries are group-state controls: they set other nodes' modes
# while the graph is being edited and have no runtime existence at all. By the
# time a saved graph reaches us their work is already done - every node it
# muted or bypassed carries that mode itself - so discarding the controls
# cannot disturb what they applied.
#
# Named individually rather than by a rule like "no cnr_id". A node without a
# package is usually a node whose package we failed to identify, and treating
# that whole class as frontend furniture would silently drop real
# dependencies.
FRONTEND_SYSTEM_NODE_TYPES = frozenset(
    {
        "MarkdownNote",
        "Note",
        "PrimitiveNode",
        "Reroute",
        "Fast Groups Bypasser (rgthree)",
        "Mute / Bypass Relay (rgthree)",
        "Mute / Bypass Repeater (rgthree)",
    }
)

AssetPolicy = Literal["supported", "blocked", "unsupported"]
AssetKind = Literal[
    "checkpoint",
    "configuration",
    "embedding",
    "lora",
    "upscaler",
    "vae",
]
IssueSeverity = Literal["advisory", "blocking"]
OperationGuess = Literal["image", "unknown", "video"]


class WorkflowPackageError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkflowPackageRequirement:
    package_id: str
    versions: tuple[str, ...]
    node_types: tuple[str, ...]
    locally_resolved: bool = False


@dataclass(frozen=True)
class WorkflowAssetReference:
    filename: str
    suffix: str
    policy: AssetPolicy
    kind: AssetKind = "checkpoint"
    source_url: str | None = None
    present_locally: bool = False


@dataclass(frozen=True)
class WorkflowPackageIssue:
    code: str
    count: int
    node_types: tuple[str, ...] = ()
    severity: IssueSeverity = "blocking"


@dataclass(frozen=True)
class WorkflowMissingNode:
    node_type: str
    count: int
    package_id: str | None


@dataclass(frozen=True)
class ComfyWorkflowPackageAnalysis:
    format_version: str
    frontend_version: str | None
    node_count: int
    link_count: int
    subgraph_count: int
    required_node_types: tuple[str, ...]
    frontend_node_types: tuple[str, ...]
    missing_node_types: tuple[str, ...]
    custom_packages: tuple[WorkflowPackageRequirement, ...]
    asset_references: tuple[WorkflowAssetReference, ...]
    issues: tuple[WorkflowPackageIssue, ...]
    missing_nodes: tuple[WorkflowMissingNode, ...]
    operation_guess: OperationGuess
    truncated: bool

    @property
    def runtime_nodes_available(self) -> bool:
        return not self.missing_node_types

    @property
    def dependencies_resolved(self) -> bool:
        blocking = {
            "blocked_asset_format",
            "conflicting_custom_node_versions",
            "missing_asset",
            "remote_url_reference",
            "unsafe_asset_reference",
            "unsupported_asset_format",
            "unidentified_custom_node_package",
            "unresolved_custom_node_package",
            "unversioned_custom_node_package",
        }
        return self.runtime_nodes_available and not any(
            issue.code in blocking for issue in self.issues
        )

    @property
    def ready(self) -> bool:
        return self.dependencies_resolved


@dataclass(frozen=True)
class _GraphScope:
    name: str
    nodes: Sequence[object]
    links: Sequence[object]
    allows_boundary_nodes: bool


@dataclass(frozen=True)
class _NodeRecord:
    node_type: str
    package_id: str | None
    package_version: str | None
    widgets: object


def analyze_comfyui_workflow_package(
    workflow: Mapping[str, object],
    *,
    available_node_types: Collection[str] = (),
    available_asset_filenames: Collection[str] = (),
    installed_package_versions: Mapping[str, Collection[str]] | None = None,
) -> ComfyWorkflowPackageAnalysis:
    """Inspect a ComfyUI v0.4 UI workflow without executing or persisting it."""
    _validate_bounded_json(workflow)
    version = _format_version(workflow.get("version"))
    scopes, subgraph_ids = _graph_scopes(workflow)
    records: list[_NodeRecord] = []
    link_count = 0
    dangling_links = 0
    for scope in scopes:
        records.extend(_validate_nodes(scope))
        dangling_links += _validate_links(scope)
        link_count += len(scope.links)
    if not records:
        raise WorkflowPackageError("empty_workflow", "workflow contains no nodes")
    if len(records) > MAX_UI_GRAPH_NODES:
        raise WorkflowPackageError("too_many_nodes", "workflow has too many nodes")
    if link_count > MAX_UI_GRAPH_LINKS:
        raise WorkflowPackageError("too_many_links", "workflow has too many links")
    structural_issues = (
        (
            WorkflowPackageIssue(
                "dangling_link",
                dangling_links,
                severity="advisory",
            ),
        )
        if dangling_links
        else ()
    )
    return _analysis(
        workflow,
        version,
        records,
        subgraph_ids,
        link_count,
        available_node_types,
        available_asset_filenames,
        installed_package_versions,
        structural_issues,
    )


def _analysis(
    workflow: Mapping[str, object],
    version: str,
    records: Sequence[_NodeRecord],
    subgraph_ids: frozenset[str],
    link_count: int,
    available_node_types: Collection[str],
    available_asset_filenames: Collection[str],
    installed_package_versions: Mapping[str, Collection[str]] | None,
    structural_issues: tuple[WorkflowPackageIssue, ...],
) -> ComfyWorkflowPackageAnalysis:
    all_types = {record.node_type for record in records}
    frontend = all_types & (FRONTEND_SYSTEM_NODE_TYPES | subgraph_ids)
    required = all_types - FRONTEND_SYSTEM_NODE_TYPES - subgraph_ids
    available = {str(value) for value in available_node_types}
    missing = required - available
    packages, package_issues = _package_requirements(
        records,
        required,
        missing,
        available,
        installed_package_versions or {},
    )
    assets, asset_issues = _asset_references(records, available_asset_filenames)
    return ComfyWorkflowPackageAnalysis(
        version,
        _frontend_version(workflow),
        len(records),
        link_count,
        len(subgraph_ids),
        tuple(sorted(required, key=str.casefold)),
        tuple(sorted(frontend, key=str.casefold)),
        tuple(sorted(missing, key=str.casefold)),
        packages,
        assets,
        tuple(
            sorted(
                (*structural_issues, *package_issues, *asset_issues),
                key=lambda item: item.code,
            )
        ),
        _missing_node_requirements(records, missing),
        _operation_guess(required),
        False,
    )


def _validate_bounded_json(value: object) -> None:
    values = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        values += 1
        if values > MAX_UI_GRAPH_VALUES:
            raise WorkflowPackageError("too_many_values", "workflow has too many values")
        if depth > MAX_UI_GRAPH_DEPTH:
            raise WorkflowPackageError("too_deep", "workflow nesting is too deep")
        if isinstance(current, Mapping):
            for key, child in current.items():
                if not isinstance(key, str) or not _bounded_printable(
                    key, MAX_UI_GRAPH_KEY_CHARACTERS
                ):
                    raise WorkflowPackageError("invalid_key", "workflow has an invalid object key")
                stack.append((child, depth + 1))
        elif isinstance(current, list | tuple):
            stack.extend((child, depth + 1) for child in current)
        elif isinstance(current, str):
            if len(current) > MAX_UI_GRAPH_STRING_CHARACTERS or "\x00" in current:
                raise WorkflowPackageError(
                    "invalid_string", "workflow has an invalid or oversized string"
                )
        elif isinstance(current, float):
            if not math.isfinite(current):
                raise WorkflowPackageError("non_finite_number", "workflow has a non-finite number")
        elif current is not None and not isinstance(current, int | bool):
            raise WorkflowPackageError("invalid_value", "workflow has a non-JSON value")
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as exc:
        raise WorkflowPackageError("invalid_json", "workflow is not valid JSON") from exc
    if len(encoded) > MAX_UI_GRAPH_BYTES:
        raise WorkflowPackageError("too_large", "workflow exceeds the UI graph size limit")


def _format_version(value: object) -> str:
    if value != 0.4 and value != "0.4":
        raise WorkflowPackageError(
            "unsupported_format", "workflow must use ComfyUI UI graph version 0.4"
        )
    return "0.4"


def _frontend_version(workflow: Mapping[str, object]) -> str | None:
    extra = workflow.get("extra")
    if not isinstance(extra, Mapping):
        return None
    value = extra.get("frontendVersion")
    if value is None:
        return None
    if not isinstance(value, str) or not _bounded_printable(value, 200):
        raise WorkflowPackageError(
            "invalid_frontend_version", "workflow has an invalid frontend version"
        )
    return value


def _graph_scopes(
    workflow: Mapping[str, object],
) -> tuple[tuple[_GraphScope, ...], frozenset[str]]:
    scopes = [
        _GraphScope(
            "root",
            _sequence(workflow.get("nodes"), "workflow nodes"),
            _sequence(workflow.get("links"), "workflow links"),
            False,
        )
    ]
    definitions = workflow.get("definitions")
    if definitions is None:
        return tuple(scopes), frozenset()
    if not isinstance(definitions, Mapping):
        raise WorkflowPackageError("invalid_subgraphs", "workflow definitions must be an object")
    subgraphs = _sequence(definitions.get("subgraphs", []), "workflow subgraphs")
    if len(subgraphs) > MAX_UI_GRAPH_SUBGRAPHS:
        raise WorkflowPackageError("too_many_subgraphs", "workflow has too many subgraphs")

    subgraph_ids: set[str] = set()
    for index, value in enumerate(subgraphs):
        if not isinstance(value, Mapping):
            raise WorkflowPackageError(
                "invalid_subgraph", "each workflow subgraph must be an object"
            )
        identifier = _identifier(value.get("id"), "subgraph id")
        if identifier in subgraph_ids:
            raise WorkflowPackageError("duplicate_subgraph", "workflow has a duplicate subgraph id")
        subgraph_ids.add(identifier)
        scopes.append(
            _GraphScope(
                f"subgraph {index + 1}",
                _sequence(value.get("nodes"), "subgraph nodes"),
                _sequence(value.get("links"), "subgraph links"),
                True,
            )
        )
    return tuple(scopes), frozenset(subgraph_ids)


def _validate_nodes(scope: _GraphScope) -> list[_NodeRecord]:
    node_ids: set[str] = set()
    records: list[_NodeRecord] = []
    for value in scope.nodes:
        if not isinstance(value, Mapping):
            raise WorkflowPackageError(
                "invalid_node", f"{scope.name} has a node that is not an object"
            )
        identifier = _identifier(value.get("id"), "node id")
        if identifier in node_ids:
            raise WorkflowPackageError("duplicate_node", f"{scope.name} has a duplicate node id")
        node_ids.add(identifier)
        node_type = value.get("type")
        if not isinstance(node_type, str) or not _bounded_printable(node_type, 200):
            raise WorkflowPackageError(
                "invalid_node_type", f"{scope.name} has an invalid node type"
            )
        properties = value.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            raise WorkflowPackageError(
                "invalid_node_properties", f"{scope.name} has invalid node properties"
            )
        records.append(
            _NodeRecord(
                node_type,
                _optional_property(properties, "cnr_id"),
                _optional_property(properties, "ver"),
                value.get("widgets_values", ()),
            )
        )
    return records


def _validate_links(scope: _GraphScope) -> int:
    node_ids = {
        _identifier(value.get("id"), "node id")
        for value in scope.nodes
        if isinstance(value, Mapping)
    }
    link_ids: set[str] = set()
    dangling_links = 0
    boundaries = {"-10", "-20"} if scope.allows_boundary_nodes else set()
    for value in scope.links:
        if isinstance(value, Mapping):
            link_id = _identifier(value.get("id"), "link id")
            origin = _identifier(value.get("origin_id"), "link origin")
            target = _identifier(value.get("target_id"), "link target")
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) < 5:
                raise WorkflowPackageError("invalid_link", f"{scope.name} has a malformed link")
            link_id = _identifier(value[0], "link id")
            origin = _identifier(value[1], "link origin")
            target = _identifier(value[3], "link target")
        else:
            raise WorkflowPackageError("invalid_link", f"{scope.name} has an invalid link")
        if link_id in link_ids:
            raise WorkflowPackageError("duplicate_link", f"{scope.name} has a duplicate link id")
        link_ids.add(link_id)
        if origin not in node_ids | boundaries or target not in node_ids | boundaries:
            dangling_links += 1
    return dangling_links


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise WorkflowPackageError("invalid_structure", f"{name} must be an array")
    return value


def _identifier(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise WorkflowPackageError("invalid_identifier", f"invalid {name}")
    text = str(value)
    if not _bounded_printable(text, MAX_UI_GRAPH_KEY_CHARACTERS):
        raise WorkflowPackageError("invalid_identifier", f"invalid {name}")
    return text


def _optional_property(properties: object, key: str) -> str | None:
    if not isinstance(properties, Mapping):
        return None
    value = properties.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not _bounded_printable(value, 200):
        raise WorkflowPackageError("invalid_node_properties", f"workflow has an invalid node {key}")
    return value


def _bounded_printable(value: str, maximum: int) -> bool:
    return bool(
        value
        and len(value) <= maximum
        and all(character >= " " and character != "\x7f" for character in value)
    )


def _package_requirements(
    records: Sequence[_NodeRecord],
    required_types: set[str],
    missing_types: set[str],
    available_types: set[str],
    installed_package_versions: Mapping[str, Collection[str]],
) -> tuple[tuple[WorkflowPackageRequirement, ...], tuple[WorkflowPackageIssue, ...]]:
    packages: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    unidentified: set[str] = set()
    unversioned: set[str] = set()
    for record in records:
        if record.node_type not in required_types:
            continue
        if not record.package_id or record.package_id == "comfy-core":
            if record.node_type in missing_types:
                unidentified.add(record.node_type)
            continue
        version = record.package_version or ""
        packages[record.package_id][version].add(record.node_type)
        if not version:
            unversioned.add(record.node_type)

    requirements = tuple(
        WorkflowPackageRequirement(
            package_id,
            tuple(sorted(version for version in versions if version)),
            tuple(
                sorted(
                    {node_type for node_types in versions.values() for node_type in node_types},
                    key=str.casefold,
                )
            ),
            {node_type for node_types in versions.values() for node_type in node_types}
            <= available_types
            and bool({version for version in versions if version})
            and {version for version in versions if version}
            <= {str(version) for version in installed_package_versions.get(package_id, ())},
        )
        for package_id, versions in sorted(packages.items(), key=lambda item: item[0].casefold())
    )
    issues: list[WorkflowPackageIssue] = []
    if unidentified:
        issues.append(
            WorkflowPackageIssue(
                "unidentified_custom_node_package",
                len(unidentified),
                tuple(sorted(unidentified, key=str.casefold)),
            )
        )
    if unversioned:
        issues.append(
            WorkflowPackageIssue(
                "unversioned_custom_node_package",
                len(unversioned),
                tuple(sorted(unversioned, key=str.casefold)),
            )
        )
    conflicting = {
        node_type
        for versions in packages.values()
        if len({version for version in versions if version}) > 1
        for node_types in versions.values()
        for node_type in node_types
    }
    if conflicting:
        issues.append(
            WorkflowPackageIssue(
                "conflicting_custom_node_versions",
                len(conflicting),
                tuple(sorted(conflicting, key=str.casefold)),
            )
        )
    unresolved = [requirement for requirement in requirements if not requirement.locally_resolved]
    if unresolved:
        node_types = {
            node_type for requirement in unresolved for node_type in requirement.node_types
        }
        issues.append(
            WorkflowPackageIssue(
                "unresolved_custom_node_package",
                len(unresolved),
                tuple(sorted(node_types, key=str.casefold)),
            )
        )
    return requirements, tuple(issues)


def _missing_node_requirements(
    records: Sequence[_NodeRecord],
    missing_types: set[str],
) -> tuple[WorkflowMissingNode, ...]:
    counts: dict[str, int] = defaultdict(int)
    packages: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.node_type not in missing_types:
            continue
        counts[record.node_type] += 1
        if record.package_id and record.package_id != "comfy-core":
            packages[record.node_type].add(record.package_id)
    return tuple(
        WorkflowMissingNode(
            node_type,
            counts[node_type],
            next(iter(packages[node_type])) if len(packages[node_type]) == 1 else None,
        )
        for node_type in sorted(counts, key=str.casefold)
    )


def _asset_references(
    records: Sequence[_NodeRecord],
    available_asset_filenames: Collection[str],
) -> tuple[tuple[WorkflowAssetReference, ...], tuple[WorkflowPackageIssue, ...]]:
    references: dict[str, WorkflowAssetReference] = {}
    issue_counts: dict[str, int] = defaultdict(int)
    available = {
        value.replace("\\", "/").casefold()
        for value in available_asset_filenames
        if isinstance(value, str)
    }
    missing_assets: set[str] = set()
    for record in records:
        if record.node_type in {"MarkdownNote", "Note"}:
            continue
        for value in _strings(record.widgets):
            candidate = value.strip()
            if not candidate:
                continue
            lowered = candidate.casefold()
            if lowered.startswith(("http://", "https://")):
                issue_counts["remote_url_reference"] += 1
                continue
            suffix = PurePosixPath(lowered.split("?", 1)[0]).suffix
            if suffix not in KNOWN_MODEL_SUFFIXES:
                continue
            if not _safe_relative_asset(candidate):
                issue_counts["unsafe_asset_reference"] += 1
                continue
            policy: AssetPolicy
            if suffix in SUPPORTED_MODEL_SUFFIXES:
                policy = "supported"
            elif suffix in BLOCKED_MODEL_SUFFIXES:
                policy = "blocked"
                issue_counts["blocked_asset_format"] += 1
            else:
                policy = "unsupported"
                issue_counts["unsupported_asset_format"] += 1
            normalized = candidate.casefold()
            present = normalized in available
            references[normalized] = WorkflowAssetReference(
                candidate,
                suffix,
                policy,
                _asset_kind(record.node_type, suffix),
                present_locally=present,
            )
            if policy == "supported" and not present:
                missing_assets.add(normalized)
    if missing_assets:
        issue_counts["missing_asset"] = len(missing_assets)
    issues = tuple(
        WorkflowPackageIssue(code, count) for code, count in sorted(issue_counts.items())
    )
    return (
        tuple(sorted(references.values(), key=lambda item: item.filename.casefold())),
        issues,
    )


def _asset_kind(node_type: str, suffix: str) -> AssetKind:
    lowered = node_type.casefold()
    if "lora" in lowered:
        return "lora"
    if "vae" in lowered:
        return "vae"
    if "upscale" in lowered:
        return "upscaler"
    if "embedding" in lowered:
        return "embedding"
    if suffix == ".json":
        return "configuration"
    return "checkpoint"


def _operation_guess(node_types: Collection[str]) -> OperationGuess:
    tokens = {token for node_type in node_types for token in _node_type_tokens(node_type)}
    if tokens & {"video", "vhs"} or any(_is_video_family_token(token) for token in tokens):
        return "video"
    if tokens & {"image", "ksampler", "vae", "checkpoint", "unet", "cliptextencode"}:
        return "image"
    return "unknown"


def _node_type_tokens(value: str) -> frozenset[str]:
    tokens: set[str] = set()
    for chunk in re.findall(r"[A-Za-z0-9]+", value):
        tokens.add(chunk.casefold())
        separated = re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])",
            " ",
            chunk,
        )
        tokens.update(part.casefold() for part in separated.split())
    return frozenset(tokens)


def _is_video_family_token(value: str) -> bool:
    if value in {"wan", "ltx", "ltxv", "hunyuan"}:
        return True
    if value.startswith("hunyuan"):
        return True
    return bool(re.fullmatch(r"(?:wan|ltx|ltxv)\d+[a-z0-9]*", value))


def _strings(value: object) -> list[str]:
    found: list[str] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, str):
            found.append(current)
        elif isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, list | tuple):
            stack.extend(current)
    return found


def _safe_relative_asset(value: str) -> bool:
    raw_parts = value.split("/")
    return bool(
        value
        and len(value) <= MAX_ASSET_REFERENCE_CHARACTERS
        and "\\" not in value
        and "?" not in value
        and "#" not in value
        and all(part not in {"", ".", ".."} and ":" not in part for part in raw_parts)
    )
