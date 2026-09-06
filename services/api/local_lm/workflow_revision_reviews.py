"""Exact local workflow review identity; runtime readiness remains a separate check."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Never

from sqlalchemy import select
from sqlalchemy.orm import Session

from .custom_nodes import reviewed_custom_node_types
from .domain import utcnow
from .models import (
    ComfyRegistryInstall,
    CustomNodeInstall,
    WorkflowDefinition,
    WorkflowRevision,
    WorkflowRevisionReview,
)

_MAX_ENTRIES = 100_000
_MAX_DEPTH = 32
_MAX_BYTES = 4 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}")
_CORE_MODULE = re.compile(r"comfy_extras\.nodes_[a-zA-Z0-9_]+")


class WorkflowReviewError(ValueError):
    """A content-free refusal; graph and stored values never enter its message."""


def _canonical(value: object, *, max_bytes: int = _MAX_BYTES, max_depth: int = _MAX_DEPTH) -> bytes:
    pending = [(value, 0)]
    entries = 0
    while pending:
        item, depth = pending.pop()
        entries += 1
        if entries > _MAX_ENTRIES or depth > max_depth:
            raise WorkflowReviewError("workflow_review_structure_invalid")
        if isinstance(item, dict | list) and entries + len(pending) + len(item) > _MAX_ENTRIES:
            raise WorkflowReviewError("workflow_review_structure_invalid")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str or len(key) > 200:
                    raise WorkflowReviewError("workflow_review_structure_invalid")
                pending.append((child, depth + 1))
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            if len(item) > 65_536:
                raise WorkflowReviewError("workflow_review_structure_invalid")
        elif type(item) is float:
            if not math.isfinite(item):
                raise WorkflowReviewError("workflow_review_structure_invalid")
        elif item is not None and type(item) not in {bool, int}:
            raise WorkflowReviewError("workflow_review_structure_invalid")
    parts: list[bytes] = []
    size = 0
    encoder = json.JSONEncoder(
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    try:
        # Each scalar is already bounded above. Stop cumulative encoding before
        # constructing an arbitrarily large string that the byte limit rejects.
        for fragment in encoder.iterencode(value):
            encoded = fragment.encode("utf-8")
            size += len(encoded)
            if size > max_bytes:
                raise WorkflowReviewError("workflow_review_structure_invalid")
            parts.append(encoded)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise WorkflowReviewError("workflow_review_structure_invalid") from None
    return b"".join(parts)


_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,200}")
_CLASS_NAME = re.compile(r"[A-Za-z0-9_ .:()+/-]{1,200}")
_PLACEHOLDER = re.compile(r"\$\{([A-Za-z0-9_.:-]{1,200})\}")
_REMOTE_VALUE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_CODE_INPUTS = frozenset({"code", "python_code", "javascript", "script", "shell", "command"})


def _invalid_structure() -> Never:
    raise WorkflowReviewError("workflow_review_structure_invalid")


def _input_placeholders(value: object, properties: dict[str, Any]) -> set[str]:
    consumed: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str):
            placeholder = _PLACEHOLDER.fullmatch(item)
            if placeholder:
                name = placeholder.group(1)
                if name not in properties:
                    _invalid_structure()
                consumed.add(name)
            elif (
                "${" in item
                or "\x00" in item
                or item.startswith(("/", "\\"))
                or re.match(r"^[A-Za-z]:", item)
                or ".." in item.replace("\\", "/").split("/")
                or _REMOTE_VALUE.search(item)
            ):
                _invalid_structure()
    return consumed


def _validate_review_structure(
    revision: WorkflowRevision, object_info: dict[str, Any] | None
) -> None:
    graph = revision.api_graph_json
    schema = revision.input_schema_json
    dependencies = revision.dependencies_json
    _canonical(graph, max_bytes=2 * 1024 * 1024)
    _canonical(schema, max_bytes=256 * 1024, max_depth=24)
    _canonical(dependencies, max_bytes=256 * 1024)
    if type(schema) is not dict or type(dependencies) is not dict:
        _invalid_structure()
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if (
        type(properties) is not dict
        or len(properties) > 256
        or any(not _IDENTIFIER.fullmatch(key) for key in properties)
        or type(required) is not list
        or len(required) > 256
        or any(type(key) is not str or key not in properties for key in required)
    ):
        _invalid_structure()
    for key, limit in (("models", 256), ("custom_nodes", 64), ("registry_packages", 64)):
        entries = dependencies.get(key, [])
        if type(entries) is not list or len(entries) > limit:
            _invalid_structure()
    if type(graph) is not dict or not graph or len(graph) > 4096:
        _invalid_structure()
    successors: dict[str, set[str]] = {key: set() for key in graph}
    indegree = dict.fromkeys(graph, 0)
    consumed: set[str] = set()
    for identifier, node in graph.items():
        if (
            not _IDENTIFIER.fullmatch(identifier)
            or type(node) is not dict
            or type(node.get("class_type")) is not str
            or not _CLASS_NAME.fullmatch(node["class_type"])
            or type(node.get("inputs")) is not dict
        ):
            _invalid_structure()
        inputs = node["inputs"]
        info = object_info.get(node["class_type"]) if object_info is not None else None
        input_contract = info.get("input") if isinstance(info, dict) else None
        if isinstance(input_contract, dict):
            required_inputs = input_contract.get("required", {})
            optional_inputs = input_contract.get("optional", {})
            if not isinstance(required_inputs, dict) or not isinstance(optional_inputs, dict):
                _invalid_structure()
            if not required_inputs.keys() <= inputs.keys():
                _invalid_structure()
            if not inputs.keys() <= required_inputs.keys() | optional_inputs.keys():
                _invalid_structure()
        for name, value in inputs.items():
            if not _IDENTIFIER.fullmatch(name) or name.lower() in _CODE_INPUTS:
                _invalid_structure()
            if isinstance(value, list):
                if (
                    len(value) != 2
                    or type(value[0]) is not str
                    or value[0] not in graph
                    or type(value[1]) is not int
                    or value[1] < 0
                ):
                    _invalid_structure()
                source, output = value
                source_node = graph[source]
                if type(source_node) is not dict or type(source_node.get("class_type")) is not str:
                    _invalid_structure()
                source_info = (
                    object_info.get(source_node["class_type"]) if object_info is not None else None
                )
                if isinstance(source_info, dict):
                    outputs = source_info.get("output")
                    if not isinstance(outputs, list) or output >= len(outputs):
                        _invalid_structure()
                if identifier not in successors[source]:
                    successors[source].add(identifier)
                    indegree[identifier] += 1
            else:
                consumed.update(_input_placeholders(value, properties))
    if not set(required) <= consumed:
        _invalid_structure()
    pending = [key for key, degree in indegree.items() if degree == 0]
    visited = 0
    while pending:
        identifier = pending.pop()
        visited += 1
        for successor in successors[identifier]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                pending.append(successor)
    if visited != len(graph):
        _invalid_structure()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def revision_identity(definition: WorkflowDefinition, revision: WorkflowRevision) -> str:
    # Recompute from execution-bearing fields, never the cached artifact column.
    # Display fields and UI layout do not change execution approval.
    return _digest(
        {
            "version": 1,
            "operation": definition.operation,
            "engine": revision.engine,
            "engine_version": revision.engine_version,
            "api_graph": revision.api_graph_json,
            "input_schema": revision.input_schema_json,
            "dependencies": revision.dependencies_json,
            "dependency_contract_sha256": revision.dependency_contract_sha256,
            "capabilities": revision.capabilities_json,
        }
    )


def _package_pin(install: CustomNodeInstall | ComfyRegistryInstall) -> dict[str, Any]:
    if not install.trusted or not install.active:
        raise WorkflowReviewError("workflow_review_node_unavailable")
    if isinstance(install, CustomNodeInstall):
        if (
            not re.fullmatch(r"[0-9a-f]{40}", install.revision)
            or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", install.tree_hash)
            or not isinstance(install.security_json, dict)
            or install.security_json.get("trusted_by_local_user") is not True
            or not install.security_json.get("reviewed_at")
        ):
            raise WorkflowReviewError("workflow_review_node_unavailable")
        return {
            "kind": "git",
            "id": install.id,
            "source": install.source_url,
            "commit": install.revision,
            "tree": install.tree_hash,
            "security": _digest(install.security_json),
            "node_types": list(reviewed_custom_node_types(install.security_json, required=True)),
            "active": True,
            "trusted": True,
        }
    if (
        not _HEX.fullmatch(install.archive_sha256)
        or not _HEX.fullmatch(install.manifest_sha256)
        or not isinstance(install.review_json, dict)
        or not install.review_json.get("reviewed_at")
    ):
        raise WorkflowReviewError("workflow_review_node_unavailable")
    return {
        "kind": "registry",
        "id": install.id,
        "source": install.repository_url,
        "package_id": install.package_id,
        "version": install.package_version,
        "archive": install.archive_sha256,
        "manifest": install.manifest_sha256,
        "security": _digest(install.review_json),
        "node_types": sorted(install.node_types_json),
        "wheel_closure": install.wheel_closure_sha256,
        "wheel_environment": install.wheel_environment_sha256,
        "active": True,
        "trusted": True,
    }


def _declared(
    install: CustomNodeInstall | ComfyRegistryInstall, dependencies: dict[str, Any]
) -> bool:
    key = "custom_nodes" if isinstance(install, CustomNodeInstall) else "registry_packages"
    values = dependencies.get(key, [])
    if not isinstance(values, list):
        return False
    for value in values:
        if isinstance(install, CustomNodeInstall):
            if isinstance(value, dict):
                identity = value.get("id") or value.get("name") or value.get("source_url")
                commit = value.get("revision")
            else:
                identity, commit = value, None
            if identity in {install.id, install.name, install.source_url} and (
                commit is None or commit == install.revision
            ):
                return True
        elif isinstance(value, dict) and value.get("package_id") == install.package_id:
            version = value.get("package_version")
            if version is None or version == install.package_version:
                return True
    return False


@dataclass(frozen=True)
class ReviewSnapshot:
    revision_sha256: str
    subject_sha256: str
    node_bindings: dict[str, Any]
    reasons: tuple[str, ...]


def _review_subject(revision_hash: str, bindings: dict[str, Any]) -> str:
    # Runtime contracts are checked afresh for each operation. A compatible
    # runtime update does not change the graph or separately approved code.
    nodes: dict[str, Any] = {}
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise WorkflowReviewError("workflow_review_changed")
        nodes[name] = {key: value for key, value in binding.items() if key != "contract"}
    return _digest({"version": 1, "revision": revision_hash, "nodes": nodes})


def build_review_snapshot(
    session: Session,
    definition: WorkflowDefinition,
    revision: WorkflowRevision,
    *,
    object_info: dict[str, Any] | None,
) -> ReviewSnapshot:
    """Bind current server node metadata; caller must prove its managed runtime.

    This function neither contacts a worker nor verifies installed code bytes.
    Approval/dispatch must perform those checks under their lifecycle authority.
    """
    _validate_review_structure(revision, object_info)
    revision_hash = revision_identity(definition, revision)
    graph = revision.api_graph_json
    if type(graph) is not dict or not graph or len(graph) > 4096:
        raise WorkflowReviewError("workflow_review_structure_invalid")
    required: set[str] = set()
    for node in graph.values():
        kind = node.get("class_type") if type(node) is dict else None
        if (
            not isinstance(kind, str)
            or not kind
            or len(kind) > 200
            or any(ord(character) < 32 or ord(character) == 127 for character in kind)
        ):
            raise WorkflowReviewError("workflow_review_structure_invalid")
        required.add(kind)
    reasons: set[str] = set()
    bindings: dict[str, Any] = {}
    installs: list[CustomNodeInstall | ComfyRegistryInstall] = [
        *session.scalars(select(CustomNodeInstall)).all(),
        *session.scalars(select(ComfyRegistryInstall)).all(),
    ]
    for kind in sorted(required):
        info = object_info.get(kind) if object_info is not None else None
        if type(info) is not dict:
            reasons.add("workflow_review_runtime_unavailable")
            continue
        module = info.get("python_module")
        if not isinstance(module, str):
            reasons.add("workflow_review_node_unavailable")
            continue
        owners = [
            install
            for install in installs
            if kind
            in (
                reviewed_custom_node_types(install.security_json)
                if isinstance(install, CustomNodeInstall)
                else install.node_types_json
            )
        ]
        core = module == "nodes" or _CORE_MODULE.fullmatch(module) is not None
        # A declared/installed package inventory claiming a core name is ambiguous;
        # do not grant core trust based only on the runtime's module label.
        if core and not owners:
            bindings[kind] = {"kind": "core", "module": module, "contract": _digest(info)}
            continue
        if len(owners) != 1 or not _declared(owners[0], revision.dependencies_json):
            reasons.add("workflow_review_node_unavailable")
            continue
        try:
            pin = _package_pin(owners[0])
        except (WorkflowReviewError, ValueError):
            reasons.add("workflow_review_node_unavailable")
            continue
        bindings[kind] = {
            "kind": "package",
            "module": module,
            "contract": _digest(info),
            "pin": pin,
        }
    subject = _review_subject(revision_hash, bindings)
    return ReviewSnapshot(revision_hash, subject, bindings, tuple(sorted(reasons)))


def review_is_current(
    session: Session, definition: WorkflowDefinition, revision: WorkflowRevision
) -> bool:
    """Revalidate durable identities; live runtime/code validation is also required."""
    review = session.get(WorkflowRevisionReview, revision.id)
    if review is None:
        return revision.trusted
    if review.state != "approved" or not revision.trusted:
        return False
    try:
        if revision_identity(definition, revision) != review.revision_sha256:
            return False
        nodes = review.node_bindings_json
        if not isinstance(nodes, dict) or not nodes:
            return False
        if _review_subject(review.revision_sha256, nodes) != review.subject_sha256:
            return False
        for binding in nodes.values():
            if not isinstance(binding, dict):
                return False
            if binding.get("kind") == "core":
                continue
            pin = binding.get("pin")
            if not isinstance(pin, dict):
                return False
            model = CustomNodeInstall if pin.get("kind") == "git" else ComfyRegistryInstall
            install = session.get(model, pin.get("id"))
            if (
                not isinstance(install, CustomNodeInstall | ComfyRegistryInstall)
                or _package_pin(install) != pin
            ):
                return False
    except (WorkflowReviewError, ValueError, TypeError):
        return False
    return True


def record_review(
    session: Session, revision: WorkflowRevision, snapshot: ReviewSnapshot, *, approved: bool
) -> WorkflowRevisionReview:
    """Persist a server-validated decision in the caller's existing transaction."""
    if approved and snapshot.reasons:
        raise WorkflowReviewError("workflow_review_node_unavailable")
    review = session.get(WorkflowRevisionReview, revision.id)
    if review is None:
        review = WorkflowRevisionReview(workflow_revision_id=revision.id)
        session.add(review)
    review.revision_sha256 = snapshot.revision_sha256
    review.subject_sha256 = snapshot.subject_sha256
    review.node_bindings_json = json.loads(_canonical(snapshot.node_bindings))
    review.state = "approved" if approved else "revoked"
    review.reviewed_at = utcnow()
    revision.trusted = approved
    session.flush()
    return review


def inherit_review(
    session: Session,
    source_definition: WorkflowDefinition,
    source: WorkflowRevision,
    target_definition: WorkflowDefinition,
    target: WorkflowRevision,
) -> None:
    """Carry exact local evidence into a byte-identical clone or restoration.

    Targets must be initially untrusted. The writer protects the fresh source
    decision and target identity through the caller's commit.
    """
    session.connection().exec_driver_sql("UPDATE workflow_revision_reviews SET state=state WHERE 0")
    for row in (source_definition, source, target_definition, target):
        session.refresh(row)
    target.trusted = False
    if revision_identity(source_definition, source) != revision_identity(
        target_definition, target
    ) or not review_is_current(session, source_definition, source):
        return
    source_review = session.get(WorkflowRevisionReview, source.id)
    if source_review is None:
        target.trusted = source.trusted
        return
    copied = record_review(
        session,
        target,
        ReviewSnapshot(
            source_review.revision_sha256,
            source_review.subject_sha256,
            source_review.node_bindings_json,
            (),
        ),
        approved=True,
    )
    copied.reviewed_at = source_review.reviewed_at


def revision_is_trusted(session: Session, revision: WorkflowRevision) -> bool:
    definition = session.get(WorkflowDefinition, revision.workflow_id)
    return definition is not None and review_is_current(session, definition, revision)
