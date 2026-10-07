"""Bind measured built-in node names to an exact runtime installation plan."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .filesystem_links import (
    AnchoredDirectory,
    AnchoredDirectoryError,
    open_child_directory,
    open_entry,
)
from .runtime_provisioning import RuntimeProvisioner, RuntimeProvisioningError
from .runtime_provisioning_plans import RuntimeProvisioningPlan
from .workflow_revision_reviews import _CORE_MODULE

_MAX_BYTES = 2 * 1024 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class WorkflowRuntimeNodeInventory:
    runtime_plan_sha256: str
    catalog_sha256: str
    node_types: tuple[str, ...]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("The runtime node catalog contains duplicate fields.")
        result[key] = value
    return result


def _read_catalog(root: Path, relative: str) -> bytes:
    path = PurePosixPath(relative)
    if (
        not 2 <= len(path.parts) <= 5
        or path.parts[0] != "runtime-reviews"
        or path.as_posix() != relative
        or any(part in {".", ".."} or "\\" in part or ":" in part for part in path.parts)
        or path.suffix != ".json"
    ):
        raise ValueError("The runtime node catalog path is invalid.")
    with ExitStack() as stack:
        anchor = stack.enter_context(AnchoredDirectory(root.absolute()))
        for part in path.parts[:-1]:
            anchor = stack.enter_context(open_child_directory(anchor, part))
        descriptor = open_entry(anchor, path.name)
        if descriptor is None:
            raise ValueError("The runtime node catalog is missing.")
        with os.fdopen(descriptor, "rb") as stream:
            payload = stream.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES:
        raise ValueError("The runtime node catalog is too large.")
    return payload


def _load_inventory(
    runtimes: RuntimeProvisioner, plan: RuntimeProvisioningPlan
) -> WorkflowRuntimeNodeInventory | None:
    if plan.engine != "comfyui" or plan.operation != "install_managed":
        return None
    if runtimes.preflight("comfyui") != plan:
        return None
    inputs = runtimes._provisioning_inputs("comfyui")
    if runtimes._json_sha256(inputs) != plan.inputs_sha256:
        return None
    asset = inputs["asset"]
    if not isinstance(asset, dict):
        return None
    reference = asset.get("workflow_node_inventory")
    if not isinstance(reference, dict) or set(reference) != {"file", "sha256", "asset_key"}:
        return None
    relative, digest = reference["file"], reference["sha256"]
    if (
        not isinstance(relative, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        or reference["asset_key"] != inputs["platform"]
    ):
        return None
    payload = _read_catalog(runtimes.manifest_path.parent, relative)
    if hashlib.sha256(payload).hexdigest() != digest:
        return None
    catalog = json.loads(payload, object_pairs_hook=_unique_object)
    if (
        not isinstance(catalog, dict)
        or set(catalog)
        != {
            "version",
            "engine",
            "release",
            "asset_key",
            "archive_sha256",
            "runtime_contract_sha256",
            "nodes",
        }
        or type(catalog["version"]) is not int
        or catalog["version"] != 1
        or catalog["engine"] != plan.engine
        or catalog["release"] != plan.release
        or catalog["release"] != inputs["definition"]["pinned_release"]
        or catalog["asset_key"] != inputs["platform"]
        or catalog["archive_sha256"] != asset["sha256"]
        or catalog["runtime_contract_sha256"] != runtimes._runtime_contract_sha256(asset)
    ):
        return None
    nodes = catalog["nodes"]
    if not isinstance(nodes, dict) or not 1 <= len(nodes) <= 8192:
        return None
    for name, module in nodes.items():
        if (
            not 1 <= len(name) <= 256
            or not name.isprintable()
            or not isinstance(module, str)
            or (module != "nodes" and _CORE_MODULE.fullmatch(module) is None)
        ):
            return None
    if inputs != runtimes._provisioning_inputs("comfyui"):
        return None
    return WorkflowRuntimeNodeInventory(plan.plan_sha256, digest, tuple(sorted(nodes)))


async def preflight_workflow_runtime_nodes(
    runtimes: RuntimeProvisioner, plan: RuntimeProvisioningPlan | None
) -> WorkflowRuntimeNodeInventory | None:
    """Return no inventory when the selected runtime lacks a matching measured catalog."""

    if plan is None:
        return None
    try:
        return await asyncio.to_thread(_load_inventory, runtimes, plan)
    except (AnchoredDirectoryError, OSError, ValueError, RecursionError, RuntimeProvisioningError):
        return None
