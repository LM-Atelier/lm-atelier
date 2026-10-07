"""A measured catalog cannot authorize a different runtime or linked content."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_filesystem_links import _make_junction
from test_workflow_runtime_targets import runtime as runtime

from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.workflow_runtime_nodes import preflight_workflow_runtime_nodes

_ASSET_KEY = "windows-x86_64-nvidia-cu13"


def _catalog(runtime: RuntimeProvisioner) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    definition = runtime._definition("comfyui")
    asset = definition["runtime_assets"][_ASSET_KEY]
    catalog = {
        "version": 1,
        "engine": "comfyui",
        "release": definition["pinned_release"],
        "asset_key": _ASSET_KEY,
        "archive_sha256": asset["sha256"],
        "runtime_contract_sha256": runtime._runtime_contract_sha256(asset),
        "nodes": {"SaveImage": "nodes", "EmptyImage": "comfy_extras.nodes_images"},
    }
    path = runtime.manifest_path.parent / "runtime-reviews/constructed-nodes.json"
    reference = {"file": "runtime-reviews/constructed-nodes.json", "asset_key": _ASSET_KEY}
    asset["workflow_node_inventory"] = reference
    _save(path, reference, json.dumps(catalog).encode())
    return path, reference, catalog


def _save(path: Path, reference: dict[str, Any], payload: bytes) -> None:
    path.write_bytes(payload)
    reference["sha256"] = hashlib.sha256(payload).hexdigest()


async def test_measured_nodes_are_bound_to_the_exact_plan_without_installing(
    runtime: RuntimeProvisioner,
) -> None:
    path, reference, _ = _catalog(runtime)
    plan = runtime.preflight("comfyui")
    inventory = await preflight_workflow_runtime_nodes(runtime, plan)
    assert inventory is not None
    assert inventory.runtime_plan_sha256 == plan.plan_sha256
    assert inventory.catalog_sha256 == reference["sha256"]
    assert inventory.node_types == ("EmptyImage", "SaveImage")
    assert path.is_file()
    assert runtime.settings.comfy_executable is None
    assert list(runtime.runtime_root.iterdir()) == []


@pytest.mark.parametrize(
    "change",
    [
        "missing-reference",
        "reference-shape",
        "reference-asset",
        "invalid-hash",
        "hash",
        "missing-file",
        "oversized",
        "invalid-json",
        "duplicate",
        "unknown-field",
        "version",
        "boolean-version",
        "engine",
        "release",
        "asset_key",
        "archive_sha256",
        "runtime_contract_sha256",
        "empty-nodes",
        "many-nodes",
        "invalid-node-name",
        "custom-module",
        "cloud-module",
        "module-prefix",
        "module-type",
        "nested-module",
        "parent-path",
        "absolute-path",
        "backslash-path",
        "stream-path",
        "collapsed-path",
    ],
)
async def test_unverified_catalogs_provide_no_nodes(
    runtime: RuntimeProvisioner, change: str
) -> None:
    path, reference, catalog = _catalog(runtime)
    if change == "missing-reference":
        del runtime._definition("comfyui")["runtime_assets"][_ASSET_KEY]["workflow_node_inventory"]
    elif change == "reference-shape":
        reference["extra"] = True
    elif change == "reference-asset":
        reference["asset_key"] = "windows-x86_64-nvidia-cu126"
    elif change == "invalid-hash":
        reference["sha256"] = "invalid"
    elif change == "hash":
        reference["sha256"] = "0" * 64
    elif change == "missing-file":
        path.unlink()
    elif change == "oversized":
        _save(path, reference, b" " * (2 * 1024 * 1024 + 1))
    elif change == "invalid-json":
        _save(path, reference, b"[")
    elif change == "duplicate":
        payload = json.dumps(catalog).replace('"version": 1', '"version": 2, "version": 1')
        _save(path, reference, payload.encode())
    elif change.endswith("-path"):
        reference["file"] = {
            "parent-path": "runtime-reviews/../constructed-nodes.json",
            "absolute-path": "/runtime-reviews/constructed-nodes.json",
            "backslash-path": "runtime-reviews/..\\constructed-nodes.json",
            "stream-path": "runtime-reviews/constructed:nodes.json",
            "collapsed-path": "runtime-reviews//constructed-nodes.json",
        }[change]
    else:
        if change == "unknown-field":
            catalog["extra"] = True
        elif change == "version":
            catalog["version"] = 2
        elif change == "boolean-version":
            catalog["version"] = True
        elif change in {
            "engine",
            "release",
            "asset_key",
            "archive_sha256",
            "runtime_contract_sha256",
        }:
            catalog[change] = "different"
        elif change == "empty-nodes":
            catalog["nodes"] = {}
        elif change == "many-nodes":
            catalog["nodes"] = {str(i): "nodes" for i in range(8193)}
        elif change == "invalid-node-name":
            catalog["nodes"] = {"\n": "nodes"}
        else:
            catalog["nodes"]["EmptyImage"] = {
                "custom-module": "custom_nodes.example",
                "cloud-module": "comfy_api_nodes.nodes_example",
                "module-prefix": "comfy_extras.nodes",
                "module-type": None,
                "nested-module": "comfy_extras.nodes_images.external",
            }[change]
        _save(path, reference, json.dumps(catalog).encode())
    assert await preflight_workflow_runtime_nodes(runtime, runtime.preflight("comfyui")) is None


@pytest.mark.parametrize(
    "change", ["no-plan", "configured", "managed", "wrong-engine", "plan-hash", "inputs"]
)
async def test_only_the_matching_fresh_install_plan_can_use_a_catalog(
    runtime: RuntimeProvisioner, change: str
) -> None:
    _catalog(runtime)
    plan = runtime.preflight("comfyui")
    if change == "no-plan":
        assert await preflight_workflow_runtime_nodes(runtime, None) is None
        return
    if change == "configured":
        plan = replace(plan, operation="reuse_configured")
    elif change == "managed":
        plan = replace(plan, operation="reuse_managed")
    elif change == "wrong-engine":
        plan = replace(plan, engine="llamacpp")
    elif change == "plan-hash":
        plan = replace(plan, plan_sha256="a" * 64)
    else:
        runtime._definition("comfyui")["runtime_assets"][_ASSET_KEY]["sha256"] = "a" * 64
    assert await preflight_workflow_runtime_nodes(runtime, plan) is None


async def test_the_packaged_catalog_matches_its_runtime_asset(runtime: RuntimeProvisioner) -> None:
    root = Path(__file__).resolve().parents[3]
    asset = runtime._definition("comfyui")["runtime_assets"][_ASSET_KEY]
    reference = asset["workflow_node_inventory"]
    relative = reference["file"]
    (runtime.manifest_path.parent / relative).write_bytes(
        (root / "packaging" / relative).read_bytes()
    )
    plan = runtime.preflight("comfyui")
    inventory = await preflight_workflow_runtime_nodes(runtime, plan)
    assert inventory is not None
    assert inventory.runtime_plan_sha256 == plan.plan_sha256
    assert inventory.catalog_sha256 == reference["sha256"]
    assert len(inventory.node_types) == 576
    assert {"EmptyImage", "SaveImage", "LoraLoader"} <= set(inventory.node_types)
    assert "ExampleNode" not in inventory.node_types


async def test_inputs_changing_during_catalog_read_invalidate_the_inventory(
    runtime: RuntimeProvisioner, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_lm import workflow_runtime_nodes

    _catalog(runtime)
    plan = runtime.preflight("comfyui")
    read = workflow_runtime_nodes._read_catalog

    def changed(root: Path, relative: str) -> bytes:
        payload = read(root, relative)
        runtime._definition("comfyui")["runtime_assets"][_ASSET_KEY]["sha256"] = "a" * 64
        return payload

    monkeypatch.setattr(workflow_runtime_nodes, "_read_catalog", changed)
    assert await preflight_workflow_runtime_nodes(runtime, plan) is None


async def test_a_runtime_blocked_after_preview_has_no_planned_node_inventory(
    runtime: RuntimeProvisioner,
) -> None:
    _catalog(runtime)
    plan = runtime.preflight("comfyui")
    runtime._definition("comfyui")["runtime_assets"][_ASSET_KEY]["security_status"] = "blocked"
    assert await preflight_workflow_runtime_nodes(runtime, plan) is None


async def test_a_linked_catalog_directory_cannot_supply_matching_bytes(
    runtime: RuntimeProvisioner, tmp_path: Path
) -> None:
    path, reference, _ = _catalog(runtime)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "nodes.json").write_bytes(path.read_bytes())
    link = path.parent / "redirected"
    if os.name == "nt":
        assert _make_junction(link, outside)
    else:
        link.symlink_to(outside, target_is_directory=True)
    reference["file"] = "runtime-reviews/redirected/nodes.json"
    assert await preflight_workflow_runtime_nodes(runtime, runtime.preflight("comfyui")) is None
