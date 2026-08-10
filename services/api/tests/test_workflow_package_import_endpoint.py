"""The ready-package import hop: analyzer-gated, runtime-compiled, untrusted."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm import api as api_module

pytestmark = pytest.mark.asyncio


def _object_info() -> dict[str, Any]:
    return {
        "Source": {
            "display_name": "Source image",
            "input": {
                "required": {
                    "label": ["STRING", {"default": "default"}],
                    "seed": ["INT", {"default": 1, "control_after_generate": True}],
                }
            },
            "input_order": {"required": ["label", "seed"]},
            "output": ["IMAGE"],
        },
        "Save": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output": [],
            "output_node": True,
        },
    }


def _ui_graph() -> dict[str, Any]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "Source",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [7]}],
                "widgets_values": ["camera", 42, "randomize"],
            },
            {
                "id": 2,
                "type": "Save",
                "mode": 0,
                "inputs": [
                    {"name": "images", "type": "IMAGE", "link": 7},
                    {
                        "name": "filename_prefix",
                        "type": "STRING",
                        "widget": {"name": "filename_prefix"},
                    },
                ],
                "outputs": [],
                "widgets_values": ["result"],
            },
        ],
        "links": [[7, 1, 0, 2, 0, "IMAGE"]],
    }


def _wire_runtime(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    async def object_info() -> dict[str, Any]:
        return _object_info()

    monkeypatch.setattr(app.state.services.engines.media, "object_info", object_info, raising=False)


def _record_registry_install(package_id: str, node_type: str) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import ComfyRegistryInstall

    with SessionLocal() as session:
        session.add(
            ComfyRegistryInstall(
                package_id=package_id,
                package_version="1.2.3",
                registry_record_id=f"registry-record-{package_id}",
                repository_url=f"https://github.com/example/{package_id}.git",
                download_url=f"https://cdn.comfy.org/{package_id}/1.2.3.zip",
                archive_sha256="a" * 64,
                manifest_sha256="b" * 64,
                installed_path=f"lm-atelier-registry_{package_id}",
                node_types_json=[node_type],
                pip_dependencies_json=[],
                review_json={"review_required": True},
                trusted=True,
                active=True,
            )
        )
        session.commit()


def _registry_graph(package_id: str, node_type: str) -> dict[str, Any]:
    graph = _ui_graph()
    graph["nodes"][0]["type"] = node_type
    graph["nodes"][0]["properties"] = {"cnr_id": package_id, "ver": "1.2.3"}
    return graph


def _record_manual_install(package_id: str, revision: str, node_type: str) -> None:
    from local_lm.db import SessionLocal
    from local_lm.models import CustomNodeInstall

    with SessionLocal() as session:
        session.add(
            CustomNodeInstall(
                id=f"node_{package_id}",
                name=package_id,
                source_url=f"https://github.com/example/{package_id}.git",
                revision=revision,
                installed_path=f"lm-atelier-node_{package_id}",
                tree_hash="f" * 40,
                trusted=True,
                active=True,
                security_json={"node_types": [node_type]},
            )
        )
        session.commit()


def _manual_graph(package_id: str, revision: str, node_type: str) -> dict[str, Any]:
    graph = _ui_graph()
    graph["nodes"][0]["type"] = node_type
    graph["nodes"][0]["properties"] = {"cnr_id": package_id, "ver": revision}
    return graph


async def test_a_stopped_runtime_refuses_before_compiling(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def failing_object_info() -> dict[str, Any]:
        raise RuntimeError("media runtime is not up")

    monkeypatch.setattr(
        app.state.services.engines.media, "object_info", failing_object_info, raising=False
    )

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": _ui_graph(), "name": "Camera save", "operation": "text_to_image"},
    )

    assert response.status_code == 503
    assert response.json()["code"] == "media-runtime-unavailable"


async def test_an_unresolved_package_refuses_with_the_analyzer_verdict(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    graph["nodes"][0]["type"] = "NotInstalledNode"
    graph["nodes"][0]["properties"] = {"cnr_id": "missing-pack", "version": "1.0.0"}

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Unresolved", "operation": "text_to_image"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "package-not-resolved"
    listed = (await client.get("/api/workflows")).json()
    assert all(workflow["name"] != "Unresolved" for workflow in listed)


async def test_import_restarts_unscoped_for_a_verified_registry_package(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_id = "comfyui-kjnodes"
    node_type = "GetNode"
    _record_registry_install(package_id, node_type)
    reads = 0
    starts = 0

    async def object_info() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        inventory = _object_info()
        if reads > 1:
            inventory[node_type] = inventory["Source"]
        return inventory

    async def launchable_packages() -> dict[tuple[str, str], frozenset[str]]:
        return {(package_id, "1.2.3"): frozenset({node_type})}

    async def start_media() -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    monkeypatch.setattr(
        app.state.services.processes,
        "trusted_comfy_registry_package_node_types",
        launchable_packages,
    )
    monkeypatch.setattr(app.state.services.processes, "start_media", start_media)

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": _registry_graph(package_id, node_type),
            "name": "KJ Nodes import",
            "operation": "text_to_image",
        },
    )

    assert response.status_code == 201, response.json()
    assert starts == 1
    assert reads == 2
    current_id = response.json()["current_revision_id"]
    current = next(
        revision for revision in response.json()["revisions"] if revision["id"] == current_id
    )
    assert current["api_graph_json"]["1"]["class_type"] == node_type


async def test_import_restarts_unscoped_for_a_reviewed_manual_package(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_id = "comfyui-kjnodes"
    revision = "e" * 40
    node_type = "GetNode"
    _record_manual_install(package_id, revision, node_type)
    reads = 0
    starts = 0

    async def object_info() -> dict[str, Any]:
        nonlocal reads
        reads += 1
        inventory = _object_info()
        if reads > 1:
            inventory[node_type] = inventory["Source"]
        return inventory

    async def no_registry_packages() -> dict[tuple[str, str], frozenset[str]]:
        return {}

    async def manual_packages() -> dict[tuple[str, str], frozenset[str]]:
        return {(package_id, revision): frozenset({node_type})}

    async def start_media() -> None:
        nonlocal starts
        starts += 1

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    monkeypatch.setattr(
        app.state.services.processes,
        "trusted_comfy_registry_package_node_types",
        no_registry_packages,
    )
    monkeypatch.setattr(
        app.state.services.processes,
        "trusted_comfy_custom_node_package_node_types",
        manual_packages,
    )
    monkeypatch.setattr(app.state.services.processes, "start_media", start_media)

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": _manual_graph(package_id, revision, node_type),
            "name": "Reviewed manual import",
            "operation": "text_to_image",
        },
    )

    assert response.status_code == 201, response.json()
    assert starts == 1
    assert reads == 2
    current_id = response.json()["current_revision_id"]
    current = next(item for item in response.json()["revisions"] if item["id"] == current_id)
    assert current["api_graph_json"]["1"]["class_type"] == node_type


async def test_import_fails_closed_when_verified_registry_nodes_cannot_load(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_id = "comfyui-videohelpersuite"
    node_type = "VHS_VideoCombine"
    _record_registry_install(package_id, node_type)

    async def object_info() -> dict[str, Any]:
        return _object_info()

    async def launchable_packages() -> dict[tuple[str, str], frozenset[str]]:
        return {(package_id, "1.2.3"): frozenset({node_type})}

    async def start_media() -> None:
        raise RuntimeError(r"C:\private\managed-runtime\failure")

    monkeypatch.setattr(
        app.state.services.engines.media,
        "object_info",
        object_info,
        raising=False,
    )
    monkeypatch.setattr(
        app.state.services.processes,
        "trusted_comfy_registry_package_node_types",
        launchable_packages,
    )
    monkeypatch.setattr(app.state.services.processes, "start_media", start_media)

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": _registry_graph(package_id, node_type),
            "name": "Must not persist",
            "operation": "text_to_video",
        },
    )

    assert response.status_code == 422
    assert response.json() == {
        "code": "workflow-package-node-refresh-failed",
        "detail": "The installed custom-node packages could not be loaded for workflow import.",
    }
    listed = (await client.get("/api/workflows")).json()
    assert all(workflow["name"] != "Must not persist" for workflow in listed)


async def test_an_unresolved_package_can_be_persisted_as_an_exact_non_executable_draft(
    client: AsyncClient,
) -> None:
    payload = {
        "ui_graph": _ui_graph(),
        "name": "Package draft",
        "operation": "image_to_image",
    }

    first = await client.post("/api/workflows/packages/drafts", json=payload)
    second = await client.post(
        "/api/workflows/packages/drafts",
        json={**payload, "name": "Renamed draft", "operation": "text_to_video"},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    draft = second.json()
    assert draft["name"] == "Renamed draft"
    # A revision's operation participates in its artifact identity. The final
    # import may choose another operation by creating a new revision, but
    # reopening the same draft must not mutate this one in place.
    assert draft["operation"] == "image_to_image"
    assert len(draft["revisions"]) == 1
    (revision,) = draft["revisions"]
    assert revision["id"] == draft["current_revision_id"]
    assert revision["trusted"] is False
    assert revision["api_graph_json"] == {}
    assert revision["dependencies_json"]["workflow_package_draft"]["graph_sha256"]
    listed = (await client.get("/api/workflows")).json()
    assert all(workflow["id"] != draft["id"] for workflow in listed)
    draft_families = (await client.get("/api/workflow-families?selector_capability=image")).json()
    assert all(
        variant["id"] != draft["id"] for family in draft_families for variant in family["variants"]
    )


async def test_a_draft_identity_collision_refuses_instead_of_reusing_another_graph(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api_module,
        "_workflow_package_draft_identity",
        lambda _canonical: ("wfpkgdraft_collision", "wfpkgdrev_collision", "a" * 64),
    )
    first = await client.post(
        "/api/workflows/packages/drafts",
        json={"ui_graph": _ui_graph(), "name": "First", "operation": "text_to_image"},
    )
    changed = _ui_graph()
    changed["nodes"][0]["widgets_values"][0] = "different"

    second = await client.post(
        "/api/workflows/packages/drafts",
        json={"ui_graph": changed, "name": "Second", "operation": "text_to_image"},
    )

    assert first.status_code == 201
    assert second.status_code == 409
    assert second.json()["code"] == "workflow-package-draft-collision"


async def test_a_ready_draft_finalizes_in_place_and_retry_is_idempotent(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    draft_response = await client.post(
        "/api/workflows/packages/drafts",
        json={"ui_graph": graph, "name": "Draft", "operation": "text_to_image"},
    )
    draft = draft_response.json()
    import_payload = {
        "ui_graph": graph,
        "name": "Final workflow",
        "operation": "image_to_image",
        "draft_workflow_id": draft["id"],
        "draft_revision_id": draft["current_revision_id"],
    }

    finalized_response = await client.post("/api/workflows/packages/import", json=import_payload)

    assert finalized_response.status_code == 201, finalized_response.json()
    finalized = finalized_response.json()
    assert finalized["id"] == draft["id"]
    assert finalized["operation"] == "image_to_image"
    assert len(finalized["revisions"]) == 2
    assert finalized["revisions"][0]["api_graph_json"] == {}
    current = next(
        revision
        for revision in finalized["revisions"]
        if revision["id"] == finalized["current_revision_id"]
    )
    assert current["trusted"] is False
    assert current["api_graph_json"]["1"]["class_type"] == "Source"
    listed = (await client.get("/api/workflows")).json()
    assert any(workflow["id"] == draft["id"] for workflow in listed)
    finalized_families = (
        await client.get("/api/workflow-families?selector_capability=image")
    ).json()
    assert (
        sum(
            variant["id"] == draft["id"]
            for family in finalized_families
            for variant in family["variants"]
        )
        == 1
    )

    retry_response = await client.post(
        "/api/workflows/packages/import",
        json={
            **import_payload,
            "draft_revision_id": finalized["current_revision_id"],
        },
    )

    assert retry_response.status_code == 201
    assert retry_response.json()["id"] == draft["id"]
    assert len(retry_response.json()["revisions"]) == 2


async def test_a_draft_cannot_authorize_a_different_graph(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    draft = (
        await client.post(
            "/api/workflows/packages/drafts",
            json={"ui_graph": graph, "name": "Draft", "operation": "text_to_image"},
        )
    ).json()
    changed = _ui_graph()
    changed["nodes"][0]["widgets_values"][0] = "substituted"

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": changed,
            "name": "Substituted",
            "operation": "text_to_image",
            "draft_workflow_id": draft["id"],
            "draft_revision_id": draft["current_revision_id"],
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-package-draft-identity-mismatch"


async def test_a_ready_package_imports_as_an_untrusted_workflow(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": _ui_graph(),
            "name": "Camera save",
            "operation": "image_to_image",
            "description": "From a shared export",
        },
    )

    assert response.status_code == 201
    workflow = response.json()
    assert workflow["name"] == "Camera save"
    assert workflow["operation"] == "image_to_image"
    (revision,) = workflow["revisions"]
    # Imported code is never trusted by arriving; review grants that later.
    assert revision["trusted"] is False
    assert revision["ui_graph_json"]["nodes"][0]["type"] == "Source"
    compiled = revision["api_graph_json"]
    assert compiled["1"]["class_type"] == "Source"
    assert compiled["2"]["inputs"]["images"] == ["1", 0]


async def test_plain_package_import_ignores_an_unsupported_frontend_version(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    graph["extra"] = {"frontendVersion": "99.0.0"}

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Headless import", "operation": "text_to_image"},
    )

    assert response.status_code == 201, response.json()


@pytest.mark.parametrize("frontend_version", [None, "1.45.22"])
async def test_package_import_refuses_uncertified_frontend_semantics(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    frontend_version: str | None,
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    if frontend_version is not None:
        graph["extra"] = {"frontendVersion": frontend_version}
    graph["nodes"].append({"id": 3, "type": "Reroute", "mode": 0, "inputs": [], "outputs": []})

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": f"Uncertified frontend {frontend_version}",
            "operation": "text_to_image",
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unsupported_frontend_version"
    assert response.json()["detail"] == (
        "workflow uses PrimitiveNode or Reroute semantics without a certified "
        "ComfyUI frontend version"
    )


async def test_package_import_accepts_certified_frontend_semantics(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    graph["extra"] = {"frontendVersion": "1.45.21"}
    graph["nodes"].append({"id": 3, "type": "Reroute", "mode": 0, "inputs": [], "outputs": []})

    response = await client.post(
        "/api/workflows/packages/import",
        json={
            "ui_graph": graph,
            "name": "Certified frontend",
            "operation": "text_to_image",
        },
    )

    assert response.status_code == 201, response.json()


async def test_compilation_refusals_keep_their_stable_codes(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    # A subgraph whose declared input nothing feeds. Its inner node types are
    # all available, so the package is still ready - the refusal has to come
    # from compilation, carrying compilation's code rather than the
    # analyzer's verdict on readiness.
    subgraph_id = "9bc44576-7290-4701-bda4-032ca796efbc"
    graph["nodes"].append({"id": 3, "type": subgraph_id, "mode": 0, "inputs": [], "outputs": []})
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": subgraph_id,
                "nodes": [{"id": 10, "type": "Source", "mode": 0, "inputs": [], "outputs": []}],
                "links": [[100, "-10", 0, 10, 0, "IMAGE"]],
            }
        ]
    }

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Subgraphs", "operation": "text_to_image"},
    )

    assert response.status_code == 422
    assert response.json()["code"] == "unconnected_subgraph_input"


async def test_a_subgraph_package_imports_now_that_it_can_be_expanded(
    client: AsyncClient, app: FastAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of expansion: authored graphs become runnable.

    Both workflows this project was asked to set up are built out of
    subgraphs, so every asset they needed could be installed and they would
    still refuse to compile.
    """

    _wire_runtime(app, monkeypatch)
    graph = _ui_graph()
    subgraph_id = "5f0b1f4c-1f3a-4b8e-9d21-6f0a2f5c7a10"
    # The Save node moves inside a subgraph; the instance takes the image and
    # hands it to the node that was there before.
    graph["nodes"] = [
        graph["nodes"][0],
        {
            "id": 3,
            "type": subgraph_id,
            "mode": 0,
            "inputs": [{"name": "images", "type": "IMAGE", "link": 7}],
            "outputs": [],
        },
    ]
    graph["links"] = [[7, 1, 0, 3, 0, "IMAGE"]]
    graph["definitions"] = {
        "subgraphs": [
            {
                "id": subgraph_id,
                "nodes": [
                    {
                        "id": 10,
                        "type": "Save",
                        "mode": 0,
                        "inputs": [
                            {"name": "images", "type": "IMAGE", "link": 100},
                            {
                                "name": "filename_prefix",
                                "type": "STRING",
                                "widget": {"name": "filename_prefix"},
                            },
                        ],
                        "outputs": [],
                        "widgets_values": ["result"],
                    }
                ],
                "links": [[100, "-10", 0, 10, 0, "IMAGE"]],
            }
        ]
    }

    response = await client.post(
        "/api/workflows/packages/import",
        json={"ui_graph": graph, "name": "Expanded", "operation": "text_to_image"},
    )

    assert response.status_code == 201, response.json()
