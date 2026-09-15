"""Workflow previews preserve freshly inspected extension inputs without installing them."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select, text
from test_comfy_registry_downloads import archive_bytes
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_install_plans import (
    URL,
    _payload,
)
from test_workflow_package_install_plans import (
    no_installation_work as no_installation_work,
)
from test_workflow_package_install_plans import (
    runtime_inventory as runtime_inventory,
)

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, WorkflowPackageInstallPlan

pytestmark = pytest.mark.asyncio


def _configure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, commit: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    from local_lm import workflow_package_extension_preflight as preflight

    inputs = _inputs(commit=commit, dependencies=True)
    monkeypatch.setattr(settings, "comfy_executable", inputs["python_executable"])
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "runtime")
    (tmp_path / "runtime").mkdir(exist_ok=True)
    (tmp_path / "runtime/main.py").write_bytes(b"neutral runtime source")
    monkeypatch.setattr(
        preflight, "probe_comfy_registry_runtime_target", inputs["interpreter_probe"]
    )
    from local_lm import workflow_runtime_targets

    async def target_probe(path: Path) -> Any:
        probe = vars(preflight)["probe_comfy_registry_runtime_target"]
        probed = await probe(path)
        return (*probed, ()) if len(probed) == 2 else probed

    monkeypatch.setattr(
        workflow_runtime_targets, "probe_comfy_registry_runtime_target", target_probe
    )

    _configure_clients(inputs, monkeypatch)
    payload = _payload()
    selected = inputs["selected"]
    for node_type in selected.node_types:
        payload["ui_graph"]["nodes"].append(
            {
                "id": max(node["id"] for node in payload["ui_graph"]["nodes"]) + 1,
                "type": node_type,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "widgets_values": [],
                "properties": {"cnr_id": selected.package_id, "ver": selected.declared_version},
            }
        )
    return inputs, payload


def _configure_clients(inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from local_lm import workflow_package_extension_preflight as preflight

    async def close() -> None:
        pass

    monkeypatch.setattr(
        preflight,
        "ComfyRegistryClient",
        lambda: SimpleNamespace(resolve=inputs["registry_client"].resolve, close=close),
    )
    monkeypatch.setattr(
        preflight,
        "ComfyRegistryWheelProjectClient",
        lambda: SimpleNamespace(fetch=inputs["project_client"].fetch, close=close),
    )
    monkeypatch.setattr(
        preflight,
        "ComfyRegistryWheelMetadataClient",
        lambda: SimpleNamespace(fetch=inputs["metadata_client"].fetch, close=close),
    )
    # Repeated requests share the neutral transport, while each real client is closed.
    original = inputs["archive_downloader"]
    import httpx

    from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader

    clients: list[ComfyRegistryArchiveDownloader] = []

    def archive() -> ComfyRegistryArchiveDownloader:
        instance = ComfyRegistryArchiveDownloader(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=inputs["state"]["content"])
            )
        )
        clients.append(instance)
        return instance

    monkeypatch.setattr(preflight, "ComfyRegistryArchiveDownloader", archive)
    inputs["clients"] = clients
    inputs["unused_archive"] = original


@pytest.mark.parametrize("commit", [False, True])
async def test_server_preflight_preserves_exact_extensions_and_counts_transitive_downloads(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: bool,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path, commit=commit)
    try:
        created = await client.post(URL, json=payload)
        assert created.status_code == 201, created.text
        plan = created.json()
        identifier = inputs["selected"].package_id
        extension = plan["extension_execution"]["plans"][identifier]
        assert plan["extension_execution"]["errors"] == {}
        assert extension["archive_bytes"] == len(inputs["state"]["content"])
        assert len(extension["closure"]["manifest"]["artifacts"]) == 2
        assert plan["total_download_bytes"] == 17 + 200 + extension["archive_bytes"]
        assert "extension-plan-unavailable" not in plan["blockers"]
        assert plan["can_accept"] and plan["blockers"] == []
        assert plan["missing_node_types"] == list(inputs["selected"].node_types)
        restored = await client.get(f"{URL}/{plan['id']}")
        assert restored.status_code == 200 and restored.json() == plan, restored.text
        repeated = await client.post(URL, json=payload)
        assert repeated.status_code == 201 and repeated.json() == plan
        with SessionLocal() as session:
            saved = session.get(WorkflowPackageInstallPlan, plan["id"])
            assert saved is not None
            assert saved.preflight_json["extension_execution"] == plan["extension_execution"]
            assert list(session.scalars(select(ComfyRegistryInstall))) == []
        assert len(inputs["clients"]) == 3
        assert all(item._client.is_closed for item in inputs["clients"])
    finally:
        await inputs["unused_archive"].close()


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("fits", [False, True])
async def test_preflight_reserves_the_exact_extension_identity_size(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: bool,
    fits: bool,
) -> None:
    from test_workflow_package_install_plans import (
        _installed_resource_slot,
        _sized_optional_declaration,
    )

    from local_lm.workflow_dependencies import (
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES,
        canonical_workflow_dependency_json,
    )

    inputs, payload = _configure(settings, monkeypatch, tmp_path, commit=commit)
    selected = inputs["selected"]
    additions = [
        _installed_resource_slot(
            "media_asset_1",
            "model_asset",
            {
                "kind": "model_asset",
                "asset_kind": "lora",
                "runtime_reference": "styles/detail.safetensors",
                "sha256": "a" * 64,
            },
        ),
        _installed_resource_slot(
            "extension_1",
            "registry_package",
            {
                "package_id": selected.package_id,
                "package_version": selected.declared_version,
                "node_types": list(selected.node_types),
                "archive_sha256": "a" * 64,
                "manifest_sha256": "b" * 64,
                "wheel_closure_sha256": "c" * 64,
                "wheel_environment_sha256": "d" * 64,
            },
        ),
        _installed_resource_slot("media_runtime_2", "runtime", {"engine": "comfyui"}),
    ]
    payload["dependencies"] = _sized_optional_declaration(
        MAX_WORKFLOW_DEPENDENCY_PAYLOAD_BYTES
        - sum(len(canonical_workflow_dependency_json(slot)) + 1 for slot in additions)
        + (0 if fits else 1)
    )
    try:
        created = await client.post(URL, json=payload)
        assert created.status_code == 201, created.text
        plan = created.json()
        assert plan["extension_execution"]["errors"] == {}
        assert ("dependency-contract-capacity-exceeded" in plan["blockers"]) is (not fits)
        assert "dependency-contract-unrepresentable" not in plan["blockers"]
        assert plan["can_accept"] is fits
        restored = await client.get(f"{URL}/{plan['id']}")
        assert restored.status_code == 200 and restored.json() == plan, restored.text
        if not fits:
            refused = await client.post(f"/api/workflow-install-offers/{plan['id']}/install")
            assert refused.status_code == 409, refused.text
    finally:
        await inputs["unused_archive"].close()


@pytest.mark.parametrize("change", ["archive", "target", "version", "saved", "runtime-missing"])
async def test_fresh_extension_drift_refuses_saved_workflow_preflight(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    try:
        created = await client.post(URL, json=payload)
        assert created.status_code == 201, created.text
        plan = created.json()
        if change == "archive":
            inputs["state"]["content"] = archive_bytes({"__init__.py": b"CHANGED = True\n"})
        elif change == "target":
            inputs["environment"]["platform_release"] += "-changed"
        elif change == "version":
            inputs["registry_client"]._resolution = replace(
                inputs["selected"], declared_version="9.9.9"
            )
        elif change == "runtime-missing":
            monkeypatch.setattr(settings, "comfy_executable", None)
        else:
            with SessionLocal() as session:
                saved = session.get(WorkflowPackageInstallPlan, plan["id"])
                assert saved is not None
                altered = copy.deepcopy(saved.preflight_json)
                altered["extension_execution"]["plans"][inputs["selected"].package_id][
                    "archive_bytes"
                ] += 1
                saved.preflight_json = altered
                session.commit()
        response = await client.get(f"{URL}/{plan['id']}")
        assert response.status_code == 409, response.text
        assert response.json()["code"] == "workflow-package-install-plan-changed"
        with SessionLocal() as session:
            assert list(session.scalars(select(ComfyRegistryInstall))) == []
    finally:
        await inputs["unused_archive"].close()


async def test_extension_refusal_retains_a_fixed_reason_and_unknown_cost(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    inputs["state"]["content"] = b"A neutral invalid archive"
    try:
        response = await client.post(URL, json=payload)
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["extension_execution"]["plans"] == {}
        assert result["extension_execution"]["errors"][inputs["selected"].package_id]
        assert result["total_download_bytes"] is None and not result["can_accept"]
        assert all(item._client.is_closed for item in inputs["clients"])
    finally:
        await inputs["unused_archive"].close()


async def test_cancellation_during_extension_resolution_saves_no_workflow_plan(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    entered = asyncio.Event()

    async def cancelled(_requirements: Any) -> None:
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(inputs["registry_client"], "resolve", cancelled)
    try:
        task = asyncio.create_task(client.post(URL, json=payload))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with SessionLocal() as session:
            assert list(session.scalars(select(WorkflowPackageInstallPlan))) == []
        assert all(item._client.is_closed for item in inputs["clients"])
    finally:
        await inputs["unused_archive"].close()


async def test_client_cannot_supply_extension_execution_inputs(client: AsyncClient) -> None:
    payload = _payload()
    payload["extension_execution"] = {"plans": {}, "errors": {}}
    response = await client.post(URL, json=payload)
    assert response.status_code == 422, response.text


@pytest.mark.parametrize("change", ["version", "nodes", "extra", "error"])
async def test_an_inspected_plan_cannot_be_attached_to_different_source_requirements(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    from local_lm.workflow_package_extension_preflight import WorkflowPackageExtensionPreflight
    from local_lm.workflow_package_install_plans import (
        WorkflowPackageInstallPlanRequest,
        create_workflow_package_install_plan,
    )

    try:
        response = await client.post(URL, json=payload)
        assert response.status_code == 201, response.text
        execution = WorkflowPackageExtensionPreflight.model_validate(
            response.json()["extension_execution"]
        )
        identifier = inputs["selected"].package_id
        if change == "version":
            payload["ui_graph"]["nodes"][-1]["properties"]["ver"] = "9.9.9"
        elif change == "nodes":
            payload["ui_graph"]["nodes"][-1]["type"] = "ChangedNeutralNode"
        elif change == "extra":
            execution.plans["unrelated-package"] = execution.plans[identifier]
        else:
            execution.errors[identifier] = "workflow-package-execution-plan-unavailable"
        with SessionLocal() as session:
            with pytest.raises(ValueError) as refused:
                create_workflow_package_install_plan(
                    session,
                    WorkflowPackageInstallPlanRequest.model_validate(payload),
                    available_node_types={"LoraLoader"},
                    available_asset_filenames=set(),
                    installed_package_versions={},
                    extension_execution=execution,
                )
            assert getattr(refused.value, "code", None) in {
                "workflow-package-resolution-changed",
                "workflow-package-extension-plan-mismatch",
            }
            session.rollback()
            assert len(list(session.scalars(select(WorkflowPackageInstallPlan)))) == 1
    finally:
        await inputs["unused_archive"].close()


@pytest.mark.parametrize("during_get", [False, True])
async def test_extension_inspection_leaves_the_database_writer_available(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    during_get: bool,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    from local_lm import workflow_package_extension_preflight as preflight

    inspected = False

    def write() -> None:
        with SessionLocal() as session:
            session.execute(text("UPDATE workflow_package_install_plans SET id = id WHERE 0"))
            session.commit()

    async def probe(python: Path) -> Any:
        nonlocal inspected
        await asyncio.to_thread(write)
        inspected = True
        return await inputs["interpreter_probe"](python)

    try:
        if during_get:
            created = await client.post(URL, json=payload)
            assert created.status_code == 201, created.text
        monkeypatch.setattr(preflight, "probe_comfy_registry_runtime_target", probe)
        response = (
            await client.get(f"{URL}/{created.json()['id']}")
            if during_get
            else await client.post(URL, json=payload)
        )
        assert response.status_code == (200 if during_get else 201), response.text
        assert inspected and response.json()["extension_execution"]["plans"]
    finally:
        await inputs["unused_archive"].close()


@pytest.mark.parametrize("invalid", ["declaration", "count"])
async def test_invalid_extension_preflight_stops_before_external_resolution(
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid: str,
) -> None:
    inputs, payload = _configure(settings, monkeypatch, tmp_path)
    if invalid == "declaration":
        payload["dependencies"] = {"version": 99, "slots": []}
    else:
        for index in range(65):
            node = copy.deepcopy(payload["ui_graph"]["nodes"][-1])
            node["id"] += 1
            node["type"] = f"NeutralExtension{index}"
            node["properties"]["cnr_id"] = f"neutral-extension-{index}"
            payload["ui_graph"]["nodes"].append(node)
    try:
        response = await client.post(URL, json=payload)
        assert response.status_code == (422 if invalid == "declaration" else 201), response.text
        assert inputs["registry_client"].requested == [] and inputs["clients"] == []
        if invalid == "count":
            execution = response.json()["extension_execution"]
            assert execution["plans"] == {}
            assert set(execution["errors"].values()) == {"workflow-package-limit-exceeded"}
    finally:
        await inputs["unused_archive"].close()
