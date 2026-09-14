"""Accepted dependency launches keep exact source identity without granting graph review."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_workflow_activations import _asset, _registry, _slot
from test_workflow_offer_packages import _accepted, _prepare, _setup

from local_lm.comfy_editor_bridge import ComfyEditorBridgeSupport
from local_lm.comfy_registry_activation_batches import _begin
from local_lm.comfy_registry_lifecycle import ComfyRegistryPreparation
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ComfyRegistryInstall,
    Job,
    WorkflowActivation,
    WorkflowInstallOffer,
    WorkflowPackageInstallPlan,
    WorkflowRevision,
    WorkflowRevisionReview,
)
from local_lm.processes import ProcessSupervisor
from local_lm.runtime_provisioning import RuntimeProvisioner
from local_lm.workflow_activations import materialize_comfy_runtime_dependency
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_offer_packages import record_workflow_offer_package
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions

pytestmark = pytest.mark.asyncio


async def _fixture(tmp_path: Path, settings: Settings) -> dict[str, Any]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "main.py").write_text(
        "raise RuntimeError('This fixture must stay inert')\n", encoding="utf-8"
    )
    settings.comfy_directory = runtime
    settings.comfy_executable = Path(sys.executable)
    context = PreparationContext(
        Path(sys.executable), settings.custom_node_dir, settings.registry_dir
    )
    content = b"Neutral accepted weights"
    payload = {
        "name": "Neutral accepted dependency launch",
        "operation": "text_to_image",
        "ui_graph": {"version": 0.4, "nodes": [], "links": []},
        "dependencies": {
            "version": 1,
            "slots": [
                _slot(
                    "style",
                    "model_asset",
                    requirements=[
                        {
                            "key": "style",
                            "constraints": {"sha256": hashlib.sha256(content).hexdigest()},
                        },
                    ],
                ),
            ],
        },
        "selections": [],
    }
    inputs, context, offer_id, saved = await _setup(
        tmp_path,
        source_payload=payload,
        preparation_context=context,
    )
    with SessionLocal() as session:
        item = _accepted(session, offer_id, saved)
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None and offer.completion_job_id is None
        completion_job_id = stage_workflow_completion_job(session, offer).id
        item.job.status = "running"
        job_id = item.job.id
        asset = _asset(session, tmp_path / "asset", suffix="source", content=content)
        asset_id = asset.id
        session.commit()

    def record(session: Session, prepared: ComfyRegistryPreparation) -> None:
        offer = session.get(WorkflowInstallOffer, offer_id)
        assert offer is not None
        record_workflow_offer_package(
            session,
            offer,
            saved,
            package_id=inputs["selected"].package_id,
            job_id=job_id,
            preparation=prepared,
        )

    prepared = await _prepare(inputs, context, saved, record)
    trusted = trust_workflow_source_extensions(
        SessionLocal,
        offer_id,
        context=context,
        media_worker_stopped=True,
    )
    assert trusted.state == "ready"
    _begin(
        SessionLocal,
        offer_id,
        (prepared,),
        {prepared.install_id: saved.extension_execution.plans[inputs["selected"].package_id]},
        context.custom_node_root,
        registry_wheel_environment_root(context.state_root),
    )
    state = {"release": "neutral-runtime"}
    ensured = []

    async def ensure(_engine: str) -> None:
        ensured.append(True)
        raise AssertionError("An accepted source must not silently provision another runtime")

    provisioner = Mock(
        spec=RuntimeProvisioner,
        status=lambda _engine: SimpleNamespace(state="ready", release=state["release"]),
        ensure=ensure,
    )
    return {
        "context": context,
        "offer_id": offer_id,
        "saved": saved,
        "prepared": prepared,
        "job_id": job_id,
        "completion_job_id": completion_job_id,
        "asset_id": asset_id,
        "provisioner": provisioner,
        "materializer": lambda requirement, selection: materialize_comfy_runtime_dependency(
            provisioner,
            requirement,
            selection,
        ),
        "runtime_state": state,
        "ensured": ensured,
    }


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "unrelated",
        "asset-bytes",
        "code-bytes",
        "offer",
        "plan",
        "job",
        "untrusted",
        "inactive",
        "runtime",
        "scope",
        "missing-result",
        "ambiguity",
        "completion-cancelled",
        "completion-failed",
        "completion-paused",
        "completion-interrupted",
        "completion-retried",
        "completion-link-missing",
    ],
)
async def test_source_scope_binds_only_accepted_and_declared_resources(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    change: str,
) -> None:
    from local_lm.workflow_source_launch import (
        prepare_workflow_source_launch_scope,
        revalidate_workflow_source_launch_scope,
    )

    fixture = await _fixture(tmp_path, settings)
    context = fixture["context"]
    with SessionLocal() as session:
        offer = session.get(WorkflowInstallOffer, fixture["offer_id"])
        assert offer is not None
        draft_id = offer.workflow_revision_id
        if change == "unrelated":
            _registry(
                session,
                context.custom_node_root,
                registry_wheel_environment_root(context.state_root),
                suffix="unrelated",
                node_types=["UnrelatedNode"],
            )
            _asset(session, tmp_path / "unrelated", suffix="other", content=b"other")
            session.commit()

    def prepare() -> Any:
        return prepare_workflow_source_launch_scope(
            SessionLocal,
            fixture["offer_id"],
            context=context,
            runtime_materializer=fixture["materializer"],
        )

    scope = prepare()
    assert (
        scope.offer_id == fixture["offer_id"] and scope.plan_sha256 == fixture["saved"].plan_sha256
    )
    assert scope.model_asset_install_ids == (fixture["asset_id"],)
    assert scope.registry_install_ids == (fixture["prepared"].install_id,)
    assert scope.runtime_keys == ("comfyui",) and len(scope.launch_sha256) == 64
    assert not hasattr(scope, "activation_id") and not hasattr(scope, "workflow_revision_id")
    assert prepare() == scope
    with SessionLocal() as session:
        draft = session.get(WorkflowRevision, draft_id)
        assert draft is not None and not draft.trusted and draft.api_graph_json == {}
        assert session.get(WorkflowRevisionReview, draft_id) is None
        assert list(session.scalars(select(WorkflowActivation))) == []
        if change in {"untrusted", "inactive"}:
            row = session.get(ComfyRegistryInstall, fixture["prepared"].install_id)
            assert row is not None
            setattr(row, "trusted" if change == "untrusted" else "active", False)
        elif change == "offer":
            offer = session.get(WorkflowInstallOffer, fixture["offer_id"])
            assert offer is not None
            offer.status = "invalidated"
        elif change == "plan":
            plan = session.get(WorkflowPackageInstallPlan, fixture["saved"].id)
            assert plan is not None
            plan.request_json = {**plan.request_json, "name": "Changed source"}
        elif change == "job":
            job = session.get(Job, fixture["job_id"])
            assert job is not None
            job.result_json = {}
        elif change.startswith("completion-"):
            job = session.get(Job, fixture["completion_job_id"])
            assert job is not None
            if change == "completion-retried":
                job.attempt += 1
            elif change == "completion-link-missing":
                offer = session.get(WorkflowInstallOffer, fixture["offer_id"])
                assert offer is not None
                offer.completion_job_id = None
            else:
                job.status = change.removeprefix("completion-")
        elif change == "missing-result":
            row = session.get(ComfyRegistryInstall, fixture["prepared"].install_id)
            assert row is not None
            session.delete(row)
        elif change == "ambiguity":
            _asset(
                session,
                tmp_path / "duplicate",
                suffix="duplicate",
                content=b"Neutral accepted weights",
            )
        session.commit()
    if change == "asset-bytes":
        (tmp_path / "asset/styles/style.safetensors").write_bytes(b"Changed weights")
    elif change == "code-bytes":
        (context.custom_node_root / fixture["prepared"].installed_path / "__init__.py").write_bytes(
            b"Changed code"
        )
    elif change == "runtime":
        fixture["runtime_state"]["release"] = "changed-runtime"
    elif change == "scope":
        scope = replace(scope, launch_sha256="f" * 64)

    def revalidate() -> None:
        revalidate_workflow_source_launch_scope(
            SessionLocal,
            scope,
            context=context,
            runtime_materializer=fixture["materializer"],
        )

    if change in {"none", "unrelated"}:
        revalidate()
    else:
        with pytest.raises(ValueError):
            revalidate()


@pytest.mark.parametrize("change", ["missing-runtime", "after-port-check"])
async def test_source_launch_refuses_unbound_provisioning_and_last_moment_changes(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from local_lm.workflow_source_launch import prepare_workflow_source_launch_scope

    fixture = await _fixture(tmp_path, settings)
    scope = prepare_workflow_source_launch_scope(
        SessionLocal,
        fixture["offer_id"],
        context=fixture["context"],
        runtime_materializer=fixture["materializer"],
    )
    processes = ProcessSupervisor(settings, runtimes=fixture["provisioner"])
    spawned = []

    async def spawn(*_args: Any, **_kwargs: Any) -> Any:
        spawned.append(True)
        raise AssertionError("The changed acceptance reached process creation")

    monkeypatch.setattr("local_lm.processes.asyncio.create_subprocess_exec", spawn)
    if change == "missing-runtime":
        settings.comfy_executable = tmp_path / "missing-python"
        with pytest.raises(RuntimeError):
            await processes.start_media(activation_scope=scope)
    else:

        async def reclaim(*_args: object) -> None:
            return None

        async def check_port(*_args: object) -> None:
            with SessionLocal() as session:
                offer = session.get(WorkflowInstallOffer, fixture["offer_id"])
                assert offer is not None
                offer.status = "invalidated"
                session.commit()

        monkeypatch.setattr(processes, "_reclaim_port_from_our_own_children", reclaim)
        monkeypatch.setattr(processes, "_ensure_port_available", check_port)
        assert settings.comfy_directory is not None
        with pytest.raises(ValueError):
            await processes._replace(
                "media",
                [str(settings.comfy_executable), str(settings.comfy_directory / "main.py")],
                "http://127.0.0.1:8188",
                launch_scope_sha256=scope.launch_sha256,
                prestart_check=lambda: processes._revalidate_source_media_scope(scope),
            )
    assert spawned == [] and fixture["ensured"] == []


@pytest.mark.parametrize("change", ["none", "offer", "runtime"])
async def test_source_start_rechecks_acceptance_after_preparing_exact_launch_paths(
    client: AsyncClient,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from local_lm.workflow_source_launch import prepare_workflow_source_launch_scope

    fixture = await _fixture(tmp_path, settings)
    scope = prepare_workflow_source_launch_scope(
        SessionLocal,
        fixture["offer_id"],
        context=fixture["context"],
        runtime_materializer=fixture["materializer"],
    )
    processes = ProcessSupervisor(settings, runtimes=fixture["provisioner"])
    commands: list[tuple[str, ...]] = []
    phases: list[str] = []

    async def spawn(*command: str, **_kwargs: Any) -> Any:
        commands.append(command)
        raise RuntimeError("Neutral launch boundary")

    async def no_port_work(*_args: object) -> None:
        return None

    def no_broad_inventory(*_args: object) -> Any:
        raise AssertionError("An accepted source queried the unrestricted inventory")

    async def phase(value: str) -> None:
        phases.append(value)
        if value != "Starting media runtime":
            return
        if change == "offer":
            with SessionLocal() as session:
                offer = session.get(WorkflowInstallOffer, fixture["offer_id"])
                assert offer is not None
                offer.status = "invalidated"
                session.commit()
        elif change == "runtime":
            fixture["runtime_state"]["release"] = "changed-before-spawn"

    monkeypatch.setattr("local_lm.processes.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr(
        "local_lm.processes.prepare_comfy_editor_bridge",
        lambda **_kwargs: SimpleNamespace(
            folder=None,
            support=ComfyEditorBridgeSupport(
                False,
                "workflow-editor-runtime-unavailable",
                "The neutral fixture does not provide the editor bridge.",
            ),
        ),
    )
    monkeypatch.setattr(processes, "_reclaim_port_from_our_own_children", no_port_work)
    monkeypatch.setattr(processes, "_ensure_port_available", no_port_work)
    monkeypatch.setattr(processes, "_trusted_comfy_registry_contract", no_broad_inventory)
    monkeypatch.setattr(processes, "_trusted_comfy_node_folders", no_broad_inventory)
    monkeypatch.setattr(processes, "_write_comfy_model_paths", no_broad_inventory)
    if change == "none":
        with pytest.raises(RuntimeError, match="Neutral launch boundary"):
            await processes.start_media(activation_scope=scope, phase_callback=phase)
    else:
        with pytest.raises(ValueError):
            await processes.start_media(activation_scope=scope, phase_callback=phase)
    assert phases[-1] == "Starting media runtime"
    assert fixture["ensured"] == []
    if change != "none":
        assert commands == []
        return
    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--whitelist-custom-nodes") + 1 :] == (
        fixture["prepared"].installed_path,
    )
    config = Path(command[command.index("--extra-model-paths-config") + 1])
    assert config.name == scope.launch_sha256 + ".yaml"
    paths = json.loads(config.read_text(encoding="utf-8"))
    assert list(paths.values()) == [
        {"base_path": str((tmp_path / "asset").resolve()), "loras": "."},
    ]
