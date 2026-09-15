"""Accepted extension bytes pass through preparation, trust and executable completion."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import traceback
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_workflow_completion_jobs import _settled
from test_workflow_offer_packages import _setup
from test_workflow_package_execution_plan import _inputs
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import models, workflow_source_completion, workflow_source_runtime
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_paths import registry_wheel_environment_root
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.db import SessionLocal
from local_lm.workflow_activations import (
    materialize_comfy_runtime_dependency,
    revalidate_workflow_activation,
)
from local_lm.workflow_completion_jobs import stage_workflow_completion_job
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_revision_reviews import review_is_current
from local_lm.workflow_source_extensions import (
    ExtensionPreparationServices,
    prepare_workflow_source_extensions,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def fresh_runtime(
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_runtime: dict[str, Any],
    mode: str,
) -> AsyncIterator[list[httpx.Request] | None]:
    if mode not in {
        "runtime-complete",
        "cancel-runtime",
        "retry-runtime",
        "recover-runtime-configuration",
        "recover-runtime-persistence",
    }:
        yield None
        return
    from test_runtime_provisioning import _write_manifest, _zip_bytes

    from local_lm import runtime_provisioning
    from local_lm.runtime_provisioning import RuntimeProvisioner

    services = app.state.services
    await services.processes.stop("media")
    services.settings.comfy_executable = None
    services.settings.comfy_directory = None
    content = _zip_bytes(
        {
            "python/python.exe": b"neutral runtime executable",
            "python/Lib/site-packages/example-1.0.dist-info/METADATA": (
                b"Name: example\nVersion: 1.0\n"
            ),
            "ComfyUI/main.py": b"neutral runtime source",
        }
    )
    manifest = tmp_path / "engines.json"
    _write_manifest(manifest, llama_content=b"unused", comfy_content=content)
    probe = {"python": "3.13.14", "comfyui": "0.28.0", "packages": {"example": "1.0"}}

    def run(command: Any, **_kwargs: Any) -> Any:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=runtime_provisioning._RUNTIME_PROBE_SENTINEL + json.dumps(probe) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)
    requests: list[httpx.Request] = []

    def serve(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "runtime.test"
        requests.append(request)
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as transport:
        provisioner = RuntimeProvisioner(
            services.settings,
            manifest_path=manifest,
            client=transport,
            environment={},
            platform_key="test-platform",
            allowed_download_hosts={"runtime.test"},
        )
        monkeypatch.setattr(services, "runtimes", provisioner)
        monkeypatch.setattr(services.processes, "runtimes", provisioner)
        if mode.startswith("recover-runtime-"):
            target = provisioner if mode.endswith("configuration") else runtime_provisioning
            attribute = (
                "_apply_configuration"
                if mode.endswith("configuration")
                else "persist_runtime_values"
            )
            original = getattr(target, attribute)
            failed = False

            def persist(*args: Any, **kwargs: Any) -> Any:
                nonlocal failed
                if not failed:
                    failed = True
                    raise OSError("Neutral runtime persistence interruption")
                return original(*args, **kwargs)

            monkeypatch.setattr(target, attribute, persist)
        try:
            yield requests
        finally:
            await provisioner.close()


@pytest.mark.parametrize(
    "mode",
    [
        "complete",
        "validation-failure",
        "trust-review",
        "trust-resume",
        "trust-resume-running",
        "trust-stop-failure",
        "trust-revoke",
        "trust-cancelled",
        "cancel-preparation",
        "cancel-launch",
        "retry-preparation",
        "retry-launch",
        "recover-activation",
        "refuse-activation-code",
        "refuse-activation-identity",
        "cancel-interrupted-activation",
        "cancel-interrupted-clean",
        "cancel-interrupted-restart",
        "cancel-interrupted-running",
        "cancel-interrupted-session",
        "cancel-interrupted-session-stopped",
        "cancel-interrupted-stop-refused",
        "cancel-interrupted-stop-active",
        "cancel-interrupted-stop-persisted",
        "runtime-complete",
        "cancel-runtime",
        "retry-runtime",
        "recover-runtime-configuration",
        "recover-runtime-persistence",
    ],
)
async def test_source_extensions_finish_or_pause_without_partial_workflow_activation(
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_runtime: dict[str, Any],
    fresh_runtime: list[httpx.Request] | None,
    mode: str,
) -> None:
    services = app.state.services
    if fresh_runtime is None:
        services.settings.comfy_executable = Path(sys.executable)
    context = PreparationContext(
        Path(sys.executable), services.settings.custom_node_dir, services.settings.registry_dir
    )
    inputs = _inputs(commit=mode.startswith("trust-"))
    source_payload = {
        "name": "Neutral extension source",
        "operation": "text_to_image",
        "ui_graph": _ui_graph(),
        "dependencies": {"version": 1, "slots": []},
        "selections": [],
    }
    _inputs_value, context, offer_id, _saved = await _setup(
        tmp_path,
        package_inputs=[inputs],
        source_payload=source_payload,
        preparation_context=context,
        runtime_plan=services.processes.runtimes.preflight("comfyui"),
        available_nodes=frozenset(source_runtime),
    )
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        completion_id = stage_workflow_completion_job(session, offer).id
        session.commit()
    for node_type in inputs["selected"].node_types:
        source_runtime[node_type] = {
            "python_module": "custom_nodes.example",
            "input": {"required": {}},
            "output": [],
        }

    async def inventory() -> frozenset[str]:
        return frozenset(source_runtime)

    validations: list[dict[str, Any]] = []
    failures: list[tuple[str, int | None]] = []
    complete = workflow_source_completion.complete_workflow_source

    async def recorded_completion(*args: Any, **kwargs: Any) -> Any:
        try:
            return await complete(*args, **kwargs)
        except Exception as exc:
            failures.extend(
                (frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)
            )
            raise

    monkeypatch.setattr(workflow_source_completion, "complete_workflow_source", recorded_completion)

    async def validate(_graph: dict[str, Any]) -> list[str]:
        validations.append(_graph)
        return ["Constructed validation failure"] if mode == "validation-failure" else []

    monkeypatch.setattr(services.processes, "comfy_node_inventory", inventory)
    monkeypatch.setattr(services.engines.media, "validate_workflow", validate)
    entered, release = asyncio.Event(), asyncio.Event()
    interrupted = False

    async def hold_phase(phase: str) -> None:
        nonlocal interrupted
        if not interrupted and mode in {f"cancel-{phase}", f"retry-{phase}"}:
            interrupted = True
            entered.set()
            await release.wait()

    original_start = services.processes.start_media

    async def start(*args: Any, **kwargs: Any) -> Any:
        worker = await original_start(*args, **kwargs)
        if kwargs.get("activation_scope") is not None:
            await hold_phase("launch")
        return worker

    monkeypatch.setattr(services.processes, "start_media", start)
    original_provision = services.runtimes.provision

    async def provision(*args: Any, **kwargs: Any) -> Any:
        result = await original_provision(*args, **kwargs)
        await hold_phase("runtime")
        return result

    monkeypatch.setattr(services.runtimes, "provision", provision)

    async def archive_response(_request: httpx.Request) -> httpx.Response:
        await hold_phase("preparation")
        return httpx.Response(200, content=inputs["state"]["content"])

    prepare = prepare_workflow_source_extensions
    async with AsyncExitStack() as cleanup:
        archive = ComfyRegistryArchiveDownloader(transport=httpx.MockTransport(archive_response))
        cleanup.push_async_callback(archive.close)
        wheels = ComfyRegistryWheelDownloader(
            transport=httpx.MockTransport(lambda _request: httpx.Response(500))
        )
        cleanup.push_async_callback(wheels.close)

        async def prepare_with_local_transport(*args: Any, **kwargs: Any) -> Any:
            return await prepare(
                *args,
                **kwargs,
                services=ExtensionPreparationServices(
                    inputs["interpreter_probe"],
                    inputs["registry_client"],
                    inputs["project_client"],
                    inputs["metadata_client"],
                    archive,
                    wheels,
                ),
            )

        monkeypatch.setattr(
            workflow_source_runtime,
            "prepare_workflow_source_extensions",
            prepare_with_local_transport,
        )
        if mode == "recover-activation" or mode.startswith(
            ("refuse-activation-", "cancel-interrupted-")
        ):
            from local_lm.comfy_registry_activation_batches import _begin
            from local_lm.workflow_source_extension_trust import trust_workflow_source_extensions

            await services.processes.stop("media")
            prepared = await prepare_with_local_transport(
                SessionLocal, offer_id, context=context, media_worker_stopped=True
            )
            trust_workflow_source_extensions(
                SessionLocal, offer_id, context=context, media_worker_stopped=True
            )
            batch, _before = _begin(
                SessionLocal,
                offer_id,
                prepared.preparations,
                prepared.execution_plans,
                context.custom_node_root,
                registry_wheel_environment_root(context.state_root),
            )
            filename = "__init__.py" if mode == "refuse-activation-code" else "cache.json"
            if mode != "cancel-interrupted-clean":
                (batch.bindings[0].installed_path / filename).write_text("{}", encoding="utf-8")
            if mode == "refuse-activation-identity":
                with SessionLocal() as session:
                    row = session.get(
                        models.ComfyRegistryInstall, prepared.preparations[0].install_id
                    )
                    assert row is not None
                    row.wheel_environment_sha256 = "f" * 64
                    session.commit()
        if mode.startswith("cancel-interrupted-"):
            from local_lm.comfy_registry_installs import ComfyRegistryInstallError
            from local_lm.workflow_completion_jobs import cancel_workflow_completion

            if mode == "cancel-interrupted-clean":
                assert services.processes._trusted_comfy_registry_contract().custom_node_folders
            else:
                with pytest.raises(ComfyRegistryInstallError):
                    services.processes._trusted_comfy_registry_contract()
            if mode == "cancel-interrupted-restart":
                with SessionLocal() as session:
                    offer = session.get(models.WorkflowInstallOffer, offer_id)
                    assert offer is not None
                    assert cancel_workflow_completion(session, offer) is not None
                    session.commit()
            else:
                running_cancel = mode in {
                    "cancel-interrupted-running",
                    "cancel-interrupted-session",
                    "cancel-interrupted-session-stopped",
                    "cancel-interrupted-stop-refused",
                    "cancel-interrupted-stop-active",
                    "cancel-interrupted-stop-persisted",
                }
                if running_cancel:
                    await services.processes.start_media()
                cancelled = await client.post(f"/api/jobs/{completion_id}/cancel")
                assert cancelled.status_code == 200, cancelled.text
                with SessionLocal() as session:
                    row = session.get(
                        models.ComfyRegistryInstall, prepared.preparations[0].install_id
                    )
                    assert row is not None and row.trusted
                    assert row.active == running_cancel
                if mode.startswith(("cancel-interrupted-session", "cancel-interrupted-stop-")):
                    from local_lm.processes import ProcessSupervisor, _ProcessIdentity

                    launches: list[list[str]] = []
                    stops: list[str] = []
                    original_stop_unlocked = services.processes._stop_unlocked

                    async def stop_unlocked(name: str) -> None:
                        if name != "media":
                            await original_stop_unlocked(name)
                            return
                        stops.append(name)
                        if mode.endswith("stop-refused"):
                            raise RuntimeError("Neutral media stop refusal")
                        if mode.endswith("stop-active"):
                            return
                        await services.processes.stop(name)

                    async def replace(
                        name: str, command: list[str], *_args: Any, **_kwargs: Any
                    ) -> None:
                        assert name == "media"
                        launches.append(command)

                    monkeypatch.setattr(services.processes, "_stop_unlocked", stop_unlocked)
                    monkeypatch.setattr(services.processes, "_replace", replace)
                    if mode.endswith("stop-persisted"):
                        monkeypatch.setattr(
                            services.processes,
                            "_matching_worker_processes",
                            lambda name: [_ProcessIdentity(43210, 1.0)] if name == "media" else [],
                        )
                    if mode.endswith("session-stopped"):
                        await services.processes.stop("media")
                    if mode.startswith("cancel-interrupted-stop-"):
                        failure = (
                            "Neutral media stop refusal"
                            if mode.endswith("stop-refused")
                            else "The media worker must stop"
                        )
                        with pytest.raises(RuntimeError, match=failure):
                            await ProcessSupervisor.start_media(services.processes)
                        assert stops == ["media"] and launches == []
                        with SessionLocal() as session:
                            row = session.get(
                                models.ComfyRegistryInstall, prepared.preparations[0].install_id
                            )
                            assert row is not None and row.active and row.trusted
                        return
                    await ProcessSupervisor.start_media(services.processes)
                    assert stops == ["media"] and len(launches) == 1
                    assert (
                        services.processes._trusted_comfy_registry_contract().custom_node_folders
                        == ()
                    )
                    return
                await services.processes.stop("media")
            services.downloads.recover_interrupted()
            recovery = services.downloads._offer_recovery_task
            assert recovery is not None
            await asyncio.wait_for(recovery, timeout=30)
            with SessionLocal() as session:
                row = session.get(models.ComfyRegistryInstall, prepared.preparations[0].install_id)
                assert row is not None and row.trusted and not row.active
                job = session.get(models.Job, completion_id)
                assert job is not None and job.status == "cancelled"
                assert not list(session.scalars(select(models.WorkflowActivation)))
            assert services.processes._trusted_comfy_registry_contract().custom_node_folders == ()
            return
        services.downloads.start_workflow_installation(offer_id)
        if mode.startswith(("cancel-", "retry-")):
            try:
                await asyncio.wait_for(entered.wait(), timeout=30)
                cancelled = await client.post(f"/api/jobs/{completion_id}/cancel")
                assert cancelled.status_code == 200, cancelled.text
                assert cancelled.json()["status"] == "cancelled"
                if mode.startswith("retry-"):
                    retried = await client.post(f"/api/jobs/{completion_id}/retry")
                    assert retried.status_code == 200, retried.text
                    assert retried.json()["id"] == completion_id
                with SessionLocal() as session:
                    assert list(session.scalars(select(models.WorkflowActivation))) == []
            finally:
                release.set()
        await _settled(app, offer_id)
        if mode.startswith("recover-runtime-"):
            with SessionLocal() as session:
                job = session.get(models.Job, completion_id)
                assert job is not None and job.status == "failed" and job.attempt == 1
                assert job.result_json == {}
                assert list(session.scalars(select(models.WorkflowActivation))) == []
                assert list(session.scalars(select(models.ComfyRegistryInstall))) == []
            retried = await client.post(f"/api/jobs/{completion_id}/retry")
            assert retried.status_code == 200 and retried.json()["id"] == completion_id, (
                retried.text
            )
            await _settled(app, offer_id)
        if mode.startswith("trust-"):
            progress = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
            assert progress.status_code == 200 and progress.json()["phase"] == "paused"
            with SessionLocal() as session:
                extension = session.scalar(select(models.ComfyRegistryInstall))
                job = session.get(models.Job, completion_id)
                assert extension is not None and job is not None
                assert job.status == "paused" and job.attempt == 1
                assert not extension.trusted and not extension.active
                extension_id = extension.id
            if mode != "trust-review":
                if mode not in {"trust-resume-running", "trust-stop-failure"}:
                    await services.processes.stop("media")
                if mode == "trust-stop-failure":

                    async def refuse_stop(_name: str) -> Any:
                        return next(
                            worker
                            for worker in services.processes.statuses()
                            if worker.name == "media"
                        )

                    monkeypatch.setattr(services.processes, "stop", refuse_stop)
                if mode == "trust-cancelled":
                    cancelled = await client.post(f"/api/jobs/{completion_id}/cancel")
                    assert cancelled.status_code == 200
                response = await client.post(
                    f"/api/workflows/packages/installs/{extension_id}/review",
                    json={"trusted": mode != "trust-revoke"},
                )
                assert response.status_code == (409 if mode == "trust-stop-failure" else 200), (
                    response.text
                )
                task = services.downloads._offer_tasks.get(offer_id)
                if task is not None:
                    await task

    if fresh_runtime is not None:
        from local_lm.runtime_config import runtime_config_path

        assert len(fresh_runtime) == 1
        assert services.runtimes.preflight("comfyui").operation == "reuse_managed"
        assert runtime_config_path(services.settings.data_dir).is_file()
        context = PreparationContext.from_settings(services.settings)
    progress = await client.get(f"/api/workflow-install-offers/{offer_id}/progress")
    assert progress.status_code == 200, progress.text
    if mode.startswith("cancel-") or mode == "trust-cancelled":
        # A cancelled installation says so and names the job a retry resumes.
        assert progress.json()["phase"] == "needs_attention"
        assert progress.json()["attention_code"] == "workflow-install-cancelled"
        assert progress.json()["retry_job_id"] == completion_id
    elif mode.startswith("retry-") or mode == "complete":
        assert progress.json()["retry_job_id"] is None
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        job = session.get(models.Job, completion_id)
        extension = session.scalar(select(models.ComfyRegistryInstall))
        assert offer is not None and job is not None
        if mode.startswith("cancel-"):
            assert interrupted and not validations
            assert offer.status == "queued" and job.status == "cancelled" and job.attempt == 1
            assert job.result_json.get("activation_id") is None
            assert extension is None or not extension.active
            assert list(session.scalars(select(models.WorkflowActivation))) == []
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None and not revision.trusted and revision.api_graph_json == {}
            assert services.processes.launch_scope_sha256("media") is None
            return
        assert extension is not None
        if mode in {"trust-review", "trust-revoke", "trust-stop-failure"}:
            assert job.status == "paused"
            assert offer.completion_error_code == "workflow-extension-review-required"
            assert not extension.trusted and not extension.active
        elif mode == "trust-cancelled":
            assert job.status == "cancelled" and job.attempt == 1
            assert not extension.active and not validations
        elif mode == "validation-failure" or mode.startswith("refuse-activation-"):
            assert len(validations) == (1 if mode == "validation-failure" else 0), (
                offer.completion_error_code,
                failures,
            )
            assert job.status == ("queued" if mode == "refuse-activation-identity" else "failed")
            assert offer.status == "queued"
            assert not extension.active
            assert not list(session.scalars(select(models.WorkflowActivation)))
        else:
            assert len(validations) == 1, (offer.completion_error_code, failures)
            assert offer.status == "completed" and job.status == "complete", (
                offer.completion_error_code
            )
            assert extension.active and extension.trusted
            assert extension.review_json["activation_batch_v1"]["state"] == "complete"
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None
            definition = session.get(models.WorkflowDefinition, revision.workflow_id)
            assert definition is not None and review_is_current(session, definition, revision)
            activation = session.get(models.WorkflowActivation, job.result_json["activation_id"])
            assert activation is not None
            scope = revalidate_workflow_activation(
                session,
                activation,
                runtime_materializer=lambda requirement, selection: (
                    materialize_comfy_runtime_dependency(
                        services.processes.runtimes, requirement, selection
                    )
                ),
                custom_node_root=context.custom_node_root,
                registry_environment_root=registry_wheel_environment_root(context.state_root),
            )
            assert scope.registry_install_ids == (extension.id,) and scope.runtime_keys == (
                "comfyui",
            )
            assert job.attempt == (
                3
                if mode.startswith(("retry-", "recover-runtime-"))
                else 2
                if mode.startswith("trust-resume")
                else 1
            )
    if mode.startswith("refuse-activation-"):
        await services.processes.stop("media")
        assert services.processes._trusted_comfy_registry_contract().custom_node_folders == ()
