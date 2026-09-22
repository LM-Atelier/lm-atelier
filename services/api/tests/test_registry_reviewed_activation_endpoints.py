from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import delete
from test_comfy_registry_reviewed_batches import _watch_files
from test_workflow_reviewed_package_plan import _close, _inputs, _prepare

from local_lm import comfy_registry_interpreter as interpreter
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.comfy_registry_target_verification import ComfyRegistryVerificationTarget
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview
from local_lm.registry_trust_policy import POLICY_ID
from local_lm.workflow_package_activation import activate_prepared_workflow_package
from local_lm.workflow_package_execution_plan import plan_workflow_package_execution
from local_lm.workflow_package_preparation import WorkflowPackagePreparationError


@pytest.fixture
async def package(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> AsyncIterator[dict[str, Any]]:
    mode = getattr(request, "param", "active")
    settings.comfy_directory = tmp_path / "runtime"
    settings.custom_node_dir.mkdir(parents=True)
    with SessionLocal() as session:
        inputs = _inputs(
            (session, app.state.services.artifacts),
            tmp_path,
            commit=False,
            inactive=mode != "active",
        )
    if mode == "empty":
        inputs["runtime"].clear()
    inputs["context"] = replace(
        inputs["context"],
        custom_node_root=settings.custom_node_dir,
        state_root=settings.registry_dir,
    )
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        preparation = await _prepare(inputs, plan)
        settings.comfy_executable = tmp_path / "selected-python.exe"
        settings.comfy_executable.write_bytes(b"Neutral interpreter boundary")
        context = replace(inputs["context"], python_executable=settings.comfy_executable)
        probe = inputs["plan_arguments"]["interpreter_probe"]
        probes: list[Path] = []
        starts: list[bool] = []

        async def actual_probe(executable: Path) -> Any:
            assert executable == settings.comfy_executable
            probes.append(executable)
            return await probe(executable)

        async def start(*args: Any, **kwargs: Any) -> Any:
            with SessionLocal() as session:
                install = session.get(ComfyRegistryInstall, preparation.install_id)
                assert install is not None
                starts.append(install.active)
            return object()

        monkeypatch.setattr(interpreter, "probe_comfy_registry_runtime_target", actual_probe)
        monkeypatch.setattr(app.state.services.processes, "start_media", start)
        yield dict(
            inputs=inputs,
            preparation=preparation,
            context=context,
            probes=probes,
            starts=starts,
            processes=app.state.services.processes,
            url=f"/api/workflows/packages/installs/{preparation.install_id}",
        )
    finally:
        await _close(inputs)


def state(package: dict[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    with SessionLocal() as session:
        install = session.get(ComfyRegistryInstall, package["preparation"].install_id)
        assert install is not None
        return install.trusted, install.active, install.review_json


@pytest.mark.parametrize("package", ["active", "inactive", "empty"], indirect=True)
async def test_explicit_review_and_activation_verify_prepared_source_packages(
    package: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _watch_files(monkeypatch)
    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 200, response.text
    assert response.json()["trusted"] and not response.json()["active"]
    response = await client.post(package["url"] + "/activate")
    assert response.status_code == 200, response.text
    assert response.json()["active"] and response.json()["activated_at"]
    assert package["starts"] == [True] and len(package["probes"]) == 3
    assert seen and state(package)[2]["trust_authority"] == "local_user"


@pytest.mark.parametrize("change", ["source-review", "runtime", "configuration"])
async def test_explicit_review_refuses_changed_source_or_runtime_evidence(
    package: dict[str, Any],
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    if change == "source-review":
        with SessionLocal() as session:
            session.execute(delete(ComfyRegistrySourceArtifactReview))
            session.commit()
    elif change == "runtime":
        package["inputs"]["runtime"][:] = [ComfyRegistryRuntimeDistribution("helper", "3.0")]
    else:
        original = interpreter.probe_comfy_registry_runtime_target

        async def probe(executable: Path) -> Any:
            result = await original(executable)
            settings.comfy_executable = executable.with_name("different-python.exe")
            return result

        monkeypatch.setattr(interpreter, "probe_comfy_registry_runtime_target", probe)
    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "registry_install_verification_failed"
    assert state(package)[:2] == (False, False) and not package["starts"]


async def test_revocation_needs_no_source_probe_or_files(
    package: dict[str, Any],
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 200
    with SessionLocal() as session:
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()

    async def refuse(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Revoking trust must not verify source or runtime files")

    monkeypatch.setattr(ComfyRegistryVerificationTarget, "verify", refuse)
    response = await client.post(package["url"] + "/review", json={"trusted": False})
    assert response.status_code == 200, response.text
    assert state(package)[:2] == (False, False)


@pytest.mark.parametrize("change", ["source-review", "configuration", "cancel"])
async def test_activation_endpoint_restores_after_startup_invalidates_the_trial(
    package: dict[str, Any],
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 200
    original = package["processes"].start_media
    entered = asyncio.Event()
    task = None

    async def start() -> Any:
        result = await original()
        if len(package["starts"]) == 1:
            if change == "source-review":
                with SessionLocal() as session:
                    session.execute(delete(ComfyRegistrySourceArtifactReview))
                    session.commit()
            elif change == "configuration":
                assert settings.comfy_executable is not None
                settings.comfy_executable = settings.comfy_executable.with_name(
                    "different-python.exe"
                )
            else:
                entered.set()
                await asyncio.Future()
        return result

    monkeypatch.setattr(package["processes"], "start_media", start)
    try:
        if change == "cancel":
            task = asyncio.create_task(client.post(package["url"] + "/activate"))
            await asyncio.wait_for(entered.wait(), 30)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            response = await client.post(package["url"] + "/activate")
            assert response.status_code == 409, response.text
        assert package["starts"] == [True, False] and not state(package)[1]
    finally:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


@pytest.mark.parametrize("package", ["active", "inactive", "empty"], indirect=True)
async def test_policy_completion_uses_the_fresh_target_for_grant_and_activation(
    package: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _watch_files(monkeypatch)
    with SessionLocal() as session:
        result = await activate_prepared_workflow_package(
            session,
            package["preparation"],
            package["inputs"]["selected"],
            context=package["context"],
            processes=package["processes"],
            session_factory=SessionLocal,
        )
    assert result.state == "active" and state(package)[:2] == (True, True)
    assert state(package)[2]["trust_authority"] == POLICY_ID
    assert seen and len(package["probes"]) == 3 and package["starts"] == [True]


async def test_policy_completion_refuses_a_changed_preparation_during_verification(
    package: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = ComfyRegistryVerificationTarget.verify

    changed = False

    async def verify(self: ComfyRegistryVerificationTarget, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        if not changed:
            changed = True
            with SessionLocal() as session:
                install = session.get(ComfyRegistryInstall, package["preparation"].install_id)
                assert install is not None
                source = self.custom_node_root / install.installed_path
                destination = source.with_name(source.name + "-replacement")
                assert source.resolve().parent == self.custom_node_root.resolve()
                assert destination.resolve().parent == self.custom_node_root.resolve()
                source.rename(destination)
                install.installed_path = destination.name
                session.commit()
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ComfyRegistryVerificationTarget, "verify", verify)
    with SessionLocal() as session, pytest.raises(WorkflowPackagePreparationError) as error:
        await activate_prepared_workflow_package(
            session,
            package["preparation"],
            package["inputs"]["selected"],
            context=package["context"],
            processes=package["processes"],
            session_factory=SessionLocal,
        )
    assert error.value.code == "registry_policy_identity_mismatch"
    assert state(package)[:2] == (False, False) and not package["starts"]


async def test_activation_endpoint_rechecks_configuration_after_the_file_snapshot(
    package: dict[str, Any],
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_lm import comfy_registry_archives as archives
    from local_lm import comfy_registry_verified_activation as activation

    response = await client.post(package["url"] + "/review", json={"trusted": True})
    assert response.status_code == 200
    original = archives.snapshot_staged_comfy_registry_files

    def snapshot(path: Path) -> Any:
        result = original(path)
        assert settings.comfy_executable is not None
        settings.comfy_executable = settings.comfy_executable.with_name("different-python.exe")
        return result

    monkeypatch.setattr(activation, "snapshot_staged_comfy_registry_files", snapshot)
    response = await client.post(package["url"] + "/activate")
    assert response.status_code == 409, response.text
    assert not state(package)[1] and not package["starts"]
