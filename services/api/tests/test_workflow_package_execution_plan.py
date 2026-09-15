"""A saved extension plan binds the code, target and dependencies actually prepared."""

from __future__ import annotations

import asyncio
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from httpx2 import AsyncClient
from sqlalchemy import select
from test_comfy_registry_closure_driver import _project, _Sources
from test_comfy_registry_downloads import archive_bytes, commit_resolution, resolution
from test_workflow_package_preparation import _Registry

from local_lm import workflow_package_preparation as preparation_module
from local_lm.comfy_registry_closure_driver import drive_comfy_registry_wheel_closure
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.comfy_registry_wheel_artifacts import current_comfy_registry_wheel_target
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement
from local_lm.db import SessionLocal
from local_lm.models import ComfyRegistryInstall
from local_lm.workflow_package_preparation import PreparationContext, prepare_workflow_package

pytestmark = pytest.mark.asyncio


def _inputs(*, commit: bool = False, dependencies: bool = False) -> dict[str, Any]:
    selected = commit_resolution() if commit else resolution()
    version = selected.declared_version
    assert version is not None
    code = b'raise RuntimeError("This inspected package must stay inert")\n'
    prefix = "package-root/" if commit else ""
    entries = {prefix + "__init__.py": code}
    if dependencies:
        if commit:
            entries[prefix + "requirements.txt"] = b"alpha==1.0\n"
        else:
            selected = replace(selected, pip_dependencies=("alpha==1.0",))
    content = archive_bytes(entries)
    alpha, alpha_metadata = _project("alpha", [("1.0", ["beta==2.0"])])
    beta, beta_metadata = _project("beta", [("2.0", [])])
    sources = _Sources({"alpha": alpha, "beta": beta}, {**alpha_metadata, **beta_metadata})
    environment, tags = current_comfy_registry_wheel_target()

    async def probe(_python: Path) -> tuple[dict[str, str], tuple[str, ...]]:
        return environment, tags

    state = {"content": content}
    archive = ComfyRegistryArchiveDownloader(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=state["content"])
        )
    )
    return {
        "selected": selected,
        "requirement": WorkflowPackageRequirement(
            selected.package_id, (version,), selected.node_types, False
        ),
        "python_executable": Path(sys.executable),
        "interpreter_probe": probe,
        "registry_client": _Registry(selected),
        "project_client": SimpleNamespace(fetch=sources.fetch_projects),
        "metadata_client": SimpleNamespace(fetch=sources.fetch_metadata),
        "archive_downloader": archive,
        "state": state,
        "environment": environment,
        "sources": sources,
    }


async def _plan(inputs: dict[str, Any]) -> Any:
    from local_lm.workflow_package_execution_plan import plan_workflow_package_execution

    return await plan_workflow_package_execution(
        **{
            key: inputs[key]
            for key in (
                "requirement",
                "python_executable",
                "interpreter_probe",
                "registry_client",
                "project_client",
                "metadata_client",
                "archive_downloader",
            )
        }
    )


@pytest.mark.parametrize("commit", [False, True])
async def test_inert_preflight_counts_direct_and_transitive_wheels_and_survives_json(
    client: AsyncClient, commit: bool
) -> None:
    from local_lm.workflow_package_execution_plan import WorkflowPackageExecutionPlan

    inputs = _inputs(commit=commit, dependencies=True)
    try:
        plan = await _plan(inputs)
        assert plan.archive.archive_sha256 == hashlib.sha256(inputs["state"]["content"]).hexdigest()
        assert plan.archive_bytes == len(inputs["state"]["content"])
        assert plan.wheel_bytes == 200 and plan.download_bytes == plan.archive_bytes + 200
        assert [item.name for item in plan.closure.manifest.artifacts] == ["alpha", "beta"]
        loaded = WorkflowPackageExecutionPlan.model_validate_json(plan.model_dump_json())
        loaded.verify()
        assert loaded == plan
        with SessionLocal() as session:
            assert list(session.scalars(select(ComfyRegistryInstall))) == []
    finally:
        await inputs["archive_downloader"].close()


@pytest.mark.parametrize("field", ["archive_bytes", "archive", "resolution", "closure"])
async def test_mutated_plan_cannot_authorize_preparation(field: str) -> None:
    inputs = _inputs()
    try:
        plan = await _plan(inputs)
        if field == "archive_bytes":
            plan.archive_bytes += 1
        elif field == "archive":
            plan.archive = replace(plan.archive, manifest_sha256="b" * 64)
        elif field == "resolution":
            plan.resolution = replace(plan.resolution, declared_version="2.0.0")
        else:
            plan.closure = replace(plan.closure, closure_sha256="b" * 64)
        with pytest.raises(ValueError) as refused:
            plan.verify()
        assert getattr(refused.value, "code", None) == "workflow-package-execution-plan-changed"
    finally:
        await inputs["archive_downloader"].close()


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize(
    "change",
    ["none", "requested", "resolution", "target", "runtime", "archive", "closure", "renewal"],
)
async def test_preparation_uses_exact_plan_or_refuses_without_leaving_code_installed(
    client: AsyncClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: bool,
    change: str,
) -> None:
    inputs = _inputs(commit=commit)
    wheels = ComfyRegistryWheelDownloader(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(500, text="An empty closure must not download wheels")
        )
    )
    try:
        plan = await _plan(inputs)
        context = PreparationContext(
            Path(sys.executable), tmp_path / "custom_nodes", tmp_path / "registry"
        )
        context.custom_node_root.mkdir()
        context.state_root.mkdir()
        if change == "resolution":
            inputs["registry_client"] = _Registry(
                replace(inputs["selected"], warnings=("changed",))
            )
        elif change == "target":
            inputs["environment"]["platform_release"] += "-changed"
        elif change == "runtime":
            original_probe = inputs["interpreter_probe"]

            async def changed_runtime(
                python: Path,
            ) -> tuple[
                dict[str, str], tuple[str, ...], tuple[ComfyRegistryRuntimeDistribution, ...]
            ]:
                environment, tags = await original_probe(python)
                return environment, tags, (ComfyRegistryRuntimeDistribution("alpha", "1.0"),)

            inputs["interpreter_probe"] = changed_runtime
        elif change == "archive":
            prefix = "package-root/" if commit else ""
            inputs["state"]["content"] = archive_bytes(
                {prefix + "__init__.py": b"CHANGED = True\n"}
            )
        elif change == "closure":
            original = drive_comfy_registry_wheel_closure

            async def changed(*args: Any, **kwargs: Any) -> Any:
                result = await original(*args, **kwargs)
                return replace(result, closure=replace(result.closure, closure_sha256="c" * 64))

            monkeypatch.setattr(preparation_module, "drive_comfy_registry_wheel_closure", changed)

        async def prepare() -> Any:
            return await prepare_workflow_package(
                SessionLocal,
                package_id="changed-package"
                if change == "requested"
                else inputs["selected"].package_id,
                version=inputs["selected"].declared_version,
                node_types=inputs["selected"].node_types,
                context=context,
                media_worker_stopped=True,
                expected_plan=plan,
                renew_install_id="unrelated-install" if change == "renewal" else None,
                wheel_downloader=wheels,
                **{
                    key: inputs[key]
                    for key in (
                        "interpreter_probe",
                        "registry_client",
                        "project_client",
                        "metadata_client",
                        "archive_downloader",
                    )
                },
            )

        if change == "none":
            prepared = await prepare()
            with SessionLocal() as session:
                install = session.get(ComfyRegistryInstall, prepared.install_id)
                assert install is not None and not install.active and not install.trusted
                assert install.archive_sha256 == plan.archive.archive_sha256
                assert install.wheel_closure_sha256 == plan.closure.closure_sha256
        else:
            with pytest.raises(ValueError):
                await prepare()
            with SessionLocal() as session:
                assert list(session.scalars(select(ComfyRegistryInstall))) == []
            assert list(context.custom_node_root.iterdir()) == []
            if change == "requested":
                assert len(inputs["registry_client"].requested) == 1
    finally:
        await inputs["archive_downloader"].close()
        await wheels.close()


@pytest.mark.parametrize("change", ["package", "version", "nodes"])
async def test_preflight_rejects_a_resolution_for_another_requirement(change: str) -> None:
    inputs = _inputs()
    updates: dict[str, Any] = (
        {"package_id": "other-package"}
        if change == "package"
        else {"declared_version": "2.0.0"}
        if change == "version"
        else {"node_types": ("OtherNode",)}
    )
    inputs["registry_client"] = _Registry(replace(inputs["selected"], **updates))
    try:
        with pytest.raises(ValueError) as refused:
            await _plan(inputs)
        assert getattr(refused.value, "code", None) == "workflow-package-resolution-unavailable"
    finally:
        await inputs["archive_downloader"].close()


@pytest.mark.parametrize("outcome", ["success", "refusal", "cancelled"])
async def test_preflight_removes_its_inert_temporary_archive(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from local_lm import workflow_package_execution_plan as planning

    inputs = _inputs()
    destinations: list[Path] = []
    download = inputs["archive_downloader"].download_and_stage

    async def tracked(*args: Any, **kwargs: Any) -> Any:
        destinations.append(args[1])
        return await download(*args, **kwargs)

    async def interrupted(*args: Any, **kwargs: Any) -> Any:
        if outcome == "cancelled":
            raise asyncio.CancelledError
        raise ValueError("Neutral closure refusal")

    monkeypatch.setattr(inputs["archive_downloader"], "download_and_stage", tracked)
    if outcome != "success":
        monkeypatch.setattr(planning, "drive_comfy_registry_wheel_closure", interrupted)
    try:
        if outcome == "success":
            await _plan(inputs)
        else:
            expected = asyncio.CancelledError if outcome == "cancelled" else ValueError
            with pytest.raises(expected):
                await _plan(inputs)
        assert len(destinations) == 1
        assert not destinations[0].exists() and not destinations[0].parent.exists()
    finally:
        await inputs["archive_downloader"].close()
