from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_comfy_registry_closure_driver import _Sources, _wheel_file
from test_comfy_registry_downloads import archive_bytes, commit_resolution, resolution
from test_comfy_registry_mixed_wheel_environments import _mixed_files
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_environments import _marker_environment, _wheel_files
from test_workflow_package_preparation import _Registry

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_closure_driver import ComfyRegistryWheelClosureDriverError
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure
from local_lm.comfy_registry_runtime import ComfyRegistryRuntimeDistribution
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.comfy_workflow_packages import WorkflowPackageRequirement
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview
from local_lm.workflow_package_execution_plan import (
    WorkflowPackageExecutionPlan,
    WorkflowPackageExecutionPlanError,
    plan_workflow_package_execution,
)
from local_lm.workflow_package_preparation import (
    PreparationContext,
    WorkflowPackagePreparationError,
    prepare_workflow_package,
)
from local_lm.workflow_trust import canonical_graph


def _inputs(
    source_context: tuple[Session, ArtifactStore],
    root: Path,
    *,
    commit: bool,
    inactive: bool = False,
) -> dict[str, Any]:
    mixed, paths = _mixed_files(source_context, root)
    remote = mixed.manifest.remote.artifacts[0]
    content = paths[remote.filename].read_bytes()
    metadata = _wheel_files(content)["alpha-1.0.dist-info/METADATA"]
    artifact = _wheel_file("alpha", "1.0", metadata)
    artifact.update(hashes={"sha256": hashlib.sha256(content).hexdigest()}, size=len(content))
    sources = _Sources(
        {"alpha": {"meta": {"api-version": "1.4"}, "name": "alpha", "files": [artifact]}},
        {remote.filename: metadata},
    )
    declaration = DECLARATION + ' ; python_version < "0"' if inactive else DECLARATION
    selected = commit_resolution() if commit else resolution(pip_dependencies=(declaration,))
    assert selected.declared_version is not None
    prefix = "package-root/" if commit else ""
    entries = {
        prefix + "__init__.py": b'raise RuntimeError("Package inspection must stay inert")\n'
    }
    if commit:
        entries[prefix + "requirements.txt"] = (declaration + "\n").encode()
    payload = archive_bytes(entries)
    archive = ComfyRegistryArchiveDownloader(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=payload))
    )
    requests: list[str] = []

    def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200, content=metadata if request.url.path.endswith(".metadata") else content
        )

    downloader = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(fetch))
    environment = _marker_environment()
    runtime = [ComfyRegistryRuntimeDistribution("helper", "2.0")]
    probes: list[Path] = []

    async def probe(
        python: Path,
    ) -> tuple[dict[str, str], tuple[str, ...], tuple[ComfyRegistryRuntimeDistribution, ...]]:
        probes.append(python)
        return dict(environment), ("py3-none-any",), tuple(runtime)

    engine = source_context[0].get_bind()

    def factory() -> Session:
        return Session(engine)

    context = PreparationContext(
        Path(sys.executable), root / "nodes", root / "state", source_context[1]
    )
    context.custom_node_root.mkdir()
    context.state_root.mkdir()
    return dict(
        plan_arguments=dict(
            requirement=WorkflowPackageRequirement(
                selected.package_id, (selected.declared_version,), selected.node_types, False
            ),
            python_executable=context.python_executable,
            interpreter_probe=probe,
            registry_client=_Registry(selected),
            project_client=SimpleNamespace(fetch=sources.fetch_projects),
            metadata_client=SimpleNamespace(fetch=sources.fetch_metadata),
            archive_downloader=archive,
            source_session_factory=factory,
            source_store=source_context[1],
        ),
        selected=selected,
        context=context,
        factory=factory,
        downloader=downloader,
        requests=requests,
        environment=environment,
        runtime=runtime,
        probes=probes,
        remote_bytes=len(content),
        local_bytes=mixed.manifest.reviewed_local[0].size_bytes,
    )


async def _close(inputs: dict[str, Any]) -> None:
    await inputs["plan_arguments"]["archive_downloader"].close()
    await inputs["downloader"].close()


async def _prepare(inputs: dict[str, Any], plan: WorkflowPackageExecutionPlan) -> Any:
    arguments = inputs["plan_arguments"]
    return await prepare_workflow_package(
        inputs["factory"],
        package_id=inputs["selected"].package_id,
        version=inputs["selected"].declared_version,
        node_types=inputs["selected"].node_types,
        context=inputs["context"],
        media_worker_stopped=True,
        expected_plan=plan,
        interpreter_probe=arguments["interpreter_probe"],
        registry_client=arguments["registry_client"],
        project_client=arguments["project_client"],
        metadata_client=arguments["metadata_client"],
        archive_downloader=arguments["archive_downloader"],
        wheel_downloader=inputs["downloader"],
    )


@pytest.mark.parametrize("commit", [False, True])
@pytest.mark.parametrize("change", ["none", "revoked", "target", "runtime"])
async def test_reviewed_source_preview_binds_real_preparation_and_network_bytes(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    commit: bool,
    change: str,
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=commit)
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        assert plan.version == 2 and isinstance(plan.closure, ComfyRegistryMixedWheelClosure)
        assert plan.wheel_bytes == inputs["remote_bytes"] + inputs["local_bytes"]
        assert plan.download_bytes == plan.archive_bytes + inputs["remote_bytes"]
        loaded = WorkflowPackageExecutionPlan.model_validate_json(plan.model_dump_json())
        loaded.verify()
        assert loaded == plan and inputs["requests"] == []
        if change == "revoked":
            with inputs["factory"]() as session:
                session.execute(delete(ComfyRegistrySourceArtifactReview))
                session.commit()
        elif change == "target":
            inputs["environment"]["platform_release"] += "-changed"
        elif change == "runtime":
            inputs["runtime"].append(ComfyRegistryRuntimeDistribution("other", "1"))
        if change == "none":
            prepared = await _prepare(inputs, loaded)
            assert prepared.wheel_closure_sha256 == plan.closure.closure_sha256
            assert len(inputs["requests"]) == 2
            assert all("alpha-1.0" in url for url in inputs["requests"])
            with inputs["factory"]() as session:
                install = session.get(ComfyRegistryInstall, prepared.install_id)
                assert install is not None and not install.trusted and not install.active
                assert "reviewed_wheel_closure" in install.review_json
            assert inputs["probes"] == [inputs["context"].python_executable] * 2
        else:
            with pytest.raises(ValueError):
                await _prepare(inputs, loaded)
            assert not list(inputs["context"].custom_node_root.iterdir())
            assert not inputs["requests"]
            with inputs["factory"]() as session:
                assert list(session.scalars(select(ComfyRegistryInstall))) == []
    finally:
        await _close(inputs)


@pytest.mark.parametrize("commit", [False, True])
async def test_inactive_source_declarations_remain_bound_without_reading_or_downloading_them(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path, commit: bool
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=commit, inactive=True)
    with inputs["factory"]() as session:
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        assert plan.version == 2 and isinstance(plan.closure, ComfyRegistryMixedWheelClosure)
        assert plan.closure.manifest.reviewed_local == ()
        assert plan.wheel_bytes == 0 and plan.download_bytes == plan.archive_bytes
        prepared = await _prepare(inputs, plan)
        assert not inputs["requests"]
        with inputs["factory"]() as session:
            install = session.get(ComfyRegistryInstall, prepared.install_id)
            assert install is not None and "reviewed_wheel_closure" in install.review_json
    finally:
        await _close(inputs)


@pytest.mark.parametrize("change", ["version", "closure", "target"])
async def test_rehashing_a_mixed_execution_plan_does_not_validate_an_incompatible_contract(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path, change: str
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=False)
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        if change == "version":
            plan.version = 1
        elif change == "closure":
            plan.closure = replace(plan.closure, closure_sha256="b" * 64)
        else:
            assert isinstance(plan.closure, ComfyRegistryMixedWheelClosure)
            remote = replace(plan.closure.manifest.remote, target_sha256="b" * 64)
            plan.closure = replace(
                plan.closure, manifest=replace(plan.closure.manifest, remote=remote)
            )
        plan.plan_sha256 = hashlib.sha256(
            canonical_graph(plan.model_dump(mode="json", exclude={"plan_sha256"})).encode()
        ).hexdigest()
        with pytest.raises(ValueError):
            plan.verify()
        if change == "version":
            with pytest.raises(WorkflowPackageExecutionPlanError):
                plan.verify_closure(plan.closure)
    finally:
        await _close(inputs)


async def test_source_review_context_preserves_remote_plan_serialization(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=False)
    selected = resolution(pip_dependencies=("alpha==1.0",))
    arguments = inputs["plan_arguments"]
    arguments["registry_client"] = _Registry(selected)
    try:
        with_context = await plan_workflow_package_execution(**arguments)
        arguments.pop("source_session_factory")
        arguments.pop("source_store")
        without_context = await plan_workflow_package_execution(**arguments)
        assert with_context.version == without_context.version == 1
        assert with_context.model_dump_json() == without_context.model_dump_json()
        assert with_context.download_bytes == with_context.archive_bytes + inputs["remote_bytes"]
        assert with_context.wheel_bytes == inputs["remote_bytes"]
        with_context.verify()
        without_context.verify()
    finally:
        await _close(inputs)


@pytest.mark.parametrize("missing", ["both", "factory", "store"])
async def test_partial_source_review_context_cannot_authorize_a_plan(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path, missing: str
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=False)
    arguments = inputs["plan_arguments"]
    if missing in {"both", "factory"}:
        arguments.pop("source_session_factory")
    if missing in {"both", "store"}:
        arguments.pop("source_store")
    try:
        with pytest.raises(ComfyRegistryWheelClosureDriverError) as error:
            await plan_workflow_package_execution(**arguments)
        assert error.value.code == "unresolved_source_dependency"
        assert not inputs["requests"]
        assert not list(inputs["context"].custom_node_root.iterdir())
        with inputs["factory"]() as session:
            assert list(session.scalars(select(ComfyRegistryInstall))) == []
    finally:
        await _close(inputs)


async def test_preparation_cannot_use_preview_review_context_as_current_authority(
    source_review_context: tuple[Session, ArtifactStore], tmp_path: Path
) -> None:
    inputs = _inputs(source_review_context, tmp_path, commit=False)
    try:
        plan = await plan_workflow_package_execution(**inputs["plan_arguments"])
        inputs["context"] = replace(inputs["context"], source_store=None)
        with pytest.raises(WorkflowPackagePreparationError) as error:
            await _prepare(inputs, plan)
        assert error.value.code == "unresolved_source_dependency"
        assert not inputs["requests"]
        assert not list(inputs["context"].custom_node_root.iterdir())
        with inputs["factory"]() as session:
            assert list(session.scalars(select(ComfyRegistryInstall))) == []
    finally:
        await _close(inputs)
