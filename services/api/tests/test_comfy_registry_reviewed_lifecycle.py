from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from test_comfy_registry_lifecycle import _ArchiveDownloader, _resolution
from test_comfy_registry_mixed_wheel_environments import _input_context, _mixed_files
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_environments import _marker_environment, _wheel_files

from local_lm import comfy_registry_lifecycle as lifecycle
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry import ComfyNodeResolution
from local_lm.comfy_registry_installs import ComfyRegistryInstallError
from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies
from local_lm.comfy_registry_mixed_wheel_closure import (
    ComfyRegistryMixedWheelClosure,
    plan_comfy_registry_mixed_wheel_closure,
)
from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
from local_lm.comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentReport,
    assemble_comfy_registry_wheel_environment,
    verify_comfy_registry_wheel_environment,
)
from local_lm.comfy_registry_wheel_inputs_v1 import build_comfy_registry_wheel_input_manifest
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview


@dataclass(frozen=True)
class Inputs:
    closure: ComfyRegistryMixedWheelClosure
    resolution: ComfyNodeResolution
    documents: dict[str, bytes]
    paths: dict[str, Path]
    nodes: Path
    state: Path


def _inputs(context: tuple[Session, ArtifactStore], root: Path) -> Inputs:
    original, paths = _mixed_files(context, root)
    resolution = _resolution(pip_dependencies=(DECLARATION, "alpha==1.0"))
    plan = plan_comfy_registry_mixed_dependencies(resolution.pip_dependencies)
    manifest = build_comfy_registry_wheel_input_manifest(
        plan.declaration_sha256, original.manifest.remote, original.manifest.reviewed_local
    )
    documents = {
        name: next(
            value
            for member, value in _wheel_files(path.read_bytes()).items()
            if member.endswith("/METADATA")
        )
        for name, path in paths.items()
    }
    closure = plan_comfy_registry_mixed_wheel_closure(
        manifest,
        documents,
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions={"helper": "2.0"},
    )
    nodes, state = root / "nodes", root / "state"
    nodes.mkdir()
    state.mkdir()
    return Inputs(closure, resolution, documents, paths, nodes, state)


def _downloader(inputs: Inputs, requests: list[str]) -> ComfyRegistryWheelDownloader:
    remote = inputs.closure.manifest.remote.artifacts[0]

    def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            content=inputs.documents[remote.filename]
            if request.url.path.endswith(".metadata")
            else inputs.paths[remote.filename].read_bytes(),
        )

    return ComfyRegistryWheelDownloader(transport=httpx.MockTransport(fetch))


async def _prepare(
    context: tuple[Session, ArtifactStore],
    inputs: Inputs,
    downloader: ComfyRegistryWheelDownloader,
    **changes: Any,
) -> lifecycle.ComfyRegistryPreparation:
    arguments: dict[str, Any] = dict(
        resolution=inputs.resolution,
        closure=inputs.closure,
        archive_downloader=_ArchiveDownloader(),
        wheel_downloader=downloader,
        python_executable=Path(sys.executable),
        custom_node_root=inputs.nodes,
        state_root=inputs.state,
        media_worker_stopped=True,
        reviewed_inputs=_input_context(context),
    )
    arguments.update(changes)
    return await lifecycle.prepare_comfy_registry_install(context[0], **arguments)


def _revoke(context: tuple[Session, ArtifactStore]) -> None:
    with Session(context[0].get_bind()) as session:
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()


async def test_preparation_reuse_and_renewal_preserve_exact_reviewed_source_evidence(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    requests: list[str] = []
    downloader = _downloader(inputs, requests)
    try:
        first = await _prepare(source_review_context, inputs, downloader)
        assert not first.reused_wheel_environment and len(requests) == 2
        session, _store = source_review_context
        install = session.get(ComfyRegistryInstall, first.install_id)
        assert install is not None and not install.trusted and not install.active
        assert install.review_json["reviewed_wheel_closure"]
        peer = replace(inputs.resolution, package_id="second-node", registry_record_id="second")
        second = await _prepare(source_review_context, inputs, downloader, resolution=peer)
        assert second.reused_wheel_environment and len(requests) == 2
        assert first.wheel_environment_path == second.wheel_environment_path
        install.trusted = True
        install.review_json = {**install.review_json, "preserved": "value"}
        session.commit()
        renewed_closure = plan_comfy_registry_mixed_wheel_closure(
            inputs.closure.manifest,
            inputs.documents,
            marker_environment=_marker_environment(),
            supported_tags=("py3-none-any",),
            runtime_distributions={"helper": "2.0", "other": "1"},
        )
        renewed = await lifecycle.renew_comfy_registry_install_environment(
            session,
            install_id=first.install_id,
            resolution=inputs.resolution,
            closure=renewed_closure,
            wheel_downloader=downloader,
            python_executable=Path(sys.executable),
            custom_node_root=inputs.nodes,
            state_root=inputs.state,
            media_worker_stopped=True,
            reviewed_inputs=_input_context(source_review_context),
        )
        assert renewed.wheel_closure_sha256 == renewed_closure.closure_sha256
        assert renewed.wheel_closure_sha256 != first.wheel_closure_sha256
        assert (
            install.trusted and not install.active and install.review_json["preserved"] == "value"
        )
        environment_root = inputs.state / "registry-wheel-environments"
        assert (environment_root / first.wheel_environment_path).is_dir()
        assert (environment_root / renewed.wheel_environment_path).is_dir()
        assert len(requests) == 4
    finally:
        await downloader.close()


@pytest.mark.parametrize("problem", ["missing", "revoked", "target"])
async def test_preparation_checks_current_reviews_before_staging_any_package(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    problem: str,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    archive = _ArchiveDownloader()
    context = _input_context(source_review_context)
    if problem == "revoked":
        _revoke(source_review_context)
    elif problem == "target":
        context = replace(context, supported_tags=("py2-none-any",))
    requests: list[str] = []
    downloader = _downloader(inputs, requests)
    try:
        with pytest.raises(lifecycle.ComfyRegistryLifecycleError):
            await _prepare(
                source_review_context,
                inputs,
                downloader,
                archive_downloader=archive,
                reviewed_inputs=None if problem == "missing" else context,
            )
        assert archive.calls == 0 and requests == []
        assert not list(inputs.nodes.iterdir())
    finally:
        await downloader.close()


async def test_reuse_rechecks_revocation_after_verifying_the_existing_environment(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    requests: list[str] = []
    downloader = _downloader(inputs, requests)
    try:
        first = await _prepare(source_review_context, inputs, downloader)
        original = verify_comfy_registry_wheel_environment

        def verify(*args: Any, **kwargs: Any) -> ComfyRegistryWheelEnvironmentReport:
            result = original(*args, **kwargs)
            _revoke(source_review_context)
            return result

        monkeypatch.setattr(lifecycle, "verify_comfy_registry_wheel_environment", verify)
        peer = replace(inputs.resolution, package_id="second-node", registry_record_id="second")
        with pytest.raises(lifecycle.ComfyRegistryLifecycleError) as error:
            await _prepare(source_review_context, inputs, downloader, resolution=peer)
        assert error.value.code == "source_review_verification_failed"
        session, _store = source_review_context
        assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 1
        assert len(list(inputs.nodes.iterdir())) == 1
        assert (
            inputs.state / "registry-wheel-environments" / first.wheel_environment_path
        ).is_dir()
        assert len(requests) == 2
    finally:
        await downloader.close()


async def test_revocation_after_assembly_rolls_back_preparation_and_owned_files(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    requests: list[str] = []
    downloader = _downloader(inputs, requests)

    async def assemble(*args: Any, **kwargs: Any) -> ComfyRegistryWheelEnvironmentReport:
        result = await assemble_comfy_registry_wheel_environment(*args, **kwargs)
        _revoke(source_review_context)
        return result

    try:
        with pytest.raises(ComfyRegistryInstallError):
            await _prepare(
                source_review_context, inputs, downloader, environment_assembler=assemble
            )
        session, _store = source_review_context
        assert session.scalar(select(func.count()).select_from(ComfyRegistryInstall)) == 0
        assert not list(inputs.nodes.iterdir())
        assert not list((inputs.state / "registry-wheel-environments").glob("registry-wheels-*"))
    finally:
        await downloader.close()


async def test_cancellation_drains_current_review_verification_before_exiting(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs(source_review_context, tmp_path)
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    original = ComfyRegistryReviewedInputContext.validate

    def verify(self: ComfyRegistryReviewedInputContext, manifest: Any) -> None:
        started.set()
        assert release.wait(10)
        try:
            original(self, manifest)
        finally:
            finished.set()

    monkeypatch.setattr(ComfyRegistryReviewedInputContext, "validate", verify)
    downloader = _downloader(inputs, [])
    task = asyncio.create_task(_prepare(source_review_context, inputs, downloader))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and not finished.is_set()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await downloader.close()
    assert finished.is_set() and not list(inputs.nodes.iterdir())
