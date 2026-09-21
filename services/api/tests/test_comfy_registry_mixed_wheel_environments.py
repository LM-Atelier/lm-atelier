from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import threading
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION, _artifact
from test_comfy_registry_wheel_environments import (
    _closure,
    _marker_environment,
    _record_bytes,
    _wheel_bytes,
    _wheel_content,
    _wheel_files,
)

from local_lm import comfy_registry_wheel_environments as module
from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_source_artifacts import record_local_source_artifact_review
from local_lm.comfy_registry_wheel_artifacts import build_comfy_registry_wheel_artifact_manifest
from local_lm.comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentError,
    assemble_comfy_registry_wheel_environment,
    verify_comfy_registry_wheel_environment,
)
from local_lm.comfy_registry_wheel_inputs_v1 import (
    build_comfy_registry_wheel_input_manifest,
    reviewed_wheel_input,
)
from local_lm.models import ComfyRegistrySourceArtifactReview

if TYPE_CHECKING:
    from local_lm.comfy_registry_mixed_wheel_closure import ComfyRegistryMixedWheelClosure
    from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext


def _mixed_files(
    context: tuple[Session, ArtifactStore],
    root: Path,
    *,
    dependency: str = "alpha>=1",
    dishonest_record: bool = False,
) -> tuple[ComfyRegistryMixedWheelClosure, dict[str, Path]]:
    from local_lm.comfy_registry_mixed_wheel_closure import plan_comfy_registry_mixed_wheel_closure

    session, store = context
    local_metadata = (
        "Metadata-Version: 2.4\nName: example-pkg\nVersion: 1.2.3\n"
        f"Requires-Dist: {dependency}\nRequires-Dist: helper>=2\n\n"
    ).encode()
    local_contents = {
        "example_pkg/__init__.py": b"VALUE = 1\n",
        "example_pkg-1.2.3.dist-info/METADATA": local_metadata,
        "example_pkg-1.2.3.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    record = "example_pkg-1.2.3.dist-info/RECORD"
    recorded = dict(local_contents)
    if dishonest_record:
        recorded.pop("example_pkg/__init__.py")
    local_contents[record] = _record_bytes(recorded, record)
    content = _wheel_bytes(local_contents)
    artifact = _artifact(session, store, payload=content)
    record_local_source_artifact_review(
        session, store, declaration=DECLARATION, artifact_id=artifact.id
    )
    session.commit()
    local = reviewed_wheel_input(
        session,
        store,
        declaration=DECLARATION,
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
    )
    remote_content = _wheel_content()
    remote = _closure(remote_content).manifest
    remote_metadata = _wheel_files(remote_content)["alpha-1.0.dist-info/METADATA"]
    remote = build_comfy_registry_wheel_artifact_manifest(
        remote.declaration_sha256,
        remote.target_sha256,
        [replace(remote.artifacts[0], metadata_sha256=hashlib.sha256(remote_metadata).hexdigest())],
    )
    inputs = build_comfy_registry_wheel_input_manifest("f" * 64, remote, [local])
    result = plan_comfy_registry_mixed_wheel_closure(
        inputs,
        {local.filename: local_metadata, remote.artifacts[0].filename: remote_metadata},
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions={"helper": "2.0"},
    )
    paths = {
        local.filename: root / local.filename,
        remote.artifacts[0].filename: root / remote.artifacts[0].filename,
    }
    paths[local.filename].write_bytes(content)
    paths[remote.artifacts[0].filename].write_bytes(remote_content)
    return result, paths


def _destination(root: Path, closure: ComfyRegistryMixedWheelClosure) -> Path:
    return root / f"registry-wheels-v3-{closure.closure_sha256}"


def _input_context(context: tuple[Session, ArtifactStore]) -> ComfyRegistryReviewedInputContext:
    from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext

    session, store = context
    engine = session.get_bind()
    return ComfyRegistryReviewedInputContext(
        lambda: Session(engine), store, _marker_environment(), ("py3-none-any",)
    )


async def _assemble(
    closure: ComfyRegistryMixedWheelClosure,
    paths: dict[str, Path],
    destination: Path,
    context: tuple[Session, ArtifactStore],
) -> module.ComfyRegistryWheelEnvironmentReport:
    return await assemble_comfy_registry_wheel_environment(
        closure,
        paths,
        python_executable=Path(sys.executable),
        destination=destination,
        media_worker_stopped=True,
        reviewed_inputs=_input_context(context),
    )


async def test_real_pip_assembles_both_origins_and_binds_source_review_identity(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    from local_lm.comfy_registry_mixed_wheel_closure import plan_comfy_registry_mixed_wheel_closure

    closure, paths = _mixed_files(source_review_context, tmp_path)
    destination = _destination(tmp_path, closure)
    report = await _assemble(closure, paths, destination, source_review_context)
    assert report.artifact_count == 2 and report.closure_sha256 == closure.closure_sha256
    assert {(item.name, item.version) for item in report.distributions} == {
        ("alpha", "1.0"),
        ("example-pkg", "1.2.3"),
    }
    assert report.runtime_distributions == closure.runtime_distributions
    assert (destination / "site-packages/example_pkg/__init__.py").read_bytes() == b"VALUE = 1\n"
    assert (
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=closure.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )
        == report
    )
    payload = json.loads((destination / "environment-manifest.json").read_text(encoding="utf-8"))
    assert payload["ownership_attestation"] == "wheel-source-record-v1"
    changed = build_comfy_registry_wheel_input_manifest(
        closure.manifest.declaration_sha256,
        closure.manifest.remote,
        [replace(closure.manifest.reviewed_local[0], review_sha256="e" * 64)],
    )
    documents = {}
    for name, path in paths.items():
        files = _wheel_files(path.read_bytes())
        documents[name] = next(value for key, value in files.items() if key.endswith("/METADATA"))
    other = plan_comfy_registry_mixed_wheel_closure(
        changed,
        documents,
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions=closure.runtime_distributions,
    )
    assert other.closure_sha256 != closure.closure_sha256
    with pytest.raises(ComfyRegistryWheelEnvironmentError):
        verify_comfy_registry_wheel_environment(
            destination,
            expected_closure_sha256=other.closure_sha256,
            expected_environment_sha256=report.environment_sha256,
        )


@pytest.mark.parametrize("problem", ["incomplete", "hash", "missing", "changed", "size", "record"])
async def test_mixed_inputs_are_refused_before_pip_when_incomplete_or_changed(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    closure, paths = _mixed_files(
        source_review_context,
        tmp_path,
        dependency="leaf>=1" if problem == "incomplete" else "alpha>=1",
        dishonest_record=problem == "record",
    )
    if problem == "hash":
        closure = replace(closure, closure_sha256="0" * 64)
    filename = closure.manifest.reviewed_local[0].filename
    if problem == "missing":
        paths.pop(filename)
    elif problem in {"changed", "size"}:
        content = paths[filename].read_bytes()
        paths[filename].write_bytes(
            content + b"x" if problem == "size" else content[:-1] + bytes([content[-1] ^ 1])
        )

    async def unexpected(_python: Path, _wheels: tuple[Path, ...], _target: Path) -> None:
        pytest.fail("Invalid mixed inputs reached pip")

    monkeypatch.setattr(module, "_run_pip", unexpected)
    destination = _destination(tmp_path, closure)
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as caught:
        await _assemble(closure, paths, destination, source_review_context)
    assert (
        caught.value.code
        == {
            "incomplete": "closure_incomplete",
            "hash": "invalid_closure",
            "missing": "missing_wheel_file",
            "changed": "wheel_hash_mismatch",
            "size": "wheel_size_mismatch",
            "record": "invalid_wheel_record",
        }[problem]
    )
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}-*"))


async def test_honest_installed_record_cannot_hide_changed_local_source_bytes(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure, paths = _mixed_files(source_review_context, tmp_path)

    async def install_changed(_python: Path, wheels: tuple[Path, ...], target: Path) -> None:
        for wheel in wheels:
            files = _wheel_files(wheel.read_bytes())
            record = next(path for path in files if path.endswith("/RECORD"))
            files.pop(record)
            if "example_pkg/__init__.py" in files:
                files["example_pkg/__init__.py"] = b"VALUE = 2\n"
            directory = record.rsplit("/", 1)[0]
            files.update(
                {
                    f"{directory}/INSTALLER": b"pip\n",
                    f"{directory}/REQUESTED": b"",
                    f"{directory}/direct_url.json": b"{}\n",
                }
            )
            files[record] = _record_bytes(files, record)
            for relative, content in files.items():
                path = target / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

    monkeypatch.setattr(module, "_run_pip", install_changed)
    destination = _destination(tmp_path, closure)
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as caught:
        await _assemble(closure, paths, destination, source_review_context)
    assert caught.value.code == "wheel_ownership_mismatch"
    assert not destination.exists()


@pytest.mark.parametrize("phase", ["stage", "cleanup", "audit"])
async def test_cancellation_waits_for_file_work_before_removing_staging(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    closure, paths = _mixed_files(source_review_context, tmp_path)
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = module._stage_wheels
    original_cleanup = module._remove_staged_wheels
    original_audit = module._audit_environment
    directories: list[Path] = []

    def hold(directory: Path) -> None:
        directories.append(directory)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5), "Staging test did not release the worker"

    def held_stage(
        wheels: Sequence[tuple[module._EnvironmentWheel, Path]], directory: Path
    ) -> tuple[tuple[Path, ...], module._WheelOwnershipPlan]:
        result = original(wheels, directory)
        hold(directory.parent)
        return result

    def held_cleanup(directory: Path) -> None:
        original_cleanup(directory)
        hold(directory.parent)

    def held_audit(
        closed: module._EnvironmentClosure,
        artifacts: Sequence[module._EnvironmentWheel],
        site_packages: Path,
        ownership: module._WheelOwnershipPlan,
    ) -> tuple[module.ComfyRegistryWheelEnvironmentReport, bytes]:
        result = original_audit(closed, artifacts, site_packages, ownership)
        hold(site_packages.parent)
        return result

    if phase == "stage":
        monkeypatch.setattr(module, "_stage_wheels", held_stage)
    elif phase == "cleanup":
        monkeypatch.setattr(module, "_remove_staged_wheels", held_cleanup)
    else:
        monkeypatch.setattr(module, "_audit_environment", held_audit)
    destination = _destination(tmp_path, closure)
    task = asyncio.create_task(_assemble(closure, paths, destination, source_review_context))
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert directories[0].is_dir()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not destination.exists()
    assert not directories[0].exists()


@pytest.mark.parametrize("when", ["before", "during"])
async def test_revoked_source_review_refuses_environment_publication(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    when: str,
) -> None:
    closure, paths = _mixed_files(source_review_context, tmp_path)
    session, _store = source_review_context
    engine = session.get_bind()
    original = module._run_pip
    installed: list[bool] = []

    def revoke() -> None:
        with Session(engine) as fresh:
            fresh.execute(delete(ComfyRegistrySourceArtifactReview))
            fresh.commit()

    async def install_then_revoke(python: Path, wheels: tuple[Path, ...], target: Path) -> None:
        await original(python, wheels, target)
        installed.append(True)
        revoke()

    monkeypatch.setattr(module, "_run_pip", install_then_revoke)
    if when == "before":
        revoke()
    destination = _destination(tmp_path, closure)
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as caught:
        await _assemble(closure, paths, destination, source_review_context)
    assert caught.value.code == "source_artifact_review_missing"
    assert installed == ([] if when == "before" else [True])
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}-*"))


async def test_local_assembly_requires_a_review_context(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    closure, paths = _mixed_files(source_review_context, tmp_path)
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as caught:
        await assemble_comfy_registry_wheel_environment(
            closure,
            paths,
            python_executable=Path(sys.executable),
            destination=_destination(tmp_path, closure),
            media_worker_stopped=True,
        )
    assert caught.value.code == "source_review_context_required"


async def test_closure_staging_and_real_assembly_share_one_mixed_identity(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    import httpx

    from local_lm.comfy_registry_mixed_closure_driver import (
        drive_comfy_registry_mixed_wheel_closure,
    )
    from local_lm.comfy_registry_wheel_downloads import ComfyRegistryWheelDownloader
    from local_lm.comfy_registry_wheel_inputs_v1 import parse_wheel_input_manifest

    planned, paths = _mixed_files(source_review_context, tmp_path)
    local_name = planned.manifest.reviewed_local[0].filename
    remote_name = planned.manifest.remote.artifacts[0].filename
    documents = {
        name: next(
            value
            for key, value in _wheel_files(path.read_bytes()).items()
            if key.endswith("/METADATA")
        )
        for name, path in paths.items()
    }

    async def no_projects(names: Sequence[str]) -> dict[str, object]:
        pytest.fail("The complete root set requested another project")

    async def metadata(manifest: object) -> dict[str, bytes]:
        return {remote_name: documents[remote_name]}

    closure = await drive_comfy_registry_mixed_wheel_closure(
        planned.manifest,
        {local_name: documents[local_name]},
        project_fetcher=no_projects,
        metadata_fetcher=metadata,
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions=planned.runtime_distributions,
    )
    assert closure == planned
    requests: list[str] = []

    def fetch(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(
            200,
            content=documents[remote_name]
            if request.url.path.endswith(".metadata")
            else paths[remote_name].read_bytes(),
        )

    context = _input_context(source_review_context)
    downloader = ComfyRegistryWheelDownloader(transport=httpx.MockTransport(fetch))
    stage = tmp_path / "stage"
    try:
        staged = await downloader.download_and_stage_inputs(
            closure.manifest,
            stage,
            session_factory=context.session_factory,
            store=context.store,
            marker_environment=context.marker_environment,
            supported_tags=context.supported_tags,
        )
    finally:
        await downloader.close()
    assert staged.artifact_manifest_sha256 == closure.manifest.manifest_sha256
    stage_payload = json.loads((stage / "stage-manifest.json").read_text(encoding="utf-8"))
    assert parse_wheel_input_manifest(stage_payload["input_manifest"]) == closure.manifest
    assert len(requests) == 2 and all(remote_name in url for url in requests)
    report = await _assemble(
        closure,
        {name: stage / name for name in paths},
        _destination(tmp_path, closure),
        source_review_context,
    )
    assert report.closure_sha256 == closure.closure_sha256 and report.artifact_count == 2


def test_review_context_uses_fresh_sessions_and_copies_its_target(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    from local_lm.comfy_registry_reviewed_inputs import ComfyRegistryReviewedInputContext
    from local_lm.comfy_registry_wheel_inputs_v1 import ComfyRegistryWheelInputError

    closure, _paths = _mixed_files(source_review_context, tmp_path)
    original, store = source_review_context
    engine = original.get_bind()
    sessions: list[Session] = []

    def fresh() -> Session:
        created = Session(engine)
        sessions.append(created)
        return created

    environment = _marker_environment()
    context = ComfyRegistryReviewedInputContext(fresh, store, environment, ("py3-none-any",))
    environment["python_version"] = "2.7"
    context.validate(closure.manifest)
    context.validate(closure.manifest)
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(session is not original for session in sessions)
    with pytest.raises(ComfyRegistryWheelInputError) as target:
        replace(context, marker_environment=environment).validate(closure.manifest)
    assert target.value.code == "wheel_input_target_mismatch" and len(sessions) == 2
    changed = build_comfy_registry_wheel_input_manifest(
        closure.manifest.declaration_sha256,
        closure.manifest.remote,
        [replace(closure.manifest.reviewed_local[0], review_sha256="e" * 64)],
    )
    with pytest.raises(ComfyRegistryWheelInputError) as identity:
        context.validate(changed)
    assert identity.value.code == "reviewed_wheel_input_changed" and len(sessions) == 3
