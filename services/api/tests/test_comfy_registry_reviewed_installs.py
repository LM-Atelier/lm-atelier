from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session
from test_comfy_registry_installs import _archive, _resolution
from test_comfy_registry_mixed_wheel_environments import _assemble, _input_context, _mixed_files
from test_comfy_registry_reviewed_wheel_staging import (
    source_review_context as source_review_context,
)
from test_comfy_registry_source_artifacts import DECLARATION
from test_comfy_registry_wheel_environments import _marker_environment, _wheel_files

from local_lm.artifacts import ArtifactStore
from local_lm.comfy_registry_installs import (
    ComfyRegistryInstallError,
    bind_comfy_registry_wheel_environment,
    persist_comfy_registry_install,
    scoped_comfy_registry_launch_contract,
    trusted_comfy_registry_launch_contract,
)
from local_lm.comfy_registry_mixed_dependencies_v1 import plan_comfy_registry_mixed_dependencies
from local_lm.comfy_registry_mixed_wheel_closure import (
    ComfyRegistryMixedWheelClosure,
    plan_comfy_registry_mixed_wheel_closure,
)
from local_lm.comfy_registry_wheel_artifacts import build_comfy_registry_wheel_artifact_manifest
from local_lm.comfy_registry_wheel_environments import (
    ComfyRegistryWheelEnvironmentError,
    ComfyRegistryWheelEnvironmentReport,
    verify_comfy_registry_wheel_environment,
)
from local_lm.comfy_registry_wheel_inputs_v1 import build_comfy_registry_wheel_input_manifest
from local_lm.models import ComfyRegistryInstall, ComfyRegistrySourceArtifactReview
from local_lm.workflow_activations import WorkflowRegistryLaunchBinding


@dataclass(frozen=True)
class Prepared:
    closure: ComfyRegistryMixedWheelClosure
    report: ComfyRegistryWheelEnvironmentReport
    install: ComfyRegistryInstall
    node_root: Path
    environment_root: Path
    destination: Path


async def _prepare(
    context: tuple[Session, ArtifactStore], root: Path, *, inactive: bool = False
) -> Prepared:
    session, _store = context
    original, paths = _mixed_files(context, root)
    declarations = (
        (DECLARATION + ' ; sys_platform == "missing-platform"',)
        if inactive
        else (DECLARATION, "alpha==1.0")
    )
    plan = plan_comfy_registry_mixed_dependencies(declarations)
    remote = original.manifest.remote
    local = original.manifest.reviewed_local
    if inactive:
        remote = build_comfy_registry_wheel_artifact_manifest(
            plan.remote.declaration_sha256, remote.target_sha256, ()
        )
        local = ()
        paths = {}
    assert remote.declaration_sha256 == plan.remote.declaration_sha256
    manifest = build_comfy_registry_wheel_input_manifest(plan.declaration_sha256, remote, local)
    metadata: dict[str, bytes] = {}
    for name, path in paths.items():
        metadata[name] = next(
            value
            for member, value in _wheel_files(path.read_bytes()).items()
            if member.endswith("/METADATA")
        )
    closure = plan_comfy_registry_mixed_wheel_closure(
        manifest,
        metadata,
        marker_environment=_marker_environment(),
        supported_tags=("py3-none-any",),
        runtime_distributions={} if inactive else {"helper": "2.0"},
    )
    environment_root = root / "environments"
    environment_root.mkdir()
    destination = environment_root / f"registry-wheels-v3-{closure.closure_sha256}"
    report = await _assemble(closure, paths, destination, context)
    node_root = root / "nodes"
    folder = node_root / "lm-atelier-registry_example"
    folder.mkdir(parents=True)
    content = b"NODE_CLASS_MAPPINGS = {}\n"
    (folder / "node.py").write_bytes(content)
    node_manifest = f"node.py\0{len(content)}\0{hashlib.sha256(content).hexdigest()}\n"
    install = persist_comfy_registry_install(
        session,
        resolution=_resolution(pip_dependencies=declarations),
        archive=_archive(
            manifest_sha256=hashlib.sha256(node_manifest.encode()).hexdigest(),
            entry_count=1,
            file_count=1,
            expanded_bytes=len(content),
            python_file_count=1,
        ),
        installed_path=folder.name,
    )
    install.review_json = {**install.review_json, "runtime_files": [], "preserved": "value"}
    return Prepared(closure, report, install, node_root, environment_root, destination)


def _bind(prepared: Prepared, context: tuple[Session, ArtifactStore]) -> None:
    bind_comfy_registry_wheel_environment(
        prepared.install,
        prepared.closure,
        prepared.report,
        prepared.destination,
        environment_root=prepared.environment_root,
        reviewed_inputs=_input_context(context),
    )


def _launch(prepared: Prepared, context: tuple[Session, ArtifactStore], scoped: bool) -> None:
    session, _store = context
    install = prepared.install
    if scoped:
        binding = WorkflowRegistryLaunchBinding(
            install.id,
            prepared.node_root / install.installed_path,
            prepared.destination / "site-packages",
            install.package_id,
            install.package_version,
            install.archive_sha256,
            install.manifest_sha256,
            prepared.closure.closure_sha256,
            prepared.report.environment_sha256,
            tuple(install.node_types_json),
        )
        result = scoped_comfy_registry_launch_contract(
            session,
            [binding],
            custom_node_root=prepared.node_root,
            environment_root=prepared.environment_root,
            reviewed_inputs=_input_context(context),
        )
    else:
        result = trusted_comfy_registry_launch_contract(
            session,
            custom_node_root=prepared.node_root,
            environment_root=prepared.environment_root,
            reviewed_inputs=_input_context(context),
        )
    assert result.site_packages == (prepared.destination / "site-packages",)
    assert result.node_types == tuple(install.node_types_json)


@pytest.mark.parametrize("scoped", [False, True])
async def test_installed_source_binding_is_persisted_and_rechecked_before_launch(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    scoped: bool,
) -> None:
    prepared = await _prepare(source_review_context, tmp_path)
    _bind(prepared, source_review_context)
    install = prepared.install
    assert (
        prepared.report.reviewed_input_manifest_sha256 == prepared.closure.manifest.manifest_sha256
    )
    payload = json.loads((prepared.destination / "environment-manifest.json").read_text())
    assert payload["version"] == 4
    assert (
        payload["reviewed_input_manifest_sha256"] == prepared.report.reviewed_input_manifest_sha256
    )
    assert install.review_json["preserved"] == "value"
    assert "reviewed_wheel_closure" in install.review_json
    install.trusted = install.active = True
    session, _store = source_review_context
    session.commit()
    session.expire_all()
    _launch(prepared, source_review_context, scoped)
    session.execute(delete(ComfyRegistrySourceArtifactReview))
    session.commit()
    with pytest.raises(ComfyRegistryInstallError, match="source dependency review"):
        _launch(prepared, source_review_context, scoped)


@pytest.mark.parametrize("problem", ["missing_context", "revoked", "declarations", "target"])
async def test_binding_refuses_unverified_source_inputs_without_changing_the_install(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    problem: str,
) -> None:
    prepared = await _prepare(source_review_context, tmp_path)
    session, _store = source_review_context
    context = _input_context(source_review_context)
    if problem == "revoked":
        session.execute(delete(ComfyRegistrySourceArtifactReview))
        session.commit()
    elif problem == "declarations":
        prepared.install.pip_dependencies_json = ["alpha==1.0"]
    elif problem == "target":
        context = replace(context, supported_tags=("py2-none-any",))
    with pytest.raises(ComfyRegistryInstallError):
        bind_comfy_registry_wheel_environment(
            prepared.install,
            prepared.closure,
            prepared.report,
            prepared.destination,
            environment_root=prepared.environment_root,
            reviewed_inputs=None if problem == "missing_context" else context,
        )
    assert prepared.install.wheel_closure_sha256 is None
    assert prepared.install.wheel_environment_sha256 is None
    assert "reviewed_wheel_closure" not in prepared.install.review_json


@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("downgrade_declarations", [False, True])
@pytest.mark.parametrize("inactive", [False, True])
async def test_environment_identity_prevents_removing_required_review_evidence(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    scoped: bool,
    downgrade_declarations: bool,
    inactive: bool,
) -> None:
    prepared = await _prepare(source_review_context, tmp_path, inactive=inactive)
    _bind(prepared, source_review_context)
    prepared.install.trusted = prepared.install.active = True
    prepared.install.review_json = {
        key: value
        for key, value in prepared.install.review_json.items()
        if key != "reviewed_wheel_closure"
    }
    if downgrade_declarations:
        prepared.install.pip_dependencies_json = ["alpha==1.0"]
    session, _store = source_review_context
    session.commit()
    with pytest.raises(ComfyRegistryInstallError, match="source dependency review"):
        _launch(prepared, source_review_context, scoped)


async def test_source_environment_cannot_downgrade_its_manifest_without_changing_identity(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
) -> None:
    prepared = await _prepare(source_review_context, tmp_path)
    path = prepared.destination / "environment-manifest.json"
    payload = json.loads(path.read_text())
    payload["version"] = 3
    payload.pop("reviewed_input_manifest_sha256")
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ComfyRegistryWheelEnvironmentError) as error:
        verify_comfy_registry_wheel_environment(
            prepared.destination,
            expected_closure_sha256=prepared.closure.closure_sha256,
            expected_environment_sha256=prepared.report.environment_sha256,
        )
    assert error.value.code == "environment_hash_mismatch"


@pytest.mark.parametrize("change", ["missing_marker", "version", "bad_marker"])
async def test_reviewed_environment_rejects_an_invalid_marker_even_with_its_current_hash(
    source_review_context: tuple[Session, ArtifactStore],
    tmp_path: Path,
    change: str,
) -> None:
    prepared = await _prepare(source_review_context, tmp_path, inactive=True)
    path = prepared.destination / "environment-manifest.json"
    payload = json.loads(path.read_text())
    if change == "missing_marker":
        payload.pop("reviewed_input_manifest_sha256")
    elif change == "version":
        payload["version"] = 3
    else:
        payload["reviewed_input_manifest_sha256"] = "invalid"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(encoded)
    with pytest.raises(ComfyRegistryWheelEnvironmentError):
        verify_comfy_registry_wheel_environment(
            prepared.destination,
            expected_closure_sha256=prepared.closure.closure_sha256,
            expected_environment_sha256=hashlib.sha256(encoded).hexdigest(),
        )
