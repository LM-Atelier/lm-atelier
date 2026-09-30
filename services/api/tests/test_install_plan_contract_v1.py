from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.api import _planned_download_fields
from local_lm.db import SessionLocal
from local_lm.model_manifests import InspectedComponent, ModelManifestInspection
from local_lm.model_planner import ResolvedInstallPlan, persist_install_plan, resolve_install_plan
from local_lm.models import InstallPlan
from local_lm.schemas import DownloadRequest
from local_lm.workflow_asset_downloads import (
    WorkflowAssetDownloadError,
    install_plan_download_request,
)


def _resolved(
    *, revision: str = "a" * 40, changes: dict[str, Any] | None = None
) -> ResolvedInstallPlan:
    file = {"filename": "model.gguf", "size": 128, "sha256": "b" * 64}
    file.update(changes or {})
    return resolve_install_plan(
        remote_id="synthetic/model",
        revision=revision,
        role="chat",
        engine="llama.cpp",
        selected_files=[file],
        inspection=ModelManifestInspection(
            architecture="llama",
            family="llama",
            components=(InspectedComponent("model.gguf", "gguf_model", "models"),),
            metadata_files=(),
        ),
    )


def _record(resolved: ResolvedInstallPlan) -> InstallPlan:
    return InstallPlan(
        id="plan",
        provider=resolved.provider,
        remote_id=resolved.remote_id,
        revision=resolved.revision,
        role=resolved.role,
        engine=resolved.engine,
        architecture=resolved.architecture,
        family=resolved.family,
        plan_hash=resolved.plan_hash,
        resolver_version=resolved.resolver_version,
        compatibility=resolved.compatibility,
        artifacts_json=[artifact.as_dict() for artifact in resolved.artifacts],
        runtime_contract_json=resolved.runtime_contract,
        activation_probe_json=resolved.activation_probe,
        status="planned",
        failure_code=resolved.failure_code,
    )


@pytest.mark.parametrize("revision", ["main", "release", "a" * 12, "", "g" * 40])
def test_new_plan_requires_an_immutable_provider_revision(revision: str) -> None:
    plan = _resolved(revision=revision)
    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "preflight_blocked"


@pytest.mark.parametrize(
    "changes",
    [
        {"size": None},
        {"size": 0},
        {"size": True},
        {"size": 1.5},
        {"sha256": None},
        {"sha256": "invalid"},
        {"source_remote_id": "synthetic/companion"},
        {
            "source_remote_id": "synthetic/companion",
            "source_revision": "main",
            "source_filename": "model.gguf",
        },
    ],
)
def test_new_plan_refuses_incomplete_required_file_evidence(changes: dict[str, Any]) -> None:
    plan = _resolved(changes=changes)
    assert plan.compatibility == "unsupported"
    assert plan.failure_code == "preflight_blocked"


def test_complete_new_plan_uses_the_same_request_in_general_and_workflow_installs() -> None:
    resolved = _resolved()
    assert resolved.compatibility == "supported"
    assert resolved.resolver_version == "install-resolver-v10"
    plan = _record(resolved)
    request = install_plan_download_request(plan).model_dump(mode="json")
    fields = _planned_download_fields(plan)
    assert fields and all(request[key] == value for key, value in fields.items())
    assert fields["expected_sha256"] == {"model.gguf": "b" * 64}


def test_legacy_general_plan_remains_usable_without_becoming_a_new_workflow_binding() -> None:
    resolved = _resolved()
    legacy = replace(
        resolved,
        resolver_version="install-resolver-v9",
        artifacts=(replace(resolved.artifacts[0], sha256=None, size_bytes=None),),
    )
    plan = _record(legacy)
    identity = (plan.plan_hash, plan.resolver_version, list(plan.artifacts_json))
    assert _planned_download_fields(plan)["expected_sha256"] == {}
    with pytest.raises(WorkflowAssetDownloadError, match="install contract changed"):
        install_plan_download_request(plan)
    assert (plan.plan_hash, plan.resolver_version, plan.artifacts_json) == identity


def test_accepted_legacy_request_still_requires_its_complete_file_evidence() -> None:
    resolved = replace(_resolved(), resolver_version="install-resolver-v9")
    plan = _record(resolved)
    request = install_plan_download_request(plan, allow_legacy_contract=True)
    assert request.expected_sha256 == {"model.gguf": "b" * 64}
    plan = _record(replace(resolved, artifacts=(replace(resolved.artifacts[0], sha256=None),)))
    with pytest.raises(WorkflowAssetDownloadError) as refused:
        install_plan_download_request(plan, allow_legacy_contract=True)
    assert refused.value.code == "unverified_install_artifact"


@pytest.mark.parametrize("change", ["version", "artifact", "runtime"])
def test_a_version_label_or_changed_evidence_cannot_reuse_a_new_plan_hash(change: str) -> None:
    resolved = _resolved()
    if change == "version":
        resolved = replace(resolved, resolver_version="install-resolver-v9")
    plan = _record(resolved)
    if change == "version":
        plan.resolver_version = "install-resolver-v10"
    elif change == "artifact":
        plan.artifacts_json = [{**plan.artifacts_json[0], "sha256": "c" * 64}]
    else:
        plan.runtime_contract_json = {**plan.runtime_contract_json, "auxiliary_kind": "lora"}
    with pytest.raises(WorkflowAssetDownloadError) as refused:
        install_plan_download_request(plan)
    assert refused.value.code == "install_plan_changed"


def test_a_new_plan_cannot_bypass_verification_by_adopting_the_legacy_version() -> None:
    plan = _record(_resolved())
    plan.resolver_version = "install-resolver-v9"
    with pytest.raises(ValueError, match="install plan evidence changed"):
        _planned_download_fields(plan)
    with pytest.raises(WorkflowAssetDownloadError) as refused:
        install_plan_download_request(plan, allow_legacy_contract=True)
    assert refused.value.code == "install_plan_changed"


def test_replanning_creates_a_new_contract_without_relabeling_a_legacy_row(app: FastAPI) -> None:
    resolved = _resolved()
    legacy = replace(resolved, resolver_version="install-resolver-v9")
    with SessionLocal() as session:
        old = persist_install_plan(session, legacy)
        session.commit()
        original = (old.id, old.plan_hash, old.resolver_version)
        new = persist_install_plan(session, resolved)
        session.commit()
        session.refresh(old)
        assert (old.id, old.plan_hash, old.resolver_version) == original
        assert new.id != old.id and new.plan_hash != old.plan_hash
        assert new.resolver_version == "install-resolver-v10"


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_download_endpoint_preserves_both_contracts_and_refuses_request_drift(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    resolved = _resolved()
    if legacy:
        resolved = replace(
            resolved,
            resolver_version="install-resolver-v9",
            artifacts=(replace(resolved.artifacts[0], sha256=None, size_bytes=None),),
        )
    with SessionLocal() as session:
        plan = persist_install_plan(session, resolved)
        session.commit()
        identity = (plan.id, plan.plan_hash, plan.resolver_version)
        request = DownloadRequest(install_plan_id=plan.id, **_planned_download_fields(plan))
    monkeypatch.setattr(app.state.services.downloads, "start", lambda _job_id: None)
    payload = request.model_dump(mode="json")
    refused = await client.post("/api/downloads", json={**payload, "revision": "c" * 40})
    assert refused.status_code == 422, refused.text
    assert "immutable plan" in refused.json()["detail"]
    accepted = await client.post("/api/downloads", json=payload)
    assert accepted.status_code == 202, accepted.text
    assert accepted.json()["payload_json"] == payload
    with SessionLocal() as session:
        stored = session.get(InstallPlan, identity[0])
        assert stored is not None
        assert (stored.id, stored.plan_hash, stored.resolver_version) == identity
