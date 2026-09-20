from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import update

from local_lm.capability_evidence import record_capability_evidence
from local_lm.db import SessionLocal
from local_lm.models import InstallPlan, Job, ModelInstall, ModelProfile, ModelSource
from local_lm.workflow_compatibility import ensure_legacy_profile_workflow


def _seed(
    app: FastAPI, *, role: str = "image", version_remote: bool = False, engine: str = "mock"
) -> tuple[str, str, str, str]:
    with SessionLocal() as session:
        installs: list[ModelInstall] = []
        for version in ["201", "202"]:
            remote_id = version if version_remote else "101"
            source = ModelSource(provider="civitai", remote_id=remote_id, revision=version)
            session.add(source)
            session.flush()
            install = ModelInstall(
                name=f"Landscape {version}",
                role=role,
                engine=engine,
                source_id=source.id,
                local_path=f"C:/neutral-fixture/{version}",
                active=True,
                manifest_json={
                    "remote_id": remote_id,
                    "revision": version,
                    "family": "neutral",
                    "files": ["model.safetensors"],
                    "expected_sha256": {"model.safetensors": version[0] * 64},
                },
            )
            session.add(install)
            session.flush()
            if version_remote:
                session.add(
                    InstallPlan(
                        provider="civitai",
                        remote_id=version,
                        revision=version,
                        role=role,
                        engine=engine,
                        plan_hash=version[-1] * 64,
                        resolver_version="test",
                        compatibility="supported",
                        status="activated",
                        artifacts_json=[
                            {
                                "path": "model.safetensors",
                                "required": True,
                                "sha256": version[0] * 64,
                                "source_remote_id": "101",
                                "source_revision": version,
                                "source_version_id": version,
                                "source_file_id": "301",
                            }
                        ],
                    )
                )
            record_capability_evidence(
                session,
                install,
                app.state.services.settings,
                None,
                component_hashes=install.manifest_json["expected_sha256"],
                runtime_build="mock",
                workflow_contract_version=None,
                details={"input_modalities": ["text"]},
            )
            installs.append(install)
        session.execute(
            update(ModelProfile).where(ModelProfile.role == role).values(is_default=False)
        )
        profile = ModelProfile(
            name="My landscape profile",
            role=role,
            engine=engine,
            model_install_id=installs[0].id,
            is_default=True,
            use_case="Landscapes",
            load_settings_json={"context_length": 4096},
            request_settings_json={"seed": 42},
        )
        session.add(profile)
        session.flush()
        ensure_legacy_profile_workflow(session, profile)
        job = Job(
            kind="download",
            status="complete",
            payload_json={"remote_id": "202" if version_remote else "101", "revision": "202"},
            result_json={"model_install_id": installs[1].id},
        )
        session.add(job)
        session.commit()
        return profile.id, installs[0].id, installs[1].id, job.id


@pytest.mark.parametrize("role", ["chat", "image", "video"])
@pytest.mark.parametrize("version_remote", [False, True])
async def test_switch_keeps_profile_choices_and_the_previous_install(
    app: FastAPI,
    client: AsyncClient,
    role: str,
    version_remote: bool,
) -> None:
    profile_id, old_id, new_id, job_id = _seed(app, role=role, version_remote=version_remote)
    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["model_install_id"] == new_id
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        old = session.get(ModelInstall, old_id)
        assert profile is not None and old is not None
        assert old.active
        assert profile.name == "My landscape profile"
        assert profile.use_case == "Landscapes"
        assert profile.is_default
        assert profile.load_settings_json == {"context_length": 4096}
        assert profile.request_settings_json == {"seed": 42}


@pytest.mark.parametrize(
    "change",
    [
        "stale_profile",
        "failed_download",
        "cancelled_download",
        "running_download",
        "other_job_kind",
        "missing_result",
        "inactive",
        "other_role",
        "other_engine",
        "other_parent",
        "same_version",
        "unverified",
        "other_family",
        "wrong_download_version",
    ],
)
async def test_switch_refuses_changed_or_unverified_updates(
    app: FastAPI,
    client: AsyncClient,
    change: str,
) -> None:
    from sqlalchemy import delete

    from local_lm.models import ModelCapabilityEvidence

    profile_id, old_id, new_id, job_id = _seed(app)
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        install = session.get(ModelInstall, new_id)
        job = session.get(Job, job_id)
        assert profile is not None and install is not None and job is not None
        source = session.get(ModelSource, install.source_id)
        assert source is not None
        if change == "stale_profile":
            profile.model_install_id = new_id
        elif change.endswith("_download"):
            job.status = change.removesuffix("_download")
        elif change == "other_job_kind":
            job.kind = "chat"
        elif change == "missing_result":
            job.result_json = {}
        elif change == "inactive":
            install.active = False
        elif change == "other_role":
            install.role = "video"
        elif change == "other_engine":
            install.engine = "comfyui"
        elif change == "other_parent":
            source.remote_id = "102"
            install.manifest_json = {**install.manifest_json, "remote_id": "102"}
            job.payload_json = {**job.payload_json, "remote_id": "102"}
        elif change == "same_version":
            source.remote_id = "102"
            source.revision = "201"
            previous = session.get(ModelInstall, old_id)
            assert previous is not None
            install.source_id = previous.source_id
            install.manifest_json = {**install.manifest_json, "revision": "201"}
            job.payload_json = {**job.payload_json, "revision": "201"}
        elif change == "unverified":
            session.execute(
                delete(ModelCapabilityEvidence).where(
                    ModelCapabilityEvidence.model_install_id == new_id,
                )
            )
        elif change == "other_family":
            install.manifest_json = {**install.manifest_json, "family": "different"}
        elif change == "wrong_download_version":
            job.payload_json = {**job.payload_json, "revision": "203"}
        session.commit()
    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == 409, response.text
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        assert profile.model_install_id == (new_id if change == "stale_profile" else old_id)
        previous = session.get(ModelInstall, old_id)
        assert previous is not None and previous.active


@pytest.mark.parametrize("snapshot_state", ["frozen", "missing", "changed"])
async def test_switch_preserves_accepted_runs_and_applies_to_future_runs(
    app: FastAPI,
    client: AsyncClient,
    snapshot_state: str,
) -> None:
    from sqlalchemy import select

    from local_lm.accepted_turn_context import accepted_context, resolve_accepted_profile
    from local_lm.models import Chat, Run, RunContextSnapshot
    from local_lm.schemas import TurnRequest
    from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections

    profile_id, old_id, new_id, job_id = _seed(app, role="chat")
    chat_response = await client.post("/api/chats", json={"title": "Model update"})
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.active_chat_profile_id = profile_id
        mirror_legacy_chat_workflow_selections(session, chat, ["chat"])
        session.commit()
    orchestrator = app.state.services.orchestrator
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(text="Explain paper boats.", mode="text"),
                freeze_context=True,
                activate_branch=False,
            )
            run_id = accepted.run.id
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            row = session.get(RunContextSnapshot, run_id)
            work = session.scalar(select(Job).where(Job.run_id == run_id))
            assert run is not None and row is not None and work is not None
            assert work.status == "queued"
            original_snapshot = row.sha256
            if snapshot_state == "missing":
                session.delete(row)
                run.provenance_json = {
                    key: value
                    for key, value in run.provenance_json.items()
                    if key != "accepted_context_sha256"
                }
            elif snapshot_state == "changed":
                row.sha256 = "0" * 64
            session.commit()
        response = await client.post(
            f"/api/profiles/{profile_id}/model-update",
            json={
                "expected_install_id": old_id,
                "download_job_id": job_id,
            },
        )
        if snapshot_state != "frozen":
            assert response.status_code == 409, response.text
            assert response.json()["code"] == "profile-update-busy"
            with SessionLocal() as session:
                profile = session.get(ModelProfile, profile_id)
                assert profile is not None and profile.model_install_id == old_id
            return
        assert response.status_code == 200, response.text
        with SessionLocal() as session:
            run = session.get(Run, run_id)
            assert run is not None
            old_snapshot = accepted_context(session, run)
            row = session.get(RunContextSnapshot, run_id)
            assert old_snapshot is not None and old_snapshot.profile is not None
            assert row is not None and row.sha256 == original_snapshot
            frozen_profile, old_install, old_scope = resolve_accepted_profile(
                session, old_snapshot.profile
            )
            assert frozen_profile.model_install_id == old_id == old_install.id
        with SessionLocal() as session:
            future = await orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(text="Explain folded paper.", mode="text"),
                freeze_context=True,
                activate_branch=False,
            )
            future_snapshot = accepted_context(session, future.run)
            assert future_snapshot is not None and future_snapshot.profile is not None
            future_profile, new_install, new_scope = resolve_accepted_profile(
                session, future_snapshot.profile
            )
            assert future_profile.id == profile_id
            assert future_profile.model_install_id == new_id == new_install.id
            assert new_scope != old_scope


async def test_switch_preserves_a_chat_vision_requirement(
    app: FastAPI, client: AsyncClient
) -> None:
    from local_lm.models import Chat

    profile_id, old_id, _, job_id = _seed(app, role="chat")
    with SessionLocal() as session:
        session.add(Chat(title="Vision selection", active_vision_profile_id=profile_id))
        session.commit()
    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "profile-update-vision-unverified"


async def test_download_lookup_keeps_an_older_update_reachable(
    app: FastAPI,
    client: AsyncClient,
) -> None:
    from datetime import timedelta

    from local_lm.domain import utcnow

    _, _, new_id, job_id = _seed(app)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        job.created_at = utcnow() - timedelta(days=1)
        session.add_all(Job(kind="download", status="complete") for _ in range(101))
        session.commit()
    recent = await client.get("/api/jobs")
    assert recent.status_code == 200
    assert job_id not in {item["id"] for item in recent.json()}
    response = await client.get(f"/api/downloads/{job_id}")
    assert response.status_code == 200, response.text
    assert response.json()["result_json"]["model_install_id"] == new_id


async def test_download_lookup_refuses_other_jobs(client: AsyncClient) -> None:
    with SessionLocal() as session:
        job = Job(kind="chat", status="complete")
        session.add(job)
        session.commit()
        job_id = job.id
    for identifier in [job_id, "missing"]:
        response = await client.get(f"/api/downloads/{identifier}")
        assert response.status_code == 404
        assert response.json()["code"] == "download-not-found"


@pytest.mark.parametrize("active", [True, False])
async def test_switch_preserves_active_workflow_profile_bindings(
    app: FastAPI,
    client: AsyncClient,
    active: bool,
) -> None:
    from local_lm.models import (
        WorkflowActivation,
        WorkflowDefinition,
        WorkflowDependencyBinding,
        WorkflowDependencySlot,
        WorkflowRevision,
    )

    profile_id, old_id, new_id, job_id = _seed(app)
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Pinned landscape", operation="text_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(workflow_id=definition.id, version=1)
        session.add(revision)
        session.flush()
        slot = WorkflowDependencySlot(
            workflow_revision_id=revision.id,
            name="model",
            resource_kind="model_profile",
            required=True,
            satisfaction="all_of",
            requirements_json=[],
            contract_sha256="a" * 64,
            ordinal=0,
        )
        activation = WorkflowActivation(
            workflow_revision_id=revision.id,
            resolver_version="1",
            dependency_contract_sha256="b" * 64,
            binding_sha256="c" * 64,
            state="ready",
            is_active=active,
        )
        session.add_all([slot, activation])
        session.flush()
        binding = WorkflowDependencyBinding(
            workflow_revision_id=revision.id,
            workflow_activation_id=activation.id,
            workflow_dependency_slot_id=slot.id,
            requirement_key="model",
            model_profile_id=profile_id,
            resource_identity_sha256="d" * 64,
        )
        session.add(binding)
        session.commit()
        binding_id = binding.id
    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == (409 if active else 200), response.text
    if active:
        assert response.json()["code"] == "profile-update-workflow-bound"
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        binding = session.get(WorkflowDependencyBinding, binding_id)
        assert profile is not None and binding is not None
        assert profile.model_install_id == (old_id if active else new_id)
        assert binding.resource_identity_sha256 == "d" * 64


async def test_new_turn_reloads_a_worker_with_the_previous_install(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock, Mock

    from local_lm.models import Chat, RunContextSnapshot
    from local_lm.schemas import TurnRequest, WorkerStatus
    from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections

    profile_id, old_id, new_id, job_id = _seed(app, role="chat", engine="llama.cpp")
    orchestrator = app.state.services.orchestrator
    fields = await orchestrator.engines.settings_for_role("chat", engine="mock")
    monkeypatch.setattr(orchestrator.engines, "settings_for_role", AsyncMock(return_value=fields))
    monkeypatch.setattr(orchestrator.engines.settings, "chat_engine", "llama.cpp")
    current = WorkerStatus(
        name="chat", state="ready", running=True, managed=True, profile_id=profile_id
    )
    loaded = current.model_copy()
    loader = AsyncMock(return_value=loaded)
    monkeypatch.setattr(orchestrator.processes, "statuses", Mock(return_value=[current]))
    monkeypatch.setattr(orchestrator.processes, "load_chat", loader)

    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == 200, response.text
    chat_response = await client.post("/api/chats", json={"title": "Updated ordinary turn"})
    assert chat_response.status_code == 201
    chat_id = chat_response.json()["id"]
    with SessionLocal() as session:
        chat = session.get(Chat, chat_id)
        assert chat is not None
        chat.active_chat_profile_id = profile_id
        mirror_legacy_chat_workflow_selections(session, chat, ["chat"])
        session.commit()
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            future = await orchestrator.create_turn(
                session,
                chat_id,
                TurnRequest(text="Explain paper boats.", mode="text"),
            )
            run_id = future.run.id
            assert session.get(RunContextSnapshot, run_id) is None
        assert await orchestrator._ensure_chat_worker(run_id) is loaded
        loader.assert_awaited_once()
        supplied_profile, supplied_install = loader.await_args_list[0].args
        assert supplied_profile.id == profile_id
        assert supplied_profile.model_install_id == new_id == supplied_install.id
        assert len(loader.await_args_list[0].kwargs["launch_scope_sha256"]) == 64


@pytest.mark.parametrize("role", ["image", "video"])
async def test_future_media_selection_uses_the_updated_install_workflow(
    app: FastAPI,
    client: AsyncClient,
    role: str,
) -> None:
    from local_lm.domain import Operation
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    profile_id, old_id, new_id, job_id = _seed(app, role=role)
    operation = Operation.TEXT_TO_IMAGE if role == "image" else Operation.TEXT_TO_VIDEO
    revisions: dict[str, str] = {}
    orchestrator = app.state.services.orchestrator
    with SessionLocal() as session:
        for install_id in [old_id, new_id]:
            definition = WorkflowDefinition(name="Landscape generation", operation=operation.value)
            session.add(definition)
            session.flush()
            revision = WorkflowRevision(
                workflow_id=definition.id,
                version=1,
                engine="mock",
                trusted=True,
                dependencies_json={"model_install_ids": [install_id]},
            )
            session.add(revision)
            session.flush()
            definition.current_revision_id = revision.id
            revisions[install_id] = revision.id
        session.commit()
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        previous = orchestrator.legacy_workflow_revision(session, profile, operation)
        assert previous is not None and previous.id == revisions[old_id]
    response = await client.post(
        f"/api/profiles/{profile_id}/model-update",
        json={
            "expected_install_id": old_id,
            "download_job_id": job_id,
        },
    )
    assert response.status_code == 200, response.text
    with SessionLocal() as session:
        profile = session.get(ModelProfile, profile_id)
        assert profile is not None
        future = orchestrator.legacy_workflow_revision(session, profile, operation)
        previous = session.get(WorkflowRevision, revisions[old_id])
        assert future is not None and future.id == revisions[new_id]
        assert previous is not None and previous.dependencies_json == {
            "model_install_ids": [old_id]
        }


@pytest.mark.parametrize("change", ["none", "install", "settings"])
async def test_chat_worker_reuse_requires_the_same_model_configuration(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    change: str,
) -> None:
    import asyncio
    from unittest.mock import AsyncMock, Mock

    from local_lm.accepted_turn_context import capture_profile, resolve_accepted_profile
    from local_lm.processes import ProcessSupervisor, WorkerRecord, _RotatingWorkerLog

    profile_id, old_id, _, job_id = _seed(app, role="chat", engine="llama.cpp")
    with SessionLocal() as session:
        captured = capture_profile(session, profile_id)
        assert captured is not None
        _, _, original_scope = resolve_accepted_profile(session, captured)
    if change == "install":
        response = await client.post(
            f"/api/profiles/{profile_id}/model-update",
            json={
                "expected_install_id": old_id,
                "download_job_id": job_id,
            },
        )
        assert response.status_code == 200, response.text
    elif change == "settings":
        with SessionLocal() as session:
            profile = session.get(ModelProfile, profile_id)
            assert profile is not None
            profile.load_settings_json = {"context_length": 8192}
            session.commit()
    with SessionLocal() as session:
        captured = capture_profile(session, profile_id)
        assert captured is not None
        _, _, current_scope = resolve_accepted_profile(session, captured)
    supervisor = ProcessSupervisor(app.state.services.settings)
    command = ["synthetic-chat-worker"]
    process = Mock(spec=asyncio.subprocess.Process, returncode=None)
    log = _RotatingWorkerLog(tmp_path / "worker-reuse.log")
    record = WorkerRecord(
        "chat",
        process,
        command,
        log,
        state="ready",
        profile_id=profile_id,
        launch_scope_sha256=original_scope,
    )
    supervisor._workers["chat"] = record
    stop = AsyncMock(side_effect=RuntimeError("Different launch reached the replacement path."))
    monkeypatch.setattr(supervisor, "_stop_unlocked", stop)
    try:
        if change == "none":
            await supervisor._replace(
                "chat",
                command,
                "http://127.0.0.1:12341/health",
                profile_id=profile_id,
                launch_scope_sha256=current_scope,
            )
            stop.assert_not_awaited()
            assert supervisor._workers["chat"] is record
        else:
            with pytest.raises(RuntimeError, match="Different launch reached the replacement path"):
                await supervisor._replace(
                    "chat",
                    command,
                    "http://127.0.0.1:12341/health",
                    profile_id=profile_id,
                    launch_scope_sha256=current_scope,
                )
            stop.assert_awaited_once_with("chat")
    finally:
        log.close()
