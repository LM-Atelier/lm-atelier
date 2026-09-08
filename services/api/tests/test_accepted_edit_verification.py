from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from test_image_edit_verification_jobs import _png, _wait_for_run

from local_lm.adapters.base import ChatEvent, ChatRequest, GeneratedAsset, MediaEvent, MediaRequest
from local_lm.adapters.mock import MockChatAdapter, MockMediaAdapter
from local_lm.db import SessionLocal
from local_lm.image_edit_verification import image_edit_verification_job_id
from local_lm.models import Chat, Job, Message, ModelInstall, ModelProfile, Run
from local_lm.schemas import JobOut, TurnRequest
from local_lm.workflow_compatibility import mirror_legacy_chat_workflow_selections


@pytest.mark.parametrize(
    "later_change",
    [
        "queue_policy",
        "queue_profile",
        "execution_policy",
        "execution_prompt",
        "execution_profile",
        "inactive",
        "install_manifest",
        "export",
        "inherit",
        "retry_prompt",
        "retry_settings",
        "retry_profile",
        "retry_snapshot",
        "retry_export",
    ],
)
async def test_image_edit_verification_uses_accepted_configuration(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, later_change: str
) -> None:
    orchestrator = app.state.services.orchestrator
    original_capabilities = MockChatAdapter.capabilities
    captured: list[ChatRequest] = []
    generated: list[MediaRequest] = []
    verified_settings: list[dict[str, Any]] = []
    executing = False
    finished = asyncio.Event()
    original_start = orchestrator.start

    def start(job_id: str, run_id: str | None) -> None:
        if later_change == "retry_snapshot" and run_id is not None:
            with SessionLocal() as session:
                run = session.get(Run, run_id)
                if run is not None and run.provenance_json.get("image_edit_verification_retry"):
                    run.settings_json = {**run.settings_json, "width": 512}
                    run.standalone_prompt = "Make the square blue instead."
                    session.commit()
        original_start(job_id, run_id)

    monkeypatch.setattr(orchestrator, "start", start)

    async def capabilities(adapter: MockChatAdapter) -> Any:
        result = await original_capabilities(adapter)
        return result.model_copy(update={"input_modalities": ["text", "image"]})

    async def assessment(
        adapter: MockChatAdapter, request: ChatRequest
    ) -> AsyncIterator[ChatEvent]:
        captured.append(request)
        yield ChatEvent(
            type="delta",
            text=json.dumps(
                {
                    "requested_change_visible": not later_change.startswith("retry_"),
                    "unrelated_content_preserved": True,
                    "retry_recommended": later_change.startswith("retry_"),
                    "direction": "increase" if later_change.startswith("retry_") else "none",
                    "confidence": 0.94,
                }
            ),
        )
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    async def media(adapter: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        generated.append(request)
        yield MediaEvent(
            type="complete",
            progress=1,
            phase="complete",
            assets=[
                GeneratedAsset(
                    content=_png((20, 180, 80)),
                    media_type="image/png",
                    kind="image",
                    name="constructed-result.png",
                    metadata={"synthetic": True},
                )
            ],
        )

    def verified(session: Any, profile: ModelProfile) -> bool:
        if executing and not captured:
            verified_settings.append(dict(profile.load_settings_json))
        return True

    monkeypatch.setattr(MockChatAdapter, "capabilities", capabilities)
    monkeypatch.setattr(MockChatAdapter, "stream", assessment)
    monkeypatch.setattr(MockMediaAdapter, "generate", media)
    monkeypatch.setattr(orchestrator, "_profile_has_verified_vision", verified)
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("source.png", _png((180, 20, 20)), "image/png")}
    )
    assert uploaded.status_code == 201
    project = (await client.post("/api/projects", json={"name": "Accepted verifier"})).json()
    chat = (
        await client.post(
            "/api/chats",
            json={
                "title": "Accepted edit verification",
                "project_id": project["id"],
                "vision_settings_json": {"verify_image_edits": True},
            },
        )
    ).json()
    with SessionLocal() as session:
        install = ModelInstall(
            id="model_accepted_verifier",
            name="Constructed verifier",
            role="chat",
            engine="mock",
            local_path="synthetic",
            active=True,
        )
        profile = ModelProfile(
            id="profile_accepted_verifier",
            name="Accepted verifier",
            role="chat",
            engine="mock",
            model_install_id=install.id,
            load_settings_json={"context_length": 4096},
        )
        other = ModelProfile(
            id="profile_later_verifier",
            name="Later verifier",
            role="chat",
            engine="mock",
            model_install_id=install.id,
        )
        stored = session.get(Chat, chat["id"])
        assert stored is not None
        stored.active_vision_profile_id = profile.id
        session.add_all([install, profile, other])
        session.flush()
        mirror_legacy_chat_workflow_selections(session, stored, ["vision"])
        session.commit()

    original_execute = orchestrator._execute_image_edit_verification

    async def execute(job_id: str, claim: Any) -> None:
        nonlocal executing
        executing = True
        with SessionLocal() as session:
            job = session.get(Job, job_id)
            assert job is not None
            run = session.get(Run, job.payload_json["source_run_id"])
            stored = session.get(Chat, chat["id"])
            profile = session.get(ModelProfile, "profile_accepted_verifier")
            install = session.get(ModelInstall, "model_accepted_verifier")
            assert (
                run is not None
                and stored is not None
                and profile is not None
                and install is not None
            )
            if later_change == "execution_policy":
                stored.vision_settings_json = {"verify_image_edits": False}
            elif later_change == "execution_prompt":
                run.standalone_prompt = "Make the square blue instead."
            elif later_change == "execution_profile":
                profile.load_settings_json = {"context_length": 2048}
            elif later_change == "inactive":
                install.active = False
            elif later_change == "install_manifest":
                install.manifest_json = {"expected_sha256": {"model": "changed"}}
            elif later_change == "retry_prompt":
                source = session.get(Message, run.user_message_id)
                assert source is not None
                for part in source.parts:
                    if part.text:
                        part.text = "Make the square blue instead."
            elif later_change == "retry_settings":
                run.settings_json = {**run.settings_json, "width": 512}
            elif later_change == "retry_profile":
                media_profile = session.get(ModelProfile, run.profile_id)
                assert media_profile is not None
                media_profile.load_settings_json = {"context_length": 2048}
            session.commit()
        try:
            await original_execute(job_id, claim)
        finally:
            finished.set()

    monkeypatch.setattr(orchestrator, "_execute_image_edit_verification", execute)
    async with app.state.services.scheduler.lease("primary"):
        with SessionLocal() as session:
            accepted = await orchestrator.create_turn(
                session,
                chat["id"],
                TurnRequest(
                    text="Make the square green.",
                    mode="image",
                    input_artifact_ids=[uploaded.json()["id"]],
                    settings={"width": 640},
                ),
                freeze_context=True,
                activate_branch=False,
            )
            stored = session.get(Chat, chat["id"])
            assert stored is not None
            if later_change == "queue_policy":
                stored.vision_settings_json = {"verify_image_edits": False}
            elif later_change == "queue_profile":
                stored.active_vision_profile_id = "profile_later_verifier"
                mirror_legacy_chat_workflow_selections(session, stored, ["vision"])
            session.commit()
    await _wait_for_run(client, accepted.run.id)
    job_id = image_edit_verification_job_id(accepted.run.id)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None, "Accepted verification was silently disabled"
    await asyncio.wait_for(finished.wait(), timeout=10)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job is not None
        assert job.payload_json["vision_profile_id"] == "profile_accepted_verifier"
        result = JobOut.model_validate(job).model_dump(mode="json")
    assert result["status"] == "complete"
    if later_change in {"inactive", "install_manifest"}:
        assert result["result_json"]["reason"] == "vision_profile_unavailable"
        assert captured == []
    else:
        assert result["result_json"]["status"] == "complete", result["result_json"]
        assert len(captured) == 1
        rendered = json.dumps(captured[0].messages)
        assert "Make the square green." in rendered
        assert "Make the square blue instead." not in rendered
        assert verified_settings == [{"context_length": 4096}]
        assert result["result_json"]["automatic_retry_executed"] is later_change.startswith(
            "retry_"
        )
    if later_change.startswith("retry_"):
        from local_lm.accepted_turn_context import accepted_context

        retry_id = result["result_json"]["retry_run_id"]
        await _wait_for_run(client, retry_id)
        assert len(generated) == 2
        assert generated[1].prompt == generated[0].prompt
        assert generated[1].parameters["width"] == generated[0].parameters["width"] == 640
        assert generated[1].parameters["denoise"] == 0.62
        with SessionLocal() as session:
            source_run = session.get(Run, accepted.run.id)
            retry_run = session.get(Run, retry_id)
            stored = session.get(Chat, chat["id"])
            assert source_run is not None and retry_run is not None and stored is not None
            source_snapshot = accepted_context(session, source_run)
            retry_snapshot = accepted_context(session, retry_run)
            assert source_snapshot is not None and retry_snapshot is not None
            assert retry_snapshot.profile == source_snapshot.profile
            assert retry_snapshot.verification_profile == source_snapshot.verification_profile
            assert retry_snapshot.workflow == source_snapshot.workflow
            assert retry_snapshot.input_artifact_ids == source_snapshot.input_artifact_ids
            assert retry_snapshot.vision_settings == source_snapshot.vision_settings
            assert stored.active_head_message_id == chat["active_head_message_id"]
            assert retry_run.provenance_json["image_edit_verification"]["reason"] == (
                "retry_limit_reached"
            )
            assert "edit_source" not in retry_run.provenance_json
    if later_change in {"export", "retry_export"}:
        from local_lm.accepted_turn_context import accepted_context

        exported = await client.post(f"/api/projects/{project['id']}/export")
        assert exported.status_code == 201, exported.text
        archive = await client.get(f"/api/artifacts/{exported.json()['id']}/content")
        imported = await client.post(
            "/api/projects/import",
            files={"archive": ("verifier.zip", archive.content, "application/zip")},
        )
        assert imported.status_code == 201, imported.text
        with SessionLocal() as session:
            imported_chat = session.scalar(
                select(Chat).where(Chat.project_id == imported.json()["id"])
            )
            assert imported_chat is not None
            imported_run = session.scalar(select(Run).where(Run.chat_id == imported_chat.id))
            assert imported_run is not None
            snapshot = accepted_context(session, imported_run)
            assert snapshot is not None and snapshot.verification_profile is not None
            assert snapshot.verification_profile.load_settings_json == {"context_length": 4096}
            assert snapshot.verification_profile.install is None
            assert snapshot.verification_profile.id != "profile_accepted_verifier"
            if later_change == "retry_export":
                imported_runs = list(
                    session.scalars(select(Run).where(Run.chat_id == imported_chat.id))
                )
                assert len(imported_runs) == 2
                imported_retry = next(
                    run
                    for run in imported_runs
                    if run.provenance_json.get("image_edit_verification_retry")
                )
                retry_snapshot = accepted_context(session, imported_retry)
                assert retry_snapshot is not None
                imported_source = session.get(Run, retry_snapshot.source_run_id)
                assert imported_source is not None and imported_source in imported_runs
                assert imported_source.id != accepted.run.id
                assert retry_snapshot.source_message_id == imported_source.user_message_id
                binding = imported_retry.provenance_json["image_edit_verification_retry"]
                assert binding["source_run_id"] == imported_source.id
                assert binding["source_message_id"] == imported_source.user_message_id
                assert retry_snapshot.media_prompt == generated[0].prompt
                assert retry_snapshot.settings["width"] == 640
                assert retry_snapshot.settings["denoise"] == 0.62
    if later_change == "inherit":
        with SessionLocal() as session:
            stored = session.get(Chat, chat["id"])
            assert stored is not None
            stored.active_vision_profile_id = "profile_later_verifier"
            mirror_legacy_chat_workflow_selections(session, stored, ["vision"])
            session.commit()
        source_id = accepted.user_message.id
        loaded = await client.get(f"/api/messages/{source_id}/edit-source")
        assert loaded.status_code == 200, loaded.text
        finished.clear()
        captured.clear()
        verified_settings.clear()
        executing = False
        async with app.state.services.scheduler.lease("primary"):
            edited = await client.post(
                f"/api/messages/{source_id}/edits",
                json={
                    "text": "Make the square green.",
                    "source_run_id": loaded.json()["source_run_id"],
                    "source_snapshot_sha256": loaded.json()["source_snapshot_sha256"],
                    "idempotency_key": "inherit-the-source-verifier",
                    "confirm_media": True,
                },
            )
            assert edited.status_code == 202, edited.text
        edited_run_id = edited.json()["run"]["id"]
        await _wait_for_run(client, edited_run_id)
        await asyncio.wait_for(finished.wait(), timeout=10)
        with SessionLocal() as session:
            job = session.get(Job, image_edit_verification_job_id(edited_run_id))
            assert job is not None
            assert job.payload_json["vision_profile_id"] == "profile_accepted_verifier"
            assert job.result_json["status"] == "complete"
        assert len(captured) == 1
        assert verified_settings == [{"context_length": 4096}]
