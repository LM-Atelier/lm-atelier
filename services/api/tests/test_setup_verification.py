from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select

from local_lm.adapters.base import ChatEvent, ChatRequest, MediaEvent, MediaRequest
from local_lm.adapters.contracts import ADAPTER_CONTRACT_VERSION
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.hardware import hardware_capability_class
from local_lm.model_planner import ACTIVATION_PROBE_VERSION, LAUNCH_CONTRACT_VERSION
from local_lm.models import (
    Artifact,
    ArtifactLibraryEntry,
    Chat,
    Job,
    Message,
    MessagePart,
    ModelCapabilityEvidence,
    ModelInstall,
    ModelProfile,
    SetupVerification,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.setup_verification import (
    SETUP_VERIFICATION_SCOPE,
    ingest_synthetic_setup_image,
    recover_terminal_setup_verifications,
    setup_verification_settings,
    synthetic_setup_image,
)

pytestmark = pytest.mark.asyncio


def seed_ready_role(
    settings: Settings,
    role: str,
    *,
    operation: str | None = None,
    input_schema: dict[str, object] | None = None,
) -> tuple[ModelInstall, ModelProfile, WorkflowRevision | None]:
    install = ModelInstall(
        id=f"model_{role}",
        name=f"Synthetic {role}",
        role=role,
        engine="mock",
        local_path=f"C:/synthetic/{role}",
        compatibility="likely",
        manifest_json={
            "files": [f"{role}.bin"],
            "expected_sha256": {f"{role}.bin": role[0] * 64},
        },
        active=True,
    )
    profile = ModelProfile(
        id=f"profile_{role}",
        model_install_id=install.id,
        name=f"Synthetic {role}",
        role=role,
        engine="mock",
        is_default=True,
    )
    evidence = ModelCapabilityEvidence(
        model_install_id=install.id,
        evidence_key=role[0] * 64,
        result="ready",
        component_hashes_json=install.manifest_json["expected_sha256"],
        runtime_build="mock-test",
        adapter_contract_version=ADAPTER_CONTRACT_VERSION,
        launch_contract_version=LAUNCH_CONTRACT_VERSION,
        workflow_contract_version=None,
        hardware_class=hardware_capability_class(settings),
        probe_version=ACTIVATION_PROBE_VERSION,
        details_json={},
    )
    definition = None
    revision = None
    if operation:
        definition = WorkflowDefinition(
            id=f"workflow_{role}",
            name=f"Synthetic {role}",
            operation=operation,
        )
        revision = WorkflowRevision(
            id=f"revision_{role}",
            workflow_id=definition.id,
            version=1,
            engine="mock",
            api_graph_json={"node": {"class_type": "Synthetic"}},
            input_schema_json=input_schema or {},
            dependencies_json={"model_install_ids": [install.id]},
            trusted=True,
        )
        definition.current_revision_id = revision.id
    with SessionLocal() as session:
        session.add(install)
        session.flush()
        session.add_all([profile, evidence])
        if definition and revision:
            session.add(definition)
            session.flush()
            session.add(revision)
        session.commit()
    return install, profile, revision


async def wait_for_role(
    client: AsyncClient,
    role: str,
    *states: str,
) -> dict:  # type: ignore[type-arg]
    deadline = asyncio.get_running_loop().time() + 8
    while asyncio.get_running_loop().time() < deadline:
        payload = (await client.get("/api/setup/readiness")).json()
        current = next(item for item in payload["roles"] if item["role"] == role)
        if current["state"] in states:
            return current
        await asyncio.sleep(0.03)
    raise AssertionError(f"{role} did not reach {states}")


@pytest.mark.parametrize(
    ("role", "operation"),
    [
        ("chat", None),
        ("image", "text_to_image"),
        ("video", "image_to_video"),
    ],
)
async def test_setup_verification_uses_isolated_queue_and_cleans_everything(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    operation: str | None,
) -> None:
    _install, profile, revision = seed_ready_role(settings, role, operation=operation)
    with SessionLocal() as session:
        source = Chat(title="User chat")
        session.add(source)
        session.flush()
        user_message = Message(chat_id=source.id, role="user", status="complete")
        user_message.parts.append(
            MessagePart(position=0, type="text", text="Synthetic source-only phrase")
        )
        session.add(user_message)
        session.commit()

    chat_requests: list[ChatRequest] = []
    media_requests: list[MediaRequest] = []
    if role == "chat":
        adapter = app.state.services.engines.chat
        original_stream = adapter.stream

        async def recording_stream(request: ChatRequest):  # type: ignore[no-untyped-def]
            chat_requests.append(request)
            async for event in original_stream(request):
                yield event

        monkeypatch.setattr(adapter, "stream", recording_stream)
    else:
        adapter = app.state.services.engines.media
        original_generate = adapter.generate

        async def recording_generate(request: MediaRequest):  # type: ignore[no-untyped-def]
            media_requests.append(request)
            async for event in original_generate(request):
                yield event

        monkeypatch.setattr(adapter, "generate", recording_generate)

    response = await client.post(f"/api/setup/verify/{role}")
    assert response.status_code == 202
    ready = await wait_for_role(client, role, "ready")
    assert ready["verification_level"] == "generation_probe"
    assert ready["checks"][-1]["code"] == "generation_verified"
    assert ready["job_id"] is None

    if role == "chat":
        assert len(chat_requests) == 1
        assert "Synthetic source-only phrase" not in str(chat_requests[0].messages)
        assert chat_requests[0].messages == [
            {"role": "user", "content": "Reply with exactly the single word ready."}
        ]
        assert chat_requests[0].settings["max_tokens"] == 8
    else:
        assert len(media_requests) == 1
        assert media_requests[0].operation == operation
        assert media_requests[0].workflow == {"node": {"class_type": "Synthetic"}}
        assert len(media_requests[0].input_paths) == (1 if operation == "image_to_video" else 0)

    with SessionLocal() as session:
        verification = session.scalar(
            select(SetupVerification).where(SetupVerification.role == role)
        )
        assert verification is not None
        assert verification.state == "ready"
        assert verification.model_install_id == f"model_{role}"
        assert verification.profile_id == profile.id
        assert verification.workflow_revision_id == (revision.id if revision else None)
        assert verification.chat_id is None
        assert verification.run_id is None
        assert verification.job_id is None
        assert (
            session.scalars(select(Chat).where(Chat.scope == SETUP_VERIFICATION_SCOPE)).all() == []
        )
        assert session.scalars(select(Artifact)).all() == []
        assert session.scalars(select(ArtifactLibraryEntry)).all() == []
        assert session.scalars(select(Job)).all() == []


async def test_setup_verification_failure_is_bounded_and_self_cleaning(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_ready_role(settings, "image", operation="text_to_image")
    adapter = app.state.services.engines.media

    async def failing_generate(_request: MediaRequest) -> AsyncIterator[MediaEvent]:
        if False:
            yield MediaEvent(type="progress")
        raise RuntimeError("sensitive synthetic engine detail")

    monkeypatch.setattr(adapter, "generate", failing_generate)
    response = await client.post("/api/setup/verify/image")
    assert response.status_code == 202

    failed = await wait_for_role(client, "image", "action_required")
    assert failed["checks"][-1]["code"] == "generation_verification_failed"
    assert failed["next_action"] == "verify_generation"
    assert "sensitive synthetic engine detail" not in repr(failed)
    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.state == "failed"
        assert verification.failure_code == "generation_failed"
        assert (
            session.scalars(select(Chat).where(Chat.scope == SETUP_VERIFICATION_SCOPE)).all() == []
        )
        assert session.scalars(select(Artifact)).all() == []
        assert session.scalars(select(Job)).all() == []


async def test_setup_verification_uses_selected_workflow_settings(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_ready_role(
        settings,
        "image",
        operation="image_to_image",
        input_schema={
            "type": "object",
            "properties": {
                "width": {"type": "integer", "readOnly": True},
                "height": {"type": "integer", "readOnly": True},
                "steps": {
                    "type": "integer",
                    "default": 4,
                    "minimum": 1,
                    "maximum": 4,
                },
            },
        },
    )
    requests: list[MediaRequest] = []
    adapter = app.state.services.engines.media
    original_generate = adapter.generate

    async def recording_generate(request: MediaRequest):  # type: ignore[no-untyped-def]
        requests.append(request)
        async for event in original_generate(request):
            yield event

    monkeypatch.setattr(adapter, "generate", recording_generate)

    response = await client.post("/api/setup/verify/image")

    assert response.status_code == 202
    await wait_for_role(client, "image", "ready")
    assert len(requests) == 1
    assert requests[0].operation == "image_to_image"
    assert len(requests[0].input_paths) == 1
    assert requests[0].parameters["steps"] == 4
    assert "width" not in requests[0].parameters
    assert "height" not in requests[0].parameters


async def test_setup_verification_cancellation_cleans_transient_state(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_ready_role(settings, "image", operation="text_to_image")
    started = asyncio.Event()

    async def slow_generate(_request: MediaRequest) -> AsyncIterator[MediaEvent]:
        started.set()
        yield MediaEvent(type="progress", progress=0.1, phase="loading")
        await asyncio.sleep(30)

    monkeypatch.setattr(app.state.services.engines.media, "generate", slow_generate)
    assert (await client.post("/api/setup/verify/image")).status_code == 202
    await asyncio.wait_for(started.wait(), timeout=2)
    running = await wait_for_role(client, "image", "in_progress")
    assert running["job_id"]
    assert (await client.post(f"/api/jobs/{running['job_id']}/cancel")).status_code == 200

    failed = await wait_for_role(client, "image", "action_required")
    assert failed["checks"][-1]["code"] == "generation_verification_failed"
    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.failure_code == "generation_cancelled"
        assert (
            session.scalars(select(Chat).where(Chat.scope == SETUP_VERIFICATION_SCOPE)).all() == []
        )
        assert session.scalars(select(Artifact)).all() == []
        assert session.scalars(select(Job)).all() == []


async def test_profile_change_invalidates_prior_generation_evidence(
    client: AsyncClient,
    settings: Settings,
) -> None:
    _install, profile, _workflow = seed_ready_role(settings, "chat")
    assert (await client.post("/api/setup/verify/chat")).status_code == 202
    assert (await wait_for_role(client, "chat", "ready"))["state"] == "ready"

    with SessionLocal() as session:
        stored = session.get(ModelProfile, profile.id)
        assert stored is not None
        stored.request_settings_json = {"temperature": 0.25}
        session.commit()

    stale = await wait_for_role(client, "chat", "action_required")
    assert stale["checks"][-1]["code"] == "generation_verification_required"
    assert stale["next_action"] == "verify_generation"


async def test_activation_change_invalidates_prior_generation_evidence(
    client: AsyncClient,
    settings: Settings,
) -> None:
    seed_ready_role(settings, "chat")
    assert (await client.post("/api/setup/verify/chat")).status_code == 202
    assert (await wait_for_role(client, "chat", "ready"))["state"] == "ready"

    with SessionLocal() as session:
        evidence = session.scalar(select(ModelCapabilityEvidence))
        assert evidence is not None
        evidence.evidence_key = "n" * 64
        session.commit()

    stale = await wait_for_role(client, "chat", "action_required")
    assert stale["checks"][-1]["code"] == "generation_verification_required"


async def test_empty_generation_does_not_verify_setup(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_ready_role(settings, "chat")

    async def empty_stream(_request: ChatRequest) -> AsyncIterator[ChatEvent]:
        yield ChatEvent(type="complete", data={"finish_reason": "stop"})

    monkeypatch.setattr(app.state.services.engines.chat, "stream", empty_stream)
    assert (await client.post("/api/setup/verify/chat")).status_code == 202
    failed = await wait_for_role(client, "chat", "action_required")
    assert failed["checks"][-1]["code"] == "generation_verification_failed"
    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.failure_code == "empty_generation"
        assert (
            session.scalars(select(Chat).where(Chat.scope == SETUP_VERIFICATION_SCOPE)).all() == []
        )


async def test_restart_recovery_removes_unclaimed_hidden_verification(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
) -> None:
    del client
    install, profile, _workflow = seed_ready_role(
        settings,
        "image",
        operation="text_to_image",
    )
    with SessionLocal() as session:
        chat = Chat(
            title="Setup verification",
            archived=True,
            scope=SETUP_VERIFICATION_SCOPE,
            routing_mode="image",
        )
        session.add(chat)
        session.flush()
        verification = SetupVerification(
            role="image",
            evidence_key="r" * 64,
            state="running",
            model_install_id=install.id,
            profile_id=profile.id,
            workflow_revision_id="revision_image",
            chat_id=chat.id,
        )
        session.add(verification)
        session.flush()
        artifact = ingest_synthetic_setup_image(
            session,
            app.state.services.artifacts,
            verification.id,
        )
        verification.input_artifact_id = artifact.id
        session.commit()

    with SessionLocal() as session:
        recover_terminal_setup_verifications(
            session,
            app.state.services.artifacts,
        )
        session.commit()

    with SessionLocal() as session:
        verification = session.scalar(select(SetupVerification))
        assert verification is not None
        assert verification.state == "failed"
        assert verification.failure_code == "application_restarted"
        assert verification.chat_id is None
        assert session.scalars(select(Artifact)).all() == []
        assert session.scalars(select(Job)).all() == []
        assert (
            session.scalars(select(Chat).where(Chat.scope == SETUP_VERIFICATION_SCOPE)).all() == []
        )


async def test_setup_verification_settings_and_input_are_bounded_and_unique() -> None:
    from local_lm.schemas import SettingField

    settings = setup_verification_settings(
        [
            SettingField(
                key="max_tokens",
                label="Maximum output",
                type="integer",
                default=4096,
                minimum=1,
                maximum=8192,
                scope="request",
            ),
            SettingField(
                key="temperature",
                label="Temperature",
                type="number",
                default=0.8,
                minimum=0,
                maximum=2,
                scope="request",
            ),
        ],
        "chat",
    )
    assert settings == {"max_tokens": 8}
    first = synthetic_setup_image("verify_one")
    second = synthetic_setup_image("verify_two")
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert second.startswith(b"\x89PNG\r\n\x1a\n")
    assert first != second
