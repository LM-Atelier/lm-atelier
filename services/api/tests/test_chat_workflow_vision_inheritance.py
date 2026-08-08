from __future__ import annotations

import base64
from typing import cast

from httpx2 import AsyncClient
from sqlalchemy.orm import Session

from local_lm.capability_evidence import record_capability_evidence
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.workflow_compatibility import reconcile_legacy_workflow_compatibility

ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _profile(
    session: Session,
    settings: Settings,
    identity: str,
    *,
    verified_vision: bool,
) -> ModelProfile:
    digest = ("a" if identity == "selected" else "b") * 64
    install = ModelInstall(
        id=f"install_{identity}",
        name=identity.title(),
        role="chat",
        engine="mock",
        local_path=f"C:/synthetic/{identity}.gguf",
        manifest_json={
            "expected_sha256": {f"{identity}.gguf": digest},
            # A downloaded declaration is intentionally not evidence. One test
            # leaves the proof absent while keeping this claim in place.
            "input_modalities": ["text", "image"],
        },
        active=True,
    )
    profile = ModelProfile(
        id=f"profile_{identity}",
        model_install_id=install.id,
        name=identity.title(),
        role="chat",
        engine="mock",
    )
    session.add_all([install, profile])
    session.flush()
    if verified_vision:
        record_capability_evidence(
            session,
            install,
            settings,
            None,
            component_hashes=dict(install.manifest_json["expected_sha256"]),
            runtime_build="mock-vision",
            workflow_contract_version=None,
            details={"input_modalities": ["text", "image"]},
        )
    return profile


def _chat(
    session: Session,
    selected: ModelProfile,
    legacy_vision: ModelProfile | None,
) -> Chat:
    chat = Chat(
        id="chat_vision_inheritance",
        title="Vision inheritance",
        routing_mode="text",
        active_chat_profile_id=selected.id,
        active_vision_profile_id=legacy_vision.id if legacy_vision else None,
    )
    session.add(chat)
    session.flush()
    reconcile_legacy_workflow_compatibility(session)
    session.commit()
    return chat


def _native_text_family(
    session: Session,
    profile: ModelProfile,
) -> tuple[WorkflowFamily, WorkflowRevision]:
    family = WorkflowFamily(name="Native multimodal chat")
    definition = WorkflowDefinition(
        family=family,
        variant_key="text",
        name="Native multimodal text",
        operation="text",
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="mock",
        dependency_contract_sha256="d" * 64,
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="chat")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    activation = WorkflowActivation(
        workflow_revision_id=revision.id,
        resolver_version="resolver-v1",
        dependency_contract_sha256=revision.dependency_contract_sha256,
        binding_sha256="e" * 64,
        state="ready",
        is_active=True,
        details_json={"launch_sha256": "f" * 64},
    )
    slot = WorkflowDependencySlot(
        workflow_revision_id=revision.id,
        name="primary",
        resource_kind="model_profile",
        required=True,
        satisfaction="all_of",
        requirements_json=[{"key": "default", "constraints": {}}],
        contract_sha256="1" * 64,
        ordinal=0,
    )
    session.add_all([activation, slot])
    session.flush()
    session.add(
        WorkflowDependencyBinding(
            workflow_revision_id=revision.id,
            workflow_activation_id=activation.id,
            workflow_dependency_slot_id=slot.id,
            requirement_key="default",
            model_profile_id=profile.id,
            resource_identity_sha256="2" * 64,
        )
    )
    session.commit()
    return family, revision


async def _accept_vision_turn(client: AsyncClient, chat_id: str) -> dict[str, object]:
    uploaded = await client.post(
        "/api/artifacts",
        files={"file": ("pixel.png", ONE_PIXEL_PNG, "image/png")},
    )
    assert uploaded.status_code == 201
    accepted = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={
            "text": "What does this image show?",
            "mode": "text",
            "input_artifact_ids": [uploaded.json()["id"]],
        },
    )
    assert accepted.status_code == 202
    return cast(dict[str, object], accepted.json()["run"])


async def test_admission_uses_verified_selected_chat_profile_for_vision(
    client: AsyncClient,
    settings: Settings,
) -> None:
    with SessionLocal() as session:
        selected = _profile(session, settings, "selected", verified_vision=True)
        legacy = _profile(session, settings, "legacy", verified_vision=True)
        chat = _chat(session, selected, legacy)
        chat_id = chat.id
        selected_id = selected.id

    accepted_run = await _accept_vision_turn(client, chat_id)

    with SessionLocal() as session:
        run = session.get(Run, accepted_run["id"])
        assert run
        assert run.profile_id == selected_id
        assert run.vision_profile_id == selected_id


async def test_native_selected_chat_family_inherits_verified_vision_profile(
    client: AsyncClient,
    settings: Settings,
) -> None:
    with SessionLocal() as session:
        selected = _profile(session, settings, "selected", verified_vision=True)
        legacy = _profile(session, settings, "legacy", verified_vision=True)
        chat = _chat(session, selected, legacy)
        family, revision = _native_text_family(session, selected)
        chat_id = chat.id
        family_id = family.id
        revision_id = revision.id
        selected_id = selected.id

    chosen = await client.put(
        f"/api/chats/{chat_id}/workflow-selections/chat",
        json={"mode": "family", "workflow_family_id": family_id},
    )
    assert chosen.status_code == 200, chosen.json()
    accepted_run = await _accept_vision_turn(client, chat_id)

    with SessionLocal() as session:
        run = session.get(Run, accepted_run["id"])
        assert run
        assert run.workflow_revision_id == revision_id
        assert run.profile_id == selected_id
        assert run.vision_profile_id == selected_id


async def test_unverified_selected_profile_retains_verified_legacy_bridge(
    client: AsyncClient,
    settings: Settings,
) -> None:
    with SessionLocal() as session:
        selected = _profile(session, settings, "selected", verified_vision=False)
        legacy = _profile(session, settings, "legacy", verified_vision=True)
        chat = _chat(session, selected, legacy)
        chat_id = chat.id
        selected_id = selected.id
        legacy_id = legacy.id

    accepted_run = await _accept_vision_turn(client, chat_id)

    with SessionLocal() as session:
        run = session.get(Run, accepted_run["id"])
        assert run
        assert run.profile_id == selected_id
        assert run.vision_profile_id == legacy_id


async def test_declared_or_stale_vision_capability_never_enables_a_profile(
    client: AsyncClient,
    settings: Settings,
) -> None:
    with SessionLocal() as session:
        selected = _profile(session, settings, "selected", verified_vision=False)
        legacy = _profile(session, settings, "legacy", verified_vision=True)
        legacy_install = session.get(ModelInstall, legacy.model_install_id)
        assert legacy_install
        legacy_install.manifest_json = {
            **legacy_install.manifest_json,
            "expected_sha256": {"legacy.gguf": "c" * 64},
        }
        chat = _chat(session, selected, legacy)
        chat_id = chat.id
        selected_id = selected.id
        assert selected.model_install_id
        selected_install = session.get(ModelInstall, selected.model_install_id)
        assert selected_install
        assert selected_install.manifest_json["input_modalities"] == ["text", "image"]

    accepted_run = await _accept_vision_turn(client, chat_id)

    with SessionLocal() as session:
        run = session.get(Run, accepted_run["id"])
        assert run
        assert run.profile_id == selected_id
        assert run.vision_profile_id is None
