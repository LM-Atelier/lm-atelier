"""The turn API applies localized edit eligibility before it records a turn."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from test_automatic_image_edit_selection import GLOBAL, LOCALIZED, _edit_family

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowSelection,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    WorkPlan,
    WorkStep,
)


async def _source_picture(client: AsyncClient) -> str:
    content = io.BytesIO()
    Image.new("RGB", (32, 24), (80, 120, 160)).save(content, format="PNG")
    response = await client.post(
        "/api/artifacts", files={"file": ("source.png", content.getvalue(), "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def _durable_counts(session: Session) -> tuple[int, ...]:
    return tuple(
        session.scalar(select(func.count()).select_from(model)) or 0
        for model in (Message, Run, Job, WorkPlan, WorkStep)
    )


def _selection_mode(session: Session, chat_id: str) -> str:
    selection = session.scalar(
        select(ChatWorkflowSelection).where(
            ChatWorkflowSelection.chat_id == chat_id,
            ChatWorkflowSelection.selector_capability == "image",
        )
    )
    assert selection is not None
    return selection.mode


def _comfy_selection(
    app: FastAPI, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> AsyncMock:
    # Selection sees ComfyUI while the existing adapters and worker starts stay mocked.
    monkeypatch.setattr(settings, "media_engine", "comfyui")
    start_media = AsyncMock()
    monkeypatch.setattr(app.state.services.processes, "start_media", start_media)
    model_root = settings.model_dir / "picture-model"
    model_root.mkdir()
    with SessionLocal() as session:
        install = ModelInstall(
            name="Picture model",
            role="image",
            engine="comfyui",
            local_path=str(model_root),
            manifest_json={},
            active=True,
        )
        session.add(install)
        session.flush()
        profile = session.scalar(
            select(ModelProfile).where(
                ModelProfile.role == "image", ModelProfile.is_default.is_(True)
            )
        )
        assert profile is not None
        profile.engine = "comfyui"
        profile.model_install_id = install.id
        session.commit()
    return start_media


@pytest.mark.parametrize("automatic", ["chat-create", "chat-update", "turn-choice"])
async def test_localized_auto_refuses_strength_edits_before_durable_turn_writes(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    automatic: str,
) -> None:
    start_media = _comfy_selection(app, settings, monkeypatch)
    with SessionLocal() as session:
        revision = _edit_family(
            session, "Color changes", instruction=False, is_default=False, use_case="change color"
        )
        family_id = revision.definition.family_id
        session.commit()
    source_id = await _source_picture(client)
    created = await client.post("/api/chats", json={"title": "Color changes"})
    assert created.status_code == 201, created.text
    assert created.json()["active_image_profile_id"] == "__auto__"
    chat_id = created.json()["id"]
    if automatic != "chat-create":
        chosen = await client.put(
            f"/api/chats/{chat_id}/workflow-selections/image",
            json={"mode": "family", "workflow_family_id": family_id},
        )
        assert chosen.status_code == 200, chosen.text
    if automatic == "chat-update":
        updated = await client.patch(
            f"/api/chats/{chat_id}", json={"active_image_profile_id": "__auto__"}
        )
        assert updated.status_code == 200, updated.text
    with SessionLocal() as session:
        before = _durable_counts(session)
        mode = _selection_mode(session, chat_id)
    assert mode == ("family" if automatic == "turn-choice" else "automatic")

    request: dict[str, object] = {
        "text": LOCALIZED,
        "mode": "image",
        "input_artifact_ids": [source_id],
    }
    if automatic == "turn-choice":
        request["workflow_selection"] = {"selector_capability": "image", "mode": "automatic"}
    async with app.state.services.scheduler.lease("primary"):
        refused = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert refused.status_code == 422, refused.text
        assert refused.json()["code"] == "workflow-instruction-edit-required"
        assert "instruction-edit workflow" in refused.json()["detail"]
        with SessionLocal() as session:
            assert _durable_counts(session) == before
            assert _selection_mode(session, chat_id) == mode
        start_media.assert_not_called()


@pytest.mark.parametrize("choice", ["global", "explicit-revision", "instruction"])
async def test_the_turn_api_preserves_eligible_edit_choices(
    app: FastAPI,
    client: AsyncClient,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    choice: str,
) -> None:
    start_media = _comfy_selection(app, settings, monkeypatch)
    with SessionLocal() as session:
        revision = _edit_family(
            session,
            "Picture edits",
            instruction=choice == "instruction",
            is_default=False,
            use_case="change color watercolor",
        )
        revision_id = revision.id
        # No model declares an instruction capability for this graph.
        assert not revision.dependencies_json
        session.commit()
    source_id = await _source_picture(client)
    created = await client.post("/api/chats", json={"title": "Picture edits"})
    assert created.status_code == 201, created.text
    chat_id = created.json()["id"]
    request: dict[str, object] = {
        "text": GLOBAL if choice == "global" else LOCALIZED,
        "mode": "image",
        "input_artifact_ids": [source_id],
    }
    if choice == "explicit-revision":
        request["workflow_selection"] = {
            "selector_capability": "image",
            "mode": "revision",
            "workflow_revision_id": revision_id,
        }
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(f"/api/chats/{chat_id}/turns", json=request)
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["run"]["workflow_revision_id"] == revision_id
        with SessionLocal() as session:
            assert _selection_mode(session, chat_id) == "automatic"
        start_media.assert_not_called()
