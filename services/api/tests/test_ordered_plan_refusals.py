"""What an ordered plan refuses before it writes anything.

An Auto turn that asks for several kinds of work in order is checked as a
whole before any of it is stored. Each refusal below names its own reason and
leaves the chat with no message and no plan.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    GenerationPreset,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowRevision,
)

STORY_THEN_PICTURE = "Write a short story about a paper boat, then create an image based on it"
STORY_PICTURE_VIDEO = (
    "Write a short story about a paper boat, then create an image based on it, "
    "then animate the image into a video"
)


async def _refused(
    client: AsyncClient,
    status: int,
    code: str,
    *,
    text: str = STORY_THEN_PICTURE,
    **fields: Any,
) -> str:
    chat = (await client.post("/api/chats", json={"title": "Ordered refusal"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={"text": text, "mode": "auto", "confirm_media": True, **fields},
    )
    assert response.status_code == status, response.text
    assert response.json()["code"] == code
    assert (await client.get(f"/api/chats/{chat['id']}")).json()["messages"] == []
    assert (await client.get("/api/work-plans", params={"chat_id": chat["id"]})).json() == []
    detail: str = response.json()["detail"]
    return detail


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        (
            {
                "workflow_selection": {"selector_capability": "image", "mode": "default"},
                "workflow_revision_id": "no-such-revision",
            },
            "Choose one turn workflow selection or exact revision.",
        ),
        (
            {"workflow_revision_id": "no-such-revision"},
            "The selected turn workflow revision no longer exists.",
        ),
        (
            {"output_count": 2},
            "A heterogeneous ordered plan cannot multiply all steps. "
            "Request variations in a separate media turn.",
        ),
        ({"ordered_settings": {"audio": {}}}, "Ordered settings contain an unsupported role."),
        (
            {"settings": {"max_tokens": 64}},
            "Ordered plans accept role-specific ordered_settings, not shared settings.",
        ),
    ],
)
async def test_a_request_the_plan_cannot_honour_is_refused_whole(
    client: AsyncClient, fields: dict[str, Any], reason: str
) -> None:
    assert await _refused(client, 422, "turn-invalid", **fields) == reason


@pytest.mark.parametrize(
    ("fields", "reason"),
    [
        ({"profile_id": "no-such-model"}, "The selected turn model no longer exists."),
        ({"preset_id": "no-such-preset"}, "The selected turn preset no longer exists."),
    ],
)
async def test_a_model_or_preset_that_is_gone_is_named(
    client: AsyncClient, fields: dict[str, Any], reason: str
) -> None:
    assert await _refused(client, 404, "turn-subject-not-found", **fields) == reason


@pytest.mark.parametrize(
    ("field", "reason"),
    [
        (
            "workflow_revision_id",
            "The selected turn workflow cannot perform an operation in this plan.",
        ),
        ("profile_id", "The selected turn model cannot perform an operation in this plan."),
        ("preset_id", "The selected turn preset cannot apply to an operation in this plan."),
    ],
)
async def test_a_choice_for_work_the_plan_does_not_do_is_refused(
    client: AsyncClient, field: str, reason: str
) -> None:
    # A story and then a picture: nothing in the plan makes a video.
    with SessionLocal() as session:
        family = WorkflowFamily(name="Ordered video workflow")
        definition = WorkflowDefinition(
            family=family, variant_key="video", name="Video variant", operation="text_to_video"
        )
        revision = WorkflowRevision(definition=definition, version=1, engine="mock", trusted=True)
        profile = ModelProfile(name="Ordered video model", role="video", engine="mock")
        preset = GenerationPreset(name="Ordered video preset", role="video")
        session.add_all([family, definition, revision, profile, preset])
        session.commit()
        chosen = {
            "workflow_revision_id": revision.id,
            "profile_id": profile.id,
            "preset_id": preset.id,
        }[field]
    assert await _refused(client, 422, "turn-invalid", **{field: chosen}) == reason


async def test_every_step_counts_against_the_pending_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import local_lm.orchestrator as orchestration

    # One free place passes the chat's own check; the plan's two steps do not fit in it.
    monkeypatch.setattr(orchestration, "MAX_PENDING_WORK_PER_CHAT", 1)
    assert (
        await _refused(client, 422, "turn-invalid")
        == "This ordered request would exceed the limit of 1 pending items in one chat."
    )


async def test_the_first_media_step_needs_a_picture_to_start_from(client: AsyncClient) -> None:
    with SessionLocal() as session:
        session.add(
            Artifact(
                id="ordered-video-input",
                sha256="d" * 64,
                kind="video",
                media_type="video/mp4",
                size_bytes=4,
                relative_path="ordered-video-input",
            )
        )
        session.commit()
    assert (
        await _refused(
            client,
            422,
            "turn-invalid",
            text="Create an image of a paper boat, then write a short story about it",
            input_artifact_ids=["ordered-video-input"],
        )
        == "The first media step requires an image-compatible input."
    )


@pytest.mark.parametrize(
    ("limit", "reason"),
    [
        (
            "max_media_plan_work_units",
            "This ordered plan is too large to queue safely. "
            "Reduce its media steps, resolution, frames, or generation steps.",
        ),
        ("max_media_plan_estimated_bytes", "This ordered plan has an unsafe storage estimate."),
        (
            "max_media_plan_duration_seconds",
            "This ordered plan requests too much total video duration.",
        ),
    ],
)
async def test_a_plan_over_a_whole_plan_limit_is_refused(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    limit: str,
    reason: str,
) -> None:
    monkeypatch.setattr(app.state.services.engines.settings, limit, 0)
    assert await _refused(client, 422, "turn-invalid", text=STORY_PICTURE_VIDEO) == reason


async def test_a_plan_needs_room_for_everything_it_will_write(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_lm.orchestrator.shutil.disk_usage", lambda _path: SimpleNamespace(free=0)
    )
    assert (
        await _refused(client, 422, "turn-invalid")
        == "There is not enough free storage for this ordered plan."
    )
