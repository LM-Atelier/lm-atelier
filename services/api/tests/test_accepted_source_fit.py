"""A saved extension keeps its source identity, workflow and retention edges."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session
from test_source_fit_image import artifact_session as artifact_session
from test_source_fit_image import ingest, png
from test_workflow_source_geometry import composited_graph

from local_lm.accepted_turn_context import (
    _digest,
    accepted_context,
    save_accepted_context,
)
from local_lm.artifact_library import referenced_artifact_ids
from local_lm.artifacts import ArtifactStore
from local_lm.domain import utcnow
from local_lm.models import (
    Chat,
    Message,
    Run,
    RunContextArtifact,
    RunContextSnapshot,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.source_fit_image import capture_source_fit_image, replay_source_fit_image
from local_lm.source_fit_recipe import SourceExtensionRecipe, plan_source_extension
from local_lm.vision import VisionSamplingPolicy


def make_run(pair: tuple[ArtifactStore, Session]) -> tuple[Run, SourceExtensionRecipe]:
    store, session = pair
    source = ingest(pair, png(orientation=6))
    image = capture_source_fit_image(session, store, source)
    workflow = composited_graph()
    recipe = plan_source_extension(
        image, canvas_width=4, canvas_height=5, api_graph=workflow, save_node_id="save"
    )
    chat = Chat(title="Source geometry fixture")
    definition = WorkflowDefinition(name="Source geometry fixture", operation="image_to_image")
    session.add_all([chat, definition])
    session.flush()
    messages = [
        Message(chat_id=chat.id, role="user"),
        Message(chat_id=chat.id, role="assistant"),
    ]
    session.add_all(messages)
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=workflow,
        input_schema_json={},
        dependencies_json={},
    )
    session.add(revision)
    session.flush()
    run = Run(
        chat_id=chat.id,
        user_message_id=messages[0].id,
        assistant_message_id=messages[1].id,
        workflow_revision_id=revision.id,
        operation="image_to_image",
        standalone_prompt="Extend the neutral color grid",
        settings_json={},
        provenance_json={},
    )
    session.add(run)
    session.flush()
    return run, recipe


def save(
    session: Session, run: Run, recipe: SourceExtensionRecipe | None, **overrides: Any
) -> None:
    options: dict[str, Any] = {
        "messages": [{"role": "user", "content": "Extend the neutral color grid"}],
        "sources": [run.user_message_id],
        "artifact_ids": {recipe.image.source_artifact_id} if recipe else set(),
        "input_artifact_ids": [recipe.image.source_artifact_id] if recipe else [],
        "visual_artifact_ids": [],
        "strict_artifact_ids": [],
        "vision_settings": {},
        "vision_sampling": VisionSamplingPolicy(
            max_images=1,
            max_video_frames=3,
            max_frame_dimension=256,
        ),
        "vision_bridge_max_tokens": 128,
        "context_limit": 512,
        "chat_engine": "mock",
        "media_engine": "comfyui",
        "media_prompt": "Extend the neutral color grid",
        "context_artifact_ids": set(),
        "source_fit": recipe,
    }
    options.update(overrides)
    save_accepted_context(session, run, **options)


def test_saved_recipe_retains_both_images_and_replays_exact_prepared_bytes(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    store, session = artifact_session
    run, recipe = make_run(artifact_session)
    save(session, run, recipe)
    session.commit()
    snapshot = accepted_context(session, run)
    assert snapshot is not None
    assert snapshot.source_fit == recipe
    expected_ids = {recipe.image.source_artifact_id, recipe.image.prepared_artifact_id}
    assert expected_ids <= set(snapshot.artifact_ids)
    assert expected_ids <= referenced_artifact_ids(session)
    assert set(session.scalars(select(RunContextArtifact.artifact_id))) == expected_ids
    prepared = replay_source_fit_image(
        session,
        store,
        snapshot.source_fit.image,
        selected_source_id=snapshot.input_artifact_ids[0],
    )
    assert (prepared.width, prepared.height) == (2, 3)
    snapshot.source_fit.route(composited_graph())
    assert snapshot.source_fit.margins() == {"top": 1, "right": 1, "bottom": 1, "left": 1}
    assert "content" not in snapshot.source_fit.model_dump()


@pytest.mark.parametrize(
    "corruption", ["first_input", "no_workflow", "operation", "engine", "graph"]
)
def test_save_refuses_a_recipe_bound_to_different_execution_inputs(
    artifact_session: tuple[ArtifactStore, Session],
    corruption: str,
) -> None:
    _, session = artifact_session
    run, recipe = make_run(artifact_session)
    options: dict[str, Any] = {}
    if corruption == "first_input":
        unrelated = ingest(artifact_session, png())
        options["input_artifact_ids"] = [unrelated.id, recipe.image.source_artifact_id]
    elif corruption == "no_workflow":
        run.workflow_revision_id = None
    elif corruption == "operation":
        run.operation = "image_to_video"
    elif corruption == "engine":
        options["media_engine"] = "mock"
    else:
        revision = session.get(WorkflowRevision, run.workflow_revision_id)
        assert revision is not None
        revision.api_graph_json = {}
    session.flush()
    with pytest.raises(ValueError, match="source_fit"):
        save(session, run, recipe, **options)
    assert session.get(RunContextSnapshot, run.id) is None


@pytest.mark.parametrize("corruption", ["payload", "retention", "removed_message"])
def test_saved_recipe_cannot_replay_after_its_context_binding_is_lost(
    artifact_session: tuple[ArtifactStore, Session],
    corruption: str,
) -> None:
    _, session = artifact_session
    run, recipe = make_run(artifact_session)
    save(session, run, recipe)
    session.commit()
    if corruption == "payload":
        row = session.get(RunContextSnapshot, run.id)
        assert row is not None
        payload = deepcopy(row.payload_json)
        payload["source_fit"]["canvas_width"] = 6
        row.payload_json = payload
    elif corruption == "retention":
        session.execute(
            delete(RunContextArtifact).where(
                RunContextArtifact.run_id == run.id,
                RunContextArtifact.artifact_id == recipe.image.prepared_artifact_id,
            )
        )
    else:
        message = session.get(Message, run.user_message_id)
        assert message is not None
        message.content_removed_at = utcnow()
    session.commit()
    with pytest.raises(ValueError, match="Accepted conversation context is unavailable"):
        accepted_context(session, run)


def test_snapshot_without_new_optional_field_still_uses_its_original_digest(
    artifact_session: tuple[ArtifactStore, Session],
) -> None:
    _, session = artifact_session
    run, _ = make_run(artifact_session)
    save(session, run, None)
    session.commit()
    row = session.get(RunContextSnapshot, run.id)
    assert row is not None
    payload = deepcopy(row.payload_json)
    payload.pop("source_fit", None)
    row.payload_json = payload
    row.sha256 = _digest(payload)
    run.provenance_json = {**run.provenance_json, "accepted_context_sha256": row.sha256}
    session.commit()
    snapshot = accepted_context(session, run)
    assert snapshot is not None
    assert snapshot.source_fit is None
