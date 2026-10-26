"""The app's own choice of a video workflow follows whether the turn brings a guide video.

A workflow reads a guide video through its graph. Chosen for a turn that brings
none, such a workflow waits for a video that never comes; a turn that brings one
looks first for a workflow that reads it.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.orm import Session
from test_automatic_image_edit_selection import _orchestrator
from test_automatic_image_edit_selection import session as session

from local_lm.domain import Operation
from local_lm.media_references import videos_required
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.schemas import TurnRequest
from local_lm.setup_readiness import _workflow_check
from local_lm.turn_workflow_kinds import ANY_TURN, WHOLE_PICTURE_EDIT, TurnWorkflowKinds

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"prompt": {"type": "string"}}}
PROMPT = "The leaves drift slowly across the frame."


def _graph(*video_slots: str) -> dict[str, Any]:
    """A picture-to-video graph that also reads the named video slots."""
    graph: dict[str, Any] = {
        "picture": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "encode": {"class_type": "CLIPTextEncode", "inputs": {"text": "${prompt}"}},
        "save": {"class_type": "SaveVideo", "inputs": {"images": ["picture", 0]}},
    }
    for index, slot in enumerate(video_slots):
        graph[f"clip_{index}"] = {"class_type": "LoadVideo", "inputs": {"file": f"${{{slot}}}"}}
    return graph


@pytest.mark.parametrize(
    "graph,required",
    [
        ({}, 0),
        (_graph(), 0),
        (_graph("input_video"), 1),
        (_graph("input_videos"), 1),
        (_graph("input_video_0"), 1),
        (_graph("input_video_2"), 3),
        (_graph("input_video", "input_video_1"), 2),
        ({"a": ["${input_video_0_style}", {"b": "${input_video}x"}]}, 0),
    ],
    ids=[
        "empty",
        "pictures-only",
        "first",
        "whole-list",
        "numbered-first",
        "numbered-third",
        "first-and-second",
        "look-alikes",
    ],
)
def test_a_graph_needs_one_video_per_slot_up_to_its_highest(
    graph: dict[str, Any], required: int
) -> None:
    assert videos_required(graph) == required


def test_a_turn_that_brings_videos_is_judged_by_the_graph() -> None:
    plain, guided = _graph(), _graph("input_video")
    second_only, pair = _graph("input_video_1"), _graph("input_video", "input_video_1")

    without = TurnWorkflowKinds(videos=0)
    assert without.admits_workflow(SCHEMA, plain)
    assert not without.admits_workflow(SCHEMA, guided)
    assert not without.prefers_workflow(SCHEMA, plain)
    assert not without.brings_input

    one = TurnWorkflowKinds(videos=1)
    # A workflow that reads no video stays a choice, so the turn is refused for
    # that reason rather than told nothing is installed.
    assert one.admits_workflow(SCHEMA, plain)
    assert one.admits_workflow(SCHEMA, guided)
    assert not one.admits_workflow(SCHEMA, second_only)
    assert one.prefers_workflow(SCHEMA, guided)
    assert not one.prefers_workflow(SCHEMA, plain)
    assert one.brings_input

    both = TurnWorkflowKinds(videos=2)
    assert both.admits_workflow(SCHEMA, pair)
    assert both.prefers_workflow(SCHEMA, pair)
    # The second video would have no slot, or the first.
    assert not both.admits_workflow(SCHEMA, guided)
    assert not both.admits_workflow(SCHEMA, second_only)


def test_a_turn_videos_play_no_part_in_is_judged_as_before() -> None:
    guided = _graph("input_video")
    assert ANY_TURN.admits_workflow(SCHEMA, guided)
    assert not ANY_TURN.prefers_workflow(SCHEMA, guided)
    assert ANY_TURN.admits(SCHEMA)
    assert WHOLE_PICTURE_EDIT.serves(SCHEMA)
    with pytest.raises(ValueError, match="judged from its graph"):
        TurnWorkflowKinds(videos=1).admits(SCHEMA)
    with pytest.raises(ValueError, match="judged from its graph"):
        TurnWorkflowKinds(videos=0).prefers(SCHEMA)


def _family(
    session: Session, name: str, graph: dict[str, Any], *, is_default: bool
) -> WorkflowRevision:
    # A family with no use case is chosen automatically only as the default.
    family = WorkflowFamily(name=name, use_case="studio clips")
    definition = WorkflowDefinition(
        family=family,
        variant_key="video",
        name=f"{name} video",
        operation=Operation.IMAGE_TO_VIDEO.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json=SCHEMA,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family, selector_capability="video", is_default=is_default
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _workflow(
    session: Session,
    name: str,
    graph: dict[str, Any],
    created: datetime,
    *,
    install_id: str | None = None,
) -> WorkflowRevision:
    """A video workflow outside any family, bound to a model when an install is named."""
    definition = WorkflowDefinition(
        name=name, operation=Operation.IMAGE_TO_VIDEO.value, created_at=created
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json=SCHEMA,
        dependencies_json={"model_install_ids": [install_id]} if install_id else {},
        trusted=True,
        created_at=created,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _chat(session: Session, mode: str | None) -> Chat:
    chat = Chat(title="Guide videos")
    session.add(chat)
    session.flush()
    if mode is not None:
        session.add(ChatWorkflowSelection(chat_id=chat.id, selector_capability="video", mode=mode))
        session.flush()
    return chat


def _chosen(
    session: Session, chat: Chat, guide_videos: int | None, *, profile_id: str | None = None
) -> WorkflowRevision | None:
    return _orchestrator()._execution_for_turn(
        session,
        chat,
        Operation.IMAGE_TO_VIDEO,
        PROMPT,
        TurnRequest(text=PROMPT, mode="video", profile_id=profile_id),
        guide_videos=guide_videos,
    )[2]


def test_auto_chooses_a_workflow_that_reads_the_guide_video_a_turn_brings(
    session: Session,
) -> None:
    plain = _family(session, "Plain", _graph(), is_default=True)
    guided = _family(session, "Guided", _graph("input_video"), is_default=False)
    chat = _chat(session, "automatic")

    assert _chosen(session, chat, guide_videos=1) == guided
    assert _chosen(session, chat, guide_videos=0) == plain
    # A caller that says nothing about videos is ranked as before.
    assert _chosen(session, chat, guide_videos=None) == plain


def test_auto_passes_over_a_workflow_waiting_for_a_video_the_turn_does_not_bring(
    session: Session,
) -> None:
    guided = _family(session, "Guided", _graph("input_video"), is_default=True)
    plain = _family(session, "Plain", _graph(), is_default=False)
    chat = _chat(session, "automatic")

    assert _chosen(session, chat, guide_videos=0) == plain
    assert _chosen(session, chat, guide_videos=1) == guided
    assert _chosen(session, chat, guide_videos=None) == guided


def test_the_newest_installed_workflow_gives_way_to_one_that_fits_the_videos(
    session: Session,
) -> None:
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 2, 1, tzinfo=UTC)
    plain = _workflow(session, "Plain", _graph(), older)
    guided = _workflow(session, "Guided", _graph("input_video"), newer)
    chat = _chat(session, None)

    assert _chosen(session, chat, guide_videos=0) == plain
    assert _chosen(session, chat, guide_videos=1) == guided
    assert _chosen(session, chat, guide_videos=None) == guided


def test_a_workflow_that_reads_the_video_comes_before_one_bound_to_the_model(
    session: Session,
) -> None:
    install = ModelInstall(
        name="Neutral video model",
        role="video",
        engine="comfyui",
        local_path="neutral-video-model",
        active=True,
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Neutral video model", role="video", engine="comfyui", model_install_id=install.id
    )
    session.add(profile)
    session.flush()
    older = datetime(2026, 1, 1, tzinfo=UTC)
    newer = datetime(2026, 2, 1, tzinfo=UTC)
    guided = _workflow(session, "Guided", _graph("input_video"), older)
    bound = _workflow(session, "Bound", _graph(), newer, install_id=install.id)
    chat = _chat(session, None)

    # The turn names the model, so a workflow bound to it would otherwise come first.
    assert _chosen(session, chat, guide_videos=1, profile_id=profile.id) == guided
    assert _chosen(session, chat, guide_videos=0, profile_id=profile.id) == bound
    assert _chosen(session, chat, guide_videos=None, profile_id=profile.id) == bound


def _readiness_workflow(
    session: Session, name: str, graph: dict[str, Any], updated: datetime, install_id: str
) -> WorkflowRevision:
    definition = WorkflowDefinition(
        name=name, operation=Operation.IMAGE_TO_VIDEO.value, updated_at=updated
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        api_graph_json=graph,
        input_schema_json=SCHEMA,
        dependencies_json={"model_install_ids": [install_id]},
        trusted=True,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    # Attaching the revision stamps the definition as just changed; set the
    # time it is ranked by afterwards, explicitly.
    session.execute(
        update(WorkflowDefinition)
        .where(WorkflowDefinition.id == definition.id)
        .values(updated_at=updated)
    )
    session.expire(definition)
    return revision


def test_the_setup_check_prefers_a_workflow_its_single_picture_can_run(
    session: Session,
) -> None:
    """The quick test hands a workflow one picture, so a guide-video workflow would fail it."""
    install = ModelInstall(
        name="Neutral video model",
        role="video",
        engine="comfyui",
        local_path="neutral-video-model",
        active=True,
    )
    session.add(install)
    session.flush()
    guided = _readiness_workflow(
        session, "Guided", _graph("input_video"), datetime(2026, 2, 1, tzinfo=UTC), install.id
    )

    selected, check = _workflow_check(session, "video", install)
    # The only ready workflow is still the role's workflow.
    assert selected == guided
    assert check.code == "workflow_ready"

    plain = _readiness_workflow(
        session, "Plain", _graph(), datetime(2026, 1, 1, tzinfo=UTC), install.id
    )
    selected, check = _workflow_check(session, "video", install)
    assert selected == plain
    assert check.code == "workflow_ready"
