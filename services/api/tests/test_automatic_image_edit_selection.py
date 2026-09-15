"""Auto sends a specific image edit to an instruction-edit workflow when one is ready."""

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from local_lm.db import Base
from local_lm.domain import Operation
from local_lm.image_edit_strength import EditScope, estimate_image_edit_strength
from local_lm.models import (
    Chat,
    ChatWorkflowSelection,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.orchestrator import ConversationOrchestrator

LOCALIZED = "Change only the sweater to green."
GLOBAL = "Turn it into a watercolor painting."


@pytest.fixture
def session() -> Generator[Session]:
    engine = create_engine("sqlite://")
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as value:
        yield value
    engine.dispose()


def _node(node_id: int, node_type: str, *outputs: str) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "outputs": [{"name": kind.lower(), "type": kind} for kind in outputs],
    }


def _graphs(
    *, instruction: bool, unused_branch: bool = False, ui_only_saved_branch: bool = False
) -> tuple[dict[str, Any], dict[str, Any]]:
    if instruction:
        # The source picture feeds the encoder that turns the words into conditioning.
        ui = {
            "nodes": [
                _node(1, "LoadImage", "IMAGE"),
                _node(2, "EditEncoder", "CONDITIONING"),
                _node(3, "Sampler", "LATENT"),
                _node(9, "SaveImage"),
            ],
            "links": [
                [10, 1, 0, 2, 0, "IMAGE"],
                [11, 2, 0, 3, 0, "CONDITIONING"],
                [12, 3, 0, 9, 0, "LATENT"],
            ],
        }
    else:
        # The source only becomes the starting latent that denoise redraws.
        ui = {
            "nodes": [
                _node(1, "LoadImage", "IMAGE"),
                _node(2, "VAEEncode", "LATENT"),
                _node(4, "CLIPTextEncode", "CONDITIONING"),
                _node(3, "Sampler", "LATENT"),
                _node(9, "SaveImage"),
            ],
            "links": [
                [10, 1, 0, 2, 0, "IMAGE"],
                [11, 2, 0, 3, 3, "LATENT"],
                [12, 4, 0, 3, 1, "CONDITIONING"],
                [13, 3, 0, 9, 0, "LATENT"],
            ],
        }
    # The executable graph the compiler writes for the same nodes.
    api: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    if instruction:
        api["2"] = {"class_type": "EditEncoder", "inputs": {"image": ["1", 0], "text": "${prompt}"}}
        api["3"] = {"class_type": "Sampler", "inputs": {"positive": ["2", 0]}}
    else:
        api["2"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0]}}
        api["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": "${prompt}"}}
        api["3"] = {
            "class_type": "Sampler",
            "inputs": {"positive": ["4", 0], "latent_image": ["2", 0], "denoise": "${denoise}"},
        }
    if unused_branch:
        # The source also reaches a conditioning node whose result nothing uses.
        ui["nodes"].append(_node(20, "EditEncoder", "CONDITIONING"))
        ui["links"].append([20, 1, 0, 20, 0, "IMAGE"])
        api["20"] = {"class_type": "EditEncoder", "inputs": {"image": ["1", 0]}}
    if ui_only_saved_branch:
        # A saved conditioning branch drawn in the UI graph that the executable
        # graph does not hold, so it never runs.
        ui["nodes"] += [
            _node(20, "EditEncoder", "CONDITIONING"),
            _node(21, "Sampler", "LATENT"),
            _node(23, "SaveImage"),
        ]
        ui["links"] += [
            [20, 1, 0, 20, 0, "IMAGE"],
            [21, 20, 0, 21, 0, "CONDITIONING"],
            [22, 21, 0, 23, 0, "LATENT"],
        ]
    return ui, api


def _edit_family(
    session: Session,
    name: str,
    *,
    instruction: bool,
    is_default: bool,
    use_case: str,
    unused_branch: bool = False,
    ui_only_saved_branch: bool = False,
) -> WorkflowRevision:
    ui, api = _graphs(
        instruction=instruction,
        unused_branch=unused_branch,
        ui_only_saved_branch=ui_only_saved_branch,
    )
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key="edit",
        name=f"{name} edit",
        operation=Operation.IMAGE_TO_IMAGE.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        ui_graph_json=ui,
        api_graph_json=api,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family, selector_capability="image", is_default=is_default
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _orchestrator() -> ConversationOrchestrator:
    return ConversationOrchestrator(
        engines=SimpleNamespace(
            settings=SimpleNamespace(media_engine="comfyui", chat_engine="mock"),
            chat=SimpleNamespace(cancel=AsyncMock()),
            media=SimpleNamespace(cancel=AsyncMock()),
        ),
        artifacts=Mock(),
        events=SimpleNamespace(publish=AsyncMock()),
        scheduler=SimpleNamespace(publish_job=AsyncMock()),
        processes=SimpleNamespace(statuses=Mock(return_value=[])),
        session_factory=Mock(),
    )


def _chat(session: Session, mode: str, family_id: str | None = None) -> Chat:
    chat = Chat(title="Edits")
    session.add(chat)
    session.flush()
    session.add(
        ChatWorkflowSelection(
            chat_id=chat.id,
            selector_capability="image",
            mode=mode,
            workflow_family_id=family_id,
        )
    )
    session.flush()
    return chat


def _chosen(session: Session, chat: Chat, prompt: str) -> str | None:
    _profile, selection, revision = _orchestrator()._profile_and_workflow_for_operation(
        session, chat, Operation.IMAGE_TO_IMAGE, prompt
    )
    assert revision is None or selection["workflow_revision_id"] == revision.id
    return revision.id if revision is not None else None


def test_auto_sends_a_specific_edit_to_the_instruction_workflow(session: Session) -> None:
    # The strength edit is the default and matches the request's words, so it
    # would win on every other ranking.
    strength = _edit_family(
        session,
        "Sweater recolor",
        instruction=False,
        is_default=True,
        use_case="change the sweater color",
    )
    instruction = _edit_family(
        session, "Picture edits", instruction=True, is_default=False, use_case="studio photos"
    )
    chat = _chat(session, "automatic")
    assert estimate_image_edit_strength(LOCALIZED).scope == EditScope.LOCALIZED

    assert _chosen(session, chat, LOCALIZED) == instruction.id
    assert _chosen(session, chat, LOCALIZED) != strength.id


def test_an_unused_conditioning_branch_does_not_promote_a_strength_edit(
    session: Session,
) -> None:
    strength = _edit_family(
        session,
        "Sweater recolor",
        instruction=False,
        is_default=True,
        use_case="change the sweater color",
        unused_branch=True,
    )
    instruction = _edit_family(
        session, "Picture edits", instruction=True, is_default=False, use_case="studio photos"
    )
    chat = _chat(session, "automatic")

    assert _chosen(session, chat, LOCALIZED) == instruction.id
    assert _chosen(session, chat, LOCALIZED) != strength.id


def test_a_saved_branch_only_in_the_ui_graph_does_not_promote_a_strength_edit(
    session: Session,
) -> None:
    strength = _edit_family(
        session,
        "Sweater recolor",
        instruction=False,
        is_default=True,
        use_case="change the sweater color",
        ui_only_saved_branch=True,
    )
    instruction = _edit_family(
        session, "Picture edits", instruction=True, is_default=False, use_case="studio photos"
    )
    chat = _chat(session, "automatic")

    assert _chosen(session, chat, LOCALIZED) == instruction.id
    assert _chosen(session, chat, LOCALIZED) != strength.id


def test_auto_keeps_the_ordinary_ranking_for_a_whole_picture_transformation(
    session: Session,
) -> None:
    strength = _edit_family(
        session,
        "Painterly restyle",
        instruction=False,
        is_default=True,
        use_case="watercolor painting",
    )
    _edit_family(
        session, "Picture edits", instruction=True, is_default=False, use_case="studio photos"
    )
    chat = _chat(session, "automatic")
    assert estimate_image_edit_strength(GLOBAL).scope == EditScope.GLOBAL

    assert _chosen(session, chat, GLOBAL) == strength.id


def test_a_chosen_workflow_is_never_replaced_by_an_instruction_edit(session: Session) -> None:
    strength = _edit_family(
        session,
        "Sweater recolor",
        instruction=False,
        is_default=True,
        use_case="change the sweater color",
    )
    _edit_family(
        session, "Picture edits", instruction=True, is_default=False, use_case="studio photos"
    )
    chosen = _chat(session, "family", strength.definition.family_id)

    assert _chosen(session, chosen, LOCALIZED) == strength.id
