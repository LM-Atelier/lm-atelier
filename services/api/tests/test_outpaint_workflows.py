"""Extending a picture past its edge, and the margins that say how far."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.outpaint_workflows import (
    MAX_MARGIN_FRACTION,
    OUTPAINT_SCHEMA_KIND,
    OUTPAINT_SETTING_KEY,
    graph_can_outpaint,
    normalize_margins,
    workflow_declares_outpaint,
)


def test_a_graph_that_pads_before_sampling_can_outpaint() -> None:
    graph = {"1": {"class_type": "ImagePadForOutpaint"}, "2": {"class_type": "KSampler"}}

    assert graph_can_outpaint(graph) is True


def test_an_ordinary_editor_cannot() -> None:
    assert graph_can_outpaint({"1": {"class_type": "KSampler"}}) is False


def test_the_declaration_is_what_the_tool_looks_for() -> None:
    schema = {
        "type": "object",
        "properties": {
            OUTPAINT_SETTING_KEY: {"type": "object", "x-lm-atelier-kind": OUTPAINT_SCHEMA_KIND}
        },
    }

    assert workflow_declares_outpaint(schema) is True
    assert workflow_declares_outpaint({"properties": {OUTPAINT_SETTING_KEY: {}}}) is False


def test_margins_are_read_per_side() -> None:
    assert normalize_margins({"top": 0.25, "right": 0.5}) == {
        "top": 0.25,
        "right": 0.5,
        "bottom": 0.0,
        "left": 0.0,
    }


def test_extending_by_nothing_is_refused() -> None:
    """It would spend a generation to return the picture that was already there."""
    with pytest.raises(ValueError, match="would not change"):
        normalize_margins({"top": 0, "right": 0, "bottom": 0, "left": 0})


@pytest.mark.parametrize(
    "margins",
    [
        {"top": -0.1},
        {"top": MAX_MARGIN_FRACTION + 0.1},
        {"top": True},
        {"top": "half"},
        "not a mapping",
        None,
    ],
)
def test_a_margin_that_is_not_a_fraction_refuses(margins: Any) -> None:
    with pytest.raises(ValueError):
        normalize_margins(margins)


def test_the_bound_is_where_the_new_region_dwarfs_the_picture() -> None:
    # At the limit the extension is twice the source on that side, which is
    # already most of what the result will be.
    assert normalize_margins({"left": MAX_MARGIN_FRACTION})["left"] == MAX_MARGIN_FRACTION


async def test_a_turn_refuses_margins_the_contract_would_not_accept(client) -> None:
    """The gap: this contract existed and nothing called it.

    Margins arrive as an ordinary object setting, and the schema layer only
    bounds a value's size and nesting - it has no opinion about the numbers
    inside. A negative margin, a margin of nine hundred, and a margin of
    "lots" were all accepted and handed to a workflow that would do something
    arbitrary with each.
    """
    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Outpainter", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            api_graph_json={"1": {"class_type": "ImagePadForOutpaint"}},
            input_schema_json={
                "type": "object",
                "properties": {
                    OUTPAINT_SETTING_KEY: {
                        "type": "object",
                        "x-lm-atelier-kind": OUTPAINT_SCHEMA_KIND,
                        # The default is what makes a schema property a
                        # setting rather than a runtime binding, and the
                        # compiler emits one. Without it the turn refuses with
                        # "unsupported settings" and every assertion below
                        # passes for the wrong reason.
                        "default": {"top": 0, "right": 0, "bottom": 0, "left": 0},
                    }
                },
            },
            dependencies_json={},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()

    source = (
        await client.post(
            "/api/artifacts",
            files={"file": ("extend.png", b"source-image", "image/png")},
        )
    ).json()
    chat = (await client.post("/api/chats", json={"title": "Extend"})).json()

    async def apply(margins: object) -> int:
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "extend the scene",
                "mode": "image",
                "input_artifact_ids": [source["id"]],
                "settings": {OUTPAINT_SETTING_KEY: margins},
            },
        )
        return response.status_code

    assert await apply({"top": -0.5}) == 422
    assert await apply({"top": 900}) == 422
    assert await apply({"top": "lots"}) == 422
    assert await apply({}) == 422
    # And the one that is actually askable still is.
    assert await apply({"top": 0.25}) == 202
