"""Extending a picture past its edge, and the margins that say how far."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_lm.outpaint_workflows import (
    MAX_MARGIN_FRACTION,
    OUTPAINT_SCHEMA_KIND,
    OUTPAINT_SETTING_KEY,
    graph_can_outpaint,
    margin_pixels,
    normalize_margins,
    oriented_size,
    pad_the_source,
    source_pad_node,
    workflow_declares_outpaint,
)


def _padder(**overrides: Any) -> dict[str, Any]:
    pad = {"image": ["load", 0], "left": 0, "top": 0, "right": 0, "bottom": 0, "feathering": 24}
    pad.update(overrides)
    return {
        "load": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "pad": {"class_type": "ImagePadForOutpaint", "inputs": pad},
    }


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


def test_the_source_pad_is_the_one_padding_step_fed_by_the_source_image() -> None:
    assert source_pad_node(_padder()) == "pad"
    assert source_pad_node({**_padder(), "another": _padder()["pad"]}) is None
    assert source_pad_node(_padder(image=["elsewhere", 0])) is None
    assert source_pad_node(_padder(image=["load", 1])) is None
    assert source_pad_node(_padder(right="${outpaint_right}")) is None
    assert source_pad_node(_padder(top=True)) is None
    fixed_source = _padder()
    fixed_source["load"]["inputs"]["image"] = "example.png"
    assert source_pad_node(fixed_source) is None
    assert source_pad_node({"load": _padder()["load"]}) is None
    assert source_pad_node("not a graph") is None


def test_margins_become_whole_pixels_along_their_own_axis() -> None:
    margins = {"top": 0.25, "right": 0.5, "bottom": 0.0, "left": 0.125}
    assert margin_pixels(margins, 400, 300) == {"top": 75, "right": 200, "bottom": 0, "left": 50}
    # A half pixel is kept rather than rounded away.
    assert margin_pixels({"top": 0.5, "right": 0.0, "bottom": 0.0, "left": 0.0}, 3, 3)["top"] == 2


def test_padding_rewrites_a_copy_and_leaves_the_rest_of_the_node_alone() -> None:
    graph = _padder()
    padded = pad_the_source(graph, {"top": 75, "right": 200, "bottom": 0, "left": 50})
    assert padded["pad"]["inputs"] == {
        "image": ["load", 0],
        "left": 50,
        "top": 75,
        "right": 200,
        "bottom": 0,
        "feathering": 24,
    }
    assert graph == _padder()
    with pytest.raises(ValueError, match="single padding step"):
        pad_the_source(
            {**graph, "another": graph["pad"]}, {"top": 1, "right": 1, "bottom": 1, "left": 1}
        )


@pytest.mark.parametrize(
    ("orientation", "expected"), [(None, (40, 30)), (1, (40, 30)), (6, (30, 40)), (8, (30, 40))]
)
def test_the_source_is_measured_as_it_is_shown(
    tmp_path: Path, orientation: int | None, expected: tuple[int, int]
) -> None:
    path = tmp_path / "source.png"
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    Image.new("RGB", (40, 30)).save(path, "PNG", exif=exif.tobytes())
    assert oriented_size(path) == expected


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
            api_graph_json=_padder(),
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
