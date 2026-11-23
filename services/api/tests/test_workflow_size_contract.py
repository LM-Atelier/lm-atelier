"""Each workflow family says what size a turn on it takes, so one code path sizes every family."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import event
from workflow_fixtures import seed_workflow_trust

from local_lm import api as api_module
from local_lm import db
from local_lm.db import SessionLocal
from local_lm.models import WorkflowDefinition, WorkflowFamily, WorkflowRevision
from local_lm.workflow_size_contract import workflow_size_contract
from local_lm.workflow_supplied_settings import bind_api_workflow_settings

_WHOLE = ["INT", {"min": 16, "max": 2048, "step": 8}]


def _node(**inputs: Any) -> dict[str, Any]:
    return {"input": {"required": inputs}, "output": ["IMAGE"]}


# What a runtime says each neutral fixture node takes. The binder reads it the
# way it reads a real runtime's node inventory when a workflow is saved.
_INVENTORY: dict[str, Any] = {
    "Canvas": _node(width=_WHOLE, height=_WHOLE),
    "MenuCanvas": _node(width=[[512, 768, 1024]], height=[[512, 768, 1024]]),
    "OneSizeCanvas": _node(width=[[1024]], height=[[768]]),
    "PinnedCanvas": _node(
        width=["INT", {"min": 1024, "max": 1024}], height=["INT", {"min": 768, "max": 768}]
    ),
    "Resize": _node(image=["IMAGE"], width=_WHOLE, height=_WHOLE),
    "ShapeCanvas": _node(
        aspect_ratio=[["1:1", "3:4", "4:3"]],
        megapixel_budget=["FLOAT", {"min": 0.0625, "max": 1}],
        round_to_multiple=["INT", {"min": 16, "max": 64, "step": 16}],
    ),
    "PinnedShapeCanvas": _node(
        aspect_ratio=[["1:1", "3:4"]],
        megapixel_budget=["FLOAT", {"min": 1, "max": 1}],
        round_to_multiple=["INT", {"min": 64, "max": 64}],
    ),
    "OneRatioCanvas": _node(aspect_ratio=[["1:1"]]),
    "RatioCanvas": _node(aspect_ratio=["FLOAT", {"min": 0.25, "max": 4, "step": 0.05}]),
    # A choice that grows inputs of its own: its "ratio" option adds an aspect ratio menu.
    "ModeCanvas": _node(
        mode=[
            "COMFY_DYNAMICCOMBO_V3",
            {
                "options": [
                    {"key": "ratio", "inputs": {"required": {"aspect_ratio": [["1:1", "16:9"]]}}}
                ]
            },
        ]
    ),
    "SizedShapeCanvas": _node(aspect_ratio=[["1:1", "16:9"]], width=_WHOLE, height=_WHOLE),
    "Number": {
        "input": {"required": {"value": ["INT", {"min": 0, "max": 4096}]}},
        "output": ["INT"],
    },
    "Save": {"input": {"required": {"images": ["IMAGE"]}}, "output": [], "output_node": True},
}

_NO_CONTROL = {
    "version": 1,
    "control": "workflow",
    "width": None,
    "height": None,
    "aspect_ratio": None,
    "megapixels": None,
    "round_to_multiple": None,
}


def _graph(**nodes: dict[str, Any]) -> dict[str, Any]:
    """A graph whose last node is saved."""
    last = list(nodes)[-1]
    return {**nodes, "save": {"class_type": "Save", "inputs": {"images": [last, 0]}}}


def _count() -> dict[str, Any]:
    """A node whose number another node's size input is linked to."""
    return {"class_type": "Number", "inputs": {"value": 1024}}


def _sides() -> dict[str, Any]:
    return _graph(canvas={"class_type": "Canvas", "inputs": {"width": 1024, "height": 768}})


def _ratio() -> dict[str, Any]:
    inputs = {"aspect_ratio": "1:1", "megapixel_budget": 0.0625, "round_to_multiple": 16}
    return _graph(canvas={"class_type": "ShapeCanvas", "inputs": inputs})


def _menu() -> dict[str, Any]:
    return _graph(canvas={"class_type": "MenuCanvas", "inputs": {"width": 768, "height": 512}})


def _ratio_and_sides() -> dict[str, Any]:
    inputs = {"aspect_ratio": "1:1", "width": 1024, "height": 1024}
    return _graph(canvas={"class_type": "SizedShapeCanvas", "inputs": inputs})


def _linked(*, resized: bool = False) -> dict[str, Any]:
    """A canvas sized by another node, and with `resized`, one editable size after it."""
    nodes: dict[str, dict[str, Any]] = {
        "number": _count(),
        "canvas": {
            "class_type": "Canvas",
            "inputs": {"width": ["number", 0], "height": ["number", 0]},
        },
    }
    if resized:
        nodes["resize"] = {
            "class_type": "Resize",
            "inputs": {"image": ["canvas", 0], "width": 640, "height": 480},
        }
    return _graph(**nodes)


def _half_linked(fixed: str) -> dict[str, Any]:
    """A canvas whose `fixed` side another node sets, leaving the other side editable."""
    inputs: dict[str, Any] = {"width": 1024, "height": 768}
    inputs[fixed] = ["number", 0]
    return _graph(number=_count(), canvas={"class_type": "Canvas", "inputs": inputs})


def _resized(**sizes: Any) -> dict[str, Any]:
    """An editable canvas, then a resize whose width and height are `sizes`."""
    return _graph(
        number=_count(),
        canvas={"class_type": "Canvas", "inputs": {"width": 1024, "height": 768}},
        resize={"class_type": "Resize", "inputs": {"image": ["canvas", 0], **sizes}},
    )


def _record(*bindings: tuple[str, str]) -> dict[str, Any]:
    """A generated settings record as a client may send it, binding parameters to the canvas."""
    return {
        "version": 1,
        "bindings": [
            {"parameter": parameter, "node_id": "canvas", "input_name": input_name}
            for parameter, input_name in bindings
        ],
    }


def _number_sides() -> tuple[dict[str, Any], dict[str, Any]]:
    """A canvas whose sides a supplied record declares as free numbers under generated keys."""
    width, height = "workflow_width_aaaaaaaaaaaa", "workflow_height_bbbbbbbbbbbb"
    inputs = {"width": "${" + width + "}", "height": "${" + height + "}"}
    side = {"type": "number", "minimum": 64, "maximum": 2048}
    schema = {
        "type": "object",
        "properties": {width: {**side, "default": 1024.0}, height: {**side, "default": 768.0}},
        "x-lm-atelier-graph-settings": _record((width, "width"), (height, "height")),
    }
    return _graph(canvas={"class_type": "Canvas", "inputs": inputs}), schema


def _declared_sides() -> dict[str, Any]:
    """A schema that declares plain sizes no input of the graph reads."""
    whole = {"type": "integer", "default": 512}
    return {"type": "object", "properties": {"width": whole, "height": whole}}


def _side(
    key: str,
    default: int,
    minimum: int,
    maximum: int,
    *,
    step: int,
    multiple_of: int | None = None,
    choices: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "default": default,
        "minimum": minimum,
        "maximum": maximum,
        "multiple_of": multiple_of,
        "step": step,
        "choices": choices or [],
    }


async def _family(
    client: AsyncClient,
    name: str,
    graph: dict[str, Any],
    schema: dict[str, Any] | None = None,
    *,
    operation: str = "text_to_image",
    bind: bool = True,
    trusted: bool = True,
) -> tuple[str, str]:
    """Save a workflow as the write path stores a bound one; give its family and revision.

    Binding here, against the fixture inventory, is what saving a ComfyUI graph
    does against the runtime. The bound schema carries its generated settings
    record, so the write path keeps it exactly as it is.
    """
    if bind:
        bound = bind_api_workflow_settings(graph, _INVENTORY, schema or {}, operation=operation)
        graph, schema = bound.api_graph, bound.input_schema
    created = await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": operation,
            "engine": "mock",
            "api_graph": graph,
            "input_schema": schema or {},
        },
    )
    assert created.status_code == 201, created.text
    workflow = created.json()
    if trusted:
        seed_workflow_trust(workflow["current_revision_id"])
    return workflow["family_id"], workflow["current_revision_id"]


def _family_of(*variants: tuple[str, str]) -> str:
    """One family holding a variant for each operation, saved as the families a profile makes."""
    with SessionLocal() as session:
        family = WorkflowFamily(name="Several operations")
        revisions = []
        for variant_key, operation in variants:
            bound = bind_api_workflow_settings(_sides(), _INVENTORY, {}, operation=operation)
            definition = WorkflowDefinition(
                family=family,
                variant_key=variant_key,
                name=f"Several operations {variant_key}",
                operation=operation,
            )
            revisions.append(
                WorkflowRevision(
                    definition=definition,
                    version=1,
                    engine="mock",
                    trusted=True,
                    api_graph_json=bound.api_graph,
                    input_schema_json=bound.input_schema,
                )
            )
        session.add_all(revisions)
        session.flush()
        for revision in revisions:
            revision.definition.current_revision_id = revision.id
        session.commit()
        return family.id


async def _size(client: AsyncClient, family_id: str) -> Any:
    family = await client.get(f"/api/workflow-families/{family_id}")
    assert family.status_code == 200, family.text
    (variant,) = family.json()["variants"]
    return variant["size"]


async def _record_of(client: AsyncClient, revision_id: str) -> Any:
    schema = await client.get(f"/api/workflow-revisions/{revision_id}/settings-schema")
    assert schema.status_code == 200, schema.text
    return schema.json()["input_schema_json"]["x-lm-atelier-graph-settings"]


async def _turn(client: AsyncClient, family_id: str, settings: dict[str, Any]) -> Any:
    chat = await client.post("/api/chats", json={"title": "Sized family"})
    assert chat.status_code == 201, chat.text
    return await client.post(
        f"/api/chats/{chat.json()['id']}/turns",
        json={
            "text": "A plain gray square on a white table",
            "mode": "image",
            "settings": settings,
            "workflow_selection": {
                "selector_capability": "image",
                "mode": "family",
                "workflow_family_id": family_id,
            },
        },
    )


@pytest.mark.parametrize(("operation", "least"), [("text_to_image", 64), ("text_to_video", 128)])
async def test_a_width_and_height_family_advertises_both_sides(
    client: AsyncClient, operation: str, least: int
) -> None:
    family_id, _ = await _family(client, "Sized canvas", _sides(), operation=operation)

    # The node's own range, narrowed to what the role allows. Its step is the
    # node's increment, not a multiple the turn enforces.
    assert await _size(client, family_id) == {
        "version": 1,
        "control": "width_height",
        "width": _side("width", 1024, least, 2048, step=8),
        "height": _side("height", 768, least, 2048, step=8),
        "aspect_ratio": None,
        "megapixels": None,
        "round_to_multiple": None,
    }


@pytest.mark.parametrize("fixed", ["width", "height"])
async def test_a_size_with_one_side_fixed_advertises_the_side_a_turn_takes(
    app: FastAPI, client: AsyncClient, fixed: str
) -> None:
    family_id, revision_id = await _family(client, "Half fixed", _half_linked(fixed))
    record = await _record_of(client, revision_id)
    assert [(entry["input_name"], entry["reason"]) for entry in record["fixed"]] == [
        (fixed, "linked")
    ]

    free = "height" if fixed == "width" else "width"
    expected = {
        "width": _side("width", 1024, 64, 2048, step=8),
        "height": _side("height", 768, 64, 2048, step=8),
    }
    # Each side is judged on its own: the fixed one is null, the free one is offered.
    assert await _size(client, family_id) == {
        "version": 1,
        "control": "width_height",
        fixed: None,
        free: expected[free],
        "aspect_ratio": None,
        "megapixels": None,
        "round_to_multiple": None,
    }
    async with app.state.services.scheduler.lease("primary"):
        refused = await _turn(client, family_id, {fixed: 512})
    assert refused.status_code == 422, refused.text
    assert "not mapped to an editable input" in refused.json()["detail"]


async def test_an_aspect_ratio_family_advertises_its_choices_and_scale(
    client: AsyncClient,
) -> None:
    family_id, _ = await _family(client, "Shaped canvas", _ratio())

    # The node names its input a budget; the turn's key is the setting's own.
    assert await _size(client, family_id) == {
        "version": 1,
        "control": "aspect_ratio",
        "width": None,
        "height": None,
        "aspect_ratio": {"key": "aspect_ratio", "default": "1:1", "choices": ["1:1", "3:4", "4:3"]},
        "megapixels": {
            "key": "megapixels",
            "default": 0.0625,
            "minimum": 0.0625,
            "maximum": 1,
            "multiple_of": None,
            "step": None,
        },
        "round_to_multiple": {
            "key": "round_to_multiple",
            "default": 16,
            "minimum": 16,
            "maximum": 64,
            "multiple_of": None,
            "step": 16,
        },
    }


async def test_a_ratio_advertises_only_the_scale_it_binds_and_lets_a_turn_choose(
    client: AsyncClient,
) -> None:
    inputs = {"aspect_ratio": "${aspect_ratio}", "megapixel_budget": 0.5, "round_to_multiple": 16}
    declared = {
        "type": "object",
        "properties": {
            "aspect_ratio": {"type": "string", "default": "1:1", "enum": ["1:1", "3:4", "4:3"]},
            "megapixels": {"type": "number", "default": 0.5, "minimum": 0.0625, "maximum": 1},
            "round_to_multiple": {"type": "integer", "default": 16, "minimum": 16, "maximum": 64},
        },
        "x-lm-atelier-graph-settings": _record(("aspect_ratio", "aspect_ratio")),
    }
    # A supplied record binds only the ratio; the budget and rounding it declares
    # are values no input reads, since the node keeps its own.
    declared_id, _ = await _family(
        client,
        "Declared scale",
        _graph(canvas={"class_type": "ShapeCanvas", "inputs": inputs}),
        declared,
        bind=False,
    )
    # Bound, but each pinned to a single value: nothing for a turn to choose.
    pinned = {"aspect_ratio": "1:1", "megapixel_budget": 1, "round_to_multiple": 64}
    pinned_id, _ = await _family(
        client,
        "Pinned scale",
        _graph(canvas={"class_type": "PinnedShapeCanvas", "inputs": pinned}),
    )

    for family_id, choices in ((declared_id, ["1:1", "3:4", "4:3"]), (pinned_id, ["1:1", "3:4"])):
        assert await _size(client, family_id) == {
            **_NO_CONTROL,
            "control": "aspect_ratio",
            "aspect_ratio": {"key": "aspect_ratio", "default": "1:1", "choices": choices},
        }


async def test_a_ratio_grown_by_a_dynamic_choice_is_advertised(client: AsyncClient) -> None:
    inputs = {"mode": "ratio", "mode.aspect_ratio": "1:1"}
    family_id, revision_id = await _family(
        client, "Mode canvas", _graph(canvas={"class_type": "ModeCanvas", "inputs": inputs})
    )
    record = await _record_of(client, revision_id)
    assert record["bindings"] == [
        {
            "parameter": "aspect_ratio",
            "node_id": "canvas",
            "input_name": "mode.aspect_ratio",
            "declared_name": "aspect_ratio",
        }
    ]

    # The input is named for the choice that grew it; the size it sets is its declared name's.
    assert await _size(client, family_id) == {
        **_NO_CONTROL,
        "control": "aspect_ratio",
        "aspect_ratio": {"key": "aspect_ratio", "default": "1:1", "choices": ["1:1", "16:9"]},
    }


async def test_a_declared_ratio_no_input_reads_is_not_advertised(client: AsyncClient) -> None:
    ratio = {"type": "string", "default": "1:1", "enum": ["1:1", "16:9"]}
    family_id, _ = await _family(
        client,
        "Declared ratio",
        _sides(),
        {"type": "object", "properties": {"aspect_ratio": ratio}},
        bind=False,
    )

    # Without a generated record the role's own sides stay; the ratio binds nothing.
    assert await _size(client, family_id) == {
        **_NO_CONTROL,
        "control": "width_height",
        "width": _side("width", 1024, 64, 4096, step=64),
        "height": _side("height", 1024, 64, 4096, step=64),
    }


async def test_a_size_offered_as_a_menu_advertises_its_choices(client: AsyncClient) -> None:
    family_id, _ = await _family(client, "Menu canvas", _menu())

    size = await _size(client, family_id)

    assert size["control"] == "width_height"
    # A menu keeps the role's range and step beside the only values it accepts.
    assert size["width"] == _side("width", 768, 64, 4096, step=64, choices=[512, 768, 1024])
    assert size["height"] == _side("height", 512, 64, 4096, step=64, choices=[512, 768, 1024])


def _fixed_sizes() -> dict[str, tuple[dict[str, Any], dict[str, Any] | None, bool]]:
    """Graphs whose size no single control chooses, each with its schema and whether to bind it."""
    read_only = {"width": {"readOnly": True}, "height": {"readOnly": True}}
    constant = {
        "width": {"type": "integer", "const": 1024},
        "height": {"type": "integer", "const": 768},
    }
    pinned = {"width": 1024, "height": 768}
    number_sides, number_schema = _number_sides()
    return {
        # A template whose graph has no size input marks the size read-only.
        "template_read_only": (_sides(), {"type": "object", "properties": read_only}, False),
        "read_only": (_sides(), {"type": "object", "properties": read_only}, True),
        "const": (_sides(), {"type": "object", "properties": constant}, False),
        "one_choice": (
            _graph(canvas={"class_type": "OneSizeCanvas", "inputs": pinned}),
            None,
            True,
        ),
        "one_number": (_graph(canvas={"class_type": "PinnedCanvas", "inputs": pinned}), None, True),
        "linked": (_linked(), None, True),
        # One editable size beside a fixed one: which of them sizes the result
        # is the graph's to say, whichever comes first.
        "editable_after_fixed": (_linked(resized=True), None, True),
        "fixed_after_editable": (_resized(width=["number", 0], height=["number", 0]), None, True),
        # A bound size beside a plain one that a later node reads.
        "beside_a_plain_key": (_resized(width="${width}", height="${height}"), None, True),
        "number_sides": (number_sides, number_schema, False),
        "one_choice_ratio": (
            _graph(canvas={"class_type": "OneRatioCanvas", "inputs": {"aspect_ratio": "1:1"}}),
            None,
            True,
        ),
        "free_number_ratio": (
            _graph(canvas={"class_type": "RatioCanvas", "inputs": {"aspect_ratio": 1.0}}),
            None,
            True,
        ),
    }


@pytest.mark.parametrize("fixed", sorted(_fixed_sizes()))
async def test_a_size_no_turn_can_choose_advertises_no_control(
    client: AsyncClient, fixed: str
) -> None:
    graph, schema, bind = _fixed_sizes()[fixed]
    family_id, _ = await _family(client, "Fixed size", graph, schema, bind=bind)

    assert await _size(client, family_id) == _NO_CONTROL


async def test_a_size_input_that_reaches_no_output_leaves_the_size_offered(
    client: AsyncClient,
) -> None:
    graph = {
        "canvas": {"class_type": "Canvas", "inputs": {"width": 1024, "height": 768}},
        # A resize nothing saves: it never runs, so its size is no second place.
        "resize": {
            "class_type": "Resize",
            "inputs": {"image": ["canvas", 0], "width": 640, "height": 480},
        },
        "save": {"class_type": "Save", "inputs": {"images": ["canvas", 0]}},
    }
    family_id, revision_id = await _family(client, "Unsaved resize", graph)
    record = await _record_of(client, revision_id)
    assert {entry["reason"] for entry in record["fixed"]} == {"disconnected"}

    assert await _size(client, family_id) == {
        **_NO_CONTROL,
        "control": "width_height",
        "width": _side("width", 1024, 64, 2048, step=8),
        "height": _side("height", 768, 64, 2048, step=8),
    }


async def test_two_inputs_for_one_size_advertise_no_control(client: AsyncClient) -> None:
    family_id, revision_id = await _family(client, "Two sizes", _resized(width=640, height=480))
    record = await _record_of(client, revision_id)
    assert sorted(binding["input_name"] for binding in record["bindings"]) == [
        "height",
        "height",
        "width",
        "width",
    ]

    assert await _size(client, family_id) == _NO_CONTROL
    # Neither key is the size on its own, and plain width is not one the turn takes.
    refused = await _turn(client, family_id, {"width": 512})
    assert refused.status_code == 422, refused.text
    assert refused.json()["code"] == "turn-invalid"
    assert "not mapped to an editable input" in refused.json()["detail"]


async def test_a_size_bound_under_a_generated_key_advertises_that_key(
    client: AsyncClient,
) -> None:
    family_id, revision_id = await _family(client, "Declared canvas", _sides(), _declared_sides())
    record = await _record_of(client, revision_id)
    # The schema already declares the plain keys, so the one native size takes a
    # generated key, and the plain keys, which no input reads, are left unbound.
    assert record["unbound_parameters"] == ["height", "width"]
    assert "fixed" not in record

    size = await _size(client, family_id)

    assert size["control"] == "width_height"
    width, height = size["width"], size["height"]
    assert width["key"].startswith("workflow_width_")
    assert height["key"].startswith("workflow_height_")
    assert width == _side(width["key"], 1024, 64, 2048, step=8)
    assert height == _side(height["key"], 768, 64, 2048, step=8)
    # Keys and bounds only: no node is named, by id or by title.
    described = json.dumps(size)
    for node in ("canvas", "Canvas"):
        assert node not in described


async def test_a_size_node_with_a_ratio_and_sides_advertises_both(client: AsyncClient) -> None:
    family_id, _ = await _family(client, "Ratio and sides", _ratio_and_sides())

    assert await _size(client, family_id) == {
        "version": 1,
        "control": "aspect_ratio_and_width_height",
        "width": _side("width", 1024, 64, 2048, step=8),
        "height": _side("height", 1024, 64, 2048, step=8),
        "aspect_ratio": {"key": "aspect_ratio", "default": "1:1", "choices": ["1:1", "16:9"]},
        "megapixels": None,
        "round_to_multiple": None,
    }


async def test_a_side_reports_the_multiple_the_turn_enforces(
    app: FastAPI, client: AsyncClient
) -> None:
    whole = {"type": "integer", "multipleOf": 16}
    schema = {
        "type": "object",
        "properties": {
            "width": {**whole, "default": 1024},
            "height": {**whole, "default": 768},
        },
    }
    graph = _graph(
        canvas={"class_type": "Canvas", "inputs": {"width": "${width}", "height": "${height}"}}
    )
    family_id, _ = await _family(client, "Multiple canvas", graph, schema, bind=False)

    size = await _size(client, family_id)

    assert size["width"] == _side("width", 1024, 64, 4096, step=16, multiple_of=16)
    async with app.state.services.scheduler.lease("primary"):
        refused = await _turn(client, family_id, {"width": 1032})
        accepted = await _turn(client, family_id, {"width": 1040})
    assert refused.status_code == 422, refused.text
    assert "width must be a multiple of 16" in refused.json()["detail"]
    assert accepted.status_code == 202, accepted.text


async def test_a_bound_sent_back_as_it_arrived_is_accepted(
    app: FastAPI, client: AsyncClient
) -> None:
    sides_id, _ = await _family(client, "Sized canvas", _sides())
    ratio_id, _ = await _family(client, "Shaped canvas", _ratio())
    sides = await _size(client, sides_id)
    ratio = await _size(client, ratio_id)
    width, height = sides["width"], sides["height"]
    scale = ratio["megapixels"], ratio["round_to_multiple"]
    turns = (
        (sides_id, {width["key"]: width["maximum"], height["key"]: height["minimum"]}),
        (ratio_id, {control["key"]: control["maximum"] for control in scale}),
    )

    async with app.state.services.scheduler.lease("primary"):
        for family_id, settings in turns:
            accepted = await _turn(client, family_id, settings)
            assert accepted.status_code == 202, accepted.text

    # A whole-number setting reports integers; a pixel budget keeps its fraction.
    assert {type(width[name]) for name in ("default", "minimum", "maximum", "step")} == {int}
    assert type(ratio["round_to_multiple"]["maximum"]) is int
    assert type(ratio["megapixels"]["maximum"]) is float


@pytest.mark.parametrize("operation", ["image_to_image", "image_to_video"])
async def test_work_that_follows_its_source_advertises_no_size(
    client: AsyncClient, operation: str
) -> None:
    family_id, revision_id = await _family(
        client, "Follows its source", _sides(), operation=operation
    )

    assert await _size(client, family_id) is None
    route = await client.get(f"/api/workflow-revisions/{revision_id}/size")
    assert route.status_code == 200, route.text
    assert route.json() is None


async def test_a_schema_the_turn_cannot_read_advertises_nothing(client: AsyncClient) -> None:
    # A width below the role's own minimum is refused whenever the settings are read.
    schema = {
        "type": "object",
        "properties": {"width": {"type": "integer", "default": 32, "minimum": 1}},
    }
    family_id, _ = await _family(client, "Unreadable size", _sides(), schema, bind=False)

    assert await _size(client, family_id) is None
    refused = await _turn(client, family_id, {})
    assert refused.status_code == 422, refused.text
    assert refused.json()["code"] == "turn-invalid"
    assert "width cannot lower the engine minimum" in refused.json()["detail"]


async def test_a_malformed_schema_leaves_the_lists_readable(client: AsyncClient) -> None:
    # A bound saved as text cannot be compared with a number. One such workflow
    # must not take down every list it appears in.
    schema = {
        "type": "object",
        "properties": {"width": {"type": "integer", "default": 1024, "minimum": "64"}},
    }
    family_id, revision_id = await _family(client, "Malformed size", _sides(), schema, bind=False)

    assert await _size(client, family_id) is None
    ready = await client.get("/api/workflow-ready-revisions", params={"revision_id": revision_id})
    assert ready.status_code == 200, ready.text
    assert [row["size"] for row in ready.json()] == [None]
    route = await client.get(f"/api/workflow-revisions/{revision_id}/size")
    assert route.status_code == 200, route.text
    assert route.json() is None


async def test_a_list_of_workflows_that_enlarge_is_sized_as_each_route_sizes_them(
    client: AsyncClient,
) -> None:
    enlargement = {
        "type": "number",
        "default": 2,
        "minimum": 1,
        "maximum": 8,
        "x-lm-atelier-kind": "upscale",
    }
    families: list[str] = []
    revisions: list[str] = []
    # Read together, and each needs its graph to tell a chosen factor from a fixed one.
    for scale_by in ("${upscale_factor}", 2):
        graph = _graph(
            canvas={"class_type": "Canvas", "inputs": {"width": 1024, "height": 768}},
            scale={
                "class_type": "ImageScaleBy",
                "inputs": {"image": ["canvas", 0], "scale_by": scale_by},
            },
        )
        schema = {"type": "object", "properties": {"upscale_factor": enlargement}}
        family_id, revision_id = await _family(client, "Enlarged canvas", graph, schema, bind=False)
        families.append(family_id)
        revisions.append(revision_id)

    listed = await client.get("/api/workflow-families", params={"family_ids": families})

    assert listed.status_code == 200, listed.text
    sizes = {
        variant["current_revision_id"]: variant["size"]
        for family in listed.json()
        for variant in family["variants"]
    }
    for revision_id in revisions:
        route = await client.get(f"/api/workflow-revisions/{revision_id}/size")
        assert sizes[revision_id] == route.json()
        assert route.json()["width"] == _side("width", 1024, 64, 4096, step=64)


async def test_every_list_and_the_revision_route_report_the_same_size(
    client: AsyncClient,
) -> None:
    for name, graph in (("Sized canvas", _sides()), ("Shaped canvas", _ratio())):
        family_id, revision_id = await _family(client, name, graph)
        route = await client.get(f"/api/workflow-revisions/{revision_id}/size")
        assert route.status_code == 200, route.text
        expected = route.json()
        assert expected["control"] != "workflow"

        unpaged = await client.get("/api/workflow-families")
        paged = await client.get(
            "/api/workflow-families", params={"family_ids": family_id, "order": "readiness"}
        )
        detail = await client.get(f"/api/workflow-families/{family_id}")
        variant_page = await client.get(
            f"/api/workflow-families/{family_id}", params={"variant_limit": 1}
        )
        patched = await client.patch(
            f"/api/workflow-families/{family_id}", json={"description": "Sized by its own node"}
        )
        ready = await client.get(
            "/api/workflow-ready-revisions", params={"revision_id": revision_id}
        )
        for response in (unpaged, paged, detail, variant_page, patched, ready):
            assert response.status_code == 200, response.text
        families = [
            next(family for family in unpaged.json() if family["id"] == family_id),
            *paged.json(),
            detail.json(),
            variant_page.json(),
            patched.json(),
        ]
        assert [family["variants"][0]["size"] for family in families] == [expected] * 5
        assert [row["size"] for row in ready.json()] == [expected]

    missing = await client.get("/api/workflow-revisions/wfrev_not_a_revision/size")
    assert missing.status_code == 404
    assert missing.json()["code"] == "workflow-revision-not-found"


async def test_every_variant_a_family_returns_carries_its_own_size(client: AsyncClient) -> None:
    family_id = _family_of(
        ("edit", "image_to_image"), ("create", "text_to_image"), ("video", "text_to_video")
    )

    detail = await client.get(f"/api/workflow-families/{family_id}")
    listed = await client.get("/api/workflow-families", params={"family_ids": family_id})
    assert detail.status_code == listed.status_code == 200
    for variants in (detail.json()["variants"], listed.json()[0]["variants"]):
        operations = [variant["operation"] for variant in variants]
        assert operations == ["image_to_image", "text_to_image", "text_to_video"]
        for variant in variants:
            route = await client.get(
                f"/api/workflow-revisions/{variant['current_revision_id']}/size"
            )
            assert variant["size"] == route.json(), variant["operation"]
        edit, picture, video = (variant["size"] for variant in variants)
        assert edit is None
        assert (picture["width"]["minimum"], video["width"]["minimum"]) == (64, 128)


@pytest.mark.parametrize(
    ("operation", "capability", "schema", "bind"),
    [
        ("image_to_image", "image", None, True),
        ("image_to_video", "video", None, True),
        # A video that declares no settings takes the video role's own size.
        ("text_to_video", "video", {}, False),
    ],
)
async def test_ready_rows_are_sized_for_their_own_operation(
    client: AsyncClient,
    operation: str,
    capability: str,
    schema: dict[str, Any] | None,
    bind: bool,
) -> None:
    _, revision_id = await _family(
        client, "Ready elsewhere", _sides(), schema, operation=operation, bind=bind
    )
    route = (await client.get(f"/api/workflow-revisions/{revision_id}/size")).json()

    ready = await client.get(
        "/api/workflow-ready-revisions",
        params={
            "selector_capability": capability,
            "operation": operation,
            "revision_id": revision_id,
        },
    )

    assert ready.status_code == 200, ready.text
    assert [row["size"] for row in ready.json()] == [route]
    if operation == "text_to_video":
        assert route["width"] == _side("width", 768, 128, 4096, step=64)
    else:
        assert route is None


async def test_sizes_are_read_together_and_only_for_the_rows_returned(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    listed = [(await _family(client, f"Counted canvas {index}", _sides()))[0] for index in range(6)]
    unreviewed, _ = await _family(client, "Counted canvas unreviewed", _sides(), trusted=False)
    computed: list[object] = []

    def counted(operation: Any, schema: Any) -> Any:
        computed.append(operation)
        return workflow_size_contract(operation, schema)

    monkeypatch.setattr(api_module, "workflow_size_contract", counted)
    reads: list[str] = []

    def record(_connection: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        # A stored settings schema read under the visibility rule, which scans the library.
        if "input_schema_json" in statement and "json_type" in statement:
            reads.append(statement)

    engine = db.engine
    event.listen(engine, "before_cursor_execute", record)
    try:
        # A readiness order projects every family once per readiness rank.
        page = await client.get(
            "/api/workflow-families",
            params={"family_ids": [*listed, unreviewed], "order": "readiness"},
        )
        page_counts = (len(reads), len(computed))
        reads.clear()
        computed.clear()
        # The unreviewed revision is projected, then left out as not ready.
        ready = await client.get(
            "/api/workflow-ready-revisions", params={"search": "Counted canvas", "limit": 200}
        )
        ready_counts = (len(reads), len(computed))
    finally:
        event.remove(engine, "before_cursor_execute", record)

    assert page.status_code == ready.status_code == 200
    assert sum(len(family["variants"]) for family in page.json()) == 7
    assert page_counts == (1, 7)
    assert len(ready.json()) == 6
    assert ready_counts == (1, 6)


def _largest(control: dict[str, Any]) -> Any:
    """The largest value a control offers, kept to its spacing, as a careful client picks it."""
    if control.get("choices"):
        return control["choices"][-1]
    spacing = control["multiple_of"] or control["step"]
    if spacing is None:
        return control["maximum"]
    return math.floor(control["maximum"] / spacing) * spacing


def _settings_for(size: dict[str, Any]) -> dict[str, Any]:
    """A turn's size settings chosen from the contract alone: the one code path a client needs."""
    controls = ("width", "height", "aspect_ratio", "megapixels", "round_to_multiple")
    return {size[name]["key"]: _largest(size[name]) for name in controls if size[name] is not None}


async def test_every_family_takes_the_size_its_contract_names(
    app: FastAPI, client: AsyncClient
) -> None:
    graphs: dict[str, tuple[dict[str, Any], dict[str, Any] | None]] = {
        "sides": (_sides(), None),
        "menu": (_menu(), None),
        "ratio": (_ratio(), None),
        "ratio and sides": (_ratio_and_sides(), None),
        "generated keys": (_sides(), _declared_sides()),
        "half size": (_half_linked("width"), None),
        "fixed": (_linked(), None),
    }
    chosen: dict[str, tuple[str, dict[str, Any]]] = {}
    for name, (graph, schema) in graphs.items():
        family_id, _ = await _family(client, name, graph, schema)
        listed = await client.get("/api/workflow-families", params={"family_ids": family_id})
        assert listed.status_code == 200, listed.text
        (variant,) = listed.json()[0]["variants"]
        chosen[name] = (family_id, _settings_for(variant["size"]))
    assert chosen["fixed"][1] == {}
    assert list(chosen["half size"][1]) == ["height"]
    assert all(settings for name, (_, settings) in chosen.items() if name != "fixed")

    async with app.state.services.scheduler.lease("primary"):
        for name, (family_id, settings) in chosen.items():
            accepted = await _turn(client, family_id, settings)
            assert accepted.status_code == 202, (name, accepted.text)
            carried = accepted.json()["run"]["settings_json"]
            assert {key: carried[key] for key in settings} == settings, name
        # Sent to the other kind of family, the same choices are refused, not adjusted.
        for sent, target, reason in (
            ("sides", "ratio", "not mapped to an editable input"),
            ("ratio", "sides", "unsupported settings: aspect_ratio"),
        ):
            refused = await _turn(client, chosen[target][0], chosen[sent][1])
            assert refused.status_code == 422, (sent, refused.text)
            assert reason in refused.json()["detail"], (sent, refused.text)
