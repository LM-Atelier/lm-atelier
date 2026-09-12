"""A declared parameter has to reach the graph, on every route that stores one.

Creation already refused a video length the graph never uses. A REVISION could
still introduce it, and the same gap existed for the edit strength, which is
declared the same way and validated against the schema alone.

What is deliberately NOT required is as important as what is. A video frame RATE
and an edit STEP COUNT are frequently fixed by the model rather than taken as
graph inputs, so demanding placeholders for those would refuse workflows that
are behaving correctly - a worse failure than the one being fixed. Those
exemptions are pinned here rather than left to be rediscovered.

An edit strength the settings path does not OFFER is exempt for a different and
stronger reason: there is no slider to be a lie. What makes a strength unoffered
is measured, never read off the schema - `readOnly` removes an ENGINE setting and
does not remove a custom property, so the same marker means opposite things
depending on whose name the parameter is.

Video length looks like the same case and is not - a declared length contract
still produces a duration control - and that asymmetry is measured below rather
than argued, because the obvious "fix" is to make the two consistent.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.domain import Operation
from local_lm.image_edit_strength import EditSettingSource, resolve_image_edit_strength
from local_lm.settings_registry import IMAGE_SETTINGS, VIDEO_SETTINGS, workflow_settings
from local_lm.workflow_edit_calibration import standard_edit_calibration

_LENGTH_CONTRACT: dict[str, Any] = {
    "version": 1,
    "frames_parameter": "frames",
    "fps_parameter": "fps",
    "fps_numerator": 8,
    "fps_denominator": 1,
    "frame_alignment": 8,
    "frame_offset": 1,
}
_LENGTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {"type": "integer", "default": 17, "minimum": 9, "maximum": 129},
        "fps": {"type": "integer", "default": 8, "minimum": 8, "maximum": 8},
    },
    "x-lm-atelier-video-length": _LENGTH_CONTRACT,
}
_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"denoise": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0}},
    "x-lm-atelier-edit-calibration": standard_edit_calibration(
        parameter="denoise", minimum=0.0, maximum=1.0, steps_parameter=None
    ),
}


# A calibration that DOES declare a step count, so the exemption has something to
# be exempt from. Without this the exemption is unpinned: a fixture built with
# steps_parameter=None cannot tell "steps are not required" from "there are no
# steps", and requiring them would pass every test.
_EDIT_SCHEMA_WITH_STEPS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "denoise": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
        "steps": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
    },
    "x-lm-atelier-edit-calibration": standard_edit_calibration(
        parameter="denoise", minimum=0.0, maximum=1.0, steps_parameter="steps"
    ),
}


# The same two declarations with the parameter marked read-only. They are not
# merely "another case": they are what separates a control that lies from a
# workflow that has told the truth about deciding its own value.
_READ_ONLY_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "denoise": {
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "readOnly": True,
        }
    },
    "x-lm-atelier-edit-calibration": standard_edit_calibration(
        parameter="denoise", minimum=0.0, maximum=1.0, steps_parameter=None
    ),
}
_READ_ONLY_LENGTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "integer",
            "default": 17,
            "minimum": 9,
            "maximum": 129,
            "readOnly": True,
        },
        "fps": {"type": "integer", "default": 8, "minimum": 8, "maximum": 8, "readOnly": True},
    },
    "x-lm-atelier-video-length": _LENGTH_CONTRACT,
}


# The same read-only marker on a parameter the workflow named itself. The engine
# has no such setting, so the overlay's custom-property branch adds it and the
# control survives the flag - which is why the exemption asks what is offered
# rather than what is declared.
_READ_ONLY_CUSTOM_EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "custom_strength": {
            "type": "number",
            "default": 0.5,
            "minimum": 0.0,
            "maximum": 1.0,
            "readOnly": True,
        }
    },
    "x-lm-atelier-edit-calibration": standard_edit_calibration(
        parameter="custom_strength", minimum=0.0, maximum=1.0, steps_parameter=None
    ),
}


def _custom_edit_graph(strength: Any) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "${image}"}},
        "2": {"class_type": "KSampler", "inputs": {"denoise": strength, "steps": 20}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }


def _video_graph(length: Any, *, fps: Any = 8) -> dict[str, Any]:
    return {
        "1": {"class_type": "EmptyLatentVideo", "inputs": {"length": length}},
        "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0], "fps": fps}},
        "3": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["2", 0]}},
    }


def _edit_graph(strength: Any, *, steps: Any = 20) -> dict[str, Any]:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "${image}"}},
        "2": {"class_type": "KSampler", "inputs": {"denoise": strength, "steps": steps}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }


async def _create(client: AsyncClient, name: str, operation: str, graph: Any, schema: Any) -> Any:
    return await client.post(
        "/api/workflows",
        json={
            "name": name,
            "operation": operation,
            "engine": "comfyui",
            "api_graph": graph,
            "input_schema": schema,
            "dependencies": {},
        },
    )


async def _revise(client: AsyncClient, workflow_id: str, graph: Any, schema: Any) -> Any:
    return await client.post(
        f"/api/workflows/{workflow_id}/revisions",
        json={"api_graph": graph, "input_schema": schema, "dependencies": {}},
    )


async def test_a_revision_cannot_introduce_a_length_the_graph_never_uses(
    app: FastAPI, client: AsyncClient
) -> None:
    """The gap creation already closed, on the route that stores revisions.

    A workflow that passes creation can be revised, and before this the revision
    route validated the contract against the schema alone - so the defect walked
    straight back in one version later.
    """
    created = await _create(
        client, "Wired at creation", "text_to_video", _video_graph("${frames}"), _LENGTH_SCHEMA
    )
    assert created.status_code in (200, 201), created.text
    workflow_id = created.json()["id"]

    refused = await _revise(client, workflow_id, _video_graph(16), _LENGTH_SCHEMA)
    assert refused.status_code == 422, refused.text
    assert "frames" in refused.json()["detail"]


async def test_a_revision_that_keeps_using_the_count_is_accepted(
    app: FastAPI, client: AsyncClient
) -> None:
    """The control: only the placeholder differs from the refusal above."""
    created = await _create(
        client, "Wired and revised", "text_to_video", _video_graph("${frames}"), _LENGTH_SCHEMA
    )
    workflow_id = created.json()["id"]

    accepted = await _revise(client, workflow_id, _video_graph("${frames}"), _LENGTH_SCHEMA)
    assert accepted.status_code in (200, 201), accepted.text


async def test_a_declared_edit_strength_the_graph_never_uses_is_refused(
    app: FastAPI, client: AsyncClient
) -> None:
    """The same gap, in the declaration that builds the change-strength slider."""
    refused = await _create(
        client, "Edit strength ignored", "image_to_image", _edit_graph(0.5), _EDIT_SCHEMA
    )
    assert refused.status_code == 422, refused.text
    assert "denoise" in refused.json()["detail"]


async def test_an_edit_workflow_that_uses_its_strength_is_accepted(
    app: FastAPI, client: AsyncClient
) -> None:
    """The control, and the case that must keep working."""
    accepted = await _create(
        client, "Edit strength driven", "image_to_image", _edit_graph("${denoise}"), _EDIT_SCHEMA
    )
    assert accepted.status_code in (200, 201), accepted.text


async def test_a_revision_cannot_introduce_an_unused_edit_strength(
    app: FastAPI, client: AsyncClient
) -> None:
    created = await _create(
        client, "Edit wired then not", "image_to_image", _edit_graph("${denoise}"), _EDIT_SCHEMA
    )
    workflow_id = created.json()["id"]

    refused = await _revise(client, workflow_id, _edit_graph(0.5), _EDIT_SCHEMA)
    assert refused.status_code == 422, refused.text
    assert "denoise" in refused.json()["detail"]


@pytest.mark.parametrize(
    ("described", "graph", "schema"),
    [
        (
            "a frame rate fixed by the model",
            _video_graph("${frames}", fps=8),
            _LENGTH_SCHEMA,
        ),
        (
            "an edit step count fixed by the schedule",
            _edit_graph("${denoise}", steps=20),
            _EDIT_SCHEMA_WITH_STEPS,
        ),
    ],
)
async def test_a_legitimately_fixed_model_parameter_is_not_refused(
    app: FastAPI,
    client: AsyncClient,
    described: str,
    graph: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """The exemption, and the reason this is not "every declared field must bind".

    A frame RATE is usually a property of the model, and an edit STEP COUNT of
    the schedule. Both are declared beside the parameter that IS driven, and
    neither needs a placeholder. Requiring one would turn away workflows that
    are behaving correctly.
    """
    operation = "text_to_video" if "frame" in described else "image_to_image"
    created = await _create(client, f"Fixed: {described}", operation, graph, schema)
    assert created.status_code in (200, 201), created.text


async def test_a_workflow_declaring_neither_contract_is_untouched(
    app: FastAPI, client: AsyncClient
) -> None:
    """Most workflows declare neither, and must be unaffected by both checks."""
    created = await _create(
        client,
        "Plain",
        "text_to_image",
        {"1": {"class_type": "KSampler", "inputs": {"seed": "${seed}"}}},
        {"type": "object", "properties": {}},
    )
    assert created.status_code in (200, 201), created.text


async def test_a_read_only_strength_fixed_in_the_graph_is_accepted(
    app: FastAPI, client: AsyncClient
) -> None:
    """The workflow that declares it decides its own strength, and is believed.

    This is the case that separates "the slider does nothing" from "there is no
    slider". A read-only property is dropped from the offered settings, so
    nothing is shown and nothing is injected, and the graph's own value is
    simply what runs. Refusing it would turn away a workflow that has been
    honest about what it controls.
    """
    accepted = await _create(
        client,
        "Strength fixed and declared read-only",
        "image_to_image",
        _edit_graph(0.5),
        _READ_ONLY_EDIT_SCHEMA,
    )
    assert accepted.status_code in (200, 201), accepted.text


async def test_a_revision_may_also_fix_a_read_only_strength(
    app: FastAPI, client: AsyncClient
) -> None:
    """The exemption has to hold on the route that stores revisions too.

    Without this, a workflow could be created read-only and then be unable to
    revise anything else about itself.
    """
    created = await _create(
        client,
        "Read-only strength revised",
        "image_to_image",
        _edit_graph(0.5),
        _READ_ONLY_EDIT_SCHEMA,
    )
    assert created.status_code in (200, 201), created.text

    revised = await _revise(client, created.json()["id"], _edit_graph(0.4), _READ_ONLY_EDIT_SCHEMA)
    assert revised.status_code in (200, 201), revised.text


async def test_a_read_only_video_length_is_still_refused(app: FastAPI, client: AsyncClient) -> None:
    """Video length is NOT exempt, and this is the case that looks like it should be.

    Read-only is the same word in both schemas and means different things, so
    the reason is measured in the test below rather than asserted here.
    """
    refused = await _create(
        client,
        "Length fixed and declared read-only",
        "text_to_video",
        _video_graph(16),
        _READ_ONLY_LENGTH_SCHEMA,
    )
    assert refused.status_code == 422, refused.text
    assert "frames" in refused.json()["detail"]


def test_read_only_removes_an_edit_control_but_not_a_duration_control() -> None:
    """Why the two are treated differently, measured rather than argued.

    A read-only edit strength leaves the settings with no strength field at all,
    so nothing can be offered or injected. A read-only frame count leaves the
    settings with a DURATION control, because the length contract appends one
    from its own declaration - so the run still resolves a frame count and
    records the seconds it believes it delivered, against a graph that ignored
    both.

    Anyone tempted to make these consistent should change this test first and
    watch what it says.
    """
    image = workflow_settings(IMAGE_SETTINGS, _READ_ONLY_EDIT_SCHEMA)
    assert not [field for field in image if field.key == "denoise"]

    video = workflow_settings(VIDEO_SETTINGS, _READ_ONLY_LENGTH_SCHEMA)
    assert not [field for field in video if field.key in {"frames", "fps"}]
    assert [field for field in video if field.key == "duration_seconds"]


async def test_a_read_only_custom_strength_is_still_refused(
    app: FastAPI, client: AsyncClient
) -> None:
    """The marker is not the exemption, and this is the case that proves it.

    `readOnly` removes an engine setting from the offered fields. It does NOT
    remove a property the workflow named itself, because the overlay's
    custom-property branch adds that one regardless. So this workflow keeps a
    control a person can move, a value can still be resolved from it, and the
    graph would ignore every bit of that - exactly the lie the check exists for.
    """
    refused = await _create(
        client,
        "Custom strength marked read-only",
        "image_to_image",
        _custom_edit_graph(0.5),
        _READ_ONLY_CUSTOM_EDIT_SCHEMA,
    )
    assert refused.status_code == 422, refused.text
    assert "custom_strength" in refused.json()["detail"]


async def test_a_custom_strength_the_graph_uses_is_accepted(
    app: FastAPI, client: AsyncClient
) -> None:
    """The control for the case above: only the placeholder differs."""
    accepted = await _create(
        client,
        "Custom strength driven",
        "image_to_image",
        _custom_edit_graph("${custom_strength}"),
        _READ_ONLY_CUSTOM_EDIT_SCHEMA,
    )
    assert accepted.status_code in (200, 201), accepted.text


def test_read_only_means_opposite_things_for_engine_and_custom_parameters() -> None:
    """The measurement the exemption rests on, stated as a table.

    Reading `readOnly` off the schema gives the right answer in one of these two
    columns and the wrong one in the other, which is why the predicate runs the
    overlay instead.
    """
    engine = workflow_settings(IMAGE_SETTINGS, _READ_ONLY_EDIT_SCHEMA)
    assert not [field for field in engine if field.key == "denoise"]

    custom = workflow_settings(IMAGE_SETTINGS, _READ_ONLY_CUSTOM_EDIT_SCHEMA)
    offered = [field for field in custom if field.key == "custom_strength"]
    assert offered and offered[0].available


def test_an_unoffered_strength_cannot_be_supplied_but_a_custom_one_can() -> None:
    """And the consequence at the consumer, which is what actually matters.

    The engine case discards a value that was already in the settings. The
    custom case keeps it and resolves a real strength - so exempting it would
    let a run send a number the graph never reads.
    """
    engine_fields = workflow_settings(IMAGE_SETTINGS, _READ_ONLY_EDIT_SCHEMA)
    engine_settings: dict[str, Any] = {"denoise": 0.9}
    assert (
        resolve_image_edit_strength(
            Operation.IMAGE_TO_IMAGE,
            "make the sky darker",
            engine_fields,
            engine_settings,
            (),
            workflow_schema=_READ_ONLY_EDIT_SCHEMA,
        )
        is None
    )
    assert "denoise" not in engine_settings

    custom_fields = workflow_settings(IMAGE_SETTINGS, _READ_ONLY_CUSTOM_EDIT_SCHEMA)
    custom_settings: dict[str, Any] = {"custom_strength": 0.9}
    resolution = resolve_image_edit_strength(
        Operation.IMAGE_TO_IMAGE,
        "make the sky darker",
        custom_fields,
        custom_settings,
        ((EditSettingSource.CHAT, {"custom_strength": 0.9}),),
        workflow_schema=_READ_ONLY_CUSTOM_EDIT_SCHEMA,
    )
    assert resolution is not None
    assert custom_settings["custom_strength"] == 0.9
