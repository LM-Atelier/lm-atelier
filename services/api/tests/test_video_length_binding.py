"""A declared length has to reach the graph that runs.

A workflow says how its length works and the product turns that into a duration
control. Nothing downstream checks the graph takes the frame count, so a
workflow that declares the contract and then hardcodes its frames offers a
control that changes nothing - and the run still records the length it thinks it
delivered.

These pin the refusal and, just as importantly, pin what must NOT be refused: a
workflow with no contract at all, and one whose frame rate is fixed by the model
rather than taken as a graph input.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.video_length import video_length_reaches_graph

_CONTRACT: dict[str, Any] = {
    "version": 1,
    "frames_parameter": "frames",
    "fps_parameter": "fps",
    "fps_numerator": 8,
    "fps_denominator": 1,
    "frame_alignment": 8,
    "frame_offset": 1,
}

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "frames": {"type": "integer", "default": 17, "minimum": 9, "maximum": 129},
        "fps": {"type": "integer", "default": 8, "minimum": 8, "maximum": 8},
    },
    "x-lm-atelier-video-length": _CONTRACT,
}


def _graph(length: Any, *, fps: Any = 8) -> dict[str, Any]:
    return {
        "1": {
            "class_type": "EmptyLatentVideo",
            "inputs": {"width": 512, "height": 512, "length": length},
        },
        "2": {"class_type": "KSampler", "inputs": {"latent_image": ["1", 0], "fps": fps}},
        "3": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["2", 0]}},
    }


def test_a_graph_that_uses_the_declared_frame_count_is_accepted() -> None:
    video_length_reaches_graph(_graph("${frames}"), _SCHEMA)


def test_a_graph_that_hardcodes_its_frame_count_is_refused() -> None:
    """The defect: the duration control would change nothing."""
    with pytest.raises(ValueError, match="frames but the graph never uses it"):
        video_length_reaches_graph(_graph(16), _SCHEMA)


def test_the_frame_rate_needs_no_placeholder() -> None:
    """Deliberately not required: fps is often fixed by the model.

    Demanding a placeholder for the rate as well would refuse workflows that are
    behaving correctly, which is a worse failure than the one being fixed.
    """
    video_length_reaches_graph(_graph("${frames}", fps=8), _SCHEMA)


def test_a_workflow_with_no_length_contract_is_untouched() -> None:
    """Most workflows declare nothing here and must stay unaffected."""
    video_length_reaches_graph(_graph(16), {"type": "object", "properties": {}})
    video_length_reaches_graph(_graph(16), None)


@pytest.mark.parametrize(
    ("where", "graph"),
    [
        ("a nested list", {"1": {"inputs": {"sizes": ["${frames}", 2]}}}),
        ("a deeper dictionary", {"1": {"inputs": {"opts": {"len": "${frames}"}}}}),
    ],
)
def test_the_placeholder_is_found_wherever_a_value_can_sit(
    where: str, graph: dict[str, Any]
) -> None:
    """The compiler substitutes anywhere a whole value matches, so this must too."""
    video_length_reaches_graph(graph, _SCHEMA)


@pytest.mark.parametrize(
    ("described", "value"),
    [
        ("a partial match inside a longer string", "length=${frames}"),
        ("the bare name without braces", "frames"),
        ("a different parameter", "${seed}"),
        ("a near miss in case", "${Frames}"),
    ],
)
def test_only_a_whole_value_placeholder_counts(described: str, value: str) -> None:
    """Matches the compiler exactly: it replaces a value only when the WHOLE
    value is the placeholder, so anything else would be a false reassurance."""
    with pytest.raises(ValueError):
        video_length_reaches_graph(_graph(value), _SCHEMA)


def test_the_message_names_the_parameter() -> None:
    """A refusal nobody can act on is a refusal that gets worked around."""
    with pytest.raises(ValueError) as raised:
        video_length_reaches_graph(_graph(16), _SCHEMA)
    assert "frames" in str(raised.value)


def test_a_differently_named_parameter_is_honoured() -> None:
    """The contract names the parameter; the check must follow it, not a literal."""
    schema = {
        "type": "object",
        "properties": {
            "num_frames": {"type": "integer", "default": 17, "minimum": 9, "maximum": 129},
            "fps": {"type": "integer", "default": 8, "minimum": 8, "maximum": 8},
        },
        "x-lm-atelier-video-length": {**_CONTRACT, "frames_parameter": "num_frames"},
    }
    video_length_reaches_graph(_graph("${num_frames}"), schema)
    with pytest.raises(ValueError, match="num_frames"):
        video_length_reaches_graph(_graph("${frames}"), schema)


def test_an_unreadable_contract_is_not_this_check_s_business() -> None:
    """workflow_video_length already refuses a malformed contract with its own
    message; this predicate must not turn that into a confusing second error."""
    with pytest.raises(ValueError) as raised:
        video_length_reaches_graph(
            _graph("${frames}"),
            {"type": "object", "properties": {}, "x-lm-atelier-video-length": {"version": 1}},
        )
    assert "never uses it" not in str(raised.value)


async def test_creation_refuses_a_declared_length_the_graph_never_uses(client) -> None:
    """The acceptance case, through POST /api/workflows rather than the validator.

    A predicate nobody calls refuses nothing, and the route is where a workflow
    actually arrives.
    """
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Declares a length it never uses",
            "operation": "text_to_video",
            "engine": "comfyui",
            "api_graph": _graph(16),
            "input_schema": _SCHEMA,
            "dependencies": {},
        },
    )
    assert created.status_code == 422, created.text
    body = created.json()
    assert body["code"] == "workflow-invalid"
    assert "frames" in body["detail"]


async def test_creation_accepts_the_same_workflow_once_it_uses_the_count(client) -> None:
    """The control: only the placeholder differs between this and the refusal."""
    created = await client.post(
        "/api/workflows",
        json={
            "name": "Uses the length it declares",
            "operation": "text_to_video",
            "engine": "comfyui",
            "api_graph": _graph("${frames}"),
            "input_schema": _SCHEMA,
            "dependencies": {},
        },
    )
    assert created.status_code in (200, 201), created.text
