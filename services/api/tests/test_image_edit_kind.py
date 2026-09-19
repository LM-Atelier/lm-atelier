"""An image edit is an instruction edit only when its graph proves it."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from local_lm.image_edit_kind import image_edit_kind


def _node(node_id: Any, node_type: str, *outputs: str, **extra: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "outputs": [{"name": kind.lower(), "type": kind} for kind in outputs],
        **extra,
    }


def _api(
    types: dict[str, str],
    links: list[tuple[str, int, str, str]],
    *,
    source: str | None = "1",
    words: str | None = "2",
    strength: str | None = None,
) -> dict[str, Any]:
    """An executable graph as the compiler writes one.

    Links are origin, output slot, target and input name. The named nodes take
    the source picture, the words and the denoise strength.
    """

    graph: dict[str, Any] = {
        node_id: {"class_type": kind, "inputs": {}} for node_id, kind in types.items()
    }
    for origin, slot, target, name in links:
        graph[target]["inputs"][name] = [origin, slot]
    if source is not None:
        graph[source]["inputs"]["image"] = "${input_image}"
    if words is not None:
        graph[words]["inputs"]["text"] = "${prompt}"
    if strength is not None:
        graph[strength]["inputs"]["denoise"] = "${denoise}"
    return graph


def _with(graph: dict[str, Any], **nodes: dict[str, Any]) -> dict[str, Any]:
    """A copy of an executable graph with nodes added or replaced by id."""

    changed = deepcopy(graph)
    changed.update(deepcopy(nodes))
    return changed


def _without(graph: dict[str, Any], *node_ids: str) -> dict[str, Any]:
    changed = deepcopy(graph)
    for node_id in node_ids:
        del changed[node_id]
    return changed


#: Words and a source picture, but no path between them and the saver.
PROMPT_AND_SAVE = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
    "2": {"class_type": "TextEncoder", "inputs": {"text": "${prompt}"}},
    "9": {"class_type": "SaveImage", "inputs": {}},
}
INSTRUCTION_TYPES = {"1": "LoadImage", "2": "EditEncoder", "3": "Sampler", "9": "SaveImage"}
INSTRUCTION_LINKS = [("1", 0, "2", "image"), ("2", 0, "3", "positive"), ("3", 0, "9", "images")]
INSTRUCTION_API = _api(INSTRUCTION_TYPES, INSTRUCTION_LINKS)


def _instruction_ui() -> dict[str, Any]:
    # The source picture is read by the encoder that turns the words into
    # conditioning, so the words steer what changes.
    return {
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


def _strength_ui() -> dict[str, Any]:
    # The source only becomes the starting latent; the words condition a
    # redraw of it.
    return {
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


def _utility_ui() -> dict[str, Any]:
    return {
        "nodes": [
            _node(1, "LoadImage", "IMAGE"),
            _node(2, "RemoveBackground", "IMAGE", "MASK"),
            _node(9, "SaveImage"),
        ],
        "links": [[10, 1, 0, 2, 0, "IMAGE"], [11, 2, 0, 9, 0, "IMAGE"]],
    }


def _instruction_in_a_subgraph() -> dict[str, Any]:
    return {
        "nodes": [
            _node(1, "LoadImage", "IMAGE"),
            _node(5, "EditBlock", "LATENT"),
            _node(9, "SaveImage"),
        ],
        "links": [[10, 1, 0, 5, 0, "IMAGE"], [11, 5, 0, 9, 0, "LATENT"]],
        "definitions": {
            "subgraphs": [
                {
                    "id": "EditBlock",
                    "nodes": [
                        _node(7, "EditEncoder", "CONDITIONING"),
                        _node(8, "Sampler", "LATENT"),
                    ],
                    "links": [
                        [70, "-10", 0, 7, 0, "IMAGE"],
                        [71, 7, 0, 8, 0, "CONDITIONING"],
                        [72, 8, 0, "-20", 0, "LATENT"],
                    ],
                }
            ]
        },
    }


def _strength_with_an_unused_conditioning_branch() -> dict[str, Any]:
    graph = _strength_ui()
    # The source also feeds a conditioning node, but nothing it produces
    # reaches the saved picture, so the workflow still redraws by strength.
    graph["nodes"].append(_node(20, "EditEncoder", "CONDITIONING"))
    graph["links"].append([20, 1, 0, 20, 0, "IMAGE"])
    return graph


def _strength_with_a_sampled_branch_that_is_never_saved() -> dict[str, Any]:
    graph = _strength_ui()
    # This branch does condition a sampler on the source, but its result is
    # only previewed, never saved.
    graph["nodes"] += [
        _node(20, "EditEncoder", "CONDITIONING"),
        _node(21, "Sampler", "LATENT"),
        _node(22, "PreviewImage"),
    ]
    graph["links"] += [
        [20, 1, 0, 20, 0, "IMAGE"],
        [21, 20, 0, 21, 0, "CONDITIONING"],
        [22, 21, 0, 22, 0, "LATENT"],
    ]
    return graph


def _instruction_with_a_muted_encoder() -> dict[str, Any]:
    graph = _instruction_ui()
    graph["nodes"][1]["mode"] = 2
    return graph


def _conditioning_declared_but_only_its_latent_used() -> dict[str, Any]:
    # The node the source reaches can produce conditioning, but only its latent
    # output feeds the sampler.
    return {
        "nodes": [
            _node(1, "LoadImage", "IMAGE"),
            _node(2, "EncodeBoth", "CONDITIONING", "LATENT"),
            _node(4, "CLIPTextEncode", "CONDITIONING"),
            _node(3, "Sampler", "LATENT"),
            _node(9, "SaveImage"),
        ],
        "links": [
            [10, 1, 0, 2, 0, "IMAGE"],
            [11, 2, 1, 3, 3, "LATENT"],
            [12, 4, 0, 3, 1, "CONDITIONING"],
            [13, 3, 0, 9, 0, "LATENT"],
        ],
    }


def _mask_only_reaches_conditioning() -> dict[str, Any]:
    # Only the loaded picture's mask output reaches the encoder.
    return {
        "nodes": [
            _node(1, "LoadImage", "IMAGE", "MASK"),
            _node(2, "EditEncoder", "CONDITIONING"),
            _node(3, "Sampler", "LATENT"),
            _node(9, "SaveImage"),
        ],
        "links": [
            [10, 1, 1, 2, 0, "MASK"],
            [11, 2, 0, 3, 0, "CONDITIONING"],
            [12, 3, 0, 9, 0, "LATENT"],
        ],
    }


def _instruction_saved_by_the_advanced_saver() -> dict[str, Any]:
    graph = _instruction_ui()
    graph["nodes"][3]["type"] = "SaveImageAdvanced"
    return graph


def _unexpandable() -> dict[str, Any]:
    graph = _instruction_in_a_subgraph()
    # A subgraph that places itself cannot be expanded exactly, so it proves nothing.
    graph["definitions"]["subgraphs"][0]["nodes"].append(_node(6, "EditBlock", "LATENT"))
    return graph


STRENGTH_TYPES = {
    "1": "LoadImage",
    "2": "VAEEncode",
    "4": "CLIPTextEncode",
    "3": "Sampler",
    "9": "SaveImage",
}
STRENGTH_LINKS = [
    ("1", 0, "2", "pixels"),
    ("2", 0, "3", "latent_image"),
    ("4", 0, "3", "positive"),
    ("3", 0, "9", "images"),
]
STRENGTH_API = _api(STRENGTH_TYPES, STRENGTH_LINKS, words="4", strength="3")
CALIBRATED_SCHEMA = {
    "type": "object",
    "properties": {"strength": {"type": "number", "default": 0.9}},
    "x-lm-atelier-edit-calibration": {
        "version": 1,
        "edit_strength": {
            "parameter": "strength",
            "minimum": 0.0,
            "maximum": 1.0,
            "recommended": {
                "minimal": 0.3,
                "localized": 0.45,
                "replacement": 0.6,
                "global": 0.8,
                "fallback": 0.5,
            },
        },
    },
}
CALIBRATED_API = deepcopy(STRENGTH_API)
CALIBRATED_API["3"]["inputs"]["denoise"] = "${strength}"
SUBGRAPH_INSTRUCTION_API = _api(
    {"1": "LoadImage", "5:7": "EditEncoder", "5:8": "Sampler", "9": "SaveImage"},
    [("1", 0, "5:7", "image"), ("5:7", 0, "5:8", "positive"), ("5:8", 0, "9", "images")],
    words="5:7",
)
UNUSED_BRANCH_API = _with(
    STRENGTH_API, **{"20": {"class_type": "EditEncoder", "inputs": {"image": ["1", 0]}}}
)
UNSAVED_BRANCH_API = _with(
    STRENGTH_API,
    **{
        "20": {"class_type": "EditEncoder", "inputs": {"image": ["1", 0]}},
        "21": {"class_type": "Sampler", "inputs": {"positive": ["20", 0]}},
        "22": {"class_type": "PreviewImage", "inputs": {"images": ["21", 0]}},
    },
)
LATENT_ONLY_API = _api(
    {"1": "LoadImage", "2": "EncodeBoth", "4": "CLIPTextEncode", "3": "Sampler", "9": "SaveImage"},
    [
        ("1", 0, "2", "image"),
        ("2", 1, "3", "latent_image"),
        ("4", 0, "3", "positive"),
        ("3", 0, "9", "images"),
    ],
    words="4",
    strength="3",
)
MASK_ONLY_API = _api(
    INSTRUCTION_TYPES,
    [("1", 1, "2", "mask"), ("2", 0, "3", "positive"), ("3", 0, "9", "images")],
)
UTILITY_API = _api(
    {"1": "LoadImage", "2": "RemoveBackground", "9": "SaveImage"},
    [("1", 0, "2", "image"), ("2", 0, "9", "images")],
    words=None,
)


@pytest.mark.parametrize(
    ("operation", "ui_graph", "api_graph", "schema", "expected"),
    [
        ("image_to_image", _instruction_ui(), INSTRUCTION_API, None, "instruction"),
        (
            "image_to_image",
            _instruction_in_a_subgraph(),
            SUBGRAPH_INSTRUCTION_API,
            None,
            "instruction",
        ),
        ("image_to_image", _strength_ui(), STRENGTH_API, None, "strength"),
        ("image_to_image", _strength_ui(), CALIBRATED_API, CALIBRATED_SCHEMA, "strength"),
        # The compiler writes the advanced saver as the ordinary one, and a graph
        # saved from the editor may keep its own name.
        (
            "image_to_image",
            _instruction_saved_by_the_advanced_saver(),
            INSTRUCTION_API,
            None,
            "instruction",
        ),
        (
            "image_to_image",
            _instruction_saved_by_the_advanced_saver(),
            _with(
                INSTRUCTION_API,
                **{"9": {"class_type": "SaveImageAdvanced", "inputs": {"images": ["3", 0]}}},
            ),
            None,
            "instruction",
        ),
        (
            "image_to_image",
            _instruction_ui(),
            _with(
                INSTRUCTION_API,
                **{"1": {"class_type": "LoadImage", "inputs": {"image": "${input_image_0}"}}},
            ),
            None,
            "instruction",
        ),
        # Negative controls: none of these may be promoted to an instruction edit.
        ("image_to_image", _utility_ui(), UTILITY_API, None, "other"),
        (
            "image_to_image",
            _instruction_ui(),
            {"9": {"class_type": "SaveImage", "inputs": {}}},
            None,
            "other",
        ),
        ("image_to_image", {}, INSTRUCTION_API, None, "other"),
        ("image_to_image", {}, STRENGTH_API, None, "strength"),
        (
            "image_to_image",
            _strength_ui(),
            _api(STRENGTH_TYPES, STRENGTH_LINKS, words="4"),
            None,
            "other",
        ),
        ("image_to_image", _unexpandable(), SUBGRAPH_INSTRUCTION_API, None, "other"),
        (
            "image_to_image",
            _strength_with_an_unused_conditioning_branch(),
            UNUSED_BRANCH_API,
            None,
            "strength",
        ),
        (
            "image_to_image",
            _strength_with_a_sampled_branch_that_is_never_saved(),
            UNSAVED_BRANCH_API,
            None,
            "strength",
        ),
        # The compiler leaves a muted node out, and a graph that still holds one
        # is not the graph the UI graph describes.
        (
            "image_to_image",
            _instruction_with_a_muted_encoder(),
            _without(INSTRUCTION_API, "2"),
            None,
            "other",
        ),
        ("image_to_image", _instruction_with_a_muted_encoder(), INSTRUCTION_API, None, "other"),
        (
            "image_to_image",
            _conditioning_declared_but_only_its_latent_used(),
            LATENT_ONLY_API,
            None,
            "strength",
        ),
        ("image_to_image", _mask_only_reaches_conditioning(), MASK_ONLY_API, None, "other"),
        ("text_to_image", _instruction_ui(), INSTRUCTION_API, None, "other"),
        ("image_to_image", _instruction_ui(), {}, None, "other"),
        # The executable graph has to hold the path; the UI graph only names types.
        ("image_to_image", _instruction_ui(), PROMPT_AND_SAVE, None, "other"),
        (
            "image_to_image",
            _instruction_ui(),
            _with(
                INSTRUCTION_API,
                **{
                    "2": {
                        "class_type": "VAEEncode",
                        "inputs": {"image": ["1", 0], "text": "${prompt}"},
                    }
                },
            ),
            None,
            "other",
        ),
        (
            "image_to_image",
            _instruction_ui(),
            _api(INSTRUCTION_TYPES, INSTRUCTION_LINKS, source=None),
            None,
            "other",
        ),
        (
            "image_to_image",
            _instruction_ui(),
            _with(
                _api(INSTRUCTION_TYPES, INSTRUCTION_LINKS, words=None),
                **{"5": {"class_type": "TextEncoder", "inputs": {"text": "${prompt}"}}},
            ),
            None,
            "other",
        ),
    ],
    ids=[
        "instruction",
        "instruction-inside-a-subgraph",
        "strength-denoise",
        "strength-calibrated-parameter",
        "instruction-saved-by-the-advanced-saver",
        "instruction-saved-by-the-advanced-saver-by-name",
        "instruction-with-a-numbered-source-slot",
        "utility-without-words",
        "words-never-bound",
        "api-only-graph-with-words",
        "api-only-graph-with-strength",
        "latent-source-without-strength",
        "unexpandable-graph",
        "strength-with-an-unused-conditioning-branch",
        "strength-with-a-sampled-branch-never-saved",
        "instruction-encoder-muted",
        "instruction-encoder-muted-but-still-compiled",
        "conditioning-declared-but-only-latent-used",
        "only-the-mask-reaches-conditioning",
        "not-an-image-edit",
        "no-compiled-graph",
        "path-only-in-the-ui-graph",
        "ui-describes-another-node-type",
        "source-picture-never-handed-to-the-graph",
        "words-off-the-saved-path",
    ],
)
def test_the_graph_decides_the_edit_kind(
    operation: str,
    ui_graph: dict[str, Any],
    api_graph: dict[str, Any],
    schema: dict[str, Any] | None,
    expected: str,
) -> None:
    assert image_edit_kind(operation, ui_graph, api_graph, schema) == expected


def test_a_saved_branch_only_in_the_ui_graph_does_not_change_the_edit_kind() -> None:
    # Execution receives the unchanged strength graph whatever the UI graph adds.
    ui = _strength_ui()
    api = deepcopy(STRENGTH_API)
    assert image_edit_kind("image_to_image", ui, api, None) == "strength"

    ui["nodes"] += [
        _node(20, "EditEncoder", "CONDITIONING"),
        _node(21, "Sampler", "LATENT"),
        _node(22, "VAEDecode", "IMAGE"),
        _node(23, "SaveImage"),
    ]
    ui["links"] += [
        [20, 1, 0, 20, 0, "IMAGE"],
        [21, 20, 0, 21, 0, "CONDITIONING"],
        [22, 21, 0, 22, 0, "LATENT"],
        [23, 22, 0, 23, 0, "IMAGE"],
    ]

    assert image_edit_kind("image_to_image", ui, api, None) == "strength"
    assert api == STRENGTH_API
