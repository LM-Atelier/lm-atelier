"""Official templates compile to what a turn can use.

ComfyUI's LTX-2.5 text-to-video template installed with three faults:
- a batch of 25 videos to a turn, because a socket accepting several types was
  not counted as the widget it draws, so every later saved value shifted;
- its prompt rewriter running every turn for nothing but a text preview;
- its untitled negative encoder handed the turn's own prompt.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.comfy_templates import _compile_ui_graph

AUDIO_LATENT = {
    "input": {
        "required": {
            "frames_number": ["INT", {"default": 97, "min": 1, "max": 1000}],
            "frame_rate": ["FLOAT,INT", {"default": 25.0, "widgetType": "FLOAT", "min": 1.0}],
            "batch_size": ["INT", {"default": 1, "min": 1, "max": 4096}],
        }
    },
    "input_order": {"required": ["frames_number", "frame_rate", "batch_size"]},
    "output": ["LATENT"],
}
SAVE = {
    "input": {"required": {"samples": ["LATENT"]}},
    "input_order": {"required": ["samples"]},
    "output": [],
    "output_node": True,
}
ENCODER = {
    "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
    "input_order": {"required": ["text", "clip"]},
    "output": ["CONDITIONING"],
}
GUIDE = {
    "input": {"required": {"positive": ["CONDITIONING"], "negative": ["CONDITIONING"]}},
    "input_order": {"required": ["positive", "negative"]},
    "output": ["LATENT"],
}
WRITER = {
    "input": {"required": {"seed": ["INT", {"default": 0}]}},
    "input_order": {"required": ["seed"]},
    "output": ["STRING"],
}
PREVIEW = {
    "input": {"required": {"source": ["*", {}]}},
    "input_order": {"required": ["source"]},
    "output": ["STRING"],
    "output_node": True,
}
READER = {
    "input": {"required": {"text": ["STRING", {"forceInput": True}]}},
    "input_order": {"required": ["text"]},
    "output": ["LATENT"],
}


def _node(node_id: int, kind: str, values: list[Any], **extra: Any) -> dict[str, Any]:
    return {"id": node_id, "type": kind, "mode": 0, "inputs": [], "outputs": [], **extra} | {
        "widgets_values": values
    }


def test_a_widget_on_a_socket_of_several_types_keeps_later_values_in_place() -> None:
    graph, schema = _compile_ui_graph(
        {"nodes": [_node(1, "AudioLatent", [97, 25, 1])], "links": []},
        {"AudioLatent": AUDIO_LATENT},
        operation="text_to_video",
    )

    assert graph["1"]["inputs"]["frame_rate"] == 25
    assert graph["1"]["inputs"]["batch_size"] == "${batch_size}"
    assert schema["properties"]["batch_size"]["default"] == 1


def _encoders(feeds: dict[int, list[str]]) -> dict[str, Any]:
    nodes = [_node(node_id, "CLIPTextEncode", [f"text {node_id}"]) for node_id in feeds]
    links: list[list[Any]] = []
    slots = {"positive": 0, "negative": 1}
    for node_id, inputs in feeds.items():
        nodes[list(feeds).index(node_id)]["outputs"] = [{"type": "CONDITIONING"}]
        for input_name in inputs:
            links.append([len(links) + 1, node_id, 0, 9, slots[input_name], "CONDITIONING"])
    guide = {
        "id": 9,
        "type": "Guide",
        "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING"},
            {"name": "negative", "type": "CONDITIONING"},
        ],
        "outputs": [],
        "widgets_values": [],
    }
    return {"nodes": [*nodes, guide], "links": links}


def test_an_untitled_encoder_feeding_only_negatives_reads_the_negative_prompt() -> None:
    graph, schema = _compile_ui_graph(
        _encoders({1: ["positive"], 2: ["negative"]}),
        {"CLIPTextEncode": ENCODER, "Guide": GUIDE},
        operation="text_to_image",
    )

    assert graph["1"]["inputs"]["text"] == "${prompt}"
    assert graph["2"]["inputs"]["text"] == "${negative_prompt}"
    # The template's own negative text stays the default, as for a titled one.
    assert schema["properties"]["negative_prompt"] == {"type": "string", "default": "text 2"}


def test_an_encoder_feeding_both_conditionings_reads_the_prompt() -> None:
    graph, _ = _compile_ui_graph(
        _encoders({1: ["positive", "negative"], 2: ["positive"]}),
        {"CLIPTextEncode": ENCODER, "Guide": GUIDE},
        operation="text_to_image",
    )

    assert graph["1"]["inputs"]["text"] == "${prompt}"
    assert graph["2"]["inputs"]["text"] == "${prompt}"


def _previewed(*, read: bool, saved: bool) -> dict[str, Any]:
    nodes = [_node(1, "Writer", [0]), _node(2, "PreviewAny", [])]
    links = [[1, 1, 0, 2, 0, "STRING"]]
    nodes[1]["inputs"] = [{"name": "source", "type": "*"}]
    if read:
        reader = _node(3, "Reader", [])
        reader["inputs"] = [{"name": "text", "type": "STRING"}]
        nodes.append(reader)
        links.append([2, 2, 0, 3, 0, "STRING"])
    if saved:
        nodes.append(_node(4, "Save", []))
        nodes[-1]["inputs"] = [{"name": "samples", "type": "LATENT"}]
        links.append([3, 3 if read else 5, 0, 4, 0, "LATENT"])
        if not read:
            nodes.append(_node(5, "AudioLatent", [97, 25, 1]))
    return {"nodes": nodes, "links": links}


INFO = {
    "Writer": WRITER,
    "PreviewAny": PREVIEW,
    "Reader": READER,
    "Save": SAVE,
    "AudioLatent": AUDIO_LATENT,
}


def test_a_text_preview_nothing_reads_is_left_out_with_its_branch() -> None:
    graph, _ = _compile_ui_graph(
        _previewed(read=False, saved=True), INFO, operation="text_to_video"
    )

    assert "2" not in graph
    assert "4" in graph


@pytest.mark.parametrize(("read", "saved"), [(True, True), (False, False)])
def test_a_text_preview_that_passes_text_on_or_is_the_only_output_stays(
    read: bool, saved: bool
) -> None:
    graph, _ = _compile_ui_graph(
        _previewed(read=read, saved=saved), INFO, operation="text_to_video"
    )

    assert graph["2"]["inputs"]["source"] == ["1", 0]
    if read:
        assert graph["3"]["inputs"]["text"] == ["2", 0]


def test_a_preview_read_only_by_an_encoder_bound_to_the_prompt_is_left_out() -> None:
    # The rewriter's text reached the encoder through its preview, and binding
    # the encoder to the turn's prompt left that preview read by nothing.
    nodes = [_node(1, "Writer", [0]), _node(2, "PreviewAny", []), _node(3, "CLIPTextEncode", [""])]
    nodes[1]["inputs"] = [{"name": "source", "type": "*"}]
    nodes[2]["inputs"] = [{"name": "text", "type": "STRING", "widget": {"name": "text"}}]
    nodes[2]["outputs"] = [{"type": "CONDITIONING"}]
    guide = _node(9, "Guide", [])
    guide["inputs"] = [
        {"name": "positive", "type": "CONDITIONING"},
        {"name": "negative", "type": "CONDITIONING"},
    ]
    save = _node(10, "Save", [])
    save["inputs"] = [{"name": "samples", "type": "LATENT"}]
    links = [
        [1, 1, 0, 2, 0, "STRING"],
        [2, 2, 0, 3, 0, "STRING"],
        [3, 3, 0, 9, 0, "CONDITIONING"],
        [4, 9, 0, 10, 0, "LATENT"],
    ]

    graph, _ = _compile_ui_graph(
        {"nodes": [*nodes, guide, save], "links": links},
        {**INFO, "CLIPTextEncode": ENCODER, "Guide": GUIDE},
        operation="text_to_image",
    )

    assert graph["3"]["inputs"]["text"] == "${prompt}"
    assert "2" not in graph
