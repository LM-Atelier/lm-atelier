"""Expansion is exact or it is a refusal - never an approximation."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.comfy_subgraphs import SubgraphExpansionError, expand_workflow


def _node(node_id: Any, node_type: str, **extra: Any) -> dict[str, Any]:
    return {"id": node_id, "type": node_type, **extra}


def _slots(*types: str) -> list[dict[str, str]]:
    return [{"type": kind} for kind in types]


def _photo_like() -> dict[str, Any]:
    """A source, a subgraph that sharpens, and a save - the common shape."""

    return {
        "nodes": [
            _node(1, "LoadImage"),
            _node(2, "Sharpen", mode=0),
            _node(3, "SaveImage"),
        ],
        "links": [
            [10, 1, 0, 2, 0, "IMAGE"],
            [11, 2, 0, 3, 0, "IMAGE"],
        ],
        "definitions": {
            "subgraphs": [
                {
                    "id": "Sharpen",
                    "nodes": [_node(7, "ImageSharpen")],
                    "links": [
                        [70, "-10", 0, 7, 0, "IMAGE"],
                        [71, 7, 0, "-20", 0, "IMAGE"],
                    ],
                }
            ]
        },
    }


def test_a_subgraph_becomes_the_nodes_it_contained() -> None:
    expanded = expand_workflow(_photo_like())

    types = sorted(str(node["type"]) for node in expanded["nodes"])
    assert types == ["ImageSharpen", "LoadImage", "SaveImage"]
    # The definition is gone, so the compiler's subgraph refusal no longer
    # applies to a graph that is now entirely runtime nodes.
    assert "definitions" not in expanded

    # The inner node is namespaced by the instance that placed it, which is
    # what lets one definition be used twice without colliding.
    inner = next(node for node in expanded["nodes"] if node["type"] == "ImageSharpen")
    assert inner["id"] == "2:7"


def test_the_boundary_is_spliced_to_what_surrounded_it() -> None:
    expanded = expand_workflow(_photo_like())
    links = {(str(link[1]), str(link[3])) for link in expanded["links"]}

    # What fed the instance now feeds the inside; what the instance fed is
    # now fed by the inside. Nothing points at the instance any more.
    assert ("1", "2:7") in links
    assert ("2:7", "3") in links
    assert not any(origin == "2" or target == "2" for origin, target in links)


def test_one_definition_placed_twice_stays_two_separate_copies() -> None:
    workflow = _photo_like()
    workflow["nodes"].append(_node(4, "Sharpen", mode=0))
    workflow["nodes"].append(_node(5, "SaveImage"))
    workflow["links"].extend([[12, 1, 0, 4, 0, "IMAGE"], [13, 4, 0, 5, 0, "IMAGE"]])

    expanded = expand_workflow(workflow)

    sharpeners = sorted(
        str(node["id"]) for node in expanded["nodes"] if node["type"] == "ImageSharpen"
    )
    assert sharpeners == ["2:7", "4:7"]


def test_a_subgraph_inside_a_subgraph_expands_all_the_way_down() -> None:
    workflow = _photo_like()
    workflow["definitions"]["subgraphs"][0]["nodes"] = [_node(7, "Inner", mode=0)]
    workflow["definitions"]["subgraphs"].append(
        {
            "id": "Inner",
            "nodes": [_node(9, "ImageSharpen")],
            "links": [[90, "-10", 0, 9, 0, "IMAGE"], [91, 9, 0, "-20", 0, "IMAGE"]],
        }
    )

    expanded = expand_workflow(workflow)

    assert [str(node["id"]) for node in expanded["nodes"] if node["type"] == "ImageSharpen"] == [
        "2:7:9"
    ]
    links = {(str(link[1]), str(link[3])) for link in expanded["links"]}
    assert ("1", "2:7:9") in links and ("2:7:9", "3") in links


def test_a_subgraph_that_contains_itself_is_refused() -> None:
    workflow = _photo_like()
    workflow["definitions"]["subgraphs"][0]["nodes"] = [_node(7, "Sharpen", mode=0)]

    with pytest.raises(SubgraphExpansionError) as caught:
        expand_workflow(workflow)
    assert caught.value.code == "recursive_subgraph"


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        # Nothing feeds the instance, so the inside has no source.
        (lambda w: w["links"].remove([10, 1, 0, 2, 0, "IMAGE"]), "unconnected_subgraph_input"),
        # Two links claim the same output boundary slot.
        (
            lambda w: w["definitions"]["subgraphs"][0]["links"].append(
                [72, 7, 1, "-20", 0, "IMAGE"]
            ),
            "ambiguous_subgraph_output",
        ),
        # The instance's output is used but the inside produces nothing.
        (
            lambda w: w["definitions"]["subgraphs"][0]["links"].remove(
                [71, 7, 0, "-20", 0, "IMAGE"]
            ),
            "unconnected_subgraph_output",
        ),
    ],
)
def test_an_undecidable_boundary_refuses_rather_than_guessing(mutate: Any, code: str) -> None:
    workflow = _photo_like()
    mutate(workflow)

    with pytest.raises(SubgraphExpansionError) as caught:
        expand_workflow(workflow)
    assert caught.value.code == code


def test_a_bypassed_node_is_routed_around_by_matching_type() -> None:
    workflow = {
        "nodes": [
            _node(1, "LoadImage", outputs=_slots("IMAGE")),
            _node(
                2,
                "Upscale",
                mode=4,
                inputs=_slots("IMAGE", "UPSCALE_MODEL"),
                outputs=_slots("IMAGE"),
            ),
            _node(3, "SaveImage", inputs=_slots("IMAGE")),
            _node(4, "LoadUpscaleModel", outputs=_slots("UPSCALE_MODEL")),
        ],
        "links": [
            [10, 1, 0, 2, 0, "IMAGE"],
            [11, 4, 0, 2, 1, "UPSCALE_MODEL"],
            [12, 2, 0, 3, 0, "IMAGE"],
        ],
    }

    expanded = expand_workflow(workflow)

    assert [str(node["id"]) for node in expanded["nodes"]] == ["1", "3", "4"]
    # The image passes through; the model input has no matching output and
    # simply stops there, which is what bypassing that node means.
    assert {(str(link[1]), str(link[3])) for link in expanded["links"]} == {("1", "3")}


def test_a_bypass_with_no_single_matching_input_is_refused() -> None:
    workflow = {
        "nodes": [
            _node(1, "LoadImage", outputs=_slots("IMAGE")),
            _node(2, "AlsoImage", outputs=_slots("IMAGE")),
            _node(3, "Blend", mode=4, inputs=_slots("IMAGE", "IMAGE"), outputs=_slots("IMAGE")),
            _node(4, "SaveImage", inputs=_slots("IMAGE")),
        ],
        "links": [
            [10, 1, 0, 3, 0, "IMAGE"],
            [11, 2, 0, 3, 1, "IMAGE"],
            [12, 3, 0, 4, 0, "IMAGE"],
        ],
    }

    # Two images arrive and one leaves. Which one the author meant is not in
    # the file, and picking one would rebuild the graph rather than expand it.
    with pytest.raises(SubgraphExpansionError) as caught:
        expand_workflow(workflow)
    assert caught.value.code == "ambiguous_bypass"


def test_only_mode_four_is_treated_as_a_bypass() -> None:
    workflow = {
        # Mode 2 is muted, not bypassed, and the rest are event and trigger
        # semantics. Rewiring any of them as a pass-through would silently
        # rebuild the graph.
        "nodes": [
            _node(1, "LoadImage", outputs=_slots("IMAGE")),
            _node(2, "Muted", mode=2, inputs=_slots("IMAGE"), outputs=_slots("IMAGE")),
            _node(3, "SaveImage", inputs=_slots("IMAGE")),
        ],
        "links": [[10, 1, 0, 2, 0, "IMAGE"], [11, 2, 0, 3, 0, "IMAGE"]],
    }

    expanded = expand_workflow(workflow)

    assert [str(node["id"]) for node in expanded["nodes"]] == ["1", "2", "3"]


def test_widget_values_survive_expansion() -> None:
    workflow = _photo_like()
    workflow["definitions"]["subgraphs"][0]["nodes"] = [
        _node(7, "ImageSharpen", widgets_values=[1.5, "high"])
    ]

    expanded = expand_workflow(workflow)

    inner = next(node for node in expanded["nodes"] if node["type"] == "ImageSharpen")
    assert inner["widgets_values"] == [1.5, "high"]


def test_a_graph_without_subgraphs_is_returned_unchanged() -> None:
    workflow = {
        "nodes": [_node(1, "LoadImage"), _node(2, "SaveImage")],
        "links": [[10, 1, 0, 2, 0, "IMAGE"]],
    }

    expanded = expand_workflow(workflow)

    assert [str(node["id"]) for node in expanded["nodes"]] == ["1", "2"]
    assert len(expanded["links"]) == 1
