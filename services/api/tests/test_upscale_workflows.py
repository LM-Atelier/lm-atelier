"""A workflow that can enlarge a picture is recognized by its graph."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.upscale_workflows import (
    UPSCALE_SCHEMA_KIND,
    UPSCALE_SETTING_KEY,
    upscale_capability,
    workflow_declares_upscale,
)


def _graph(*class_types: str) -> dict[str, Any]:
    return {str(index): {"class_type": name} for index, name in enumerate(class_types)}


def test_a_trained_upscaler_is_a_model_pass() -> None:
    graph = _graph("UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage")

    assert upscale_capability(graph) == "model"


def test_a_plain_resize_is_not_dressed_up_as_one() -> None:
    """Enhance over a plain resample would promise detail it cannot deliver."""
    assert upscale_capability(_graph("ImageScale", "SaveImage")) == "resample"
    assert upscale_capability(_graph("ImageScaleBy")) == "resample"


def test_a_graph_carrying_both_delivers_the_model_pass() -> None:
    # The resample is a stage; what the workflow delivers is the model pass.
    graph = _graph("ImageScale", "ImageUpscaleWithModel")

    assert upscale_capability(graph) == "model"


def test_a_graph_that_enlarges_nothing_says_so() -> None:
    assert upscale_capability(_graph("CheckpointLoaderSimple", "KSampler")) is None


def test_nodes_are_found_however_deeply_the_graph_nests_them() -> None:
    nested: dict[str, Any] = {
        "definitions": {"subgraphs": [{"nodes": [{"class_type": "ImageUpscaleWithModel"}]}]}
    }

    assert upscale_capability(nested) == "model"


@pytest.mark.parametrize("graph", [{}, {"nodes": []}, {"nodes": "not a list"}])
def test_an_empty_or_malformed_graph_is_not_an_upscaler(graph: dict[str, Any]) -> None:
    assert upscale_capability(graph) is None


def test_the_declared_setting_is_what_the_tool_needs() -> None:
    schema = {
        "type": "object",
        "properties": {
            UPSCALE_SETTING_KEY: {"type": "number", "x-lm-atelier-kind": UPSCALE_SCHEMA_KIND}
        },
    }

    assert workflow_declares_upscale(schema) is True


def test_a_setting_of_the_right_name_but_the_wrong_kind_is_not_it() -> None:
    """The declaration decides, never the spelling - as with the mask."""
    schema = {"type": "object", "properties": {UPSCALE_SETTING_KEY: {"type": "number"}}}

    assert workflow_declares_upscale(schema) is False


@pytest.mark.parametrize("schema", [None, {}, {"properties": None}, {"properties": {}}])
def test_a_workflow_declaring_nothing_cannot_be_asked_to_enlarge(schema: Any) -> None:
    assert workflow_declares_upscale(schema) is False
