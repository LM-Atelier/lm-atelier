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
