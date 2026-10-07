"""A workflow that can cut a subject out says so, or it is not offered."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.matting_workflows import (
    MATTING_SCHEMA_KIND,
    MATTING_SETTING_KEY,
    workflow_declares_matting,
)


def test_the_declaration_is_what_the_tool_needs() -> None:
    schema = {
        "type": "object",
        "properties": {
            MATTING_SETTING_KEY: {"type": "boolean", "x-lm-atelier-kind": MATTING_SCHEMA_KIND}
        },
    }

    assert workflow_declares_matting(schema) is True


def test_a_property_of_the_right_name_but_the_wrong_kind_is_not_it() -> None:
    """The declaration decides, never the spelling - as with the mask."""

    schema = {"type": "object", "properties": {MATTING_SETTING_KEY: {"type": "boolean"}}}

    assert workflow_declares_matting(schema) is False


def test_another_kind_under_the_same_name_is_not_it() -> None:
    """A workflow reusing the name for its own setting has not made the promise."""

    schema = {
        "type": "object",
        "properties": {MATTING_SETTING_KEY: {"type": "number", "x-lm-atelier-kind": "upscale"}},
    }

    assert workflow_declares_matting(schema) is False


@pytest.mark.parametrize("schema", [None, {}, {"properties": None}, {"properties": {}}])
def test_a_workflow_declaring_nothing_cannot_be_asked_to_isolate(schema: Any) -> None:
    assert workflow_declares_matting(schema) is False


def test_a_graph_that_could_matte_but_says_nothing_is_still_not_offered() -> None:
    """The point of the rule, written down.

    A revision carrying the background-removal nodes is capable, and it is
    still not offered, because the tool's promise and the workflow's promise
    have to be the same promise. Recognizing this from the graph is the thing
    this contract refuses to do.
    """

    capable_but_silent = {
        "type": "object",
        "properties": {"denoise": {"type": "number"}},
    }

    assert workflow_declares_matting(capable_but_silent) is False
