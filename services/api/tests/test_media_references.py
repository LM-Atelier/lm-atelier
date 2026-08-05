"""How many reference images a workflow can use, and refusing to exceed it."""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.media_references import UNBOUNDED, exceeds_capacity, reference_capacity


def _graph(*placeholders: str) -> dict[str, Any]:
    return {"nodes": [{"inputs": {"image": value}} for value in placeholders]}


def test_a_graph_naming_the_whole_list_takes_everything() -> None:
    assert reference_capacity(_graph("${input_images}")) == UNBOUNDED


def test_a_graph_naming_only_the_first_takes_one() -> None:
    assert reference_capacity(_graph("${input_image}")) == 1


def test_numbered_slots_bound_the_count() -> None:
    assert reference_capacity(_graph("${input_image_0}", "${input_image_1}")) == 2


def test_slots_and_a_first_input_coexist() -> None:
    # Slot 0 is the same picture as the plain first input, so naming both does
    # not make the graph take three.
    capacity = reference_capacity(_graph("${input_image}", "${input_image_0}", "${input_image_1}"))
    assert capacity == 2


def test_a_graph_naming_no_image_takes_none() -> None:
    """A real answer: this is text-to-image being handed a reference."""
    assert reference_capacity(_graph("a prompt", "${seed}")) == 0


@pytest.mark.parametrize("supplied", [0, 1])
def test_one_reference_never_exceeds_anything(supplied: int) -> None:
    # Even a text-to-image graph is left alone here; using one attachment as
    # an edit source is the ordinary path and is decided elsewhere.
    assert exceeds_capacity(_graph("a prompt"), supplied) is None


def test_refuses_when_more_were_given_than_the_graph_can_use() -> None:
    """The defect: four references, a graph that takes one, and no complaint.

    Silently conditioning on the first is indistinguishable from a bad model,
    so the count that was actually usable comes back to be said out loud.
    """
    assert exceeds_capacity(_graph("${input_image}"), 4) == 1


def test_allows_exactly_as_many_as_the_slots_declare() -> None:
    graph = _graph("${input_image_0}", "${input_image_1}", "${input_image_2}")

    assert exceeds_capacity(graph, 3) is None
    assert exceeds_capacity(graph, 4) == 3


def test_a_graph_taking_the_whole_list_never_refuses() -> None:
    assert exceeds_capacity(_graph("${input_images}"), 40) is None


def test_finds_placeholders_however_deeply_the_graph_nests_them() -> None:
    nested = {"a": [{"b": {"c": ["${input_image_0}", {"d": "${input_image_1}"}]}}]}

    assert reference_capacity(nested) == 2
