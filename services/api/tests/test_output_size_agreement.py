"""Warning that a picture is not the size you asked for, and refusing to guess.

The state worth showing a person is one of five, and four of them must never
become a warning. So most of these are about the refusals rather than the
comparison: a check that only proved "different sizes warn" would be satisfied by
a function that warns at every opportunity, which is the failure this area has
already produced three times.

These exercise the decision in isolation. What they cannot show is that the
decision is reached with the right arguments at the seam - that the settings come
from the accepted run rather than the artifact, and that the confirmation is the
executed graph's rather than the stored one's. That belongs to the change that
wires it and is asserted there.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_lm.output_origin import record_for, stated_origin
from local_lm.output_size_agreement import disagrees, size_agreement

SAVED = record_for(stated_origin("9", "output", "images"), "comfyui")
THROWAWAY = record_for(stated_origin("4", "temp", "images"), "comfyui")
UNATTRIBUTED = record_for(None, "comfyui")


def measured(width: int, height: int) -> dict[str, Any]:
    return {"v": 1, "state": "measured", "raster_width": width, "raster_height": height}


def asked_for(width: int, height: int) -> dict[str, Any]:
    return {"width": width, "height": height, "steps": 20}


def test_a_picture_of_the_size_you_asked_for_agrees() -> None:
    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(1024, 768),
        origin=SAVED,
        binding_confirmed=True,
    )

    assert answer["state"] == "agreed"
    assert disagrees(answer) is False
    assert (answer["requested_width"], answer["raster_width"]) == (1024, 1024)


def test_a_picture_of_another_size_is_the_thing_worth_saying() -> None:
    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(768, 768),
        origin=SAVED,
        binding_confirmed=True,
    )

    assert answer["state"] == "disagreed"
    assert disagrees(answer) is True
    # Both pairs are carried, because "it is the wrong size" is not something a
    # person can act on and "you asked for 1024x768 and got 768x768" is.
    assert answer["requested_height"] == 768
    assert answer["raster_width"] == 768


def test_a_preview_nodes_throwaway_is_never_judged() -> None:
    """The file that is not yours, at a size that would otherwise warn.

    A preview node routinely writes something of a different size from the save
    branch. Judging it would warn on a run where nothing at all went wrong, and
    the person cannot tell which of the two pictures the warning is about.
    """

    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(512, 384),
        origin=THROWAWAY,
        binding_confirmed=True,
    )

    assert answer == {"v": 1, "state": "not_assessed", "reason": "throwaway"}


def test_a_graph_that_was_not_confirmed_is_not_judged_even_when_it_differs() -> None:
    """The legitimate doubling, which must not become a warning.

    A hires-fix upscale or a scale after the decode produces a picture larger
    than the declared pair on purpose. The executed-graph confirmation is what
    refuses to vouch for those, so an unconfirmed binding has to stop the
    comparison BEFORE the sizes are looked at - not after, reporting a mismatch
    it then has to explain away.
    """

    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(2048, 1536),
        origin=SAVED,
        binding_confirmed=False,
    )

    assert answer == {"v": 1, "state": "not_assessed", "reason": "binding_unconfirmed"}


@pytest.mark.parametrize("supplied", [1, "true", "yes", [True], {}, None])
def test_only_exactly_true_counts_as_a_confirmed_binding(supplied: object) -> None:
    """A caller that has not looked must not read as having looked.

    Every one of these is truthy or falsey by accident rather than by decision.
    Accepting any of them would let a caller that forgot to run the confirmation
    produce warnings backed by nothing.
    """

    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(512, 512),
        origin=SAVED,
        binding_confirmed=supplied,
    )

    assert answer["state"] == "not_assessed"
    assert answer["reason"] == "binding_unconfirmed"


def test_a_file_we_cannot_attribute_is_not_judged() -> None:
    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement=measured(512, 512),
        origin=UNATTRIBUTED,
        binding_confirmed=True,
    )

    assert answer == {"v": 1, "state": "not_assessed", "reason": "origin_unknown"}


def test_a_file_that_could_not_be_measured_is_not_judged() -> None:
    answer = size_agreement(
        requested_settings=asked_for(1024, 768),
        measurement={"v": 1, "state": "unmeasured", "about": "file", "reason": "not_a_png"},
        origin=SAVED,
        binding_confirmed=True,
    )

    assert answer == {"v": 1, "state": "not_assessed", "reason": "unmeasured"}


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"width": 1024},
        {"height": 768},
        {"width": 0, "height": 768},
        {"width": -1024, "height": 768},
        {"width": "1024", "height": "768"},
        {"width": 1024.0, "height": 768.0},
        None,
        "1024x768",
    ],
)
def test_a_run_that_did_not_ask_for_a_size_has_nothing_to_disagree_with(settings: object) -> None:
    answer = size_agreement(
        requested_settings=settings,
        measurement=measured(512, 512),
        origin=SAVED,
        binding_confirmed=True,
    )

    assert answer == {"v": 1, "state": "not_assessed", "reason": "no_size_requested"}


def test_a_boolean_width_is_not_read_as_the_number_one() -> None:
    """`True` is an int in Python, and a settings map carrying one is a defect.

    Read as 1 it would produce a confident mismatch against every real picture,
    and the reader would be sent to look at a generation rather than at whatever
    wrote a boolean into a size.
    """

    answer = size_agreement(
        requested_settings={"width": True, "height": True},
        measurement=measured(1, 1),
        origin=SAVED,
        binding_confirmed=True,
    )

    assert answer["state"] == "not_assessed"
    assert answer["reason"] == "no_size_requested"


@pytest.mark.parametrize(
    "record",
    [
        {"v": 1, "state": "agreed"},
        {"v": 1, "state": "not_assessed", "reason": "throwaway"},
        {"v": 1, "state": "not_assessed", "reason": "binding_unconfirmed"},
        {"state": "disagreed but not really"},
        None,
        "disagreed",
    ],
)
def test_nothing_but_a_disagreement_reads_as_one(record: object) -> None:
    assert disagrees(record) is False
