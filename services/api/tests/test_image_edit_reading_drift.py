"""Two readings that disagree about something whose pixels did not move."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from test_image_edit_verification_orchestration import _TEST_CLAIM, _picture, _verification_world

from local_lm.image_edit_difference import ChangedArea, compare_region
from local_lm.image_edit_verification import (
    LOCAL_CHANGE_THRESHOLD,
    MIN_DRIFT_REGION,
    ChangeAttribution,
    InventoryChange,
    parse_subject_location,
    reading_contradicted,
    without_contradicted,
)

SQUARE = ChangedArea(left=0.25, top=0.25, right=0.75, bottom=0.75)
WHOLE = ChangedArea(left=0.0, top=0.0, right=1.0, bottom=1.0)
CORNER = ChangedArea(left=0.0, top=0.0, right=0.125, bottom=0.125)


def box(area: ChangedArea | None) -> str:
    if area is None:
        return json.dumps({"present": False})
    return json.dumps(
        {
            "present": True,
            "left": area.left,
            "top": area.top,
            "right": area.right,
            "bottom": area.bottom,
        }
    )


def test_a_location_is_a_box_inside_the_picture_or_nothing() -> None:
    placed = parse_subject_location(box(SQUARE))
    assert placed.area() == SQUARE
    assert parse_subject_location(box(None)).area() is None
    # A box whose edges enclose nothing places nothing, rather than failing.
    assert parse_subject_location(box(ChangedArea(0.5, 0.2, 0.5, 0.4))).area() is None


@pytest.mark.parametrize(
    "answer",
    [
        '{"present": false, "left": 0.1, "top": 0.1, "right": 0.2, "bottom": 0.2}',
        '{"present": true, "left": 0.1, "top": 0.1, "right": 1.2, "bottom": 0.2}',
        '{"present": true, "left": 0.1, "top": 0.1, "right": 0.2}',
        '{"present": true, "left": true, "top": 0.1, "right": 0.2, "bottom": 0.2}',
        '{"present": true, "left": 0.1, "top": 0.1, "right": 0.2, "bottom": 0.2, "why": "x"}',
        '{"present": "yes"}',
        "[0.1, 0.1, 0.2, 0.2]",
        "not json",
    ],
)
def test_a_location_outside_the_contract_is_refused(answer: str) -> None:
    with pytest.raises(ValueError):
        parse_subject_location(answer)


def test_a_region_is_measured_with_the_requested_box_left_out() -> None:
    source, result = _picture(None), _picture((16, 16, 48, 48))
    around, measured = compare_region(source, result, WHOLE, [SQUARE.widened(1 / 32)])
    assert around.comparable and measured > MIN_DRIFT_REGION
    assert reading_contradicted(around, measured)
    inside, _ = compare_region(source, result, SQUARE)
    assert not reading_contradicted(inside, 0.25)
    nothing_left, empty = compare_region(source, result, SQUARE, [SQUARE])
    assert empty == 0.0 and not nothing_left.comparable
    assert not reading_contradicted(nothing_left, empty)


def test_a_region_too_small_or_moved_too_much_contradicts_nothing() -> None:
    source, result = _picture(None), _picture((16, 16, 48, 48))
    still, measured = compare_region(source, result, WHOLE, [SQUARE.widened(1 / 32)])
    assert not reading_contradicted(still, MIN_DRIFT_REGION / 2)
    assert still.largest_local_difference is not None
    assert still.largest_local_difference < LOCAL_CHANGE_THRESHOLD
    moved, measured = compare_region(source, result, WHOLE)
    assert moved.largest_local_difference is not None
    assert moved.largest_local_difference >= LOCAL_CHANGE_THRESHOLD
    assert not reading_contradicted(moved, measured)


def test_dropping_a_contradicted_reading_keeps_the_request_pointing_at_its_change() -> None:
    changes = (
        InventoryChange("floor", "dark grey, flat", "beige, smooth"),
        InventoryChange("mug", "blue", "green"),
        InventoryChange("sphere", "glossy", "shiny"),
    )
    attribution = ChangeAttribution(
        subject_present=True, operation="change", requested=(1,), as_asked=True
    )
    kept, renumbered = without_contradicted(changes, attribution, {0, 2})
    assert kept == (changes[1],)
    assert renumbered.requested == (0,)
    with pytest.raises(ValueError):
        without_contradicted(changes, attribution, {1})


def test_widening_stays_inside_the_picture_and_overlap_needs_shared_area() -> None:
    assert CORNER.widened(0.25) == ChangedArea(0.0, 0.0, 0.375, 0.375)
    assert SQUARE.overlaps(WHOLE)
    assert not CORNER.overlaps(SQUARE)
    assert not ChangedArea(0.0, 0.0, 0.25, 0.25).overlaps(SQUARE)


FLOOR_DRIFT = (
    '[{"subject": "mug", "appearance": "blue"}, '
    '{"subject": "floor", "appearance": "dark grey, flat"}]',
    '[{"subject": "mug", "appearance": "green"}, '
    '{"subject": "floor", "appearance": "beige, smooth"}]',
    # Sorted, the differences are the floor (0) and the mug (1).
    '{"subject_present": true, "operation": "change", "requested": [1], "as_asked": true}',
)


async def verify(answers: tuple[str, ...], *patches: tuple[int, int, int, int]) -> Any:
    job, orchestrator, world = _verification_world(lose_at="never", answers=answers)
    pictures = {"artifact-source": _picture(None), "artifact-result": _picture(*patches)}
    orchestrator.artifacts.verified_bytes = Mock(  # type: ignore[method-assign]
        side_effect=lambda artifact, *, maximum_bytes: pictures[artifact.id]
    )
    retry = SimpleNamespace(
        run=SimpleNamespace(id="run-retry", work_plan_id="plan-retry", provenance_json={})
    )
    orchestrator._create_image_edit_verification_retry = AsyncMock(return_value=retry)  # type: ignore[method-assign]
    await orchestrator._execute_image_edit_verification(job.id, _TEST_CLAIM)
    return orchestrator._persist_image_edit_verification.call_args.args[2], world


async def test_a_reading_slip_about_an_unmoved_floor_no_longer_refuses_a_precise_edit() -> None:
    record, world = await verify(
        FLOOR_DRIFT
        + (
            box(SQUARE),  # where the mug is
            box(WHOLE),  # where the floor is
            '{"subjects": [0]}',  # the one changed area holds the mug, now number 0
        ),
        (16, 16, 48, 48),
    )
    assert record["reason"] == "accepted"
    assert record["assessment"]["unrelated_content_preserved"] is True
    [reading] = record["contradicted_readings"]
    assert reading["subject"] == "floor"
    assert reading["largest_local_difference"] < LOCAL_CHANGE_THRESHOLD
    assert "Find this thing in the attached picture" in world["questions"][3]
    assert len(world["questions"]) == 6


async def test_a_real_replacement_still_refuses_the_edit_beside_a_reading_slip() -> None:
    record, world = await verify(
        (
            '[{"subject": "mug", "appearance": "blue"}, '
            '{"subject": "floor", "appearance": "dark grey, flat"}, '
            '{"subject": "sphere", "appearance": "glossy"}]',
            '[{"subject": "mug", "appearance": "green"}, '
            '{"subject": "floor", "appearance": "beige, smooth"}, '
            '{"subject": "dog", "appearance": "brown"}]',
            # Sorted: dog (0), floor (1), mug (2), sphere (3).
            '{"subject_present": true, "operation": "change", "requested": [2], "as_asked": true}',
            box(SQUARE),  # the mug
            box(CORNER),  # the dog, in the result
            box(WHOLE),  # the floor
            box(CORNER),  # the sphere, in the source
        ),
        (16, 16, 48, 48),
        (0, 0, 8, 8),
    )
    # The corner really changed, so neither it nor the floor around it is a slip.
    assert "contradicted_readings" not in record
    assert record["assessment"]["unrelated_content_preserved"] is False
    assert record["reason"] != "accepted"
    assert len(world["questions"]) == 7


@pytest.mark.parametrize(
    "floor",
    [
        pytest.param("not json", id="unreadable"),
        pytest.param(box(None), id="not-found"),
        pytest.param(box(SQUARE), id="inside-the-requested-box"),
    ],
)
async def test_a_reading_that_cannot_be_measured_apart_still_counts(floor: str) -> None:
    record, _world = await verify(FLOOR_DRIFT + (box(SQUARE), floor), (16, 16, 48, 48))
    assert "contradicted_readings" not in record
    assert record["assessment"]["unrelated_content_preserved"] is False


async def test_an_unplaced_requested_thing_contradicts_nothing() -> None:
    record, world = await verify(FLOOR_DRIFT + (box(None),), (16, 16, 48, 48))
    assert "contradicted_readings" not in record
    assert record["assessment"]["unrelated_content_preserved"] is False
    # Nothing beside the mug is measured once the mug cannot be placed.
    assert len(world["questions"]) == 4
