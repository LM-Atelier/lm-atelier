"""What it costs to reach a requested shape from a picture you already have.

The arithmetic is small, so these are mostly about the properties that make the
answer USABLE rather than about individual sums: that extending keeps every
pixel, that cropping invents none, that the two are exact inverses of the same
ratio question, and that the same request twice gives the same rectangle.

The 4:3-to-16:9 case and its inverse are the ones the acceptance names, so they
are checked by hand rather than only by property.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from local_lm.source_fit import (
    MAX_FIT_FRACTION,
    SourceFitError,
    fit_source_to_shape,
)

SIDES = ("top", "right", "bottom", "left")


def fit(source: tuple[int, int], target: tuple[int, int], mode: str):
    return fit_source_to_shape(
        source_width=source[0],
        source_height=source[1],
        target_width=target[0],
        target_height=target[1],
        mode=mode,
    )


def test_a_four_three_source_extends_sideways_to_reach_sixteen_nine() -> None:
    """The case the acceptance names, worked by hand.

    1024x768 at 16:9 wants 1365 wide (768 * 16/9 truncated), so 341 pixels of
    new region. Split 170 left and 171 right - the extra pixel goes right, and
    the margins are fractions of the WIDTH because they are horizontal.
    """
    answer = fit((1024, 768), (16, 9), "extend")

    assert answer.mode == "extend"
    assert answer.kept == (0, 0, 1024, 768), "extending keeps the whole picture"
    assert answer.margins["top"] == 0.0
    assert answer.margins["bottom"] == 0.0
    assert answer.margins["left"] == pytest.approx(170 / 1024)
    assert answer.margins["right"] == pytest.approx(171 / 1024)


def test_the_inverse_extends_upward_and_downward() -> None:
    """A wide source asked for a tall shape grows on the other axis.

    1920x1080 at 3:4 wants 2560 tall, so 1480 of new region, split 740 and 740 -
    and these are fractions of the HEIGHT, which is the distinction that would
    be invisible on a square picture.
    """
    answer = fit((1920, 1080), (3, 4), "extend")

    assert answer.margins["left"] == 0.0
    assert answer.margins["right"] == 0.0
    assert answer.margins["top"] == pytest.approx(740 / 1080)
    assert answer.margins["bottom"] == pytest.approx(740 / 1080)


def test_cropping_the_same_pair_removes_the_overhang_instead() -> None:
    """1024x768 to 16:9 by crop keeps 1024x576, centred vertically."""
    answer = fit((1024, 768), (16, 9), "crop")

    assert answer.mode == "crop"
    assert answer.kept == (0, 96, 1024, 576)
    assert all(answer.margins[side] == 0.0 for side in SIDES), (
        "a crop removes rather than invents, so there is no region to paint"
    )


def test_a_shape_that_already_agrees_costs_nothing() -> None:
    """A shape that already agrees must cost nothing.

    The production code compares exact rationals, and this test does NOT prove
    that choice matters: IEEE division is correctly rounded, so within the
    dimensions this product admits no pair is equal as rationals and unequal as
    floats, and swapping the comparison to floats leaves this whole module
    green. Said here rather than implied, so nobody reads this case as evidence
    for something it cannot show.
    """
    for mode in ("extend", "crop"):
        answer = fit((1920, 1080), (16, 9), mode)
        assert all(answer.margins[side] == 0.0 for side in SIDES)
        assert answer.kept == (0, 0, 1920, 1080)


@pytest.mark.parametrize(
    ("source", "target"),
    [((1024, 768), (16, 9)), ((1920, 1080), (3, 4)), ((999, 501), (1, 1))],
)
def test_extending_never_loses_a_pixel_and_cropping_never_invents_one(
    source: tuple[int, int], target: tuple[int, int]
) -> None:
    """The property that separates the two modes, stated once."""
    extended = fit(source, target, "extend")
    cropped = fit(source, target, "crop")

    assert extended.kept == (0, 0, source[0], source[1])
    assert any(extended.margins[side] > 0 for side in SIDES)

    left, top, width, height = cropped.kept
    assert width <= source[0] and height <= source[1]
    assert left + width <= source[0] and top + height <= source[1]
    assert all(cropped.margins[side] == 0.0 for side in SIDES)


@pytest.mark.parametrize(
    ("source", "target"),
    [((1025, 768), (16, 9)), ((1024, 769), (16, 9)), ((333, 777), (5, 3))],
)
def test_an_odd_remainder_is_split_the_same_way_every_time(
    source: tuple[int, int], target: tuple[int, int]
) -> None:
    """Deterministic, because a rectangle that moves between runs is uncheckable.

    Nobody can see which side gained the extra pixel. What matters is that the
    same request twice gives the same answer, and that the extra pixel is
    accounted for rather than lost.
    """
    first = fit(source, target, "extend")
    second = fit(source, target, "extend")
    assert first.margins == second.margins
    assert first.kept == second.kept

    horizontal = first.margins["left"] > 0 or first.margins["right"] > 0
    span = source[0] if horizontal else source[1]
    near = first.margins["left"] if horizontal else first.margins["top"]
    far = first.margins["right"] if horizontal else first.margins["bottom"]
    # Round rather than truncate: these came back through a float division.
    near_pixels = round(near * span)
    far_pixels = round(far * span)
    assert far_pixels - near_pixels in (0, 1), "the extra pixel goes to one side, not both"


def test_the_extended_result_really_is_the_requested_shape() -> None:
    """Checks the ANSWER against the request, not against its own derivation."""
    for source, target in (((1024, 768), (16, 9)), ((1920, 1080), (3, 4)), ((640, 640), (21, 9))):
        answer = fit(source, target, "extend")
        width = (
            source[0]
            + round(answer.margins["left"] * source[0])
            + round(answer.margins["right"] * source[0])
        )
        height = (
            source[1]
            + round(answer.margins["top"] * source[1])
            + round(answer.margins["bottom"] * source[1])
        )
        # One pixel of slack, and only because the requested ratio may not divide
        # the source exactly; the shape is right to within that.
        assert abs(Fraction(width, height) - Fraction(*target)) < Fraction(1, 100)


def test_a_crop_that_would_keep_nothing_is_refused() -> None:
    with pytest.raises(SourceFitError, match="nothing of the picture"):
        fit((10000, 1), (1, 1000), "crop")


def test_an_extension_that_would_invent_more_than_it_kept_is_refused() -> None:
    """The ceiling exists so a fit cannot quietly become a hallucination."""
    with pytest.raises(SourceFitError, match="more than it kept"):
        fit((100, 1000), (10, 1), "extend")


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"source_width": 0}, "zero is not a picture"),
        ({"source_height": -10}, "negative is not a picture"),
        ({"target_width": 1.5}, "a float is not a whole number of pixels"),
        ({"target_height": True}, "a boolean is not a dimension"),
        ({"source_width": "1024"}, "a string that looks like a number is not one"),
    ],
)
def test_a_dimension_that_is_not_one_is_refused(kwargs: dict[str, object], why: str) -> None:
    request: dict[str, object] = {
        "source_width": 1024,
        "source_height": 768,
        "target_width": 16,
        "target_height": 9,
        "mode": "extend",
    }
    request.update(kwargs)
    with pytest.raises(SourceFitError):
        fit_source_to_shape(**request)  # type: ignore[arg-type]


def test_a_mode_nobody_defined_is_refused() -> None:
    with pytest.raises(SourceFitError, match="extension or a crop"):
        fit((1024, 768), (16, 9), "squash")


def test_the_payload_is_readable_without_the_dataclass() -> None:
    """It is stored, so it has to survive being JSON."""
    payload = fit((1024, 768), (16, 9), "crop").payload()

    assert payload["v"] == 1
    assert payload["mode"] == "crop"
    assert payload["kept"] == {"left": 0, "top": 96, "width": 1024, "height": 576}
    assert set(payload["margins"]) == set(SIDES)


def test_the_ceiling_matches_what_outpainting_already_refuses() -> None:
    """Two ceilings that disagreed would let this module propose a refused margin."""
    from local_lm.outpaint_workflows import MAX_MARGIN_FRACTION

    assert MAX_FIT_FRACTION == MAX_MARGIN_FRACTION
