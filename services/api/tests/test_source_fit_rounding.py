"""What the crop rectangle rounds to, pinned so a consumer can rely on it.

The kept rectangle is exactly the requested ratio only when the exact edge is a
whole number. When it is not - 168.75 for 9:16 from a 400x300 source - the
remainder is dropped and the rectangle is up to a pixel short on one axis. That
is under a pixel and it is still real, and anything that scales this rectangle
to an exact-ratio output is applying slightly different factors per axis.

This is pinned in its own file rather than folded into the property tests
beside it, because those are deliberately about what holds for every shape and
this is about the exact cases where exactness fails. A reader looking for
"is the crop exact" should find the answer stated rather than inferred from a
tolerance.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from local_lm.source_fit import fit_source_to_shape


def crop(source: tuple[int, int], target: tuple[int, int]) -> tuple[int, int, int, int]:
    answer = fit_source_to_shape(
        source_width=source[0],
        source_height=source[1],
        target_width=target[0],
        target_height=target[1],
        mode="crop",
    )
    return answer.kept


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ((16, 9), (0, 37, 400, 225)),
        ((1, 1), (50, 0, 300, 300)),
    ],
)
def test_a_whole_edge_crops_to_exactly_the_requested_ratio(
    target: tuple[int, int], expected: tuple[int, int, int, int]
) -> None:
    """Where the exact edge is a whole number, nothing is lost to rounding."""
    kept = crop((400, 300), target)

    assert kept == expected
    assert Fraction(kept[2], kept[3]) == Fraction(*target)


@pytest.mark.parametrize(
    ("target", "expected", "exact_edge"),
    [
        ((9, 16), (116, 0, 168, 300), Fraction(675, 4)),
        ((3, 2), (0, 17, 400, 266), Fraction(800, 3)),
    ],
)
def test_a_fractional_edge_truncates_and_misses_the_ratio(
    target: tuple[int, int], expected: tuple[int, int, int, int], exact_edge: Fraction
) -> None:
    """And where it is not, the rectangle is short and says so here.

    Recorded as the exact rectangle it should be, so the size of what is being
    dropped is visible rather than buried in a tolerance.
    """
    kept = crop((400, 300), target)

    assert kept == expected
    assert Fraction(kept[2], kept[3]) != Fraction(*target)
    # Truncated, never rounded up: the rectangle stays inside the source.
    assert int(exact_edge) in (kept[2], kept[3])
    assert exact_edge - int(exact_edge) > 0


def test_the_cropped_edge_is_always_the_truncated_exact_edge() -> None:
    """The truncation rule itself, over a grid rather than four hand cases.

    Two properties together, because either alone is weak. The rectangle stays
    inside the picture - so a consumer is never handed an edge the source does
    not have - AND the cropped edge is exactly the floor of the rational edge,
    never a rounding of it. That second half is what makes the first one a rule
    rather than a coincidence of the sizes chosen.
    """
    checked = 0
    for width in range(97, 130):
        for height in range(61, 80):
            for target in ((16, 9), (9, 16), (3, 2), (2, 3), (1, 1)):
                ratio = Fraction(*target)
                x, y, kept_width, kept_height = crop((width, height), target)

                assert x >= 0 and y >= 0
                assert x + kept_width <= width
                assert y + kept_height <= height

                if Fraction(width, height) > ratio:
                    assert kept_height == height
                    assert kept_width == int(Fraction(height) * ratio)
                elif Fraction(width, height) < ratio:
                    assert kept_width == width
                    assert kept_height == int(Fraction(width) / ratio)
                else:
                    assert (kept_width, kept_height) == (width, height)
                checked += 1

    # An empty or tiny grid would make every assertion above vacuous.
    assert checked == 33 * 19 * 5
