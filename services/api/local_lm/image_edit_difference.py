"""Decide whether an edit changed the picture at all, without asking a model.

The vision verifier answers "did the requested change happen". It cannot be
trusted to answer "did *anything* happen": a model that is not
instruction-edit tuned can return an unchanged image, and a verifier asked
to confirm a described edit has been observed approving one at high
confidence. A pixel comparison cannot be fooled that way, costs no inference,
and runs before the model is consulted.

Deliberately narrow about what it proves:

- Below the threshold means **nothing visible changed**. That is conclusive,
  and no model opinion should override it.
- Above the threshold means only **something changed** - not that the
  requested change happened, and not that unrelated content was preserved.
  Those remain questions for the verifier.

An edit check measures locally, with compare_edit. A whole-picture average
dilutes a small edit by everything that stayed, so however low its threshold,
a small enough real change reads as none. compare_edit instead compares each
part of the picture separately, only where the edit was asked, and calls the
result unchanged only when every part is.
"""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageMath, UnidentifiedImageError

# Small enough that re-encoding noise and one-pixel resampling differences
# average away, large enough that a genuine local edit (a recoloured object,
# a changed garment) still moves the mean well past the threshold.
COMPARISON_SIZE = (64, 64)
# Mean absolute difference per channel value, 0-255. Chosen an order of
# magnitude above observed re-encode noise (well under 1.0) and an order
# below a real edit (typically 5+).
UNCHANGED_THRESHOLD = 2.0
#: A mask value above this counts as selected.
SELECTED_LEVEL = 127
#: An edit check works on the source reduced by a whole factor until its
#: longer side is at most this, which keeps the reduction an exact average.
WORKING_LONG_SIDE = 512
#: The parts an edit check compares separately: this many across and down.
LOCAL_GRID = 32
#: How finely a mask of another size is sampled per source pixel, so a
#: selection that covers part of a pixel still counts that pixel.
MASK_SAMPLES = 4


@dataclass(frozen=True)
class ChangedArea:
    """Where one connected run of changed parts sits, in fractions of the picture.

    Fractions rather than pixels because the comparison works on a reduced grid
    while a reader works on the original, and the two pictures are the same
    shape by the time anything is compared. A reader that wants to look at one
    area multiplies by that picture's own width and height.
    """

    left: float
    top: float
    right: float
    bottom: float

    def provenance(self) -> dict[str, float]:
        return {
            "left": round(self.left, 4),
            "top": round(self.top, 4),
            "right": round(self.right, 4),
            "bottom": round(self.bottom, 4),
        }


@dataclass(frozen=True)
class ImageDifference:
    """How much two images differ, and whether that counts as a change.

    `largest_local_difference` is set by compare_edit: the largest difference
    any one part of the compared area showed, which is what decides `changed`
    there. compare_images leaves it unset and decides on the whole average.
    """

    mean_absolute_difference: float
    changed: bool
    comparable: bool
    largest_local_difference: float | None = None
    #: How many separate areas of the picture changed, counted by compare_edit
    #: from its own grid. A reader that knows how many things were reported
    #: changed can tell "everything that moved was named" from "something else
    #: moved too"; an aggregate difference cannot say that.
    changed_regions: int | None = None
    #: Where each of those areas sits. Counting told a reader that something
    #: unnamed moved; this says where to look, which is what lets a reader ask
    #: what is in it rather than only how many there were.
    changed_areas: tuple[ChangedArea, ...] | None = None

    def provenance(self) -> dict[str, object]:
        recorded: dict[str, object] = {
            "mean_absolute_difference": round(self.mean_absolute_difference, 4),
            "changed": self.changed,
            "comparable": self.comparable,
            "threshold": UNCHANGED_THRESHOLD,
        }
        if self.largest_local_difference is not None:
            recorded["largest_local_difference"] = round(self.largest_local_difference, 4)
        if self.changed_regions is not None:
            recorded["changed_regions"] = self.changed_regions
        if self.changed_areas is not None:
            recorded["changed_areas"] = [area.provenance() for area in self.changed_areas]
        return recorded


INCOMPARABLE = ImageDifference(mean_absolute_difference=0.0, changed=True, comparable=False)


def compare_images(source: bytes, result: bytes) -> ImageDifference:
    """Compare two encoded images at a small fixed size.

    Returns `comparable=False` when either image cannot be read, so an
    unreadable file never masquerades as "unchanged" and stop the pipeline on
    a false certainty.
    """

    try:
        source_pixels = _normalized(source)
        result_pixels = _normalized(result)
    except (OSError, UnidentifiedImageError, ValueError):
        return ImageDifference(
            mean_absolute_difference=0.0,
            changed=True,
            comparable=False,
        )
    total = sum(abs(left - right) for left, right in zip(source_pixels, result_pixels, strict=True))
    mean = total / len(source_pixels) if source_pixels else 0.0
    return ImageDifference(
        mean_absolute_difference=mean,
        changed=mean > UNCHANGED_THRESHOLD,
        comparable=True,
    )


def compare_edit(
    source: bytes, result: bytes, *, mask: bytes | None = None, invert: bool = False
) -> ImageDifference:
    """Measure an edit where it was asked, part by part.

    The result is brought to the source's size, both are reduced by the same
    whole factor, and the picture is divided into LOCAL_GRID parts across and
    down. Each part compares the average colour of source and result over its
    selected pixels only, weighted by how much of each pixel the selection
    covers, so a change in pixels the selection excludes - outside it, or in a
    hole inside it - never counts, and a pixel the selection only partly
    covers still does. The result is unchanged only when every part holding
    any selected pixel is under the threshold, so a small change is not
    averaged away by the rest of the picture.

    Without a mask the whole picture is selected. An unreadable image or mask,
    or a mask that selects nothing, is incomparable rather than unchanged.
    """

    try:
        before, after, weights = _working_pictures(source, result, mask, invert=invert)
    except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError):
        return INCOMPARABLE
    grid = (min(LOCAL_GRID, weights.width), min(LOCAL_GRID, weights.height))
    coverage = list(weights.resize(grid, Image.Resampling.BOX).get_flattened_data())
    parts = [0.0] * len(coverage)
    for before_band, after_band in zip(before.split(), after.split(), strict=True):
        weighted = ImageMath.lambda_eval(
            lambda names: (names["before"] - names["after"]) * names["weights"],
            before=before_band.convert("F"),
            after=after_band.convert("F"),
            weights=weights,
        )
        if not isinstance(weighted, Image.Image):
            return INCOMPARABLE
        sums = weighted.resize(grid, Image.Resampling.BOX).get_flattened_data()
        for index, (total, covered) in enumerate(zip(sums, coverage, strict=True)):
            if isinstance(total, float) and isinstance(covered, float) and covered > 0:
                # Signed first and averaged, then its size: noise cancels
                # inside a part, a real change does not.
                parts[index] += abs(total / covered) / 3
    measured = [
        part
        for part, covered in zip(parts, coverage, strict=True)
        if isinstance(covered, float) and covered > 0
    ]
    if not measured:
        return INCOMPARABLE
    largest = max(measured)
    areas = _changed_areas(parts, coverage, grid)
    return ImageDifference(
        mean_absolute_difference=sum(measured) / len(measured),
        changed=largest > UNCHANGED_THRESHOLD,
        comparable=True,
        largest_local_difference=largest,
        changed_regions=len(areas),
        changed_areas=areas,
    )


def _changed_areas(
    parts: Sequence[float], coverage: Sequence[object], grid: tuple[int, int]
) -> tuple[ChangedArea, ...]:
    """Where each separate area of the grid changed, touching parts counted once.

    Two things changed in two places is two areas; one thing spanning several
    parts is still one. The count alone lets a reader ask whether every area
    that moved was among the things reported changed; the bounds let it ask the
    harder question, which is what is inside one.

    Each area is returned as the box enclosing its parts, in fractions of the
    picture, ordered from the top left so that two runs over the same picture
    describe its areas in the same order.
    """

    across, down = grid
    changed = {
        index
        for index, (part, covered) in enumerate(zip(parts, coverage, strict=True))
        if isinstance(covered, float) and covered > 0 and part > UNCHANGED_THRESHOLD
    }
    areas: list[ChangedArea] = []
    while changed:
        start = min(changed)
        changed.remove(start)
        members = [start]
        frontier = [start]
        while frontier:
            index = frontier.pop()
            row, column = divmod(index, across)
            for neighbour_row, neighbour_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if 0 <= neighbour_row < down and 0 <= neighbour_column < across:
                    neighbour = neighbour_row * across + neighbour_column
                    if neighbour in changed:
                        changed.remove(neighbour)
                        frontier.append(neighbour)
                        members.append(neighbour)
        rows = [index // across for index in members]
        columns = [index % across for index in members]
        areas.append(
            ChangedArea(
                left=min(columns) / across,
                top=min(rows) / down,
                right=(max(columns) + 1) / across,
                bottom=(max(rows) + 1) / down,
            )
        )
    return tuple(sorted(areas, key=lambda area: (area.top, area.left)))


def _working_pictures(
    source: bytes, result: bytes, mask: bytes | None, *, invert: bool
) -> tuple[Image.Image, Image.Image, Image.Image]:
    """Bring source, result and selection weights onto one grid, each reduced exactly."""

    with Image.open(io.BytesIO(source)) as opened:
        before = opened.convert("RGB")
    with Image.open(io.BytesIO(result)) as opened:
        after = opened.convert("RGB")
    if after.size != before.size:
        after = after.resize(before.size, Image.Resampling.BOX)
    factor = max(1, math.ceil(max(before.size) / WORKING_LONG_SIDE))
    if mask is None:
        weights = Image.new("F", before.size, 1.0)
    else:
        weights = _selection_weights(mask, before.size, invert=invert)
    return before.reduce(factor), after.reduce(factor), weights.reduce(factor)


def _selection_weights(mask: bytes, size: tuple[int, int], *, invert: bool) -> Image.Image:
    """Weigh each source pixel by how much of it the selection covers, from 0 to 1."""

    with Image.open(io.BytesIO(mask)) as opened:
        levels = opened.convert("L")
    # 255 where selected. Inverting selects everything the mask leaves out.
    selected = levels.point(lambda value: 0 if (value > SELECTED_LEVEL) == invert else 255)
    if selected.size != size:
        # Sample each source pixel at several points and average them, so a
        # pixel the selection only partly covers keeps part of its weight
        # instead of being assigned wholly in or out by its centre.
        width, height = size
        fine = selected.resize(
            (width * MASK_SAMPLES, height * MASK_SAMPLES), Image.Resampling.NEAREST
        )
        selected = fine.reduce(MASK_SAMPLES)
    return selected.convert("F").point(lambda value: value / 255.0)


def _normalized(payload: bytes) -> list[int]:
    with Image.open(io.BytesIO(payload)) as image:
        # Colour, not luminance. Greyscale was the first attempt and it hid
        # the exact case this exists for: a blue object recoloured burgundy
        # keeps almost the same brightness, so a luminance comparison called
        # a real edit "unchanged". A fixed size lets differing output
        # dimensions still compare.
        converted = image.convert("RGB").resize(COMPARISON_SIZE)
        return list(converted.tobytes())
