"""Fitting a picture you already have into the shape you asked for.

A reference is whatever shape it is. When somebody asks for an output of a
different shape, something has to give, and there are exactly two honest answers:
keep all of the source and invent the rest (EXTEND), or keep the shape and lose
part of the source (CROP). This module decides how much, and nothing else - it
does not pick between them, does not touch a graph, and does not know what a
workflow is.

WHAT AN EXTENSION MARGIN MEANS, because a fraction is a fraction OF something
and getting the denominator wrong would be wrong in a way every test written
against the guess would agree with. The only thing in the tree that writes these
values is the Studio's drag handles, and it divides by the picture's own size on
the axis being dragged:

    apps/web/src/StudioExtendHandles.tsx:51-53
    span = axis === "x" ? size.width : size.height
    moved = ((along - drag.from) * sign) / span

So a LEFT or RIGHT margin is a fraction of the source's WIDTH and a TOP or
BOTTOM margin is a fraction of its HEIGHT - per axis, not one dimension for all
four sides. This module emits against that contract. If something ever consumes
these fractions with a different denominator, the disagreement is with that
file, and it should be settled there rather than by quietly changing the
arithmetic here.

A NAME THAT ALREADY EXISTS NEARBY, so the two are not confused later.
`output_geometry.ResolvedOutputGeometry` has a `source_fit` field, but it is a
different thing: its vocabulary is `"workflow_native" | None`, it names WHICH
policy a request resolved to, and its companion `source_fit_applied` is pinned
`Literal[False]` as a deliberate not-yet marker. This module does not resolve a
policy and does not set that field - it answers what a chosen fit COSTS, in
pixels. When the two are eventually joined, that field's vocabulary will have to
grow to admit an extension and a crop, and `source_fit_applied` will stop being
a constant; neither belongs here.

Nothing in this module reads a workflow, a run or a database. It is arithmetic
over four integers, so it can be trusted about the one thing it claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

#: The largest ratio difference worth bridging. Beyond this the invented region
#: dwarfs the picture it was inferred from, or the crop keeps so little of the
#: source that calling the result a fit is a stretch. Mirrors the outpainting
#: ceiling in `outpaint_workflows.MAX_MARGIN_FRACTION`.
MAX_FIT_FRACTION = 2.0

#: A record version, so a later reader can tell an old record from a new one
#: rather than inferring shape from which keys happen to be present.
RECORD_VERSION = 1

Mode = Literal["extend", "crop"]


class SourceFitError(ValueError):
    """The fit was not asked for in terms this module can answer."""


@dataclass(frozen=True)
class SourceFit:
    """How to reach the requested shape from the source, in exact terms."""

    mode: Mode
    #: Per-side, against the per-axis denominator described in the module
    #: docstring. Zero on every side means the shapes already agree.
    margins: dict[str, float]
    #: The rectangle of the SOURCE that survives, in source pixels. For an
    #: extension this is always the whole picture; for a crop it is what is
    #: kept, and the difference is the whole point of recording it.
    kept: tuple[int, int, int, int]

    def payload(self) -> dict[str, Any]:
        left, top, width, height = self.kept
        return {
            "v": RECORD_VERSION,
            "mode": self.mode,
            "margins": dict(self.margins),
            "kept": {"left": left, "top": top, "width": width, "height": height},
        }


def _dimension(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceFitError(f"The {name} must be a whole number of pixels.")
    if value <= 0:
        raise SourceFitError(f"The {name} must be greater than zero.")
    return value


def _split(total: int) -> tuple[int, int]:
    """Divide an odd remainder deterministically: the extra pixel goes last.

    Centring is not quite possible on an odd difference, and the choice has to
    be made SOMEWHERE - so it is made here, once, rather than left to whichever
    caller rounds first. Left and top take the smaller half; right and bottom
    take the extra pixel. A person cannot see which side gained it, but two
    runs of the same request must agree, and a recorded rectangle that shifts by
    a pixel between them is a record nobody can check.
    """

    half = total // 2
    return half, total - half


def fit_source_to_shape(
    *,
    source_width: object,
    source_height: object,
    target_width: object,
    target_height: object,
    mode: Mode,
) -> SourceFit:
    """Work out what reaching the requested shape costs, in exact pixels.

    Only the target's RATIO is used, never its size: scaling to a requested
    resolution is a different job done by a different part of the system, and
    conflating them here would make this answer depend on something it has no
    business knowing.
    """

    if mode not in ("extend", "crop"):
        raise SourceFitError("A fit is either an extension or a crop.")
    width = _dimension(source_width, "source width")
    height = _dimension(source_height, "source height")
    target_w = _dimension(target_width, "requested width")
    target_h = _dimension(target_height, "requested height")

    # Exact rationals rather than floats. Being honest about what this buys:
    # IEEE division is correctly rounded, so two equal ratios round to the same
    # double for every dimension this product admits (the cap is a million, and
    # the products stay far inside 53 bits). I looked for a pair that is equal
    # as rationals and unequal as floats within that range and there is none, so
    # NO TEST HERE DISTINGUISHES THE TWO and swapping in floats passes the whole
    # module. They are kept because the intent - a shape that already agrees
    # must cost nothing - should be expressed exactly rather than rest on a
    # property of the input range that a later cap change could quietly remove.
    source_ratio = Fraction(width, height)
    target_ratio = Fraction(target_w, target_h)

    if source_ratio == target_ratio:
        return SourceFit(
            mode=mode,
            margins={"top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0},
            kept=(0, 0, width, height),
        )

    if mode == "extend":
        return _extend(width, height, source_ratio, target_ratio)
    return _crop(width, height, source_ratio, target_ratio)


def _extend(width: int, height: int, source_ratio: Fraction, target_ratio: Fraction) -> SourceFit:
    """Keep every pixel and grow the short axis, centred."""

    if source_ratio < target_ratio:
        # Too tall for the shape: the picture needs to get wider.
        grown = int(Fraction(height) * target_ratio)
        extra = grown - width
        near, far = _split(extra)
        margins = {
            "top": 0.0,
            "right": far / width,
            "bottom": 0.0,
            "left": near / width,
        }
    else:
        grown = int(Fraction(width) / target_ratio)
        extra = grown - height
        near, far = _split(extra)
        margins = {
            "top": near / height,
            "right": 0.0,
            "bottom": far / height,
            "left": 0.0,
        }
    _refuse_a_runaway(margins)
    return SourceFit(mode="extend", margins=margins, kept=(0, 0, width, height))


def _crop(width: int, height: int, source_ratio: Fraction, target_ratio: Fraction) -> SourceFit:
    """Keep the shape and lose the overhang, centred.

    Margins are all zero here and that is not an oversight: a crop removes
    rather than invents, so there is no region for a workflow to paint. The
    rectangle is the answer.

    THE RECTANGLE IS TRUNCATED, AND IT IS NOT ALWAYS THE REQUESTED RATIO. The
    exact edge is a rational, and `int()` drops whatever is left of it, so the
    kept rectangle is exactly the requested ratio only when that edge is already
    a whole number:

        400x300 to 16:9   kept 400x225   exactly 16/9
        400x300 to 9:16   kept 168x300   14/25, because 168.75 truncates
        400x300 to 3:2    kept 400x266   200/133, because 266.67 truncates

    The error is under one pixel on one axis and it is still real - 0.5600
    against 0.5625 - so a consumer that needs an exact ratio cannot get it from
    this rectangle, and one that scales this rectangle to an exact-ratio output
    is applying slightly different factors per axis. That is a distortion,
    small but not nothing.

    Truncating rather than rounding is deliberate: it keeps the rectangle
    INSIDE the source on the cropped axis, so no consumer is ever handed an edge
    the picture does not have. Rounding 168.75 up to 169 would exceed the exact
    ratio instead, which trades one kind of wrongness for another.

    Whoever needs exactness has three ways out and none of them belongs here:
    shrink both axes to the largest exactly-ratio rectangle, choose the output
    size FROM this rectangle so the scale is uniform by construction, or keep a
    fractional viewport and resample it. `source_crop` takes the third road with
    a rational viewport and one exact scale; this module stays the integer cost
    estimate it has always been.
    """

    if source_ratio > target_ratio:
        kept_width = int(Fraction(height) * target_ratio)
        left, _ = _split(width - kept_width)
        kept = (left, 0, kept_width, height)
    else:
        kept_height = int(Fraction(width) / target_ratio)
        top, _ = _split(height - kept_height)
        kept = (0, top, width, kept_height)
    if kept[2] <= 0 or kept[3] <= 0:
        raise SourceFitError("Cropping to that shape would leave nothing of the picture.")
    return SourceFit(
        mode="crop",
        margins={"top": 0.0, "right": 0.0, "bottom": 0.0, "left": 0.0},
        kept=kept,
    )


def _refuse_a_runaway(margins: dict[str, float]) -> None:
    for side, margin in margins.items():
        if margin > MAX_FIT_FRACTION:
            raise SourceFitError(
                f"Extending the {side} by more than {MAX_FIT_FRACTION:g} times the "
                "picture would invent more than it kept."
            )
