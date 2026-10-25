"""Match an extension's new area to the colour of the source it continues.

A model asked to paint past a picture's edge chooses its own exposure and white
balance, and the new area can come out a few levels cooler, warmer or brighter,
so the edge shows. A workflow that pastes its decode back over the padded
canvas through the padding's own mask leaves, in the feathered band just inside
each extended edge, the source times (1 - m) plus the decode times m, with m
known from the padding alone. Where the decode carries most of that mix it is
read back out, pixel by pixel, and compared with the source at the same place:
the same picture there, so what differs is the model's colour. That difference
is taken back from the band at the share the workflow painted, and from the new
area by an amount that falls to nothing at the canvas's outer edge, so a
deliberately different far region, such as an added sky, keeps its own colour.

Pure work on pixels: nothing here opens a session or a file.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageMath

#: How much of an edge each reading covers; positions between are blended.
SEGMENT_PX = 64
#: A difference below this many levels is noise, not a colour cast.
DEAD_ZONE = 2
#: The most, in levels of a channel, the colour is shifted back.
MAX_SHIFT = 24
#: The decode is read back only where it carries at least this share of the
#: mix, so a level lost when the picture was saved costs at most two there.
MEASURED_SHARE = 0.5
#: Fewer pixels than this read back in a stretch say nothing about it.
MIN_SAMPLES = 16

SIDES = ("top", "right", "bottom", "left")
Shift = tuple[float, float, float]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class _Frame:
    """The source within the extended picture, and how the padding feathered it."""

    width: int
    height: int
    margins: Mapping[str, int]
    feather: int

    def padded(self, side: str) -> bool:
        return self.margins[side] > 0

    def inward(self, side: str, row: int, column: int) -> int:
        """ComfyUI's distance in from a padded edge: from 0 at the top and left, 1 at the others."""
        if side == "top":
            return row
        if side == "bottom":
            return self.height - row
        if side == "left":
            return column
        return self.width - column


def match_extension_colour(
    picture: Image.Image,
    source: Image.Image,
    margins: Mapping[str, int],
    feather: int,
) -> tuple[Image.Image, dict[str, Any]]:
    """The extended picture with the model's colour cast taken back, and what was done.

    `picture` is the canvas asked for, made by pasting the decode over the
    padded source through the padding's mask; `source` is the source alone,
    `margins` the pixels added on each side and `feather` the feather that
    mask was made with, zero where ComfyUI would not feather at all.
    """
    frame = _Frame(width=source.width, height=source.height, margins=margins, feather=feather)
    padded = [side for side in SIDES if frame.padded(side)]
    if feather <= 0 or not padded:
        return picture, {"applied": False, "reason": "not_feathered"}
    shifts = {side: _shifts(picture, source, frame, side) for side in padded}
    out = picture.convert("RGB")
    for box, inside in _boxes(frame):
        _correct(out, frame, shifts, box, inside)
    return out, {
        "applied": True,
        "feather_px": feather,
        "segment_px": SEGMENT_PX,
        "max_shift": MAX_SHIFT,
        "sides": {
            side: {
                "largest_shift": [
                    _largest([value[channel] for value in values]) for channel in range(3)
                ]
            }
            for side, values in shifts.items()
        },
    }


def _largest(values: Sequence[float]) -> int:
    """The shift furthest from none, in whole levels."""
    largest: float = max(values, key=math.fabs)
    return round(largest)


def _shifts(picture: Image.Image, source: Image.Image, frame: _Frame, side: str) -> list[Shift]:
    """At each position along one edge, how far the decode's colour is from the source's."""
    length = frame.height if side in {"left", "right"} else frame.width
    # The decode's share at a distance d is ((f - d) / f) ** 2.
    deepest = frame.feather * (1 - math.sqrt(MEASURED_SHARE))
    first = 0 if side in {"top", "left"} else 1
    distances = [d for d in range(first, frame.feather) if d < deepest]
    others = [other for other in SIDES if other != side and frame.padded(other)]
    painted, original = picture.load(), source.load()
    assert painted is not None and original is not None
    left, top = frame.margins["left"], frame.margins["top"]
    readings: list[Shift] = []
    for start in range(0, length, SEGMENT_PX):
        samples: tuple[list[float], list[float], list[float]] = ([], [], [])
        for position in range(start, min(length, start + SEGMENT_PX)):
            for d in distances:
                row, column = _pixel(frame, side, position, d)
                if any(frame.inward(other, row, column) < d for other in others):
                    continue  # nearer another edge, so mixed by that edge's distance
                share = ((frame.feather - d) / frame.feather) ** 2
                made = painted[left + column, top + row]
                kept = original[column, row]
                assert isinstance(made, tuple) and isinstance(kept, tuple)
                for channel, values in enumerate(samples):
                    # ComfyUI truncates as it saves; half a level puts the mix back in the middle.
                    values.append((made[channel] + 0.5 - kept[channel]) / share)
        r, g, b = (_reading(values) if len(values) >= MIN_SAMPLES else 0.0 for values in samples)
        readings.append((r, g, b))
    centres = [min(length - 1, start + SEGMENT_PX // 2) for start in range(0, length, SEGMENT_PX)]
    smoothed = _smoothed(readings)
    return [_interpolated(position, centres, smoothed) for position in range(length)]


def _pixel(frame: _Frame, side: str, position: int, d: int) -> tuple[int, int]:
    """The source's row and column at a distance in from an edge, a position along it."""
    if side == "top":
        return d, position
    if side == "bottom":
        return frame.height - d, position
    if side == "left":
        return position, d
    return position, frame.width - d


def _reading(values: list[float]) -> float:
    middle = statistics.median(values)
    if abs(middle) < DEAD_ZONE:
        return 0.0
    return max(-MAX_SHIFT, min(MAX_SHIFT, middle))


def _smoothed(readings: Sequence[Shift]) -> list[Shift]:
    """Each stretch as the middle of three readings, those at the ends included.

    A stretch that disagrees with the two beside it is more likely something
    the model drew differently than a cast. An end stretch has one neighbour,
    so it is read with the next one in; of only two, the gentler is taken.
    """
    count = len(readings)
    smoothed: list[Shift] = []
    for index in range(count):
        values: list[float] = []
        for channel in range(3):
            if count == 1:
                values.append(readings[0][channel])
            elif count == 2:
                values.append(min((reading[channel] for reading in readings), key=abs))
            else:
                low = min(max(index - 1, 0), count - 3)
                trio = sorted(readings[i][channel] for i in range(low, low + 3))
                values.append(trio[1])
        smoothed.append((values[0], values[1], values[2]))
    return smoothed


def _interpolated(position: int, centres: Sequence[int], values: Sequence[Shift]) -> Shift:
    if position <= centres[0]:
        return values[0]
    for index in range(1, len(centres)):
        if position <= centres[index]:
            share = (position - centres[index - 1]) / (centres[index] - centres[index - 1])
            low, high = values[index - 1], values[index]
            r, g, b = (low[c] + (high[c] - low[c]) * share for c in range(3))
            return r, g, b
    return values[-1]


def _boxes(frame: _Frame) -> list[tuple[Box, bool]]:
    """The regions of the picture that change, in its own pixels, and whether each is the source's.

    Inside the source only the feathered band changes. Every box of the new
    area around it changes, the corners both its sides touch included.
    """
    f, width, height = frame.feather, frame.width, frame.height
    left, top = frame.margins["left"], frame.margins["top"]
    upper = f if frame.padded("top") else 0
    lower = height - f + 1 if frame.padded("bottom") else height
    band: list[Box] = []
    if frame.padded("top"):
        band.append((0, 0, width, f))
    if frame.padded("bottom"):
        band.append((0, height - f + 1, width, height))
    if frame.padded("left"):
        band.append((0, upper, f, lower))
    if frame.padded("right"):
        band.append((width - f + 1, upper, width, lower))
    boxes = [((left + x0, top + y0, left + x1, top + y1), True) for x0, y0, x1, y1 in band]
    spans = (
        ((0, left), (left, left + width), (left + width, left + width + frame.margins["right"])),
        ((0, top), (top, top + height), (top + height, top + height + frame.margins["bottom"])),
    )
    for row, (y0, y1) in enumerate(spans[1]):
        for column, (x0, x1) in enumerate(spans[0]):
            if (row, column) != (1, 1):
                boxes.append(((x0, y0, x1, y1), False))
    return [(box, inside) for box, inside in boxes if box[2] > box[0] and box[3] > box[1]]


def _correct(
    out: Image.Image,
    frame: _Frame,
    shifts: Mapping[str, Sequence[Shift]],
    box: Box,
    inside: bool,
) -> None:
    """Take the shift back from one region of the picture.

    Each extended edge's shift counts by how far into its band a pixel lies,
    and fully anywhere past that edge, so near a corner of the source the
    shift turns from one edge's to the other's without a step. Inside the
    source it is taken back at the share the padding gave the decode; in the
    new area fully beside the source, and less with each step away from it.
    """
    f, width, height = frame.feather, frame.width, frame.height
    size = (box[2] - box[0], box[3] - box[1])
    rows = [y - frame.margins["top"] for y in range(box[1], box[3])]
    columns = [x - frame.margins["left"] for x in range(box[0], box[2])]
    inward: dict[str, Image.Image] = {}
    weights: dict[str, Image.Image] = {}
    for side in shifts:
        if side in {"top", "bottom"}:
            distances = [frame.inward(side, row, 0) for row in rows]
            inward[side] = _down(distances, size)
            weights[side] = _down([max(f - max(d, 0), 0) for d in distances], size)
        else:
            distances = [frame.inward(side, 0, column) for column in columns]
            inward[side] = _across(distances, size)
            weights[side] = _across([max(f - max(d, 0), 0) for d in distances], size)
    total = _sum(list(weights.values()))
    if inside:
        nearest = _least(list(inward.values()))
        amount = _math(lambda a: (a["max"](f - a["d"], 0.0) / f) ** 2, d=nearest)
    else:
        down = _down(
            [_fade(frame, "top", -1 - row) * _fade(frame, "bottom", row - height) for row in rows],
            size,
        )
        across = _across(
            [
                _fade(frame, "left", -1 - column) * _fade(frame, "right", column - width)
                for column in columns
            ],
            size,
        )
        amount = _math(lambda a: a["x"] * a["y"], x=down, y=across)
    corrections = []
    for channel in range(3):
        weighted = []
        for side, values in shifts.items():
            if side in {"top", "bottom"}:
                along = _across(
                    [values[_clamp(column, width)][channel] for column in columns], size
                )
            else:
                along = _down([values[_clamp(row, height)][channel] for row in rows], size)
            weighted.append(_math(lambda a: a["w"] * a["s"], w=weights[side], s=along))
        corrections.append(
            _math(lambda a: a["s"] / a["t"] * a["m"], s=_sum(weighted), t=total, m=amount)
        )
    _subtract(out, box, corrections)


def _fade(frame: _Frame, side: str, step: int) -> float:
    """How much of the shift is taken back a number of steps past an edge, from 0 beside it."""
    if step < 0 or not frame.padded(side):
        return 1.0
    return 1 - step / frame.margins[side]


def _clamp(position: int, length: int) -> int:
    return min(max(position, 0), length - 1)


def _subtract(out: Image.Image, box: Box, corrections: Sequence[Image.Image]) -> None:
    region = out.crop(box)
    channels = [
        # Converting to eight bits clips and truncates, so half a level rounds.
        _math(
            lambda a: a["p"] - a["c"] + 0.5,
            p=region.getchannel(index).convert("F"),
            c=correction,
        ).convert("L")
        for index, correction in enumerate(corrections)
    ]
    out.paste(Image.merge("RGB", channels), box[:2])


def _across(values: Sequence[float], size: tuple[int, int]) -> Image.Image:
    """A row of values, repeated down a region."""
    image = Image.new("F", (len(values), 1))
    image.putdata([float(value) for value in values])
    return image.resize(size, Image.Resampling.NEAREST)


def _down(values: Sequence[float], size: tuple[int, int]) -> Image.Image:
    """A column of values, repeated across a region."""
    image = Image.new("F", (1, len(values)))
    image.putdata([float(value) for value in values])
    return image.resize(size, Image.Resampling.NEAREST)


def _sum(images: Sequence[Image.Image]) -> Image.Image:
    total = images[0]
    for image in images[1:]:
        total = _math(lambda a: a["x"] + a["y"], x=total, y=image)
    return total


def _least(images: Sequence[Image.Image]) -> Image.Image:
    least = images[0]
    for image in images[1:]:
        least = _math(lambda a: a["min"](a["x"], a["y"]), x=least, y=image)
    return least


def _math(expression: Callable[[dict[str, Any]], Any], **images: Image.Image) -> Image.Image:
    result = ImageMath.lambda_eval(expression, **images)
    assert isinstance(result, Image.Image)
    return result
