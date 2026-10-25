"""An extension's new area takes the colour of the source it continues."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping

import pytest
from PIL import Image, ImageChops

from local_lm.outpaint_colour import MAX_SHIFT, match_extension_colour

Colour = tuple[int, int, int]
SKY = (90, 140, 200)
GROUND = (70, 110, 50)
POLE = (40, 35, 30)
NONE = {"top": 0, "right": 0, "bottom": 0, "left": 0}


def scene(x: int, y: int, horizon: int = 96) -> Colour:
    """A picture defined past its own edges: sky over ground, a pole, a little grain."""
    base = POLE if 40 <= x < 46 else SKY if y < horizon else GROUND
    grain = (x * 7 + y * 13) % 9 - 4
    r, g, b = (value + grain for value in base)
    return r, g, b


def picture_of(size: tuple[int, int], colour: Callable[[int, int], Colour]) -> Image.Image:
    image = Image.new("RGB", size)
    image.putdata([colour(x, y) for y in range(size[1]) for x in range(size[0])])
    return image


def comfy_extended(
    handed: Image.Image,
    pads: Mapping[str, int],
    feather: int,
    decode: Callable[[int, int], Colour],
) -> Image.Image:
    """The padded canvas as ComfyUI saves it, its decode pasted back through the padding's mask.

    The mask is ImagePadForOutpaint's own: inside the source, ((f - d) / f) ** 2
    from the nearest padded edge, d counted from 0 at the top and left and 1 at
    the others, and nothing unless twice the feather fits the source; outside
    it, 1. ImageCompositeMasked mixes the decode in by that mask, and SaveImage
    truncates to eight bits.
    """
    width, height = handed.size
    left, top = pads["left"], pads["top"]
    size = (width + left + pads["right"], height + top + pads["bottom"])
    feathered = feather > 0 and feather * 2 < height and feather * 2 < width
    kept = handed.convert("RGB").load()
    assert kept is not None
    pixels: list[Colour] = []
    for y in range(size[1]):
        for x in range(size[0]):
            made = decode(x, y)
            i, j = y - top, x - left
            if not (0 <= i < height and 0 <= j < width):
                pixels.append(made)
                continue
            d = min(
                i if pads["top"] else height,
                height - i if pads["bottom"] else height,
                j if pads["left"] else width,
                width - j if pads["right"] else width,
            )
            m = ((feather - d) / feather) ** 2 if feathered and d < feather else 0.0
            source = kept[j, i]
            assert isinstance(source, tuple)
            r, g, b = (math.floor(source[c] * (1 - m) + made[c] * m) for c in range(3))
            pixels.append((r, g, b))
    image = Image.new("RGB", size)
    image.putdata(pixels)
    return image


def cast_by(shift: Colour, colour: Callable[[int, int], Colour]) -> Callable[[int, int], Colour]:
    def shifted(x: int, y: int) -> Colour:
        r, g, b = (
            max(0, min(255, value + s)) for value, s in zip(colour(x, y), shift, strict=True)
        )
        return r, g, b

    return shifted


def placed(
    colour: Callable[[int, int], Colour], margins: Mapping[str, int]
) -> Callable[[int, int], Colour]:
    """A picture's colours at canvas coordinates, its source at the margins' corner."""

    def at(x: int, y: int) -> Colour:
        return colour(x - margins["left"], y - margins["top"])

    return at


def largest_difference(a: Image.Image, b: Image.Image) -> int:
    extrema = ImageChops.difference(a.convert("RGB"), b.convert("RGB")).getextrema()
    return max(int(band[1]) for band in extrema if isinstance(band, tuple))


def median_shift(
    after: Image.Image, before: Image.Image, box: tuple[int, int, int, int]
) -> list[float]:
    """How far each channel moved in a region, by its median."""
    moved = []
    for channel in range(3):
        now = after.crop(box).getchannel(channel).tobytes()
        was = before.crop(box).getchannel(channel).tobytes()
        moved.append(statistics.median(n - w for n, w in zip(now, was, strict=True)))
    return moved


def _extend(
    margins: Mapping[str, int],
    feather: int,
    decode: Callable[[int, int], Colour],
    size: tuple[int, int] = (256, 192),
    source: Callable[[int, int], Colour] = scene,
) -> tuple[Image.Image, Image.Image, Image.Image, dict[str, object]]:
    original = picture_of(size, source)
    made = comfy_extended(original, margins, feather, decode)
    out, record = match_extension_colour(made, original, margins, feather)
    return original, made, out, record


# A narrow feather leaves a miscounted row several levels off, not a fraction of one.
@pytest.mark.parametrize("feather", [64, 8])
def test_the_band_reads_as_the_source_again_and_the_new_area_meets_it(feather: int) -> None:
    """The horizon crosses the right edge; the model continued it a few levels off."""
    margins = {**NONE, "right": 96}
    cast = (12, -8, 16)
    continued = placed(scene, margins)

    original, _, out, record = _extend(margins, feather, cast_by(cast, continued))

    assert record["applied"] is True
    assert largest_difference(out.crop((0, 0, 256, 192)), original) <= 1
    beside = picture_of((1, 192), lambda _x, y: scene(256, y))
    assert largest_difference(out.crop((256, 0, 257, 192)), beside) <= 1
    # At the canvas's outer edge almost all of the model's colour stays.
    outer = picture_of((1, 192), lambda _x, y: scene(351, y))
    left_on = [round(value * (1 - 95 / 96)) for value in cast]
    assert [
        round(v) for v in median_shift(out.crop((351, 0, 352, 192)), outer, (0, 0, 1, 192))
    ] == [value - taken for value, taken in zip(cast, left_on, strict=True)]


@pytest.mark.parametrize("feather", [64, 8])
def test_a_fine_pattern_at_the_edge_is_read_exactly(feather: int) -> None:
    """A pattern that changes every pixel is the same picture in the decode, so it cancels."""
    margins = {**NONE, "right": 64, "bottom": 64}
    cast = (10, -6, 14)

    def checks(x: int, y: int) -> Colour:
        return (60, 70, 80) if (x + y) % 2 else (190, 180, 170)

    original, _, out, record = _extend(
        margins, feather, cast_by(cast, placed(checks, margins)), source=checks
    )

    assert largest_difference(out.crop((0, 0, 256, 192)), original) <= 1
    sides = record["sides"]
    assert isinstance(sides, dict)
    assert sides["right"]["largest_shift"] == list(cast)
    assert sides["bottom"]["largest_shift"] == list(cast)


@pytest.mark.parametrize("feather", [48, 8, 1])
def test_left_and_top_are_read_from_their_first_row(feather: int) -> None:
    """ComfyUI counts the top and left from 0, so the first row inside is all decode."""
    margins = {**NONE, "left": 80, "top": 72}
    cast = (-10, 12, 8)
    continued = placed(scene, margins)

    original, _, out, _ = _extend(margins, feather, cast_by(cast, continued))

    assert largest_difference(out.crop((80, 72, 336, 264)), original) <= 1
    beside = picture_of((1, 192), lambda _x, y: scene(-1, y))
    above = picture_of((256, 1), lambda x, _y: scene(x, -1))
    assert largest_difference(out.crop((79, 72, 80, 264)), beside) <= 1
    assert largest_difference(out.crop((80, 71, 336, 72)), above) <= 1


def test_a_corner_turns_from_one_sides_shift_to_the_others() -> None:
    """Each side cast differently: the correction runs between them without a step."""
    margins = {"top": 40, "right": 56, "bottom": 48, "left": 64}
    casts: dict[str, Colour] = {
        "top": (12, 0, -6),
        "right": (-8, 10, 0),
        "bottom": (0, -12, 10),
        "left": (14, 6, -10),
    }
    width, height = 256, 192

    def cast_at(x: int, y: int) -> Colour:
        i, j = y - margins["top"], x - margins["left"]
        near = {"top": i, "bottom": height - i, "left": j, "right": width - j}
        outside = [
            side for side, d in near.items() if d < 0 or (side in {"bottom", "right"} and d <= 0)
        ]
        sides = outside or [min(near, key=lambda side: near[side])]
        r, g, b = (round(sum(casts[side][c] for side in sides) / len(sides)) for c in range(3))
        return r, g, b

    continued = placed(scene, margins)

    def decode(x: int, y: int) -> Colour:
        return cast_by(cast_at(x, y), continued)(x, y)

    _, made, out, record = _extend(margins, 32, decode)

    sides = record["sides"]
    assert isinstance(sides, dict)
    for side, cast in casts.items():
        assert sides[side]["largest_shift"] == list(cast)
    # What was taken from each pixel, around a middle grey so it reads either way.
    for channel in range(3):
        taken = ImageChops.subtract(made.getchannel(channel), out.getchannel(channel), offset=128)
        values = list(taken.tobytes())
        width = taken.width
        steps = [
            abs(values[y * width + x] - values[(y + dy) * width + x + dx])
            for y in range(taken.height - 1)
            for x in range(width - 1)
            for dx, dy in ((1, 0), (0, 1))
        ]
        assert max(steps) <= 2
        # The canvas's own corners keep the model's colour.
        for x, y in ((0, 0), (width - 1, 0), (0, taken.height - 1), (width - 1, taken.height - 1)):
            assert abs(values[y * width + x] - 128) <= 1


def test_a_line_beside_the_edge_that_the_model_left_out_is_outvoted() -> None:
    """A pole just inside the edge, painted over as sky: most of the band still reads the cast."""
    margins = {**NONE, "right": 64}
    cast = (10, -6, 12)

    def poled(x: int, y: int) -> Colour:
        return POLE if 244 <= x < 250 else scene(x, y)

    def decode(x: int, y: int) -> Colour:
        return cast_by(cast, placed(scene, margins))(x, y)

    _, made, out, record = _extend(margins, 64, decode, source=poled)

    sides = record["sides"]
    assert isinstance(sides, dict)
    assert sides["right"]["largest_shift"] == list(cast)
    assert median_shift(out, made, (256, 0, 257, 192)) == [-c for c in cast]


def test_a_uniform_cast_is_taken_back_evenly_where_two_bands_meet() -> None:
    """ComfyUI mixes a corner of the band by its nearer edge; so is the cast taken back there."""
    margins = {**NONE, "right": 64, "bottom": 48}
    cast = (20, 0, -12)

    original, _, out, _ = _extend(margins, 40, cast_by(cast, placed(scene, margins)))

    assert largest_difference(out.crop((0, 0, 256, 192)), original) <= 1
    corner = picture_of((1, 1), lambda _x, _y: scene(256, 192))
    assert largest_difference(out.crop((256, 192, 257, 193)), corner) <= 1


def test_a_new_area_already_in_step_with_its_source_is_left_alone() -> None:
    margins = {**NONE, "right": 64}
    for cast in ((0, 0, 0), (1, -1, 1)):
        _, made, out, record = _extend(margins, 64, cast_by(cast, placed(scene, margins)))

        assert out.tobytes() == made.tobytes()
        sides = record["sides"]
        assert isinstance(sides, dict)
        assert sides["right"]["largest_shift"] == [0, 0, 0]


def test_a_large_cast_is_taken_back_only_so_far() -> None:
    margins = {**NONE, "right": 64}
    cast = (40, -40, 40)

    _, made, out, record = _extend(margins, 64, cast_by(cast, placed(scene, margins)))

    sides = record["sides"]
    assert isinstance(sides, dict)
    assert sides["right"]["largest_shift"] == [MAX_SHIFT, -MAX_SHIFT, MAX_SHIFT]
    assert median_shift(out, made, (256, 0, 257, 192)) == [-MAX_SHIFT, MAX_SHIFT, -MAX_SHIFT]


def test_each_stretch_of_the_edge_is_matched_on_its_own() -> None:
    margins = {**NONE, "right": 64}
    continued = placed(scene, margins)

    def decode(x: int, y: int) -> Colour:
        return cast_by((10, 10, 10) if y < 128 else (-10, -10, -10), continued)(x, y)

    _, made, out, _ = _extend(margins, 64, decode, size=(256, 256))

    assert median_shift(out, made, (256, 0, 257, 64)) == [-10, -10, -10]
    assert median_shift(out, made, (256, 192, 257, 256)) == [10, 10, 10]


@pytest.mark.parametrize("first", [0, 64, 192], ids=["first", "middle", "last"])
def test_something_drawn_differently_at_one_stretch_is_not_taken_for_a_cast(first: int) -> None:
    """One stretch of the band the model redrew, at either end too: its neighbours outvote it."""
    margins = {**NONE, "right": 64}
    continued = placed(scene, margins)

    def decode(x: int, y: int) -> Colour:
        redrawn = first <= y < first + 64 and x < 256
        return cast_by((46, 46, 46) if redrawn else (6, 6, 6), continued)(x, y)

    _, made, out, _ = _extend(margins, 64, decode, size=(256, 256))

    assert median_shift(out, made, (256, first, 257, first + 64)) == [-6, -6, -6]


def test_a_narrow_extension_of_a_small_picture_is_matched() -> None:
    margins = {**NONE, "right": 3}
    cast = (9, -9, 9)

    original, _, out, _ = _extend(margins, 16, cast_by(cast, placed(scene, margins)), size=(96, 80))

    assert largest_difference(out.crop((0, 0, 96, 80)), original) <= 1
    beside = picture_of((1, 80), lambda _x, y: scene(96, y))
    assert largest_difference(out.crop((96, 0, 97, 80)), beside) <= 1


def test_a_picture_the_padding_did_not_feather_is_left_as_made() -> None:
    margins = {**NONE, "right": 64}

    _, made, out, record = _extend(margins, 0, cast_by((12, 12, 12), placed(scene, margins)))

    assert out.tobytes() == made.tobytes()
    assert record == {"applied": False, "reason": "not_feathered"}
