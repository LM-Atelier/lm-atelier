"""Light and color adjustments: the pixels the studio's preview promises."""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image, ImageStat

from local_lm.studio_adjustments import ColorAdjustments, adjust_colors, channel_tables


def _pixels(text: str) -> list[tuple[int, int, int]]:
    """Pixels written as "r,g,b" separated by spaces."""
    triples = [tuple(int(part) for part in pixel.split(",")) for pixel in text.split()]
    return [(r, g, b) for r, g, b in triples]


# The same pixels and results studioAdjustments.test.ts checks the browser's
# preview against. If either side's arithmetic moves, one of the two suites
# fails. The last three pixels sit where the saturation mix's single-precision
# rounding changes the answer.
PIXELS = _pixels(
    "200,100,50 10,240,128 128,128,128 255,0,255 0,0,0 "
    "255,255,255 37,91,203 0,170,0 210,253,126 169,45,156"
)
CASES: list[tuple[str, dict[str, int], str]] = [
    (
        "brighter",
        {"brightness": 40},
        "255,132,66 13,255,169 169,169,169 255,0,255 0,0,0 "
        "255,255,255 49,120,255 0,224,0 255,255,166 223,59,206",
    ),
    (
        "flatter",
        {"contrast": -30},
        "186,105,65 32,219,128 128,128,128 231,24,231 24,24,24 "
        "231,231,231 54,98,189 24,162,24 195,229,126 161,60,151",
    ),
    (
        "grey",
        {"saturation": -100},
        "124,124,124 158,158,158 128,128,128 105,105,105 0,0,0 "
        "255,255,255 88,88,88 100,100,100 226,226,226 95,95,95",
    ),
    (
        "richer",
        {"saturation": 60},
        "245,85,5 0,255,110 128,128,128 255,0,255 0,0,0 "
        "255,255,255 6,92,255 0,212,0 200,255,66 213,15,192",
    ),
    (
        "paler",
        {"saturation": -41},
        "168,109,80 70,206,140 128,128,128 193,43,193 0,0,0 "
        "255,255,255 57,89,155 41,141,41 216,241,167 138,65,130",
    ),
    (
        "livelier",
        {"saturation": 8},
        "206,98,44 0,246,125 128,128,128 255,0,255 0,0,0 "
        "255,255,255 32,91,212 0,175,0 208,255,117 174,40,160",
    ),
    (
        "warmer",
        {"warmth": 50},
        "215,99,46 11,237,118 138,126,118 255,0,235 0,0,0 "
        "255,251,235 40,90,187 0,168,0 226,249,116 182,44,144",
    ),
    (
        "magenta",
        {"tint": 60},
        "227,95,57 11,227,145 145,121,145 255,0,255 0,0,0 "
        "255,241,255 42,86,230 0,161,0 238,240,143 192,43,177",
    ),
    (
        "greener",
        {"tint": -100},
        "160,108,40 8,255,102 102,138,102 204,0,204 0,0,0 "
        "204,255,204 30,98,162 0,184,0 168,255,101 135,49,125",
    ),
    (
        "tinted and warmed",
        {"tint": 35, "warmth": -45, "brightness": 15},
        "232,107,66 12,255,169 148,137,169 255,0,255 0,0,0 "
        "255,255,255 43,98,255 0,183,0 243,255,167 196,48,207",
    ),
    (
        "everything",
        {"brightness": -20, "contrast": 35, "saturation": 25, "warmth": -40},
        "200,69,7 0,251,114 102,107,124 255,0,255 0,0,0 "
        "238,247,255 0,66,250 0,168,0 186,252,95 166,1,177",
    ),
]


def _row(pixels: list[tuple[int, int, int]], mode: str = "RGB") -> Image.Image:
    image = Image.new("RGB", (len(pixels), 1))
    image.putdata(pixels)
    return image.convert(mode)


@pytest.mark.parametrize(("name", "sliders", "expected"), CASES, ids=[case[0] for case in CASES])
def test_the_pixels_the_preview_promises(name: str, sliders: dict[str, Any], expected: str) -> None:
    result = adjust_colors(_row(PIXELS), ColorAdjustments(**sliders))

    assert [result.getpixel((x, 0)) for x in range(len(PIXELS))] == _pixels(expected)


def test_every_slider_at_zero_changes_nothing() -> None:
    assert ColorAdjustments().is_neutral()
    for table in channel_tables(ColorAdjustments()):
        assert table == list(range(256))


def test_transparency_is_kept_as_it_was() -> None:
    picture = _row([(200, 100, 50), (200, 100, 50)], "RGBA")
    picture.putpixel((0, 0), (200, 100, 50, 0))
    picture.putpixel((1, 0), (200, 100, 50, 77))

    result = adjust_colors(picture, ColorAdjustments(brightness=60, saturation=-50))

    assert result.mode == "RGBA"
    assert [result.getpixel((x, 0))[3] for x in range(2)] == [0, 77]


def test_warmth_changes_the_color_and_not_the_light() -> None:
    grey = Image.new("RGB", (8, 8), (128, 128, 128))

    warm = adjust_colors(grey, ColorAdjustments(warmth=80))
    cool = adjust_colors(grey, ColorAdjustments(warmth=-80))

    red, _, blue = warm.getpixel((0, 0))
    assert red > blue
    cool_red, _, cool_blue = cool.getpixel((0, 0))
    assert cool_blue > cool_red
    for picture in (warm, cool):
        assert abs(ImageStat.Stat(picture.convert("L")).mean[0] - 128) < 2


def test_tint_moves_green_against_magenta_and_not_the_light() -> None:
    grey = Image.new("RGB", (8, 8), (128, 128, 128))

    magenta = adjust_colors(grey, ColorAdjustments(tint=80))
    green = adjust_colors(grey, ColorAdjustments(tint=-80))

    red, middle, blue = magenta.getpixel((0, 0))
    assert red == blue > middle
    red, middle, blue = green.getpixel((0, 0))
    assert middle > red == blue
    # Measured with the same luminance weights the gains are evened out by.
    # Pillow's own grey uses older weights, which count green for less.
    for picture in (magenta, green):
        red, middle, blue = picture.getpixel((0, 0))
        assert abs(0.2126 * red + 0.7152 * middle + 0.0722 * blue - 128) < 1
