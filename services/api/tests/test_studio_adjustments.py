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


# A picture with edges in it, for sharpness, which reads each pixel's
# neighbors. studioAdjustments.test.ts holds the browser's copy to the same
# pixels and results, opaque and as a cutout: in the cutout some pixels are
# partly transparent, and some wholly so with a color hidden under them.
SHARP_WIDTH = 6
SHARP_COLORS = _pixels(
    "12,40,200 60,60,60 200,30,30 250,250,250 0,0,0 90,180,40 "
    "30,30,30 220,200,10 128,128,128 15,90,210 240,120,60 33,66,99 "
    "255,255,255 10,10,10 180,40,160 70,200,120 0,255,0 128,0,255 "
    "45,45,200 150,150,20 5,5,5 250,10,120 100,100,100 200,200,200 "
    "77,22,11 0,128,255 255,128,0 60,30,90 210,210,40 18,180,180"
)
SHARP_ALPHAS = [
    *(255, 255, 255, 255, 0, 255),
    *(255, 128, 255, 64, 255, 255),
    *(0, 255, 200, 255, 17, 255),
    *(255, 255, 255, 255, 128, 0),
    *(255, 255, 0, 255, 255, 255),
]
SHARP_CASES: list[tuple[str, dict[str, int], bool, str]] = [
    (
        "crisper",
        {"sharpness": 60},
        False,
        "12,40,200,255 60,60,60,255 200,30,30,255 250,250,250,255 0,0,0,255 90,180,40,255 "
        "30,30,30,255 255,255,0,255 125,138,138,255 0,66,255,255 255,119,39,255 33,66,99,255 "
        "255,255,255,255 0,0,0,255 221,15,199,255 50,247,126,255 0,255,0,255 128,0,255,255 "
        "45,45,200,255 184,186,0,255 0,0,0,255 255,0,142,255 85,79,94,255 200,200,200,255 "
        "77,22,11,255 0,128,255,255 255,128,0,255 60,30,90,255 210,210,40,255 18,180,180,255",
    ),
    (
        "softer",
        {"sharpness": -45},
        False,
        "12,40,200,255 60,60,60,255 200,30,30,255 250,250,250,255 0,0,0,255 90,180,40,255 "
        "30,30,30,255 176,155,37,255 130,119,120,255 59,108,175,255 177,120,75,255 33,66,99,255 "
        "255,255,255,255 57,48,41,255 148,58,130,255 84,164,115,255 44,202,47,255 128,0,255,255 "
        "45,45,200,255 124,122,51,255 54,32,33,255 196,43,102,255 111,115,104,255 200,200,200,255 "
        "77,22,11,255 0,128,255,255 255,128,0,255 60,30,90,255 210,210,40,255 18,180,180,255",
    ),
    (
        "crisper, warmer and paler",
        {"sharpness": 100, "warmth": 30, "saturation": -20},
        False,
        "20,41,162,255 62,60,57,255 183,40,39,255 253,248,241,255 0,0,0,255 102,170,58,255 "
        "30,30,29,255 255,255,15,255 130,144,140,255 0,46,225,255 255,132,55,255 39,63,87,255 "
        "254,253,245,255 0,0,0,255 227,20,191,255 68,255,138,255 0,255,0,255 120,13,208,255 "
        "50,48,165,255 210,204,0,255 0,0,0,255 255,0,140,255 77,65,82,255 206,198,192,255 "
        "71,25,16,255 20,122,215,255 234,131,30,255 59,33,78,255 213,204,68,255 41,169,163,255",
    ),
    (
        "a crisper cutout",
        {"sharpness": 60},
        True,
        "12,40,200,255 60,60,60,255 200,30,30,255 250,250,250,255 0,0,0,0 90,180,40,255 "
        "30,30,30,255 255,255,0,128 122,140,141,255 0,57,253,64 255,117,27,255 33,66,99,255 "
        "255,255,255,0 0,0,0,255 223,17,203,200 36,255,125,255 0,255,0,17 128,0,255,255 "
        "45,45,200,255 199,195,0,255 0,0,0,255 255,0,138,255 78,95,91,128 255,255,255,0 "
        "77,22,11,255 0,128,255,255 255,204,0,0 60,30,90,255 210,210,40,255 18,180,180,255",
    ),
    (
        "the softest cutout",
        {"sharpness": -100},
        True,
        "12,40,200,255 60,60,60,255 200,30,30,255 250,250,250,255 0,0,0,0 90,180,40,255 "
        "30,30,30,255 97,75,65,128 138,107,105,255 159,144,137,64 147,124,114,255 33,66,99,255 "
        "0,0,0,0 84,65,56,255 108,78,88,200 126,107,111,255 132,95,130,17 128,0,255,255 "
        "45,45,200,255 68,74,83,255 92,56,75,255 134,67,90,255 136,107,114,128 0,0,0,0 "
        "77,22,11,255 0,128,255,255 0,0,0,0 60,30,90,255 210,210,40,255 18,180,180,255",
    ),
]


def _row(pixels: list[tuple[int, int, int]], mode: str = "RGB") -> Image.Image:
    image = Image.new("RGB", (len(pixels), 1))
    image.putdata(pixels)
    return image.convert(mode)


def _sharpness_picture(cutout: bool) -> Image.Image:
    size = (SHARP_WIDTH, len(SHARP_COLORS) // SHARP_WIDTH)
    if not cutout:
        opaque = Image.new("RGB", size)
        opaque.putdata(SHARP_COLORS)
        return opaque
    picture = Image.new("RGBA", size)
    picture.putdata(
        [(*color, alpha) for color, alpha in zip(SHARP_COLORS, SHARP_ALPHAS, strict=True)]
    )
    return picture


def _grid(picture: Image.Image) -> list[tuple[int, ...]]:
    """Every pixel, row by row, with its transparency."""
    data = picture.convert("RGBA").tobytes()
    return [tuple(data[index : index + 4]) for index in range(0, len(data), 4)]


def _reds(picture: Image.Image) -> list[list[int]]:
    """The red channel, row by row."""
    data = picture.getchannel("R").tobytes()
    width = picture.width
    return [list(data[row * width : (row + 1) * width]) for row in range(picture.height)]


@pytest.mark.parametrize(("name", "sliders", "expected"), CASES, ids=[case[0] for case in CASES])
def test_the_pixels_the_preview_promises(name: str, sliders: dict[str, Any], expected: str) -> None:
    result = adjust_colors(_row(PIXELS), ColorAdjustments(**sliders))

    assert [result.getpixel((x, 0)) for x in range(len(PIXELS))] == _pixels(expected)


def test_every_slider_at_zero_changes_nothing() -> None:
    assert ColorAdjustments().is_neutral()
    assert not ColorAdjustments(sharpness=1).is_neutral()
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


@pytest.mark.parametrize(
    ("name", "sliders", "cutout", "expected"), SHARP_CASES, ids=[case[0] for case in SHARP_CASES]
)
def test_the_sharpened_pixels_the_preview_promises(
    name: str, sliders: dict[str, int], cutout: bool, expected: str
) -> None:
    result = adjust_colors(_sharpness_picture(cutout), ColorAdjustments(**sliders))

    assert result.mode == ("RGBA" if cutout else "RGB")
    assert _grid(result) == [
        tuple(int(part) for part in pixel.split(",")) for pixel in expected.split()
    ]


def test_sharpening_steepens_an_edge_and_leaves_the_pictures_own_border() -> None:
    edge = Image.new("RGB", (4, 3), (50, 50, 50))
    edge.paste((200, 200, 200), (2, 0, 4, 3))

    result = adjust_colors(edge, ColorAdjustments(sharpness=100))

    # Beside the edge the softened copy is 88 and 163, and 100 doubles each
    # pixel's difference from it: 88 - 2 * 38 and 163 + 2 * 37.
    assert _reds(result) == [[50, 50, 200, 200], [50, 12, 237, 200], [50, 50, 200, 200]]


def test_the_softest_setting_is_the_softened_copy() -> None:
    dot = Image.new("RGB", (5, 5), (0, 0, 0))
    dot.putpixel((2, 2), (255, 255, 255))

    result = adjust_colors(dot, ColorAdjustments(sharpness=-100))

    # The dot spread by the 1-2-1 weights over 16, rounded half up.
    assert _reds(result) == [
        [0, 0, 0, 0, 0],
        [0, 16, 32, 16, 0],
        [0, 32, 64, 32, 0],
        [0, 16, 32, 16, 0],
        [0, 0, 0, 0, 0],
    ]


def test_a_color_hidden_under_transparency_does_not_bleed_into_a_cutout() -> None:
    cutout = Image.new("RGBA", (5, 5), (128, 128, 128, 255))
    cutout.paste((255, 255, 255, 0), (3, 0, 5, 5))

    result = adjust_colors(cutout, ColorAdjustments(sharpness=100))

    # The grey beside the transparent pixels stays grey: the white under them
    # would otherwise pull the softened copy up and darken the edge.
    middle_row = _grid(result)[10:15]
    assert middle_row[:3] == [(128, 128, 128, 255)] * 3
    assert [pixel[3] for pixel in middle_row[3:]] == [0, 0]
