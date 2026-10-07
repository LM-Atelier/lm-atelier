"""Light and color adjustments: the pixels the studio's preview promises."""

from __future__ import annotations

from itertools import pairwise
from typing import TypedDict

import pytest
from PIL import Image, ImageStat

from local_lm.schemas import StudioColorAdjustments
from local_lm.studio_adjustments import (
    GRAIN_TILE,
    ColorAdjustments,
    CurvePoint,
    adjust_colors,
    channel_tables,
    grain_level,
    level_ends,
    tone_curve,
    vignette_mask,
)


class SliderValues(TypedDict, total=False):
    brightness: int
    contrast: int
    highlights: int
    shadows: int
    whites: int
    blacks: int
    saturation: int
    warmth: int
    tint: int
    sharpness: int
    vibrance: int
    vignette: int
    grain: int


class CurvePointValues(TypedDict):
    x: int
    y: int


class AdjustmentValues(SliderValues, total=False):
    curve: list[CurvePointValues]


def _pixel(picture: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    """One pixel's channels: every picture read this way has more than one."""
    value = picture.getpixel(xy)
    assert isinstance(value, tuple)
    return value


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
CASES: list[tuple[str, AdjustmentValues, str]] = [
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
    (
        "lifted shadows",
        {"shadows": 100},
        "209,137,82 19,241,160 160,160,160 255,0,255 0,0,0 "
        "255,255,255 64,129,211 0,189,0 217,253,158 188,76,180",
    ),
    (
        "deeper shadows",
        {"shadows": -60},
        "194,78,31 4,240,109 109,109,109 255,0,255 0,0,0 "
        "255,255,255 21,68,198 0,159,0 206,253,107 157,27,142",
    ),
    (
        "recovered highlights",
        {"highlights": -100},
        "166,76,42 10,227,96 96,96,96 255,0,255 0,0,0 "
        "255,255,255 32,70,170 0,132,0 179,251,95 131,38,119",
    ),
    (
        "brighter highlights",
        {"highlights": 45},
        "215,111,54 10,246,142 142,142,142 255,0,255 0,0,0 "
        "255,255,255 39,100,218 0,187,0 224,254,140 186,48,173",
    ),
    (
        "vivid",
        {"vibrance": 60},
        "219,94,31 9,241,126 128,128,128 255,0,255 0,0,0 "
        "255,255,255 26,91,221 0,184,0 205,254,96 192,30,174",
    ),
    (
        "muted",
        {"vibrance": -60},
        "181,106,68 19,235,130 128,128,128 255,0,255 0,0,0 "
        "255,255,255 47,90,179 20,156,20 215,244,156 146,60,137",
    ),
    (
        "vivid and paler",
        {"vibrance": 80, "saturation": -30},
        "202,99,47 34,230,131 128,128,128 223,22,223 0,0,0 "
        "255,255,255 36,91,203 14,170,14 208,251,119 173,41,160",
    ),
    (
        "toned and graded",
        {"shadows": 50, "highlights": -40, "contrast": 20, "saturation": -30, "warmth": 25},
        "184,112,74 48,222,136 135,131,127 210,31,210 0,0,0 "
        "255,255,255 54,95,164 29,148,29 223,247,156 151,62,134",
    ),
    (
        "faded",
        {"blacks": 60, "whites": -30},
        "193,116,77 46,224,137 137,137,137 236,38,236 38,38,38 "
        "236,236,236 67,109,196 38,170,38 201,234,136 169,73,159",
    ),
    (
        "deeper blacks and brighter whites",
        {"blacks": -50, "whites": 70},
        "240,97,26 0,255,137 137,137,137 255,0,255 0,0,0 "
        "255,255,255 7,84,244 0,197,0 254,255,134 196,19,177",
    ),
    (
        "brighter with white held down",
        {"brightness": 40, "whites": -80},
        "204,106,53 11,204,135 135,135,135 204,0,204 0,0,0 "
        "204,204,204 39,96,204 0,179,0 204,204,133 178,48,165",
    ),
    (
        "an s-curve",
        {"curve": [{"x": 64, "y": 48}, {"x": 192, "y": 208}]},
        "216,90,35 7,244,128 128,128,128 255,0,255 0,0,0 "
        "255,255,255 25,79,218 0,184,0 224,254,125 182,31,166",
    ),
    (
        "a lifted middle",
        {"curve": [{"x": 128, "y": 160}]},
        "217,129,65 13,244,160 160,160,160 255,0,255 0,0,0 "
        "255,255,255 48,118,219 0,196,0 224,254,158 195,59,185",
    ),
    (
        "a steep turn, where the slopes are scaled down",
        {"curve": [{"x": 40, "y": 120}, {"x": 60, "y": 130}, {"x": 200, "y": 140}]},
        "140,131,129 33,218,132 132,132,132 255,0,255 0,0,0 "
        "255,255,255 115,131,141 0,135,0 148,251,132 135,126,134",
    ),
    (
        "a curve under faded ends",
        {
            "curve": [{"x": 64, "y": 48}, {"x": 192, "y": 208}],
            "blacks": 40,
            "whites": -20,
            "contrast": 15,
        },
        "214,99,50 26,241,134 134,134,134 242,26,242 26,26,26 "
        "242,242,242 41,88,217 26,186,26 221,242,132 185,47,170",
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
SHARP_CASES: list[tuple[str, SliderValues, bool, str]] = [
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

# The same picture again, for the vignette, which reads where each pixel is.
VIGNETTE_CASES: list[tuple[str, SliderValues, bool, str]] = [
    (
        "darker edges",
        {"vignette": 60},
        False,
        "7,25,122,255 49,49,49,255 182,27,27,255 227,227,227,255 0,0,0,255 55,110,25,255 "
        "25,25,25,255 217,197,10,255 128,128,128,255 15,90,210,255 236,118,59,255 27,55,82,255 "
        "229,229,229,255 10,10,10,255 180,40,160,255 70,200,120,255 0,255,0,255 115,0,229,255 "
        "37,37,166,255 148,148,20,255 5,5,5,255 250,10,120,255 98,98,98,255 166,166,166,255 "
        "47,13,7,255 0,104,207,255 232,117,0,255 55,27,82,255 170,170,33,255 11,110,110,255",
    ),
    (
        "lighter edges",
        {"vignette": -45},
        False,
        "83,103,216,255 87,87,87,255 204,45,45,255 250,250,250,255 36,36,36,255 138,202,103,255 "
        "59,59,59,255 220,201,13,255 128,128,128,255 15,90,210,255 240,122,62,255 62,90,119,255 "
        "255,255,255,255 10,10,10,255 180,40,160,255 70,200,120,255 0,255,0,255 138,19,255,255 "
        "72,72,207,255 151,151,23,255 5,5,5,255 250,10,120,255 102,102,102,255 207,207,207,255 "
        "129,90,82,255 36,146,255,255 255,137,17,255 73,45,101,255 216,216,70,255 87,202,202,255",
    ),
    (
        "a cutout with darker edges",
        {"vignette": 100},
        True,
        "4,14,70,255 41,41,41,255 170,25,25,255 212,212,212,255 0,0,0,0 32,63,14,255 "
        "21,21,21,255 214,195,10,128 128,128,128,255 15,90,210,64 234,117,58,255 24,47,71,255 "
        "212,212,212,0 10,10,10,255 180,40,160,200 70,200,120,255 0,255,0,17 106,0,212,255 "
        "32,32,143,255 146,146,19,255 5,5,5,255 250,10,120,255 97,97,97,128 143,143,143,0 "
        "27,7,4,255 0,88,175,255 217,109,0,0 51,25,76,255 144,144,27,255 7,63,63,255",
    ),
    (
        "crisper, livelier and darker at the edges",
        {"sharpness": 40, "vibrance": 50, "vignette": 30},
        False,
        "7,32,173,255 55,55,55,255 208,20,20,255 239,239,239,255 0,0,0,255 64,152,18,255 "
        "28,28,28,255 253,239,0,255 124,136,134,255 0,75,255,255 253,113,28,255 21,62,103,255 "
        "242,242,242,255 0,0,0,255 233,5,204,255 30,247,112,255 0,255,0,255 121,0,242,255 "
        "37,37,203,255 175,178,0,255 0,0,0,255 255,0,136,255 89,84,95,255 183,183,183,255 "
        "74,13,2,255 0,116,231,255 244,122,0,255 62,23,102,255 193,193,25,255 9,152,152,255",
    ),
]

# The same picture again, for grain, which reads where each pixel is too.
GRAIN_CASES: list[tuple[str, SliderValues, bool, str]] = [
    (
        "grain",
        {"grain": 60},
        False,
        "2,30,190,255 61,61,61,255 203,33,33,255 255,255,255,255 16,16,16,255 91,181,41,255 "
        "40,40,40,255 218,198,8,255 123,123,123,255 3,78,198,255 238,118,58,255 42,75,108,255 "
        "255,255,255,255 21,21,21,255 176,36,156,255 72,202,122,255 11,255,11,255 120,0,247,255 "
        "53,53,208,255 152,152,22,255 7,7,7,255 255,24,134,255 88,88,88,255 199,199,199,255 "
        "69,14,3,255 12,140,255,255 255,136,8,255 57,27,87,255 211,211,41,255 23,185,185,255",
    ),
    (
        "a cutout with the most grain",
        {"grain": 100},
        True,
        "0,23,183,255 61,61,61,255 206,36,36,255 255,255,255,255 26,26,26,0 91,181,41,255 "
        "47,47,47,255 216,196,6,128 120,120,120,255 0,70,190,64 236,116,56,255 48,81,114,255 "
        "255,255,255,0 29,29,29,255 173,33,153,200 73,203,123,255 19,255,19,17 114,0,241,255 "
        "59,59,214,255 154,154,24,255 9,9,9,255 255,33,143,255 81,81,81,128 198,198,198,0 "
        "63,8,0,255 21,149,255,255 255,142,14,0 55,25,85,255 212,212,42,255 27,189,189,255",
    ),
    (
        "a film look",
        {"contrast": 10, "saturation": -15, "warmth": 10, "grain": 35},
        False,
        "4,29,172,255 55,55,54,255 190,33,33,255 255,255,255,255 9,9,9,255 96,176,48,255 "
        "29,29,29,255 222,201,28,255 126,125,123,255 10,78,185,255 235,122,67,255 36,65,94,255 "
        "255,255,255,255 9,9,8,255 170,41,148,255 81,198,124,255 29,246,29,255 115,5,221,255 "
        "47,46,185,255 152,149,31,255 1,1,1,255 238,23,121,255 91,91,90,255 206,204,201,255 "
        "62,11,1,255 22,131,239,255 244,136,27,255 51,23,77,255 216,213,58,255 31,178,176,255",
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
def test_the_pixels_the_preview_promises(
    name: str, sliders: AdjustmentValues, expected: str
) -> None:
    result = adjust_colors(_row(PIXELS), ColorAdjustments.from_request(sliders))

    assert [result.getpixel((x, 0)) for x in range(len(PIXELS))] == _pixels(expected)


def test_every_slider_at_zero_changes_nothing() -> None:
    assert ColorAdjustments().is_neutral()
    assert not ColorAdjustments(sharpness=1).is_neutral()
    assert not ColorAdjustments(highlights=-1).is_neutral()
    assert not ColorAdjustments(shadows=1).is_neutral()
    assert not ColorAdjustments(vibrance=-1).is_neutral()
    assert not ColorAdjustments(vignette=1).is_neutral()
    assert not ColorAdjustments(whites=-1).is_neutral()
    assert not ColorAdjustments(blacks=1).is_neutral()
    assert not ColorAdjustments(curve=(CurvePoint(100, 101),)).is_neutral()
    # Points on the straight line from black to white change no level.
    assert ColorAdjustments(curve=(CurvePoint(64, 64), CurvePoint(200, 200))).is_neutral()
    for table in channel_tables(ColorAdjustments()):
        assert table == list(range(256))


def test_shadows_move_the_dark_tones_most_and_highlights_the_light_ones() -> None:
    # 85 is a third of white and 170 two thirds. Shadows at 100 adds
    # 255 * 1/3 * (2/3)^2 = 37.8 to 85 and 255 * 2/3 * (1/3)^2 = 18.9 to 170;
    # highlights at -100 takes the same amounts the other way round.
    lifted = channel_tables(ColorAdjustments(shadows=100))[1]
    recovered = channel_tables(ColorAdjustments(highlights=-100))[1]

    assert [lifted[level] for level in (0, 85, 170, 255)] == [0, 123, 189, 255]
    assert [recovered[level] for level in (0, 85, 170, 255)] == [0, 66, 132, 255]


@pytest.mark.parametrize("shadows", [-100, -37, 0, 64, 100])
@pytest.mark.parametrize("highlights", [-100, -51, 0, 23, 100])
def test_no_two_levels_swap_places_however_the_tone_sliders_are_set(
    shadows: int, highlights: int
) -> None:
    for table in channel_tables(ColorAdjustments(shadows=shadows, highlights=highlights)):
        assert table[0] == 0 and table[255] == 255
        assert all(low <= high for low, high in pairwise(table))


def test_whites_and_blacks_move_white_and_black_a_quarter_of_the_range_at_most() -> None:
    lifted = channel_tables(ColorAdjustments(blacks=100))[1]
    deepened = channel_tables(ColorAdjustments(blacks=-100))[1]
    brightened = channel_tables(ColorAdjustments(whites=100))[1]
    dimmed = channel_tables(ColorAdjustments(whites=-100))[1]

    # Black lifts to 63.75 and white keeps its place; the rest spread evenly.
    assert [lifted[level] for level in (0, 128, 255)] == [64, 160, 255]
    # Every level up to the lowest quarter's top becomes black.
    assert set(deepened[:65]) == {0} and deepened[255] == 255
    # Every level from the highest quarter's foot becomes white.
    assert brightened[0] == 0 and set(brightened[192:]) == {255}
    assert [dimmed[level] for level in (0, 128, 255)] == [0, 96, 191]


def test_white_and_black_hold_however_far_the_other_sliders_take_the_picture() -> None:
    glaring = ColorAdjustments(brightness=100, contrast=100, whites=-100)
    murky = ColorAdjustments(brightness=-100, contrast=100, blacks=100)

    assert {max(table) for table in channel_tables(glaring)} == {191}
    assert {min(table) for table in channel_tables(murky)} == {64}


@pytest.mark.parametrize("blacks", [-100, -33, 0, 41, 100])
@pytest.mark.parametrize("whites", [-100, -67, 0, 12, 100])
def test_no_two_levels_swap_places_however_whites_and_blacks_are_set(
    blacks: int, whites: int
) -> None:
    _, black, _, white = level_ends(blacks, whites)
    alone = channel_tables(ColorAdjustments(blacks=blacks, whites=whites))
    toned = channel_tables(ColorAdjustments(blacks=blacks, whites=whites, contrast=-45, shadows=30))

    assert all(table[0] == int(black + 0.5) and table[255] == int(white + 0.5) for table in alone)
    for table in (*alone, *toned):
        assert all(low <= high for low, high in pairwise(table))


def test_whites_and_blacks_run_from_minus_100_to_100() -> None:
    for name in ("whites", "blacks"):
        assert getattr(StudioColorAdjustments(**{name: -100}), name) == -100
        with pytest.raises(ValueError):
            StudioColorAdjustments(**{name: -101})
        with pytest.raises(ValueError):
            StudioColorAdjustments(**{name: 101})


def test_the_tone_curve_passes_through_its_points_and_keeps_black_and_white() -> None:
    points = (CurvePoint(40, 120), CurvePoint(60, 130), CurvePoint(200, 140))
    curve = tone_curve(points)
    assert curve is not None

    assert [curve.level(point.x) for point in points] == [120.0, 130.0, 140.0]
    assert (curve.level(0), curve.level(255)) == (0.0, 255.0)
    for table in channel_tables(ColorAdjustments(curve=points)):
        assert [table[point.x] for point in points] == [120, 130, 140]
        assert (table[0], table[255]) == (0, 255)
    assert tone_curve(()) is None


@pytest.mark.parametrize(
    "points",
    [
        [(64, 48), (192, 208)],
        [(40, 120), (60, 130), (200, 140)],
        [(10, 90), (20, 95), (30, 200), (240, 201), (250, 254), (254, 255)],
    ],
)
def test_a_rising_curve_never_swaps_two_levels_or_overshoots_its_points(
    points: list[tuple[int, int]],
) -> None:
    table = channel_tables(ColorAdjustments(curve=tuple(CurvePoint(x, y) for x, y in points)))[0]

    assert all(low <= high for low, high in pairwise(table))
    ends = [(0, 0), *points, (255, 255)]
    for (left, low), (right, high) in pairwise(ends):
        assert all(low <= table[level] <= high for level in range(left, right + 1))


def test_a_curve_runs_left_to_right_with_at_most_six_points_inside_the_range() -> None:
    assert StudioColorAdjustments(curve=[{"x": 1, "y": 0}, {"x": 254, "y": 255}]).curve
    for curve in (
        [{"x": 80, "y": 10}, {"x": 80, "y": 90}],
        [{"x": 90, "y": 10}, {"x": 80, "y": 90}],
        [{"x": 0, "y": 10}],
        [{"x": 255, "y": 10}],
        [{"x": 10, "y": 256}],
        [{"x": 10 * index + 10, "y": 50} for index in range(7)],
    ):
        with pytest.raises(ValueError):
            StudioColorAdjustments(curve=curve)


def test_transparency_is_kept_as_it_was() -> None:
    picture = _row([(200, 100, 50), (200, 100, 50)], "RGBA")
    picture.putpixel((0, 0), (200, 100, 50, 0))
    picture.putpixel((1, 0), (200, 100, 50, 77))

    result = adjust_colors(picture, ColorAdjustments(brightness=60, saturation=-50))

    assert result.mode == "RGBA"
    assert [_pixel(result, (x, 0))[3] for x in range(2)] == [0, 77]


def test_warmth_changes_the_color_and_not_the_light() -> None:
    grey = Image.new("RGB", (8, 8), (128, 128, 128))

    warm = adjust_colors(grey, ColorAdjustments(warmth=80))
    cool = adjust_colors(grey, ColorAdjustments(warmth=-80))

    red, _, blue = _pixel(warm, (0, 0))
    assert red > blue
    cool_red, _, cool_blue = _pixel(cool, (0, 0))
    assert cool_blue > cool_red
    for picture in (warm, cool):
        assert abs(ImageStat.Stat(picture.convert("L")).mean[0] - 128) < 2


def test_tint_moves_green_against_magenta_and_not_the_light() -> None:
    grey = Image.new("RGB", (8, 8), (128, 128, 128))

    magenta = adjust_colors(grey, ColorAdjustments(tint=80))
    green = adjust_colors(grey, ColorAdjustments(tint=-80))

    red, middle, blue = _pixel(magenta, (0, 0))
    assert red == blue > middle
    red, middle, blue = _pixel(green, (0, 0))
    assert middle > red == blue
    # Measured with the same luminance weights the gains are evened out by.
    # Pillow's own grey uses older weights, which count green for less.
    for picture in (magenta, green):
        red, middle, blue = _pixel(picture, (0, 0))
        assert abs(0.2126 * red + 0.7152 * middle + 0.0722 * blue - 128) < 1


@pytest.mark.parametrize(
    ("name", "sliders", "cutout", "expected"), SHARP_CASES, ids=[case[0] for case in SHARP_CASES]
)
def test_the_sharpened_pixels_the_preview_promises(
    name: str, sliders: SliderValues, cutout: bool, expected: str
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


def test_vibrance_moves_a_muted_color_about_as_far_as_saturation_and_a_vivid_one_less() -> None:
    muted, vivid = (150, 130, 120), (230, 40, 30)
    picture = _row([muted, vivid])

    livelier = adjust_colors(picture, ColorAdjustments(vibrance=100))
    richer = adjust_colors(picture, ColorAdjustments(saturation=100))

    def moved(result: Image.Image, x: int, before: tuple[int, int, int]) -> int:
        return sum(abs(a - b) for a, b in zip(_pixel(result, (x, 0)), before, strict=True))

    # The muted color has a spread of 30, so it keeps 225/255 of the mix; the
    # vivid one has 200 and keeps 55/255 of it.
    assert moved(livelier, 0, muted) >= moved(richer, 0, muted) * 0.8
    assert moved(livelier, 1, vivid) <= moved(richer, 1, vivid) * 0.3


@pytest.mark.parametrize(
    ("name", "sliders", "cutout", "expected"),
    VIGNETTE_CASES,
    ids=[case[0] for case in VIGNETTE_CASES],
)
def test_the_vignetted_pixels_the_preview_promises(
    name: str, sliders: SliderValues, cutout: bool, expected: str
) -> None:
    result = adjust_colors(_sharpness_picture(cutout), ColorAdjustments(**sliders))

    assert result.mode == ("RGBA" if cutout else "RGB")
    assert _grid(result) == [
        tuple(int(part) for part in pixel.split(",")) for pixel in expected.split()
    ]


def test_the_vignette_leaves_the_middle_and_gathers_evenly_toward_the_corners() -> None:
    grey = Image.new("RGB", (64, 48), (160, 160, 160))

    reds = _reds(adjust_colors(grey, ColorAdjustments(vignette=100)))

    assert reds[23][31] == reds[24][32] == 160
    # Mirrored left to right and top to bottom, as a centred vignette is.
    assert all(row == row[::-1] for row in reds)
    assert reds == reds[::-1]
    # Darker at every step out along the diagonal, and darkest in the corner.
    diagonal = [reds[24 + step * 3 // 4][32 + step] for step in range(32)]
    assert all(outer <= inner for inner, outer in pairwise(diagonal))
    assert reds[0][0] == min(min(row) for row in reds) < reds[24][0] < 160


def test_a_vignette_below_zero_lightens_the_edges_instead() -> None:
    grey = Image.new("RGB", (64, 48), (100, 100, 100))

    reds = _reds(adjust_colors(grey, ColorAdjustments(vignette=-100)))

    assert reds[24][32] == 100
    assert reds[0][0] > reds[24][0] > 100


def test_a_picture_one_pixel_wide_or_tall_takes_the_vignette_too() -> None:
    # Only the middle of a strip is inside the start of the vignette.
    assert list(vignette_mask(1, 3).tobytes()) == list(vignette_mask(3, 1).tobytes())
    assert list(vignette_mask(3, 1).tobytes()) == [12, 0, 12]
    assert list(vignette_mask(1, 1).tobytes()) == [0]


@pytest.mark.parametrize(
    ("name", "sliders", "cutout", "expected"),
    GRAIN_CASES,
    ids=[case[0] for case in GRAIN_CASES],
)
def test_the_grained_pixels_the_preview_promises(
    name: str, sliders: SliderValues, cutout: bool, expected: str
) -> None:
    result = adjust_colors(_sharpness_picture(cutout), ColorAdjustments(**sliders))

    assert result.mode == ("RGBA" if cutout else "RGB")
    assert _grid(result) == [
        tuple(int(part) for part in pixel.split(",")) for pixel in expected.split()
    ]


def test_the_grain_levels_are_the_ones_the_browser_works_out() -> None:
    # The same numbers are pinned in studioAdjustments.test.ts.
    assert [grain_level(x, 0) for x in range(8)] == [59, 133, 151, 193, 232, 132, 185, 110]
    assert (grain_level(255, 255), grain_level(3, 7)) == (126, 172)


def test_grain_repeats_every_tile_and_keeps_the_picture_s_light() -> None:
    grey = Image.new("RGB", (GRAIN_TILE * 2, 40), (128, 128, 128))

    grained = adjust_colors(grey, ColorAdjustments(grain=100))
    reds = _reds(grained)

    # The same step on all three channels, so grey stays grey.
    assert all(len(set(_pixel(grained, (x, y)))) == 1 for y in range(40) for x in range(0, 512, 37))
    assert all(row[:GRAIN_TILE] == row[GRAIN_TILE:] for row in reds)
    assert len({value for row in reds for value in row}) > 40
    # Up as often as down: the picture's light is kept.
    assert abs(ImageStat.Stat(grained).mean[0] - 128) < 1
    assert max(value for row in reds for value in row) <= 128 + 32


def test_grain_is_added_or_not_and_counts_as_a_change() -> None:
    assert not ColorAdjustments(grain=1).is_neutral()
    picture = _sharpness_picture(False)
    assert _grid(adjust_colors(picture, ColorAdjustments(grain=0))) == _grid(picture)
    with pytest.raises(ValueError):
        StudioColorAdjustments(grain=-1)
    with pytest.raises(ValueError):
        StudioColorAdjustments(grain=101)
