"""Light and color adjustments, made the way the studio previews them.

The studio shows an adjustment on the picture while its sliders move, so the
arithmetic here is the arithmetic the browser runs, in the same order: one
lookup table per channel for shadows, highlights, warmth, tint, brightness,
contrast, the tone curve, whites and blacks, then saturation as a mix toward
each pixel's grey,
then vibrance as the same mix kept in proportion to how muted each pixel is,
then sharpness as a mix away from a softened copy of the picture, then the
vignette, a mix toward black or white that grows toward the corners, and last
grain, the same small step up or down on all three channels of each pixel,
taken from a fixed tile of noise. Every step is integer
arithmetic or floating-point arithmetic with a stated rounding, which the
browser repeats exactly, so the preview is the picture an apply makes. The
browser's copy lives in studioAdjustments.ts, and the two are checked against
the same pixels.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from functools import cache
from operator import itemgetter
from typing import Any

from PIL import Image, ImageChops, ImageFilter

from .studio_relight import NEUTRAL_KELVIN, warmth_gains

#: Each slider runs from -100 to 100, with 0 changing nothing.
ADJUSTMENT_LIMIT = 100
#: How far one step of the warmth slider moves the white point, in kelvin. At
#: 100 the picture is balanced for 4000 K light, and at -100 for 9000 K.
KELVIN_PER_WARMTH_STEP = 25
#: How far the tint slider at either end moves green against red and blue,
#: before the gains are evened out to keep the brightness.
TINT_REACH = 0.15
#: How far the whites and blacks sliders at either end move white or black, as
#: a share of the range: at 100 blacks lifts black a quarter of the way up, and
#: at -100 every level in the lowest quarter becomes black.
LEVELS_REACH = 0.25
#: The softened copy sharpness mixes against: each pixel's neighborhood of
#: nine, weighted 1-2-1 each way. The weights sum to 16, so every sum the
#: kernel makes in single precision is exact and the browser's integers match.
_SOFTENED = ImageFilter.Kernel((3, 3), (1, 2, 1, 2, 4, 2, 1, 2, 1), 16)
#: How far toward black, or toward white, the vignette takes the corners at
#: either end of its slider.
VIGNETTE_REACH = 0.8
#: Where the vignette begins and where it is whole, as the squared distance from
#: the middle of the picture, on which the middle of each edge is 1 and each
#: corner 2. Nothing changes inside the first, so the middle keeps its light.
VIGNETTE_START = 0.25
VIGNETTE_FULL = 1.75
#: How finely that squared distance is counted: this many steps to the middle
#: of an edge, in whole numbers the browser counts the same way.
VIGNETTE_STEPS = 4096
#: How far the grain slider at its top moves a pixel up or down, as a share of
#: the grain's own spread of 256 levels.
GRAIN_REACH = 0.25
#: The grain repeats every this many pixels across and down.
GRAIN_TILE = 256
#: The most points a tone curve passes through between its fixed ends.
CURVE_POINTS = 6


@dataclass(frozen=True)
class CurvePoint:
    """A point the tone curve passes through: a level, and the level it becomes."""

    x: int
    y: int


@dataclass(frozen=True)
class ColorAdjustments:
    """Where each slider stands, from -100 to 100."""

    brightness: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    #: Above zero the lightest levels brighten into white; below, white dims to a grey.
    whites: int = 0
    #: Above zero black lifts to a grey; below, the darkest levels deepen into black.
    blacks: int = 0
    saturation: int = 0
    warmth: int = 0
    tint: int = 0
    sharpness: int = 0
    vibrance: int = 0
    vignette: int = 0
    #: From 0 to 100 only: grain is added or not; there is no taking it away.
    grain: int = 0
    #: The tone curve's points between black and white, left to right; black
    #: and white themselves stay where they are.
    curve: tuple[CurvePoint, ...] = ()

    @classmethod
    def from_request(cls, values: Mapping[str, Any]) -> ColorAdjustments:
        """The adjustments a request names, with its curve's points as points."""

        points = tuple(CurvePoint(**point) for point in values.get("curve", ()))
        return cls(**{**values, "curve": points})

    def is_neutral(self) -> bool:
        return not (
            self.brightness
            or self.contrast
            or self.highlights
            or self.shadows
            or self.whites
            or self.blacks
            or self.saturation
            or self.warmth
            or self.tint
            or self.sharpness
            or self.vibrance
            or self.vignette
            or self.grain
            or any(point.x != point.y for point in self.curve)
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rounded(value: float) -> int:
    # Half rounds up, as the browser's Math.round does; Python's round would
    # send some halves down and the two would disagree by one level.
    return min(255, max(0, math.floor(value + 0.5)))


def tint_gains(tint: int) -> tuple[float, float, float]:
    """The red, green and blue gains that move the picture toward magenta or green.

    A positive tint raises red and blue against green, which is magenta, and a
    negative one does the reverse. The gains are evened out by their luma
    weight, as warmth's are, so a tint changes the color and not the light.
    """

    shift = TINT_REACH * tint / ADJUSTMENT_LIMIT
    gains = (1 + shift, 1 - shift, 1 + shift)
    luma = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2]
    return (gains[0] / luma, gains[1] / luma, gains[2] / luma)


def toned(value: int, shadows: float, highlights: float) -> float:
    """A level moved by the shadows and highlights sliders, each from -1 to 1.

    Taking the level as a share of white, shadows moves it by the share times
    the square of what is left above it, which reaches furthest a third of the
    way up, and highlights by the square of the share times what is left,
    which reaches furthest two thirds of the way up. Either slider alone moves
    no level by more than 4/27 of the range. Black and white stay where they
    are, and however the two are set together, no two levels swap places.
    """

    share = value / 255
    left = 1 - share
    return value + 255 * (shadows * share * left * left + highlights * share * share * left)


def level_ends(blacks: int, whites: int) -> tuple[float, float, float, float]:
    """The levels the blacks and whites sliders make black and white, and what they make them.

    In order: the level that becomes black and the level black becomes, then
    the level that becomes white and the level white becomes. Above zero,
    blacks lifts black to a grey, and below, it deepens the darkest levels into
    black. Whites does the same at the other end: above zero the lightest
    levels brighten into white, and below, white dims to a grey. Either moves
    its end by at most LEVELS_REACH of the range.
    """

    black = 255 * LEVELS_REACH * abs(blacks) / ADJUSTMENT_LIMIT
    white = 255 * LEVELS_REACH * abs(whites) / ADJUSTMENT_LIMIT
    black_from, black_to = (0.0, black) if blacks > 0 else (black, 0.0)
    white_from, white_to = (255 - white, 255.0) if whites > 0 else (255.0, 255 - white)
    return (black_from, black_to, white_from, white_to)


@dataclass(frozen=True)
class ToneCurve:
    """A tone curve ready to read: its points, black and white included, and its slope at each."""

    xs: tuple[float, ...]
    ys: tuple[float, ...]
    slopes: tuple[float, ...]

    def level(self, value: float) -> float:
        """The curve's height at `value`, from 0 to 255.

        Between two points it is the cubic that meets both with the slopes
        found for them. The powers are written out as products, since a power
        function may round differently from the browser's.
        """

        segment = 0
        while value > self.xs[segment + 1]:
            segment += 1
        left, right = self.xs[segment], self.xs[segment + 1]
        width = right - left
        t = (value - left) / width
        t2 = t * t
        t3 = t2 * t
        return (
            (2 * t3 - 3 * t2 + 1) * self.ys[segment]
            + (t3 - 2 * t2 + t) * width * self.slopes[segment]
            + (-2 * t3 + 3 * t2) * self.ys[segment + 1]
            + (t3 - t2) * width * self.slopes[segment + 1]
        )


def tone_curve(points: tuple[CurvePoint, ...]) -> ToneCurve | None:
    """The curve through black, the points and white, or none when there are no points.

    Each point's slope is found as Fritsch and Carlson find it for a curve
    that never overshoots its points: the mean of the two straight lines to
    its neighbours, zero where the curve turns back, and scaled down where it
    would carry the curve past the next point. Only sums, products, quotients
    and one square root, in an order the browser repeats.
    """

    if not points:
        return None
    xs = (0.0, *(float(point.x) for point in points), 255.0)
    ys = (0.0, *(float(point.y) for point in points), 255.0)
    secants = [(ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]) for k in range(len(xs) - 1)]
    slopes = [secants[0]]
    for k in range(1, len(secants)):
        before, after = secants[k - 1], secants[k]
        slopes.append((before + after) / 2 if before * after > 0 else 0.0)
    slopes.append(secants[-1])
    for k, secant in enumerate(secants):
        if secant == 0:
            slopes[k] = 0.0
            slopes[k + 1] = 0.0
            continue
        into = slopes[k] / secant
        out = slopes[k + 1] / secant
        reach = into * into + out * out
        if reach > 9:
            scale = 3 / math.sqrt(reach)
            slopes[k] = scale * into * secant
            slopes[k + 1] = scale * out * secant
    return ToneCurve(xs, ys, tuple(slopes))


def channel_tables(adjustments: ColorAdjustments) -> tuple[list[int], list[int], list[int]]:
    """The three channels' lookup tables for tone, color, brightness, contrast, whites and blacks.

    Each level is first moved by the shadows and highlights, so they act on the
    picture's own tones, then by the warmth and tint gains. Brightness scales
    every channel by two to the power of its slider over 100, so 100 doubles
    the light and -100 halves it. Contrast does the same to the distance from
    middle grey. The level as it then stands, held within the range, passes
    through the tone curve, which keeps black and white where they are. Last
    it is moved so that the levels the whites and blacks take as white and
    black land where they put them, with the levels between spread evenly.
    Holding it first makes the white and black they set the picture's
    lightest and darkest, however far the other sliders took it. Nothing is
    rounded until the end, so the steps do not lose detail to each other.
    """

    warmth = warmth_gains(NEUTRAL_KELVIN - KELVIN_PER_WARMTH_STEP * adjustments.warmth)
    tint = tint_gains(adjustments.tint)
    gains = [warmth[index] * tint[index] for index in range(3)]
    brightness = 2 ** (adjustments.brightness / ADJUSTMENT_LIMIT)
    contrast = 2 ** (adjustments.contrast / ADJUSTMENT_LIMIT)
    shadows = adjustments.shadows / ADJUSTMENT_LIMIT
    highlights = adjustments.highlights / ADJUSTMENT_LIMIT
    curve = tone_curve(adjustments.curve)
    black_from, black_to, white_from, white_to = level_ends(adjustments.blacks, adjustments.whites)
    spread = (white_to - black_to) / (white_from - black_from)
    tables: list[list[int]] = []
    for gain in gains:
        table = []
        for value in range(256):
            lit = toned(value, shadows, highlights) * gain * brightness
            level = min(255.0, max(0.0, (lit - 127.5) * contrast + 127.5))
            if curve is not None:
                level = curve.level(level)
            table.append(_rounded(black_to + (level - black_from) * spread))
        tables.append(table)
    return (tables[0], tables[1], tables[2])


def saturated(color: Image.Image, saturation: int) -> Image.Image:
    """Each channel moved toward or away from its pixel's grey by the slider.

    -100 leaves grey and 100 doubles every distance from it. The grey is
    Pillow's integer luma, and the move is its blend, which works in single
    precision and truncates; the browser repeats both.
    """

    if saturation == 0:
        return color
    grey = color.convert("L").convert("RGB")
    return Image.blend(grey, color, 1 + saturation / ADJUSTMENT_LIMIT)


def vibrant(color: Image.Image, vibrance: int) -> Image.Image:
    """Saturation that moves muted colors most and leaves vivid ones nearly alone.

    Each pixel is mixed toward or away from its grey as the saturation slider
    would mix it, and the mix is kept in proportion to how muted the pixel is:
    its spread, the brightest channel less the dimmest, taken from white. A
    pixel with no spread takes the whole mix and one at full spread keeps its
    color, so muted colors richen before vivid ones clip. The proportion is
    Pillow's composite, which rounds in whole numbers the browser repeats.
    """

    if vibrance == 0:
        return color
    red, green, blue = color.split()
    brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    dimmest = ImageChops.darker(ImageChops.darker(red, green), blue)
    muted = ImageChops.invert(ImageChops.subtract(brightest, dimmest))
    return Image.composite(saturated(color, vibrance), color, muted)


def sharpened(color: Image.Image, alpha: Image.Image | None, sharpness: int) -> Image.Image:
    """Each channel moved toward or away from a softened copy of the picture.

    -100 leaves the softened copy and 100 doubles every difference from it,
    which steepens each edge. Pixels on the picture's own edge have no full
    neighborhood, so the kernel leaves them as they are. Where the picture
    has transparency, the copy is made from each color weighted by its
    opacity and divided back out, so a color hidden under a transparent
    pixel does not bleed into the visible ones beside it.
    """

    if sharpness == 0:
        return color
    if alpha is None:
        soft = color.filter(_SOFTENED)
    else:
        weighted = Image.merge("RGBA", (*color.split(), alpha)).convert("RGBa")
        soft = weighted.filter(_SOFTENED).convert("RGBA").convert("RGB")
    return Image.blend(soft, color, 1 + sharpness / ADJUSTMENT_LIMIT)


def _distance_steps(size: int) -> list[int]:
    """Each column's or row's squared distance from the middle, in whole steps.

    Measured from the centre of each pixel as a share of half the picture's
    width or height, squared and floored, so the middle of an edge is
    VIGNETTE_STEPS away and the middle of the picture none. Exact at any size;
    the browser works it in exact integers too.
    """

    area = size * size
    return [(VIGNETTE_STEPS * (2 * index + 1 - size) ** 2) // area for index in range(size)]


def _vignette_weights() -> list[int]:
    """How much of the vignette each squared distance takes, from 0 to 255.

    None up to VIGNETTE_START, all of it from VIGNETTE_FULL, and a smooth step
    between, so no ring shows where it begins. Only sums, products and
    divisions, each rounded as the browser rounds it.
    """

    weights = []
    for step in range(2 * VIGNETTE_STEPS):
        share = (step / VIGNETTE_STEPS - VIGNETTE_START) / (VIGNETTE_FULL - VIGNETTE_START)
        share = min(1.0, max(0.0, share))
        weights.append(math.floor(255 * share * share * (3 - 2 * share) + 0.5))
    return weights


_VIGNETTE_WEIGHTS = _vignette_weights()


def vignette_mask(width: int, height: int) -> Image.Image:
    """How much of the vignette each pixel takes, as a greyscale picture.

    A row depends only on its distance from the middle, so each distinct row
    is worked out once and repeated.
    """

    across = _distance_steps(width)
    pick = itemgetter(*across)
    lines: dict[int, bytes] = {}
    data = bytearray()
    for down in _distance_steps(height):
        line = lines.get(down)
        if line is None:
            picked = pick(_VIGNETTE_WEIGHTS[down : down + VIGNETTE_STEPS])
            line = lines[down] = bytes(picked if width > 1 else (picked,))
        data += line
    return Image.frombytes("L", (width, height), bytes(data))


def vignetted(color: Image.Image, vignette: int) -> Image.Image:
    """The edges darkened above zero, or lightened below, most at the corners."""

    if vignette == 0:
        return color
    reach = abs(vignette) / ADJUSTMENT_LIMIT * VIGNETTE_REACH
    if vignette > 0:
        table = [_rounded(value * (1 - reach)) for value in range(256)]
    else:
        table = [_rounded(value + (255 - value) * reach) for value in range(256)]
    return Image.composite(color.point(table * 3), color, vignette_mask(*color.size))


def grain_level(x: int, y: int) -> int:
    """The grain at one place of its tile, from 0 to 255.

    A whole-number hash of the place, worked in 32 bits the browser repeats
    with the same multiplications and shifts, and the mean of two of its bytes,
    so middling levels are commoner than extremes, as they are in film grain.
    The added constant keeps the first place of every tile from hashing to
    nothing, which would put the darkest level on a regular grid.
    """

    mixed = (x * 374761393 + y * 668265263 + 0x9E3779B9) & 0xFFFFFFFF
    mixed = ((mixed ^ (mixed >> 13)) * 1274126177) & 0xFFFFFFFF
    mixed ^= mixed >> 16
    return ((mixed & 255) + ((mixed >> 8) & 255)) >> 1


@cache
def grain_tile() -> Image.Image:
    """The tile of grain levels, made once and laid across a picture of any size."""

    tile = Image.new("L", (GRAIN_TILE, GRAIN_TILE))
    tile.putdata([grain_level(x, y) for y in range(GRAIN_TILE) for x in range(GRAIN_TILE)])
    return tile


def grain_offsets(grain: int) -> list[int]:
    """How far each grain level moves a pixel at this slider value, plus 128."""

    reach = grain / ADJUSTMENT_LIMIT * GRAIN_REACH
    return [_rounded(128 + (level - 128) * reach) for level in range(256)]


def grained(color: Image.Image, grain: int) -> Image.Image:
    """Grain laid over the picture: each pixel moved the same way on all three channels."""

    if grain <= 0:
        return color
    offsets = grain_tile().point(grain_offsets(grain))
    width, height = color.size
    field = Image.new("L", (width, height))
    for top in range(0, height, GRAIN_TILE):
        for left in range(0, width, GRAIN_TILE):
            field.paste(offsets, (left, top))
    # A clipped whole-number add: each channel plus the offset, less the 128 it carries.
    return Image.merge(
        "RGB", [ImageChops.add(band, field, scale=1.0, offset=-128) for band in color.split()]
    )


def adjust_colors(picture: Image.Image, adjustments: ColorAdjustments) -> Image.Image:
    """The picture with the adjustments applied; transparency is kept as it was."""

    alpha = picture.getchannel("A") if picture.mode == "RGBA" else None
    color = picture.convert("RGB") if alpha is not None else picture
    red, green, blue = color.split()
    tables = channel_tables(adjustments)
    adjusted = Image.merge(
        "RGB", (red.point(tables[0]), green.point(tables[1]), blue.point(tables[2]))
    )
    adjusted = saturated(adjusted, adjustments.saturation)
    adjusted = vibrant(adjusted, adjustments.vibrance)
    adjusted = sharpened(adjusted, alpha, adjustments.sharpness)
    adjusted = vignetted(adjusted, adjustments.vignette)
    adjusted = grained(adjusted, adjustments.grain)
    if alpha is None:
        return adjusted
    adjusted.putalpha(alpha)
    return adjusted
