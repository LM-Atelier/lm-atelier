"""Light and color adjustments, made the way the studio previews them.

The studio shows an adjustment on the picture while its sliders move, so the
arithmetic here is the arithmetic the browser runs, in the same order: one
lookup table per channel for warmth, brightness and contrast, then saturation
as a mix toward each pixel's grey. Every step is integer arithmetic or
floating-point arithmetic with a stated rounding, which the browser repeats
exactly, so the preview is the picture an apply makes. The browser's copy lives
in studioAdjustments.ts, and the two are checked against the same pixels.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from PIL import Image

from .studio_relight import NEUTRAL_KELVIN, warmth_gains

#: Each slider runs from -100 to 100, with 0 changing nothing.
ADJUSTMENT_LIMIT = 100
#: How far one step of the warmth slider moves the white point, in kelvin. At
#: 100 the picture is balanced for 4000 K light, and at -100 for 9000 K.
KELVIN_PER_WARMTH_STEP = 25


@dataclass(frozen=True)
class ColorAdjustments:
    """Where each slider stands, from -100 to 100."""

    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    warmth: int = 0

    def is_neutral(self) -> bool:
        return not (self.brightness or self.contrast or self.saturation or self.warmth)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rounded(value: float) -> int:
    # Half rounds up, as the browser's Math.round does; Python's round would
    # send some halves down and the two would disagree by one level.
    return min(255, max(0, math.floor(value + 0.5)))


def channel_tables(adjustments: ColorAdjustments) -> tuple[list[int], list[int], list[int]]:
    """The red, green and blue lookup tables for warmth, brightness and contrast.

    Brightness scales every channel by two to the power of its slider over 100,
    so 100 doubles the light and -100 halves it. Contrast does the same to the
    distance from middle grey. Nothing is clamped until the one rounding at
    the end, so the steps do not lose detail to each other.
    """

    gains = warmth_gains(NEUTRAL_KELVIN - KELVIN_PER_WARMTH_STEP * adjustments.warmth)
    brightness = 2 ** (adjustments.brightness / ADJUSTMENT_LIMIT)
    contrast = 2 ** (adjustments.contrast / ADJUSTMENT_LIMIT)
    tables: list[list[int]] = []
    for gain in gains:
        table = []
        for value in range(256):
            lit = value * gain * brightness
            table.append(_rounded((lit - 127.5) * contrast + 127.5))
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
    if alpha is None:
        return adjusted
    adjusted.putalpha(alpha)
    return adjusted
