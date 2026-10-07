"""Prepare a centered crop with exact aspect and one scale on both axes.

The viewport uses rational pixel-edge coordinates; it is not an integer crop
rounded to a different aspect. Pixels wholly outside its enclosing integer
rectangle are removed before Lanczos filtering, so unrelated outside pixels
cannot bleed into the crop. The remaining fractional edge cells participate
in the recorded resampling.

This module only prepares bounded immutable bytes. It neither stores them nor
authorizes a workflow. Admission must retain those exact bytes and bind the
recipe, and the execution route must prove that it does not crop them again.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from io import BytesIO
from typing import Any

from PIL import Image
from PIL import __version__ as pillow_version

from .output_geometry import MAX_DIMENSION
from .source_fit_image import MAX_SOURCE_BYTES, MAX_SOURCE_PIXELS, PreparedSourceImage


@dataclass(frozen=True, slots=True)
class SourceCropPlan:
    source_size: tuple[int, int]
    canvas_size: tuple[int, int]
    rectangle: tuple[Fraction, Fraction, Fraction, Fraction]
    scale: Fraction

    @property
    def sample_bounds(self) -> tuple[int, int, int, int]:
        left, top, width, height = self.rectangle
        x, y = left.numerator // left.denominator, top.numerator // top.denominator
        right, bottom = left + width, top + height
        end_x = -(-right.numerator // right.denominator)
        end_y = -(-bottom.numerator // bottom.denominator)
        return x, y, end_x - x, end_y - y

    def payload(self) -> dict[str, Any]:
        def rational(value: Fraction) -> list[int]:
            return [value.numerator, value.denominator]

        left, top, width, height = self.rectangle
        x, y, sampled_width, sampled_height = self.sample_bounds
        return {
            "v": 1,
            "mode": "crop",
            "source": dict(zip(("width", "height"), self.source_size, strict=True)),
            "canvas": dict(zip(("width", "height"), self.canvas_size, strict=True)),
            "rectangle": {
                "left": rational(left),
                "top": rational(top),
                "width": rational(width),
                "height": rational(height),
            },
            "sample_bounds": {
                "left": x,
                "top": y,
                "width": sampled_width,
                "height": sampled_height,
            },
            "scale": rational(self.scale),
            "resampler": "lanczos",
            "coordinate_convention": "pixel-edges-v1",
        }


def _size(width: object, height: object) -> tuple[int, int]:
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= MAX_DIMENSION
        or not 1 <= height <= MAX_DIMENSION
        or width * height > MAX_SOURCE_PIXELS
    ):
        raise ValueError("source_crop_dimensions")
    return width, height


def plan_source_crop(
    source_width: object, source_height: object, canvas_width: object, canvas_height: object
) -> SourceCropPlan:
    source = _size(source_width, source_height)
    canvas = _size(canvas_width, canvas_height)
    scale = max(Fraction(canvas[0], source[0]), Fraction(canvas[1], source[1]))
    width, height = Fraction(canvas[0]) / scale, Fraction(canvas[1]) / scale
    return SourceCropPlan(
        source_size=source,
        canvas_size=canvas,
        rectangle=((source[0] - width) / 2, (source[1] - height) / 2, width, height),
        scale=scale,
    )


@dataclass(frozen=True, slots=True)
class PreparedSourceCrop:
    source: PreparedSourceImage = field(repr=False)
    plan: SourceCropPlan
    sha256: str
    resampler_version: str
    content: bytes = field(repr=False)


def prepare_source_crop(
    source: PreparedSourceImage, canvas_width: object, canvas_height: object
) -> PreparedSourceCrop:
    plan = plan_source_crop(source.width, source.height, canvas_width, canvas_height)
    if (
        type(source.content) is not bytes
        or len(source.content) > MAX_SOURCE_BYTES
        or hashlib.sha256(source.content).hexdigest() != source.sha256
        or source.source_artifact_id != f"sha256:{source.source_sha256}"
    ):
        raise ValueError("source_crop_source_binding")
    try:
        with Image.open(BytesIO(source.content)) as original:
            if (
                original.format != "PNG"
                or original.mode != "RGB"
                or original.size != plan.source_size
                or original.getexif()
                or original.info
            ):
                raise ValueError("source_crop_source_binding")
            original.load()
            x, y, width, height = plan.sample_bounds
            left, top, kept_width, kept_height = plan.rectangle
            # Clip whole outside cells first. A resize with a box on the original
            # image alone can sample those cells through the filter's support.
            with original.crop((x, y, x + width, y + height)) as bounded:
                box = (
                    float(left - x),
                    float(top - y),
                    float(left - x + kept_width),
                    float(top - y + kept_height),
                )
                with bounded.resize(plan.canvas_size, Image.Resampling.LANCZOS, box=box) as cropped:
                    output = BytesIO()
                    cropped.save(output, format="PNG", compress_level=6)
                    content = output.getvalue()
    except (OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("source_crop_source_binding") from None
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError("source_crop_dimensions")
    return PreparedSourceCrop(
        source=source,
        plan=plan,
        sha256=hashlib.sha256(content).hexdigest(),
        resampler_version=f"pillow-{pillow_version}-lanczos-pixel-edges-v1",
        content=content,
    )
