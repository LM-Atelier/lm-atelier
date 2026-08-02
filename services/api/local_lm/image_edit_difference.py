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
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

# Small enough that re-encoding noise and one-pixel resampling differences
# average away, large enough that a genuine local edit (a recoloured object,
# a changed garment) still moves the mean well past the threshold.
COMPARISON_SIZE = (64, 64)
# Mean absolute difference per channel value, 0-255. Chosen an order of
# magnitude above observed re-encode noise (well under 1.0) and an order
# below a real edit (typically 5+).
UNCHANGED_THRESHOLD = 2.0


@dataclass(frozen=True)
class ImageDifference:
    """How much two images differ, and whether that counts as a change."""

    mean_absolute_difference: float
    changed: bool
    comparable: bool

    def provenance(self) -> dict[str, object]:
        return {
            "mean_absolute_difference": round(self.mean_absolute_difference, 4),
            "changed": self.changed,
            "comparable": self.comparable,
            "threshold": UNCHANGED_THRESHOLD,
        }


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


def _normalized(payload: bytes) -> list[int]:
    with Image.open(io.BytesIO(payload)) as image:
        # Colour, not luminance. Greyscale was the first attempt and it hid
        # the exact case this exists for: a blue object recoloured burgundy
        # keeps almost the same brightness, so a luminance comparison called
        # a real edit "unchanged". A fixed size lets differing output
        # dimensions still compare.
        converted = image.convert("RGB").resize(COMPARISON_SIZE)
        return list(converted.tobytes())
