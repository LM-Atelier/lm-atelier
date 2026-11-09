"""The shape a source picture is shown in, read from its verified bytes.

A workflow is asked for the shape a picture is shown in, which is not always the
shape it is stored in: a photo taken on its side carries an orientation that
turns it a quarter, and LoadImage applies that turn before anything else sees
the picture. Only the header is read, never the pixels. The answer is a shape to
offer, and any size chosen from it is checked again when the turn is accepted.
"""

from __future__ import annotations

from io import BytesIO
from typing import Final

from PIL import Image, TiffImagePlugin
from pydantic import Field

from .artifacts import ArtifactStore
from .models import Artifact
from .schemas import ApiModel

MAX_PICTURE_BYTES: Final = 64 * 1024 * 1024
_ORIENTATION_TAG: Final = 0x0112
#: The EXIF orientations that turn a picture a quarter, swapping its width and height.
_QUARTER_TURNS: Final = frozenset({5, 6, 7, 8})
#: The TIFF tag that says how many samples each pixel stores.
_SAMPLES_PER_PIXEL: Final = 277


def media_runtime_reads(image: Image.Image) -> bool:
    """Whether the managed media runtime reads this picture's layout as this app does.

    That runtime brings its own Pillow, 12.3, which refuses two TIFF layouts
    this app's Pillow reads: grey whose alpha is premultiplied (La), and grey
    with a second sample it does not name, which this app's Pillow drops. A
    workflow is handed its source as it is stored, so such a picture would pass
    a check here and fail in the runtime.
    """
    if image.mode == "La":
        return False
    if isinstance(image, TiffImagePlugin.TiffImageFile):
        samples: object = image.tag_v2.get(_SAMPLES_PER_PIXEL, 1)
        return samples == len(image.getbands())
    return True


class MatchSourceRequest(ApiModel):
    source_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def shown_size(store: ArtifactStore, artifact: Artifact) -> tuple[int, int]:
    """The picture's width and height as it is shown, not as it is stored."""
    if not artifact.media_type.casefold().startswith("image/"):
        raise ValueError("picture_shape_unsupported")
    try:
        content = store.verified_bytes(artifact, maximum_bytes=MAX_PICTURE_BYTES)
    except (OSError, ValueError):
        raise ValueError("picture_shape_unavailable") from None
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            orientation = image.getexif().get(_ORIENTATION_TAG)
            readable = media_runtime_reads(image)
    except Exception:
        # A malformed file can make Pillow raise almost anything here: a short
        # chunk is a ValueError, broken orientation data a SyntaxError.
        raise ValueError("picture_shape_unavailable") from None
    if not readable:
        raise ValueError("picture_shape_unavailable")
    return (height, width) if orientation in _QUARTER_TURNS else (width, height)
