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

from PIL import Image
from pydantic import Field

from .artifacts import ArtifactStore
from .models import Artifact
from .schemas import ApiModel

MAX_PICTURE_BYTES: Final = 64 * 1024 * 1024
_ORIENTATION_TAG: Final = 0x0112
#: The EXIF orientations that turn a picture a quarter, swapping its width and height.
_QUARTER_TURNS: Final = frozenset({5, 6, 7, 8})


class MatchSourceRequest(ApiModel):
    source_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")


def shown_size(store: ArtifactStore, artifact: Artifact) -> tuple[int, int]:
    """The picture's width and height as it is shown, not as it is stored."""
    if not artifact.media_type.casefold().startswith("image/"):
        raise ValueError("picture_shape_unsupported")
    try:
        content = store.verified_bytes(artifact, maximum_bytes=MAX_PICTURE_BYTES)
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            orientation = image.getexif().get(_ORIENTATION_TAG)
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ValueError("picture_shape_unavailable") from None
    return (height, width) if orientation in _QUARTER_TURNS else (width, height)
