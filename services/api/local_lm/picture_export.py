"""A stored picture written out in the format someone asks for.

The stored bytes stay as they are; an export is made from them each time. It
turns the picture upright, the way it is shown, so a file that relied on a
camera's orientation tag does not open sideways in a program that ignores the
tag. It keeps the color profile where the pixels are still in that profile's
space. JPEG has no transparency, so a transparent picture is laid on white
first, rather than coming out on black, which is what dropping it would leave.
Sixteen-bit grey stays sixteen-bit in a PNG; JPEG and WebP hold eight bits, so
there it is scaled down rather than clipped to white.
"""

from __future__ import annotations

import io
from pathlib import PurePath
from typing import Any, Final, Literal

from PIL import Image, ImageOps

from .artifacts import ArtifactStore
from .models import Artifact
from .studio_region_edit import MAX_BLEND_READ_BYTES, RegionEditError, decode_picture

ExportFormat = Literal["png", "jpeg", "webp"]
#: How each format is named to Pillow, its media type, and its file extension.
EXPORT_FORMATS: Final[dict[str, tuple[str, str, str]]] = {
    "png": ("PNG", "image/png", "png"),
    "jpeg": ("JPEG", "image/jpeg", "jpg"),
    "webp": ("WEBP", "image/webp", "webp"),
}
#: The quality a lossy format is written at when none is asked for.
DEFAULT_EXPORT_QUALITY = 90
#: Modes Pillow opens a sixteen-bit grey picture in.
_SIXTEEN_BIT_GREY: Final = frozenset({"I", "I;16", "I;16B", "I;16L", "I;16N"})


class PictureExportError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def export_picture(
    payload: bytes, file_format: ExportFormat, quality: int = DEFAULT_EXPORT_QUALITY
) -> bytes:
    """The picture in `file_format`, upright, with its profile where it still applies."""

    try:
        return _export(payload, file_format, quality)
    except PictureExportError:
        raise
    except RegionEditError as exc:
        if exc.code == "region-image-too-large":
            raise PictureExportError(
                "picture-export-too-large", "This picture is too large to export in another format."
            ) from exc
        raise PictureExportError(
            "picture-export-unreadable", "This picture could not be read."
        ) from exc
    except Exception as exc:
        # A malformed file can make Pillow raise almost anything while it
        # decodes, turns or converts the picture: a short header is a
        # ValueError, a broken orientation block a SyntaxError.
        raise PictureExportError(
            "picture-export-unreadable", "This picture could not be read."
        ) from exc


def _export(payload: bytes, file_format: ExportFormat, quality: int) -> bytes:
    decoded = decode_picture(payload, "picture")
    upright = ImageOps.exif_transpose(decoded) or decoded
    if upright.mode in _SIXTEEN_BIT_GREY:
        if file_format == "png":
            grey = upright.convert("I;16")
            # Conversion carries the source's text and tags along, as below.
            grey.info = {}
            return _encoded(grey, {"format": "PNG"})
        # Converting straight to eight bits clips every value above 255.
        upright = upright.convert("I").point(lambda value: value / 257).convert("L")
    # A grey or print profile would misdescribe the RGB the picture becomes.
    profile = upright.info.get("icc_profile") if upright.mode in {"RGB", "RGBA"} else None
    has_alpha = upright.mode in {"RGBA", "LA", "PA"} or "transparency" in upright.info
    picture = upright.convert("RGBA" if has_alpha else "RGB")
    if file_format == "jpeg" and has_alpha:
        flat = Image.new("RGB", picture.size, (255, 255, 255))
        flat.paste(picture, mask=picture.getchannel("A"))
        picture = flat
    # Conversion carries the source's metadata along, including the orientation
    # tag that no longer applies to upright pixels.
    picture.info = {}
    options: dict[str, Any] = {"format": EXPORT_FORMATS[file_format][0]}
    if isinstance(profile, bytes):
        options["icc_profile"] = profile
    if file_format != "png":
        options["quality"] = quality
    return _encoded(picture, options)


def _encoded(picture: Image.Image, options: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, **options)
    return buffer.getvalue()


def export_stored_picture(
    store: ArtifactStore, artifact: Artifact, file_format: ExportFormat, quality: int
) -> bytes:
    """Read the picture's verified bytes and export them.

    Slow for a large picture and touching no session, so a caller can run it away
    from anything that must stay responsive.
    """

    try:
        payload = store.verified_bytes(artifact, maximum_bytes=MAX_BLEND_READ_BYTES)
    except (ValueError, OSError) as exc:
        raise PictureExportError(
            "picture-export-unreadable", "This picture could not be read."
        ) from exc
    return export_picture(payload, file_format, quality)


def export_file_name(original_name: str | None, file_format: ExportFormat) -> str:
    stem = PurePath(original_name).stem if original_name else "picture"
    return f"{stem[:200]}.{EXPORT_FORMATS[file_format][2]}"
