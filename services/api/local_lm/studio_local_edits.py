"""Rotate, flip, crop and resize a Studio picture without a model.

These edits need no workflow and no graphics card, and they should not look as
though one ran. The result is made here, from the stored bytes of the picture
being edited: turns, flips and crops move pixels exactly, and a resize
resamples them once, by a recorded method. It is stored like any other picture
and recorded in the session as one more step, so the filmstrip, compare and a
later edit all treat it the same way. Its provenance says what was done to
which picture and names no model, because none ran.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Any

from PIL import Image, ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifact_library import ensure_library_entry
from .artifacts import ArtifactStore
from .domain import ArtifactKind, MessageRole, MessageStatus, PartType
from .models import Artifact, Chat, Message, MessagePart
from .schemas import StudioLocalEditOperation
from .studio_region_edit import (
    MAX_BLEND_PIXELS,
    MAX_BLEND_READ_BYTES,
    RegionEditError,
    decode_picture,
    encode_png,
)

#: What each edit is called in the session, which is what the filmstrip shows.
_DESCRIPTIONS: dict[StudioLocalEditOperation, str] = {
    "rotate_clockwise": "Rotate right",
    "rotate_counterclockwise": "Rotate left",
    "flip_horizontal": "Flip horizontally",
    "flip_vertical": "Flip vertically",
    "crop": "Crop",
    "resize": "Resize",
}
#: How the edited picture is named in the library, after the source's own name.
_NAME_SUFFIXES: dict[StudioLocalEditOperation, str] = {
    "rotate_clockwise": "rotated right",
    "rotate_counterclockwise": "rotated left",
    "flip_horizontal": "flipped",
    "flip_vertical": "flipped vertically",
    "crop": "cropped",
    "resize": "resized",
}
# Pillow names a rotation by its counterclockwise angle.
_TRANSPOSITIONS = {
    "rotate_clockwise": Image.Transpose.ROTATE_270,
    "rotate_counterclockwise": Image.Transpose.ROTATE_90,
    "flip_horizontal": Image.Transpose.FLIP_LEFT_RIGHT,
    "flip_vertical": Image.Transpose.FLIP_TOP_BOTTOM,
}
#: How a resize resamples, recorded with the result. Lanczos keeps edges crisp
#: when a picture shrinks and interpolates smoothly when it grows.
RESIZE_RESAMPLER = "lanczos"


class LocalEditError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class CropBox:
    """A rectangle in the picture's own pixels, as the picture is seen upright."""

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class PictureSize:
    """The size a picture is resized to, in pixels."""

    width: int
    height: int


def render_local_edit(
    payload: bytes,
    operation: StudioLocalEditOperation,
    crop: CropBox | None = None,
    size: PictureSize | None = None,
) -> bytes:
    """The edited picture as PNG, made from the stored bytes without a model."""

    try:
        decoded = decode_picture(payload, "picture")
    except RegionEditError as exc:
        # The decoder words its refusals for a selection, which this is not.
        if exc.code == "region-image-too-large":
            raise LocalEditError(
                "studio-edit-too-large", "This picture is too large to edit here."
            ) from exc
        raise LocalEditError("studio-edit-unreadable", "This picture could not be read.") from exc
    # Upright as the person saw it, since that is the picture they turned or cut.
    upright = ImageOps.exif_transpose(decoded) or decoded
    # A profile describes the colours it came with. Kept only where the pixels
    # stay in that space, since a grey or print profile would misdescribe RGB.
    profile = upright.info.get("icc_profile") if upright.mode in {"RGB", "RGBA"} else None
    has_alpha = upright.mode in {"RGBA", "LA", "PA"} or "transparency" in upright.info
    picture = upright.convert("RGBA" if has_alpha else "RGB")
    if operation == "crop":
        if crop is None:
            raise LocalEditError("studio-crop-missing", "Choose the part of the picture to keep.")
        inside = (
            crop.width >= 1
            and crop.height >= 1
            and crop.left >= 0
            and crop.top >= 0
            and crop.left + crop.width <= picture.width
            and crop.top + crop.height <= picture.height
        )
        if not inside:
            raise LocalEditError(
                "studio-crop-outside-picture", "The part to keep must lie inside the picture."
            )
        result = picture.crop((crop.left, crop.top, crop.left + crop.width, crop.top + crop.height))
    elif operation == "resize":
        result = _resized(picture, size)
    else:
        result = picture.transpose(_TRANSPOSITIONS[operation])
    # Conversion carries the source's metadata along, and saving falls back to
    # its profile when none is given, so a dropped profile would come back.
    result.info = {}
    return encode_png(result, profile if isinstance(profile, bytes) else None)


def _resized(picture: Image.Image, size: PictureSize | None) -> Image.Image:
    if size is None:
        raise LocalEditError("studio-resize-missing", "Choose the new size.")
    if size.width < 1 or size.height < 1:
        raise LocalEditError(
            "studio-resize-empty", "A picture needs at least one pixel across and down."
        )
    if size.width * size.height > MAX_BLEND_PIXELS:
        raise LocalEditError("studio-edit-too-large", "That size is too large to make here.")
    if (size.width, size.height) == picture.size:
        raise LocalEditError("studio-resize-unchanged", "The picture is already that size.")
    # Pillow resamples transparent pictures with their colours premultiplied,
    # so a hidden colour under a transparent pixel cannot bleed into the edge.
    return picture.resize((size.width, size.height), Image.Resampling.LANCZOS)


def picture_in_session(session: Session, studio: Chat, artifact_id: str) -> bool:
    """Whether a picture is one this session holds: its source or a finished result.

    A preview is temporary and a removed step is gone, so neither can be edited.
    """

    origin = studio.origin_json
    if isinstance(origin, dict) and origin.get("source_artifact_id") == artifact_id:
        return True
    parts = session.scalars(
        select(MessagePart)
        .join(Message, Message.id == MessagePart.message_id)
        .where(
            Message.chat_id == studio.id,
            Message.role == MessageRole.ASSISTANT.value,
            Message.status == MessageStatus.COMPLETE.value,
            Message.content_removed_at.is_(None),
            MessagePart.type == PartType.IMAGE.value,
            MessagePart.artifact_id == artifact_id,
        )
    )
    return any(part.metadata_json.get("preview") is not True for part in parts)


def edited_picture_name(source: Artifact, operation: StudioLocalEditOperation) -> str:
    stem = PurePath(source.original_name).stem if source.original_name else "Studio picture"
    return f"{stem} ({_NAME_SUFFIXES[operation]}).png"[:240]


def record_local_edit(
    session: Session,
    studio: Chat,
    source: Artifact,
    operation: StudioLocalEditOperation,
    edited: bytes,
    store: ArtifactStore,
    crop: CropBox | None = None,
    size: PictureSize | None = None,
) -> Artifact:
    """Store the edited picture and append it to the session as one more step.

    The step has the shape of an apply: a request naming the picture it
    changed, first, and a finished answer holding the result. That shape is
    what the filmstrip and compare already read, so nothing there needs to
    know that no model ran.
    """

    record: dict[str, Any] = {"operation": operation, "source_artifact_id": source.id}
    if crop is not None:
        record["crop"] = asdict(crop)
    if size is not None:
        record["size"] = asdict(size)
        record["resampler"] = RESIZE_RESAMPLER
    result = store.ingest_bytes(
        session,
        edited,
        kind=ArtifactKind.IMAGE,
        media_type="image/png",
        original_name=edited_picture_name(source, operation),
        metadata={"studio_local_edit": record},
    )
    ensure_library_entry(session, result)
    request = Message(
        chat_id=studio.id,
        parent_id=studio.active_head_message_id,
        role=MessageRole.USER.value,
        status=MessageStatus.COMPLETE.value,
        parts=[
            MessagePart(position=0, type=PartType.TEXT.value, text=_DESCRIPTIONS[operation]),
            MessagePart(
                position=1,
                type=PartType.IMAGE.value,
                artifact_id=source.id,
                metadata_json={"input_reference": True, "input_reference_source": "explicit"},
            ),
        ],
    )
    session.add(request)
    session.flush()
    answer = Message(
        chat_id=studio.id,
        parent_id=request.id,
        role=MessageRole.ASSISTANT.value,
        status=MessageStatus.COMPLETE.value,
        parts=[
            MessagePart(position=0, type=PartType.IMAGE.value, artifact_id=result.id),
            MessagePart(
                position=1,
                type=PartType.GENERATION_METADATA.value,
                metadata_json={"provenance": {"local_edit": record}},
            ),
        ],
    )
    session.add(answer)
    session.flush()
    studio.active_head_message_id = answer.id
    return result


def edited_picture(
    store: ArtifactStore,
    source: Artifact,
    operation: StudioLocalEditOperation,
    crop: CropBox | None = None,
    size: PictureSize | None = None,
) -> bytes:
    """Read the source's verified bytes and make the edited picture from them.

    Slow for a large picture, and touching no session, so a caller can run it
    away from anything that must stay responsive.
    """

    try:
        payload = store.verified_bytes(source, maximum_bytes=MAX_BLEND_READ_BYTES)
    except (ValueError, OSError) as exc:
        raise LocalEditError("studio-edit-unreadable", "This picture could not be read.") from exc
    return render_local_edit(payload, operation, crop, size)
