"""Edits the studio makes to a picture itself, without a model.

Rotating, straightening, flipping, cropping, resizing, adjusting light and
color, blurring, pixelating or painting a marked area, adding words and
changing the canvas all happen here. They need no workflow and no graphics card, and they should not
look as though one ran. Each result is made from the stored bytes of the
picture being edited. Turns, flips, crops and canvas changes move pixels
exactly; a resize resamples them once, by a recorded method, and so does a
straightening, which then keeps the box the turned picture still covers; an adjustment maps
each pixel's color by the arithmetic the studio previews it with; a blur, a
pixelation or a paint changes only the marked area, through the same selection
a region edit uses; and words are the very overlay the browser drew and
showed, laid over the picture.

The result is stored like any other picture and recorded in the session as one
more step, so the filmstrip, compare and a later edit all treat it the same
way. Its provenance says what was done to which picture and names no model,
because none ran.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import PurePath
from typing import Any

from PIL import Image, ImageFilter, ImageOps
from sqlalchemy import select
from sqlalchemy.orm import Session

from .artifact_library import ensure_library_entry
from .artifacts import ArtifactStore
from .domain import ArtifactKind, MessageRole, MessageStatus, PartType
from .models import Artifact, Chat, Message, MessagePart
from .schemas import StudioLocalEditOperation, StudioPerspective
from .studio_adjustments import ColorAdjustments, adjust_colors
from .studio_region_edit import (
    MAX_BLEND_PIXELS,
    MAX_BLEND_READ_BYTES,
    RegionEditError,
    decode_picture,
    encode_png,
    exif_orientation,
    selection_alpha,
)

#: What each edit is called in the session, which is what the filmstrip shows.
_DESCRIPTIONS: dict[StudioLocalEditOperation, str] = {
    "rotate_clockwise": "Rotate right",
    "rotate_counterclockwise": "Rotate left",
    "flip_horizontal": "Flip horizontally",
    "flip_vertical": "Flip vertically",
    "straighten": "Straighten",
    "perspective": "Correct the perspective",
    "crop": "Crop",
    "resize": "Resize",
    "adjust": "Adjust light and color",
    "blur": "Blur part of the picture",
    "pixelate": "Pixelate part of the picture",
    "paint": "Paint over part of the picture",
    "caption": "Add text",
    "canvas": "Change the canvas size",
}
#: How the edited picture is named in the library, after the source's own name.
_NAME_SUFFIXES: dict[StudioLocalEditOperation, str] = {
    "rotate_clockwise": "rotated right",
    "rotate_counterclockwise": "rotated left",
    "flip_horizontal": "flipped",
    "flip_vertical": "flipped vertically",
    "straighten": "straightened",
    "perspective": "perspective corrected",
    "crop": "cropped",
    "resize": "resized",
    "adjust": "adjusted",
    "blur": "blurred",
    "pixelate": "pixelated",
    "paint": "painted",
    "caption": "with text",
    "canvas": "canvas changed",
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
class PerspectiveCorners:
    """Where the corners of what should be a rectangle lie, in the picture's own
    pixels as it is seen upright."""

    top_left: tuple[int, int]
    top_right: tuple[int, int]
    bottom_right: tuple[int, int]
    bottom_left: tuple[int, int]

    def in_order(self) -> tuple[tuple[int, int], ...]:
        """The four corners from the top left, going clockwise."""
        return (self.top_left, self.top_right, self.bottom_right, self.bottom_left)

    def as_dict(self) -> dict[str, dict[str, int]]:
        return {
            name: {"x": point[0], "y": point[1]}
            for name, point in zip(
                ("top_left", "top_right", "bottom_right", "bottom_left"),
                self.in_order(),
                strict=True,
            )
        }


def perspective_corners(corners: StudioPerspective) -> PerspectiveCorners:
    """The four corners a request names, as points."""

    return PerspectiveCorners(
        (corners.top_left.x, corners.top_left.y),
        (corners.top_right.x, corners.top_right.y),
        (corners.bottom_right.x, corners.bottom_right.y),
        (corners.bottom_left.x, corners.bottom_left.y),
    )


@dataclass(frozen=True)
class PictureSize:
    """The size a picture is resized to, in pixels."""

    width: int
    height: int


@dataclass(frozen=True)
class CanvasChange:
    """A new canvas size, where the picture sits on it, and what fills the rest."""

    width: int
    height: int
    anchor: str = "center"
    fill: str = "transparent"


#: Each anchor as a column and a row: 0 at the start, 1 in the middle, 2 at the end.
_ANCHORS = {
    "top_left": (0, 0),
    "top": (1, 0),
    "top_right": (2, 0),
    "left": (0, 1),
    "center": (1, 1),
    "right": (2, 1),
    "bottom_left": (0, 2),
    "bottom": (1, 2),
    "bottom_right": (2, 2),
}
_FILLS = {
    "transparent": (0, 0, 0, 0),
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
}


@dataclass(frozen=True)
class CaptionOverlay:
    """Words the browser drew at the picture's size, as uploaded picture bytes."""

    overlay: bytes
    #: The uploaded overlay, named in the step so the words can be retraced.
    overlay_artifact_id: str


@dataclass(frozen=True)
class SelectionPaint:
    """The marked area as uploaded selection bytes, and the paint laid over it."""

    mask: bytes
    color: tuple[int, int, int]
    #: How much of the paint covers the picture, from 1 to 100 percent.
    opacity: int
    #: The uploaded selection, named in the step so the paint can be retraced.
    mask_artifact_id: str


@dataclass(frozen=True)
class SelectionBlur:
    """The marked area as uploaded selection bytes, and the blur radius in pixels."""

    mask: bytes
    radius: int
    #: The uploaded selection, named in the step so the blur can be retraced.
    mask_artifact_id: str


@dataclass(frozen=True)
class SelectionPixelate:
    """The marked area as uploaded selection bytes, and the block size in pixels."""

    mask: bytes
    block: int
    #: The uploaded selection, named in the step so the pixelation can be retraced.
    mask_artifact_id: str


def render_local_edit(
    payload: bytes,
    operation: StudioLocalEditOperation,
    crop: CropBox | None = None,
    size: PictureSize | None = None,
    adjustments: ColorAdjustments | None = None,
    blur: SelectionBlur | None = None,
    canvas: CanvasChange | None = None,
    paint: SelectionPaint | None = None,
    caption: CaptionOverlay | None = None,
    pixelate: SelectionPixelate | None = None,
    straighten: float | None = None,
    perspective: PerspectiveCorners | None = None,
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
    orientation = exif_orientation(decoded)
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
    elif operation == "straighten":
        result = _straightened(picture, straighten)
    elif operation == "perspective":
        result = _perspective_corrected(picture, perspective)
    elif operation == "resize":
        result = _resized(picture, size)
    elif operation == "adjust":
        if adjustments is None or adjustments.is_neutral():
            raise LocalEditError("studio-adjust-unchanged", "Move a slider to change the picture.")
        result = adjust_colors(picture, adjustments)
    elif operation == "blur":
        result = _blurred(picture, blur, orientation)
    elif operation == "pixelate":
        result = _pixelated(picture, pixelate, orientation)
    elif operation == "canvas":
        result = _on_canvas(picture, canvas)
    elif operation == "paint":
        result = _painted(picture, paint, orientation)
    elif operation == "caption":
        result = _captioned(picture, caption)
    else:
        result = picture.transpose(_TRANSPOSITIONS[operation])
    # Conversion carries the source's metadata along, and saving falls back to
    # its profile when none is given, so a dropped profile would come back.
    result.info = {}
    return encode_png(result, profile if isinstance(profile, bytes) else None)


def _straightened(picture: Image.Image, degrees: float | None) -> Image.Image:
    if degrees is None or degrees == 0:
        raise LocalEditError(
            "studio-straighten-unchanged", "Turn the picture by some angle to straighten it."
        )
    width, height = picture.size
    turn = math.radians(abs(degrees))
    # The turned picture still covers a box of its own shape centered where it
    # turns; this is the largest such box.
    scale = min(
        width / (width * math.cos(turn) + height * math.sin(turn)),
        height / (width * math.sin(turn) + height * math.cos(turn)),
    )
    # Two pixels in from each side, so the resampling never reaches the
    # empty corners the turn leaves outside the picture.
    scale -= 4 / min(width, height)
    kept_width = max(1, math.floor(width * scale))
    kept_height = max(1, math.floor(height * scale))
    # Turned with its colors premultiplied, as a blur is, so no hidden color
    # under a transparent pixel is carried into its neighbours.
    premultiplied = picture.mode == "RGBA"
    source = picture.convert("RGBa") if premultiplied else picture
    # Pillow turns counterclockwise for a positive angle.
    turned = source.rotate(-degrees, resample=Image.Resampling.BICUBIC)
    left = (width - kept_width) // 2
    top = (height - kept_height) // 2
    kept = turned.crop((left, top, left + kept_width, top + kept_height))
    return kept.convert("RGBA") if premultiplied else kept


def perspective_size(corners: PerspectiveCorners) -> tuple[int, int]:
    """The corrected picture's size: the longer of each pair of opposite sides.

    The longer side is the nearer one, so keeping it keeps the most detail.
    Each length is the square root of a whole number, which is never a half,
    so rounding it cannot differ from the size the browser shows.
    """

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners.in_order()
    width = max(_length(x1 - x0, y1 - y0), _length(x2 - x3, y2 - y3))
    height = max(_length(x3 - x0, y3 - y0), _length(x2 - x1, y2 - y1))
    return (math.floor(width + 0.5), math.floor(height + 0.5))


def _length(across: int, down: int) -> float:
    # The square root of an exact whole number, as the browser takes it.
    return math.sqrt(across * across + down * down)


def _square_to_quad(
    points: tuple[tuple[int, int], ...], size: tuple[int, int]
) -> tuple[float, ...]:
    """Pillow's coefficients taking each point of the corrected picture onto the source.

    The map from a unit square onto four corners, as Heckbert gives it, then
    scaled so that the corrected picture's own pixels are what it takes in.
    """

    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    width, height = size
    sum_x = x0 - x1 + x2 - x3
    sum_y = y0 - y1 + y2 - y3
    across = (x1 - x2) * (y3 - y2) - (x3 - x2) * (y1 - y2)
    g = (sum_x * (y3 - y2) - (x3 - x2) * sum_y) / across
    h = ((x1 - x2) * sum_y - sum_x * (y1 - y2)) / across
    return (
        (x1 - x0 + g * x1) / width,
        (x3 - x0 + h * x3) / height,
        x0,
        (y1 - y0 + g * y1) / width,
        (y3 - y0 + h * y3) / height,
        y0,
        g / width,
        h / height,
    )


def _perspective_corrected(picture: Image.Image, corners: PerspectiveCorners | None) -> Image.Image:
    if corners is None:
        raise LocalEditError(
            "studio-perspective-missing", "Place the four corners of what should be square."
        )
    width, height = picture.size
    points = corners.in_order()
    if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
        raise LocalEditError(
            "studio-perspective-outside-picture", "Each corner must lie on the picture."
        )
    if points == ((0, 0), (width, 0), (width, height), (0, height)):
        raise LocalEditError(
            "studio-perspective-unchanged",
            "Move the corners onto what should be square to correct the perspective.",
        )
    # Every turn from one side to the next goes the same way, clockwise as the
    # picture is seen, which holds only for four corners in order around a
    # shape whose sides do not cross.
    for index in range(4):
        (ax, ay), (bx, by), (cx, cy) = (points[(index + step) % 4] for step in range(3))
        if (bx - ax) * (cy - by) - (by - ay) * (cx - bx) <= 0:
            raise LocalEditError(
                "studio-perspective-crossed",
                "Keep the corners in their places: top left, top right, bottom right and"
                " bottom left, with no side crossing another.",
            )
    size = perspective_size(corners)
    if size[0] * size[1] > MAX_BLEND_PIXELS:
        raise LocalEditError("studio-edit-too-large", "That size is too large to make here.")
    # Pillow resamples a transparent picture with its colors premultiplied, so
    # a hidden color under a transparent pixel cannot bleed into the edge.
    return picture.transform(
        size,
        Image.Transform.PERSPECTIVE,
        _square_to_quad(points, size),
        Image.Resampling.BICUBIC,
    )


def _blurred(picture: Image.Image, blur: SelectionBlur | None, orientation: int) -> Image.Image:
    if blur is None:
        raise LocalEditError("studio-blur-missing", "Mark the part of the picture to blur.")
    try:
        alpha = selection_alpha(blur.mask, picture.size, orientation)
    except RegionEditError as exc:
        raise LocalEditError(exc.code, str(exc)) from exc
    # Blurred with its colors premultiplied, so the hidden color under a
    # transparent pixel cannot bleed into the pixels beside it.
    premultiplied = picture.mode == "RGBA"
    source = picture.convert("RGBa") if premultiplied else picture
    blurred = source.filter(ImageFilter.GaussianBlur(blur.radius))
    return Image.composite(blurred.convert("RGBA") if premultiplied else blurred, picture, alpha)


def _pixelated(
    picture: Image.Image, pixelate: SelectionPixelate | None, orientation: int
) -> Image.Image:
    if pixelate is None:
        raise LocalEditError("studio-pixelate-missing", "Mark the part of the picture to pixelate.")
    try:
        alpha = selection_alpha(pixelate.mask, picture.size, orientation)
    except RegionEditError as exc:
        raise LocalEditError(exc.code, str(exc)) from exc
    # Blocks start at the picture's top-left corner, and each takes the mean of
    # the pixels it covers, so a block cut short by the edge averages only what
    # it has. Premultiplied, as for a blur, so no hidden color comes through.
    premultiplied = picture.mode == "RGBA"
    source = picture.convert("RGBa") if premultiplied else picture
    means = source.reduce(pixelate.block)
    blocks = means.resize(
        (means.width * pixelate.block, means.height * pixelate.block), Image.Resampling.NEAREST
    ).crop((0, 0, picture.width, picture.height))
    return Image.composite(blocks.convert("RGBA") if premultiplied else blocks, picture, alpha)


def _painted(picture: Image.Image, paint: SelectionPaint | None, orientation: int) -> Image.Image:
    if paint is None:
        raise LocalEditError("studio-paint-missing", "Mark the part of the picture to paint.")
    try:
        alpha = selection_alpha(paint.mask, picture.size, orientation)
    except RegionEditError as exc:
        raise LocalEditError(exc.code, str(exc)) from exc
    # Half rounds up, as the canvas's preview of the paint does.
    cover = alpha.point([(value * paint.opacity + 50) // 100 for value in range(256)])
    overlay = Image.new("RGBA", picture.size, (*paint.color, 0))
    overlay.putalpha(cover)
    # Laid over, so paint on a transparent part shows as paint rather than
    # mixing with the color hidden under it.
    painted = Image.alpha_composite(picture.convert("RGBA"), overlay)
    return painted if picture.mode == "RGBA" else painted.convert("RGB")


def _captioned(picture: Image.Image, caption: CaptionOverlay | None) -> Image.Image:
    if caption is None:
        raise LocalEditError("studio-caption-missing", "Write the words to add.")
    try:
        drawn = decode_picture(caption.overlay, "drawn words")
    except RegionEditError as exc:
        raise LocalEditError(
            "studio-caption-unreadable", "The drawn words could not be read."
        ) from exc
    # Drawn over the picture as the person saw it, so it must be that size;
    # stretching it would move every letter from where the preview put it.
    if drawn.size != picture.size:
        raise LocalEditError(
            "studio-caption-size-mismatch",
            "The words were drawn for a picture of another size. Write them again.",
        )
    words = Image.alpha_composite(picture.convert("RGBA"), drawn.convert("RGBA"))
    return words if picture.mode == "RGBA" else words.convert("RGB")


def _placed(room: int, part: int) -> int:
    # The share of the room before the picture, rounded toward zero, so an odd
    # pixel always goes after it, whether the canvas grows or shrinks.
    return room * part // 2 if room >= 0 else -(-room * part // 2)


def _on_canvas(picture: Image.Image, canvas: CanvasChange | None) -> Image.Image:
    if canvas is None:
        raise LocalEditError("studio-canvas-missing", "Choose the new canvas size.")
    if canvas.width < 1 or canvas.height < 1:
        raise LocalEditError(
            "studio-canvas-empty", "A canvas needs at least one pixel across and down."
        )
    if canvas.width * canvas.height > MAX_BLEND_PIXELS:
        raise LocalEditError("studio-edit-too-large", "That size is too large to make here.")
    if (canvas.width, canvas.height) == picture.size:
        raise LocalEditError("studio-canvas-unchanged", "The canvas is already that size.")
    # The picture keeps its own transparency; the fill only covers new ground.
    mode = "RGBA" if canvas.fill == "transparent" or picture.mode == "RGBA" else "RGB"
    fill = _FILLS[canvas.fill]
    result = Image.new(mode, (canvas.width, canvas.height), fill if mode == "RGBA" else fill[:3])
    column, row = _ANCHORS[canvas.anchor]
    offset = (
        _placed(canvas.width - picture.width, column),
        _placed(canvas.height - picture.height, row),
    )
    result.paste(picture.convert(mode), offset)
    return result


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
    adjustments: ColorAdjustments | None = None,
    blur: SelectionBlur | None = None,
    canvas: CanvasChange | None = None,
    paint: SelectionPaint | None = None,
    caption: CaptionOverlay | None = None,
    pixelate: SelectionPixelate | None = None,
    straighten: float | None = None,
    perspective: PerspectiveCorners | None = None,
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
    if straighten is not None:
        record["straighten"] = {"degrees": straighten, "resampler": "bicubic"}
    if perspective is not None:
        record["perspective"] = {"corners": perspective.as_dict(), "resampler": "bicubic"}
    if size is not None:
        record["size"] = asdict(size)
        record["resampler"] = RESIZE_RESAMPLER
    if adjustments is not None:
        record["adjustments"] = adjustments.as_dict()
    if blur is not None:
        record["blur"] = {
            "radius": blur.radius,
            "filter": "gaussian",
            "mask_artifact_id": blur.mask_artifact_id,
        }
    if pixelate is not None:
        record["pixelate"] = {
            "block": pixelate.block,
            "mask_artifact_id": pixelate.mask_artifact_id,
        }
    if canvas is not None:
        record["canvas"] = asdict(canvas)
    if paint is not None:
        record["paint"] = {
            "color": "#{:02x}{:02x}{:02x}".format(*paint.color),
            "opacity": paint.opacity,
            "mask_artifact_id": paint.mask_artifact_id,
        }
    if caption is not None:
        record["caption"] = {"overlay_artifact_id": caption.overlay_artifact_id}
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


def paint_color(value: str) -> tuple[int, int, int]:
    """A "#rrggbb" color as its red, green and blue levels."""

    return (int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16))


def marked_area(store: ArtifactStore, artifact: Artifact) -> bytes:
    """The uploaded selection's verified bytes, read within the bound a picture is."""

    try:
        return store.verified_bytes(artifact, maximum_bytes=MAX_BLEND_READ_BYTES)
    except (ValueError, OSError) as exc:
        raise LocalEditError(
            "studio-marked-area-unreadable", "The marked area could not be read."
        ) from exc


def edited_picture(
    store: ArtifactStore,
    source: Artifact,
    operation: StudioLocalEditOperation,
    crop: CropBox | None = None,
    size: PictureSize | None = None,
    adjustments: ColorAdjustments | None = None,
    blur: SelectionBlur | None = None,
    canvas: CanvasChange | None = None,
    paint: SelectionPaint | None = None,
    caption: CaptionOverlay | None = None,
    pixelate: SelectionPixelate | None = None,
    straighten: float | None = None,
    perspective: PerspectiveCorners | None = None,
) -> bytes:
    """Read the source's verified bytes and make the edited picture from them.

    Slow for a large picture, and touching no session, so a caller can run it
    away from anything that must stay responsive.
    """

    try:
        payload = store.verified_bytes(source, maximum_bytes=MAX_BLEND_READ_BYTES)
    except (ValueError, OSError) as exc:
        raise LocalEditError("studio-edit-unreadable", "This picture could not be read.") from exc
    return render_local_edit(
        payload,
        operation,
        crop,
        size,
        adjustments,
        blur,
        canvas,
        paint,
        caption,
        pixelate,
        straighten,
        perspective,
    )
