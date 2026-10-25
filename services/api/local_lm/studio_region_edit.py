"""Put an edited picture back into its source through the selection that asked for it.

An instruction-edit model re-renders every pixel of the picture it is given,
even when the words only asked about one part of it, and it hands the result
back at its own working size. For a tool whose promise is that only the
selection changes, that is not good enough. Cropping the selection out before
such an edit is worse: the model returns the crop at a different shape, and
anything written in it stretches.

So the edit runs on the whole picture, and afterwards the result is resized
to the source and composited over it through the selection's alpha. Outside
the selection the stored picture is the source's own pixels, inside it is the
edit, and along a feathered edge it is a mix of the two.

A workflow with its own mask input is different when every picture it saves
is its source edited in place: it returns exactly the shape it was given. So
a small selection is edited on a crop around it, in the shape of the
workflow's working size, and placed back the same way.

Pure work on bytes: nothing here opens a session or a file.
"""

from __future__ import annotations

import hashlib
import io
import math
import warnings
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageChops, ImageFilter, ImageOps, ImageStat, UnidentifiedImageError

from .studio_masks import (
    MaskContractError,
    MaskGeometry,
    MaskSelection,
    mask_geometry,
    mask_provenance,
)
from .workflow_source_geometry import SelectionRoute

#: How far the edited picture's shape may drift from the source's and still be
#: placed back. Edit models round their working size to a multiple of their
#: patch size, which moves the aspect ratio by about one percent. Past this the
#: result was cropped or reframed, and stretching it back would move what the
#: selection covers.
MAX_SHAPE_DRIFT = 0.02
#: The largest picture a blend decodes. Three pictures this size are held at
#: once, so the bound is what keeps one edit from exhausting memory.
MAX_BLEND_PIXELS = 40_000_000
#: The largest stored file read for a blend: a lossless picture at the pixel
#: bound above, with room to spare.
MAX_BLEND_READ_BYTES = 256 * 1024 * 1024
RESULT_RESAMPLER = "lanczos"
MASK_RESAMPLER = "bilinear"
BLEND_MEDIA_TYPE = "image/png"
#: The largest selection read here before a workflow is handed it: the
#: Studio's own largest mask. A larger one is handed as it was drawn.
MAX_WORKFLOW_MASK_PIXELS = 8192 * 8192
#: A selection is edited on a crop only when the crop, with its surroundings,
#: is at most this share of the picture. Past it the whole picture is as good.
MAX_CROP_SHARE = 0.5
#: Room left around a selection on each side, as a share of its own size, so
#: the workflow sees what the selection sits in.
CROP_CONTEXT = 0.5
MIN_CROP_CONTEXT_PX = 32
#: About the size image workflows are made for; a smaller crop is enlarged to it.
CROP_WORKING_PIXELS = 1024 * 1024
#: A crop's sides are whole blocks of this many pixels, which every latent
#: model's encoder divides evenly, so the edit decodes to exactly its size.
CROP_ALIGN = 64
#: Below this share of the crop the surroundings say too little about colour
#: to match the edit to them.
MIN_COLOUR_RING_SHARE = 0.01
#: The most, in levels of each channel, an edit is shifted back toward what
#: surrounds it. A model's encode and decode move colours by a few levels; a
#: larger change all around the selection may be the workflow's own.
MAX_COLOUR_SHIFT = 8
#: How far a crop may be stretched more one way than the other on its way to
#: the working size, past which the selection is edited whole.
MAX_CROP_STRETCH = 0.01
#: How far from the selection colour is measured. Resizing spreads the edit a
#: few pixels past the selection's edge, and measured there it would read as
#: the surroundings' own colour.
COLOUR_RING_GAP_PX = 4
_EXIF_ORIENTATION = 0x0112


class RegionEditError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class RegionEdit:
    """One blend selection and the exact stored bytes it is placed on."""

    selection: MaskSelection
    source_artifact_id: str
    source: bytes
    mask: bytes


@dataclass(frozen=True)
class PreparedSelection:
    """A selection checked against its source before anything is generated."""

    geometry: MaskGeometry
    #: The fraction of the source the edit may change, in source pixels.
    coverage: float


@dataclass(frozen=True)
class RegionBlend:
    """The stored picture, and what the run records about how it was made."""

    content: bytes
    media_type: str
    record: dict[str, Any]


def prepare_selection(edit: RegionEdit) -> PreparedSelection:
    """Refuse a selection that cannot be placed on its source, before the edit runs.

    Checked ahead of generation so a selection drawn on another picture, or one
    that covers nothing, costs no model time.
    """
    source = load_source(edit.source)
    alpha = _selection_alpha(
        decode_picture(edit.mask, "selection"),
        source.image.size,
        invert=edit.selection.invert,
        orientation=source.orientation,
    )
    return alpha.prepared


def blend_through_selection(edit: RegionEdit, result: bytes) -> RegionBlend:
    """Composite the edit over the source through the selection, as a PNG.

    PNG because it is lossless: re-encoding as JPEG would change the pixels
    outside the selection, which is the one thing this promises not to do.
    """
    source = load_source(edit.source)
    alpha = _selection_alpha(
        decode_picture(edit.mask, "selection"),
        source.image.size,
        invert=edit.selection.invert,
        orientation=source.orientation,
    )
    edited, fitted = fit_result(
        source,
        result,
        "The edited picture came back in a different shape, so it cannot be placed "
        "back into the selection.",
    )
    blended = Image.composite(fitted, source.image, alpha.image)
    return RegionBlend(
        content=encode_png(blended, source.icc_profile),
        media_type=BLEND_MEDIA_TYPE,
        record={
            "mode": "blend",
            "source_artifact_id": edit.source_artifact_id,
            "selection": mask_provenance(
                edit.selection, alpha.prepared.geometry, coverage=alpha.prepared.coverage
            ),
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "result_width": edited.width,
            "result_height": edited.height,
            "result_resampler": RESULT_RESAMPLER,
            "width": source.image.width,
            "height": source.image.height,
        },
    )


@dataclass(frozen=True)
class SourcePicture:
    """The source as the person saw it, ready to composite over."""

    #: Upright, in RGB, or RGBA when the source has transparency.
    image: Image.Image
    orientation: int
    icc_profile: bytes | None


def load_source(payload: bytes) -> SourcePicture:
    decoded = decode_picture(payload, "source")
    icc_profile = decoded.info.get("icc_profile")
    return SourcePicture(
        image=_composable(_oriented(decoded)),
        orientation=_orientation(decoded),
        icc_profile=icc_profile if isinstance(icc_profile, bytes) else None,
    )


def fit_result(
    source: SourcePicture, result: bytes, reshaped_message: str
) -> tuple[Image.Image, Image.Image]:
    """The decoded result, and a copy resized to the source in the source's mode.

    A result whose shape drifted past MAX_SHAPE_DRIFT is refused rather than
    stretched. The result has no transparency of its own, so for a source with
    transparency the source's alpha decides it: a cut-out stays cut out.
    """
    edited = decode_picture(result, "edited picture")
    source_ratio = source.image.width / source.image.height
    result_ratio = edited.width / edited.height
    if abs(result_ratio - source_ratio) > MAX_SHAPE_DRIFT * source_ratio:
        raise RegionEditError("region-result-reshaped", reshaped_message)
    fitted = edited.convert("RGB").resize(source.image.size, Image.Resampling.LANCZOS)
    if source.image.mode == "RGBA":
        fitted.putalpha(source.image.getchannel("A"))
    return edited, fitted


def encode_png(image: Image.Image, icc_profile: bytes | None) -> bytes:
    buffer = io.BytesIO()
    options: dict[str, Any] = {"format": "PNG"}
    if icc_profile is not None:
        options["icc_profile"] = icc_profile
    image.save(buffer, **options)
    return buffer.getvalue()


@dataclass(frozen=True)
class _Alpha:
    image: Image.Image
    prepared: PreparedSelection


def decode_picture(payload: bytes, label: str) -> Image.Image:
    try:
        with warnings.catch_warnings():
            # Between Pillow's pixel limit and twice it, a crafted header only
            # warns. The same refusal as an outright bomb, so a small file
            # cannot claim a huge decode.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(payload))
            if image.width < 1 or image.height < 1:
                raise RegionEditError("region-image-unreadable", f"The {label} has no pixels.")
            if image.width * image.height > MAX_BLEND_PIXELS:
                raise RegionEditError(
                    "region-image-too-large",
                    f"The {label} is too large to edit through a selection.",
                )
            image.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ) as exc:
        raise RegionEditError("region-image-unreadable", f"The {label} could not be read.") from exc
    return image


def selection_alpha(mask: bytes, size: tuple[int, int], orientation: int) -> Image.Image:
    """A drawn selection as an alpha at the picture's upright size.

    Placed the way a region edit places it, and refused the same ways: a
    selection drawn on another picture, or one that covers nothing.
    """

    return _selection_alpha(
        decode_picture(mask, "selection"), size, invert=False, orientation=orientation
    ).image


def exif_orientation(image: Image.Image) -> int:
    """The orientation a camera recorded, 1 when it recorded none that is valid."""

    return _orientation(image)


def _orientation(image: Image.Image) -> int:
    value = image.getexif().get(_EXIF_ORIENTATION, 1)
    return value if isinstance(value, int) and 1 <= value <= 8 else 1


def _oriented(image: Image.Image) -> Image.Image:
    """The source as the person saw it, which is what the selection was drawn on."""
    return ImageOps.exif_transpose(image) or image


def _composable(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


@dataclass(frozen=True)
class SelectionCrop:
    """The part of a picture a workflow is given for a small selection, and where it goes."""

    edit: RegionEdit
    #: Which of the workflow's pictures are edits of the crop, to be placed back.
    route: SelectionRoute
    #: Left, top, right and bottom, in the upright source's pixels.
    box: tuple[int, int, int, int]
    #: What the workflow is given: the crop enlarged to about its working size.
    picture: bytes
    #: The selection cropped the same way, in its brightness and its alpha.
    mask: bytes
    size: tuple[int, int]


def crop_selection(edit: RegionEdit, route: SelectionRoute) -> SelectionCrop | None:
    """The crop a small selection is edited on, or None when the whole picture is as good.

    A workflow draws the picture it is given at its own working size, so a
    selection a few dozen pixels across is drawn with a few dozen pixels and
    comes back soft. Given only the selection and its surroundings, enlarged
    to that size, it draws the selection with all of them. A picture too large
    to place an edit back into, or one that does not read whole here, is
    edited whole instead, where the whole picture's own checks decide.
    """
    upright = _upright_shape(edit.source)
    if upright is None or upright[0][0] * upright[0][1] > MAX_BLEND_PIXELS:
        return None
    try:
        source = load_source(edit.source)
        alpha = _selection_alpha(
            decode_picture(edit.mask, "selection"),
            source.image.size,
            invert=edit.selection.invert,
            orientation=source.orientation,
        )
    except RegionEditError:
        return None
    bounds = alpha.image.getbbox()
    if bounds is None:
        return None
    context = _context_box(bounds, source.image.size)
    size = _working_size(context[2] - context[0], context[3] - context[1])
    box = _shaped_box(context, size, source.image.size)
    if box is None:
        return None
    width, height = box[2] - box[0], box[3] - box[1]
    if width * height > MAX_CROP_SHARE * source.image.width * source.image.height:
        return None
    picture = source.image.crop(box).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    selected = alpha.image.crop(box).resize(size, Image.Resampling.BILINEAR)
    # In brightness and in alpha, as the whole picture's selection is handed.
    mask = Image.merge("RGBA", (selected, selected, selected, selected))
    return SelectionCrop(
        edit=edit,
        route=route,
        box=box,
        picture=encode_png(picture, source.icc_profile),
        mask=encode_png(mask, None),
        size=size,
    )


def stitch_selection(crop: SelectionCrop, result: bytes) -> RegionBlend:
    """Place a crop's edit back into its picture through the selection, as a PNG.

    The edit is brought back to the crop's size, and shifted back by however
    little the model moved the colours around the selection, so it does not
    stand out from what surrounds it. Outside the selection the stored picture
    is the source's own pixels.
    """
    source = load_source(crop.edit.source)
    alpha = _selection_alpha(
        decode_picture(crop.edit.mask, "selection"),
        source.image.size,
        invert=crop.edit.selection.invert,
        orientation=source.orientation,
    )
    edited = decode_picture(result, "edited picture")
    sent_ratio = crop.size[0] / crop.size[1]
    if abs(edited.width / edited.height - sent_ratio) > MAX_SHAPE_DRIFT * sent_ratio:
        raise RegionEditError(
            "region-result-reshaped",
            "The edited part came back in a different shape, so it cannot be placed back "
            "into the selection.",
        )
    left, top, right, bottom = crop.box
    original = source.image.crop(crop.box)
    selection = alpha.image.crop(crop.box)
    fitted = edited.convert("RGB").resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    # What the workflow was given, brought back the same way, so resampling
    # alone reads as no change.
    given = (
        decode_picture(crop.picture, "crop")
        .convert("RGB")
        .resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    )
    matched, colour_shift = _matched_colours(fitted, given, selection)
    if original.mode == "RGBA":
        matched.putalpha(original.getchannel("A"))
    stitched = source.image.copy()
    stitched.paste(Image.composite(matched, original, selection), (left, top))
    return RegionBlend(
        content=encode_png(stitched, source.icc_profile),
        media_type=BLEND_MEDIA_TYPE,
        record={
            "mode": "crop",
            "source_artifact_id": crop.edit.source_artifact_id,
            "selection": mask_provenance(
                crop.edit.selection, alpha.prepared.geometry, coverage=alpha.prepared.coverage
            ),
            "box": list(crop.box),
            "working_width": crop.size[0],
            "working_height": crop.size[1],
            "colour_shift": colour_shift,
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "result_width": edited.width,
            "result_height": edited.height,
            "result_resampler": RESULT_RESAMPLER,
            "width": source.image.width,
            "height": source.image.height,
        },
    )


def _context_box(
    selected: tuple[int, int, int, int], size: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, top, right, bottom = selected
    across = max(MIN_CROP_CONTEXT_PX, round((right - left) * CROP_CONTEXT))
    down = max(MIN_CROP_CONTEXT_PX, round((bottom - top) * CROP_CONTEXT))
    return (
        max(0, left - across),
        max(0, top - down),
        min(size[0], right + across),
        min(size[1], bottom + down),
    )


def _shaped_box(
    box: tuple[int, int, int, int], size: tuple[int, int], picture: tuple[int, int]
) -> tuple[int, int, int, int] | None:
    """The box widened or heightened to the working size's shape, inside the picture.

    Each side of the working size is rounded to whole blocks on its own, so a
    long, thin crop would otherwise be stretched more one way than the other,
    and anything the workflow draws in it would come back squashed. None where
    the shape does not fit in the picture closely enough.
    """
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    wanted_width = min(picture[0], max(width, round(height * size[0] / size[1])))
    wanted_height = min(picture[1], max(height, round(width * size[1] / size[0])))
    stretch = (size[0] / wanted_width) / (size[1] / wanted_height)
    if abs(stretch - 1) > MAX_CROP_STRETCH:
        return None
    left = min(max(0, left - (wanted_width - width) // 2), picture[0] - wanted_width)
    top = min(max(0, top - (wanted_height - height) // 2), picture[1] - wanted_height)
    return left, top, left + wanted_width, top + wanted_height


def _working_size(width: int, height: int) -> tuple[int, int]:
    """A crop's size enlarged toward the working size, in whole blocks."""
    scale = max(1.0, math.sqrt(CROP_WORKING_PIXELS / (width * height)))

    def side(length: int) -> int:
        return max(CROP_ALIGN, round(length * scale / CROP_ALIGN) * CROP_ALIGN)

    return side(width), side(height)


def _matched_colours(
    edited: Image.Image, given: Image.Image, selection: Image.Image
) -> tuple[Image.Image, list[int]]:
    """The edit shifted back by how far its surroundings drifted, and that shift.

    Measured only where nothing is selected, a little way from the selection,
    which the edit was not asked to change, against the crop the workflow was
    given: what is taken out is the model's drift, not the edit. The drift is
    the middle of the differences there, so a change over less than half the
    surroundings, such as light the edit cast, moves nothing. A drift past
    MAX_COLOUR_SHIFT is taken back by that much only.
    """
    grown = selection.filter(ImageFilter.MaxFilter(2 * COLOUR_RING_GAP_PX + 1))
    ring = grown.point(lambda value: 255 if value == 0 else 0)
    if ImageStat.Stat(ring).mean[0] / 255 < MIN_COLOUR_RING_SHARE:
        return edited, [0, 0, 0]
    # Centred on 128, so a drift either way of up to 127 levels is kept exactly.
    difference = ImageChops.subtract(edited, given, scale=1.0, offset=128)
    drift = [value - 128 for value in ImageStat.Stat(difference, ring).median]
    shift = [max(-MAX_COLOUR_SHIFT, min(MAX_COLOUR_SHIFT, -value)) for value in drift]
    if not any(shift):
        return edited, [0, 0, 0]
    table = [min(255, max(0, value + offset)) for offset in shift for value in range(256)]
    return edited.point(table), shift


def _drawn_selection(mask: Image.Image) -> Image.Image:
    """Where a drawn mask selects, as one channel: white is selected.

    A mask with transparency carries the selection in its alpha. One that is
    opaque everywhere carries nothing there, so, like a mask without alpha, it
    is read by brightness: a black-and-white mask selects its white.
    """
    if mask.mode in {"RGBA", "LA", "PA"} or "transparency" in mask.info:
        alpha = mask.convert("RGBA").getchannel("A")
        if alpha.getextrema() != (255, 255):
            return alpha
    return mask.convert("L")


def workflow_selection_mask(edit: RegionEdit) -> bytes | None:
    """The selection a workflow's mask input is handed, or None to hand the drawn file.

    Read as a blend reads it, from transparency or, on an opaque mask, from
    brightness; checked against the source's upright shape; and turned round
    when the selection asks to be. It is handed at the size it was drawn,
    carrying the selection in its brightness and in its alpha, the two places
    the Studio's own masks carry it, so a workflow reading either finds it.

    Only the source's header is read, never its pixels, so a picture too large
    to blend can still be edited this way. A mask too large to read here, or a
    source whose header cannot be read, is handed as drawn.
    """
    upright = _upright_shape(edit.source)
    mask = _readable_selection(edit.mask)
    if upright is None or mask is None:
        return None
    (width, height), orientation = upright
    selected = _drawn_selection(mask)
    try:
        mask_geometry(
            source_width=width,
            source_height=height,
            mask_width=selected.width,
            mask_height=selected.height,
            orientation=orientation,
            resampler=MASK_RESAMPLER,
            threshold=0,
        )
    except MaskContractError as exc:
        raise RegionEditError(exc.code, str(exc)) from exc
    if edit.selection.invert:
        selected = ImageOps.invert(selected)
    if selected.getextrema()[1] == 0:
        raise RegionEditError(
            "region-selection-empty",
            "The selection covers nothing; select the part of the picture to change.",
        )
    return encode_png(Image.merge("RGBA", (selected, selected, selected, selected)), None)


def _upright_shape(payload: bytes) -> tuple[tuple[int, int], int] | None:
    """A picture's upright size and recorded orientation, from its header alone."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(payload))
            orientation = _orientation(image)
            width, height = image.size
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
        return None
    # Orientations 5 to 8 turn the picture on its side.
    return ((height, width) if orientation >= 5 else (width, height)), orientation


def _readable_selection(payload: bytes) -> Image.Image | None:
    """A drawn selection, decoded, or None when it is too large or unreadable to read here."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(payload))
            if not 0 < image.width * image.height <= MAX_WORKFLOW_MASK_PIXELS:
                return None
            image.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
    ):
        return None
    return image


def _selection_alpha(
    mask: Image.Image, size: tuple[int, int], *, invert: bool, orientation: int
) -> _Alpha:
    alpha = _drawn_selection(mask)
    try:
        geometry = mask_geometry(
            source_width=size[0],
            source_height=size[1],
            mask_width=alpha.width,
            mask_height=alpha.height,
            orientation=orientation,
            resampler=MASK_RESAMPLER,
            # Every partly selected pixel takes part of the edit; nothing is
            # thresholded away.
            threshold=0,
        )
    except MaskContractError as exc:
        raise RegionEditError(exc.code, str(exc)) from exc
    if not geometry.is_exact:
        alpha = alpha.resize(size, Image.Resampling.BILINEAR)
    if invert:
        alpha = ImageOps.invert(alpha)
    coverage = ImageStat.Stat(alpha).mean[0] / 255
    if coverage <= 0:
        raise RegionEditError(
            "region-selection-empty",
            "The selection covers nothing; select the part of the picture to change.",
        )
    return _Alpha(image=alpha, prepared=PreparedSelection(geometry=geometry, coverage=coverage))
