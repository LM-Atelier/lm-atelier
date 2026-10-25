"""Make an extension on a canvas of whole blocks, and bring it back to the canvas asked for.

ComfyUI encodes a picture in whole blocks of eight pixels and crops whatever is
left over, centred. A canvas that is not a whole number of blocks therefore
comes back shifted against its source by up to half a block, which blurs every
line across the feathered seam, and short of its right or bottom edge, where
the padding's flat grey shows. So Extend runs on a canvas of whole blocks and
the result is cut back to the canvas that was asked for.

The padding grows on the outer edge of a side being extended. Along an axis
nobody extended, the source itself grows instead, by repeating its last column
or row, because ComfyUI feathers every edge it pads into the source. The
workflow is handed that grown source as it is read here: upright, one frame,
eight bits a channel. A picture that would be read differently is sent as it
is stored, on the canvas that was asked for. Once cut back, a picture the
workflow pasted back over its padded canvas through the padding's own mask has
its new area matched to the colour of the source along each extended edge.

Pure work on bytes: nothing here opens a session or a file.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from PIL import Image, UnidentifiedImageError

from .outpaint_colour import match_extension_colour
from .source_fit_restore import RestoredPicture
from .studio_region_edit import RegionEditError, decode_picture, encode_png, load_source
from .workflow_source_geometry import SelectionRoute

#: ComfyUI's largest padding on one side.
MAX_PAD_PX = 16384
#: Modes ComfyUI and this module read the same way.
_EIGHT_BIT = frozenset({"1", "L", "LA", "P", "PA", "RGB", "RGBA"})
#: EXIF orientations that mirror the picture, which ComfyUI does not.
_MIRRORED = frozenset({2, 4, 5, 7})
_ORIENTATION_TAG = 0x0112
_SIDES = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class ExtensionFinish:
    """How one extension is made on a canvas of whole blocks, and cut back."""

    source_artifact_id: str
    #: The source as the workflow is handed it: upright, grown on any axis nobody extended.
    prepared: bytes
    #: The saves whose pictures are the extended canvas.
    route: SelectionRoute
    #: The source's upright width and height.
    size: tuple[int, int]
    #: Pixels per side that were asked for.
    requested: Mapping[str, int]
    #: Pixels per side the workflow's padding runs with.
    working: Mapping[str, int]
    #: Columns on the right and rows at the bottom the source itself grew by.
    grown: Mapping[str, int]
    #: The saves that paste the decode over the padded canvas through the padding's mask.
    blended_saves: frozenset[str] = field(default_factory=frozenset)
    #: How far the padding feathers the source, when the run states it as a whole number.
    feathering: int | None = None

    def blends(self, origin: object) -> bool:
        """Whether this output is a picture one of the blended saves wrote."""
        return (
            self.route.saved(origin)
            and isinstance(origin, Mapping)
            and origin.get("node_id") in self.blended_saves
        )

    @property
    def canvas(self) -> tuple[int, int]:
        """The size the workflow's picture comes back at."""
        return (
            self.size[0] + self.grown["right"] + self.working["left"] + self.working["right"],
            self.size[1] + self.grown["bottom"] + self.working["top"] + self.working["bottom"],
        )


def prepare_source(source: bytes, grown: Mapping[str, int]) -> tuple[bytes, tuple[int, int]] | None:
    """The source as the workflow is handed it, with its upright size, or None to send it as stored.

    A picture with more than one frame, more than eight bits a channel or a
    mirroring orientation is read differently by ComfyUI than here, so it is
    not replaced.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(source)) as image:
                if (
                    getattr(image, "n_frames", 1) != 1
                    or image.mode not in _EIGHT_BIT
                    or image.getexif().get(_ORIENTATION_TAG) in _MIRRORED
                ):
                    return None
        loaded = load_source(source)
    except (
        RegionEditError,
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return None
    picture = loaded.image
    width, height = picture.size
    grown_picture = Image.new(picture.mode, (width + grown["right"], height + grown["bottom"]))
    grown_picture.paste(picture, (0, 0))
    if grown["right"]:
        column = picture.crop((width - 1, 0, width, height))
        grown_picture.paste(
            column.resize((grown["right"], height), Image.Resampling.NEAREST), (width, 0)
        )
    if grown["bottom"]:
        row = grown_picture.crop((0, height - 1, grown_picture.width, height))
        grown_picture.paste(
            row.resize((grown_picture.width, grown["bottom"]), Image.Resampling.NEAREST),
            (0, height),
        )
    return encode_png(grown_picture, loaded.icc_profile), (width, height)


def finish_extension(
    finish: ExtensionFinish, result: bytes, blended: bool = False
) -> RestoredPicture | None:
    """The extension cut back to the canvas that was asked for.

    None keeps the picture as the workflow made it: one that does not read, or
    that is not the canvas the workflow ran on, is not this extension's canvas.
    Its colour is matched only when `blended` says a save pasted it over the
    padded canvas through the padding's mask, and the run states the feather:
    only then is the decode known in the band where it mixes with the source.
    """
    try:
        made = decode_picture(result, "extended picture")
    except RegionEditError:
        return None
    if made.size != finish.canvas:
        return None
    requested, working = finish.requested, finish.working
    left = working["left"] - requested["left"]
    top = working["top"] - requested["top"]
    width = finish.size[0] + requested["left"] + requested["right"]
    height = finish.size[1] + requested["top"] + requested["bottom"]
    picture = made.crop((left, top, left + width, top + height))
    colour: dict[str, Any]
    if not blended:
        colour = {"applied": False, "reason": "not_blended"}
    elif finish.feathering is None:
        colour = {"applied": False, "reason": "feather_unknown"}
    else:
        handed = decode_picture(finish.prepared, "prepared source").convert("RGB")
        picture, colour = match_extension_colour(
            picture.convert("RGB"),
            handed.crop((0, 0, *finish.size)),
            requested,
            feathered_px(finish.feathering, handed.size),
        )
    return RestoredPicture(
        content=encode_png(picture, None),
        record={
            "version": 1,
            "source_artifact_id": finish.source_artifact_id,
            "requested": {side: requested[side] for side in _SIDES},
            "working": {side: working[side] for side in _SIDES},
            "grown": {"right": finish.grown["right"], "bottom": finish.grown["bottom"]},
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "colour": colour,
            "result_width": made.width,
            "result_height": made.height,
            "width": width,
            "height": height,
        },
    )


def stated_feathering(value: object, settings: Mapping[str, Any]) -> int | None:
    """The feather a padding runs with, when written or set as a whole number; else None.

    A feather another node supplies, or one the run would pass on as anything
    but a whole number, is not known exactly here.
    """
    if isinstance(value, str) and value.startswith("$" + "{") and value.endswith("}"):
        value = settings.get(value[2:-1])
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def feathered_px(feathering: int, size: tuple[int, int]) -> int:
    """How far ComfyUI's padding feathers into the picture it pads: nothing unless twice it fits."""
    if feathering <= 0 or feathering * 2 >= size[0] or feathering * 2 >= size[1]:
        return 0
    return feathering
