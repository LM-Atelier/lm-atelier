"""Put the accepted source back into an extended picture, pixel for pixel.

A cropped picture needs nothing put back: its whole canvas is the edit. It is
only brought back to its canvas when the VAE rounded the canvas down.

An outpainting workflow repaints a feathered band inside the source edge and
sends the rest of the source through its VAE, so the picture it saves is close
to the source but is not the source. The accepted margins say exactly where the
source sits on the canvas, so the kept picture takes every one of those pixels
from the prepared source and only the added canvas from the workflow.

A picture this cannot read, or one in an encoding or size the source cannot be
placed on exactly, is left as the workflow saved it. The source-fit judgement
then says why it was not assessed, rather than the run losing its result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from PIL import Image

from .output_measurement import Budget, measure_output
from .source_crop_recipe import SourceCropRecipe
from .source_fit_image import PreparedSourceImage
from .source_fit_recipe import SourceExtensionRecipe
from .studio_region_edit import RESULT_RESAMPLER, RegionEditError, decode_picture, encode_png

#: A VAE crops the canvas down to its own multiple, which is never this large.
MAX_VAE_CROP: Final = 64
_UNSUPPORTED_INFO: Final = ("exif", "icc_profile", "gamma", "chromaticity", "transparency")


@dataclass(frozen=True, slots=True)
class SourceRestore:
    """The accepted recipe and the prepared bytes its source is restored from."""

    recipe: SourceExtensionRecipe
    prepared: PreparedSourceImage


@dataclass(frozen=True, slots=True)
class RestoredPicture:
    content: bytes
    record: dict[str, Any]


def restores(restore: SourceRestore, origin: object) -> bool:
    """Whether this output is the picture the recipe's own save node wrote."""
    return (
        isinstance(origin, Mapping)
        and origin.get("state") == "attributed"
        and origin.get("node_id") == restore.recipe.save_node_id
        and origin.get("output_type") == "output"
        and origin.get("collection") == "images"
    )


def restore_source(restore: SourceRestore, result: bytes) -> RestoredPicture | None:
    """The picture with its source pasted back, or nothing when there is nothing to do."""
    recipe = restore.recipe
    canvas = (recipe.canvas_width, recipe.canvas_height)
    # Held to the same reading as the judgement that follows: a file that
    # does not measure cleanly is not repaired into one that does.
    measured = measure_output(result, Budget())
    if measured.get("state") != "measured" or measured.get("animated") is not False:
        return None
    try:
        extended = decode_picture(result, "extended picture")
        source = decode_picture(restore.prepared.content, "source picture")
    except RegionEditError:
        return None
    crop = (canvas[0] - extended.width, canvas[1] - extended.height)
    if (
        extended.format != "PNG"
        or extended.mode not in {"RGB", "RGBA", "L", "LA"}
        or getattr(extended, "n_frames", 1) != 1
        or any(key in extended.info for key in _UNSUPPORTED_INFO)
        or not all(0 <= side < MAX_VAE_CROP for side in crop)
        or source.size != (recipe.image.width, recipe.image.height)
    ):
        return None
    margins = recipe.margins()
    box = (
        margins["left"],
        margins["top"],
        margins["left"] + source.width,
        margins["top"] + source.height,
    )
    mode = "RGBA" if extended.mode in {"RGBA", "LA"} else "RGB"
    finished = extended.convert(mode)
    source_pixels = source.convert(mode)
    resized = crop != (0, 0)
    if resized:
        # A VAE may round the canvas to its own multiple. Resized back to the
        # canvas, as a composite with resize_source would, so the source lands
        # where the margins placed it.
        finished = finished.resize(canvas, Image.Resampling.LANCZOS)
    elif finished.crop(box).tobytes() == source_pixels.tobytes():
        return None
    finished.paste(source_pixels, box[:2])
    return RestoredPicture(
        content=encode_png(finished, None),
        record={
            "source_artifact_id": recipe.image.source_artifact_id,
            "prepared_artifact_id": recipe.image.prepared_artifact_id,
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "result_width": extended.width,
            "result_height": extended.height,
            "result_resampler": RESULT_RESAMPLER if resized else None,
            "left": margins["left"],
            "top": margins["top"],
            "width": canvas[0],
            "height": canvas[1],
        },
    )


def fits(recipe: SourceCropRecipe, origin: object) -> bool:
    """Whether this output is the picture the crop recipe's own save node wrote."""
    return (
        isinstance(origin, Mapping)
        and origin.get("state") == "attributed"
        and origin.get("node_id") == recipe.save_node_id
        and origin.get("output_type") == "output"
        and origin.get("collection") == "images"
    )


def fit_to_canvas(recipe: SourceCropRecipe, result: bytes) -> RestoredPicture | None:
    """A cropped edit brought back to its canvas when the VAE rounded the canvas down.

    The crop is uploaded at the canvas size, so a workflow that keeps its
    source's size saves the canvas as it is, and there is nothing to do. A VAE
    that works in blocks trims a canvas that is not a whole number of them, by
    less than one block on each side; that picture is resized back to the
    canvas, so what is kept is the size that was asked for. Anything else is
    left as the workflow saved it, and the size check says so.
    """
    canvas = (recipe.canvas_width, recipe.canvas_height)
    measured = measure_output(result, Budget())
    if measured.get("state") != "measured" or measured.get("animated") is not False:
        return None
    try:
        picture = decode_picture(result, "edited picture")
    except RegionEditError:
        return None
    trim = (canvas[0] - picture.width, canvas[1] - picture.height)
    if (
        picture.format != "PNG"
        or picture.mode not in {"RGB", "RGBA", "L", "LA"}
        or getattr(picture, "n_frames", 1) != 1
        or any(key in picture.info for key in _UNSUPPORTED_INFO)
        or not all(0 <= side < MAX_VAE_CROP for side in trim)
        or trim == (0, 0)
    ):
        return None
    mode = "RGBA" if picture.mode in {"RGBA", "LA"} else "RGB"
    resized = picture.convert(mode).resize(canvas, Image.Resampling.LANCZOS)
    return RestoredPicture(
        content=encode_png(resized, None),
        record={
            "source_artifact_id": recipe.image.source_artifact_id,
            "cropped_artifact_id": recipe.cropped_artifact_id,
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "result_width": picture.width,
            "result_height": picture.height,
            "result_resampler": RESULT_RESAMPLER,
            "width": canvas[0],
            "height": canvas[1],
        },
    )
