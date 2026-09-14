"""Put an edited picture back into its source through the selection that asked for it.

An instruction-edit model re-renders every pixel of the picture it is given,
even when the words only asked about one part of it, and it hands the result
back at its own working size. For a tool whose promise is that only the
selection changes, that is not good enough. Cropping the selection out before
the edit is worse: the model returns the crop at a different shape, and
anything written in it stretches.

So the edit runs on the whole picture, and afterwards the result is resized
to the source and composited over it through the selection's alpha. Outside
the selection the stored picture is the source's own pixels, inside it is the
edit, and along a feathered edge it is a mix of the two.

Pure work on bytes: nothing here opens a session or a file.
"""

from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from .studio_masks import (
    MaskContractError,
    MaskGeometry,
    MaskSelection,
    mask_geometry,
    mask_provenance,
)

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


def _selection_alpha(
    mask: Image.Image, size: tuple[int, int], *, invert: bool, orientation: int
) -> _Alpha:
    if mask.mode in {"RGBA", "LA", "PA"} or "transparency" in mask.info:
        alpha = mask.convert("RGBA").getchannel("A")
    else:
        alpha = mask.convert("L")
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
