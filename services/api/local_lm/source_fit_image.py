"""Prepare bounded source pixels and their exact in-memory upload content.

This initial decoder accepts only opaque, single-frame, eight-bit PNG without
color-conversion metadata. Orientation is materialized in fresh RGB pixels.
Nothing reopens a source pathname after ArtifactStore verifies its bytes.

The result is data, not an admission receipt or permission to execute a graph.
Runtime compatibility, selected source/revision authority, accepted geometry
and output verification still belong to the caller.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from typing import Final, Literal

from PIL import Image, ImageOps, PngImagePlugin
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from .artifact_library import begin_artifact_write_fence
from .artifacts import ArtifactStore
from .domain import ArtifactKind
from .models import Artifact
from .output_measurement import Budget, measure_output

MAX_SOURCE_BYTES: Final = 64 * 1024 * 1024
MAX_SOURCE_PIXELS: Final = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class PreparedSourceImage:
    source_artifact_id: str
    source_sha256: str
    sha256: str
    width: int
    height: int
    content: bytes = field(repr=False)


def prepare_source_fit_image(store: ArtifactStore, artifact: Artifact) -> PreparedSourceImage:
    """Return canonical bytes measured from a single verified artifact read."""
    if artifact.kind not in {ArtifactKind.IMAGE, ArtifactKind.INPUT}:
        raise ValueError("source_fit_image_unsupported")
    try:
        content = store.verified_bytes(artifact, maximum_bytes=MAX_SOURCE_BYTES)
    except (OSError, ValueError):
        raise ValueError("source_fit_image_unavailable") from None
    source_digest = hashlib.sha256(content).hexdigest()
    canonical, width, height = _canonical_png(content)
    return PreparedSourceImage(
        source_artifact_id=f"sha256:{source_digest}",
        source_sha256=source_digest,
        sha256=hashlib.sha256(canonical).hexdigest(),
        width=width,
        height=height,
        content=canonical,
    )


def _canonical_png(content: bytes) -> tuple[bytes, int, int]:
    # The existing bounded container/IDAT walk refuses truncation, animation,
    # trailing bytes and oversized decompression before allocating an image.
    measured = measure_output(
        content,
        Budget(inflate_input=MAX_SOURCE_BYTES, decoded=MAX_SOURCE_PIXELS * 4 + 65536),
    )
    if (
        measured.get("state") != "measured"
        or measured.get("animated") is not False
        or content[24] != 8
        or measured["raster_width"] * measured["raster_height"] > MAX_SOURCE_PIXELS
    ):
        raise ValueError("source_fit_image_unsupported")
    try:
        with Image.open(BytesIO(content)) as source:
            if (
                not isinstance(source, PngImagePlugin.PngImageFile)
                or source.n_frames != 1
                or source.mode not in {"L", "LA", "RGB", "RGBA"}
                or any(key in source.info for key in ("icc_profile", "gamma", "chromaticity"))
            ):
                raise ValueError
            source.load()
            if source.size != (measured["raster_width"], measured["raster_height"]):
                raise ValueError
            orientation = source.getexif().get(274, 1)
            if type(orientation) is not int or orientation not in range(1, 9):
                raise ValueError
            oriented = ImageOps.exif_transpose(source)
            # Transparency would require a separately bound mask policy.
            # Do not silently composite it onto a color or discard its alpha.
            if "A" in oriented.getbands() or "transparency" in oriented.info:
                with oriented.convert("RGBA") as rgba:
                    if rgba.getchannel("A").getextrema() != (255, 255):
                        raise ValueError
            # A fresh image strips EXIF, profiles and text. A subsequent
            # decoder receives the same orientation and ordinary RGB bytes.
            with (
                oriented.convert("RGB") as rgb,
                Image.frombytes("RGB", rgb.size, rgb.tobytes()) as canonical,
            ):
                output = BytesIO()
                canonical.save(output, format="PNG", compress_level=6)
                result = output.getvalue()
                if len(result) > MAX_SOURCE_BYTES:
                    raise ValueError
                return result, canonical.width, canonical.height
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("source_fit_image_unsupported") from None


class SourceFitImageRecord(BaseModel):
    """Serializable image identity, not a workflow or request authorization.

    The accepted context must retain both IDs and bind this record to its first
    spatial input. Capturing these bytes alone creates no retention edge.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: Literal[1] = 1
    source_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    prepared_artifact_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    width: int = Field(ge=1, le=MAX_SOURCE_PIXELS)
    height: int = Field(ge=1, le=MAX_SOURCE_PIXELS)


def capture_source_fit_image(
    session: Session, store: ArtifactStore, source: Artifact
) -> SourceFitImageRecord:
    """Store exact prepared bytes in the caller's acceptance transaction."""
    prepared = prepare_source_fit_image(store, source)
    return retain_prepared_source_fit_image(session, store, prepared)


def retain_prepared_source_fit_image(
    session: Session, store: ArtifactStore, prepared: PreparedSourceImage
) -> SourceFitImageRecord:
    """Retain already validated bytes after the caller has approved the canvas.

    Preparation and graph/geometry validation can fail without writing an artifact.
    Retaining the same immutable buffer avoids a second source read between them.
    """
    # Keep the existing row and its role/metadata intact on deduplication.
    # Ingest normally promotes a preview, which is useful for a completed
    # output but would reclassify that image as INPUT here. Hold the same
    # writer reservation through lookup and the caller's retention edges.
    begin_artifact_write_fence(session)
    retained = session.get(Artifact, f"sha256:{prepared.sha256}")
    if retained is None:
        retained = store.ingest_bytes(
            session,
            prepared.content,
            kind=ArtifactKind.INPUT,
            media_type="image/png",
        )
    else:
        try:
            existing = store.verified_bytes(retained, maximum_bytes=MAX_SOURCE_BYTES)
        except (OSError, ValueError):
            raise ValueError("source_fit_image_unavailable") from None
        if existing != prepared.content:
            raise ValueError("source_fit_image_binding")
    return SourceFitImageRecord(
        source_artifact_id=prepared.source_artifact_id,
        prepared_artifact_id=retained.id,
        width=prepared.width,
        height=prepared.height,
    )


def replay_source_fit_image(
    session: Session,
    store: ArtifactStore,
    record: SourceFitImageRecord,
    *,
    selected_source_id: str,
) -> PreparedSourceImage:
    """Read the retained upload bytes without decoding a mutable source again.

    The caller supplies the first input from its validated accepted context,
    whose retention and removal checks remain necessary. A record passed by
    itself is not evidence of permission to access either artifact.
    """
    if record.source_artifact_id != selected_source_id:
        raise ValueError("source_fit_image_binding")
    source = session.get(Artifact, record.source_artifact_id)
    prepared = session.get(Artifact, record.prepared_artifact_id)
    if source is None or prepared is None:
        raise ValueError("source_fit_image_unavailable")
    if source.sha256 != record.source_artifact_id.removeprefix("sha256:"):
        raise ValueError("source_fit_image_binding")
    try:
        content = store.verified_bytes(prepared, maximum_bytes=MAX_SOURCE_BYTES)
    except (OSError, ValueError):
        raise ValueError("source_fit_image_unavailable") from None
    measured = measure_output(
        content,
        Budget(inflate_input=MAX_SOURCE_BYTES, decoded=MAX_SOURCE_PIXELS * 3 + 65536),
    )
    if (
        measured.get("state") != "measured"
        or measured.get("animated") is not False
        or measured.get("exif_present") is not False
        or content[24:26] != bytes((8, 2))
        or measured["raster_width"] != record.width
        or measured["raster_height"] != record.height
        or record.width * record.height > MAX_SOURCE_PIXELS
    ):
        raise ValueError("source_fit_image_binding")
    return PreparedSourceImage(
        source_artifact_id=record.source_artifact_id,
        source_sha256=source.sha256,
        sha256=prepared.sha256,
        width=record.width,
        height=record.height,
        content=content,
    )
