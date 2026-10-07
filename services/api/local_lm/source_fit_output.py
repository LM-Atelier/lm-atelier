"""Bounded per-output comparison with the exact source uploaded by a run.

The caller supplies its accepted recipe, dispatched graph and verified upload
bytes. Results belong to that run's outputs and message parts, never the shared
artifact. A checker is used sequentially across one completed batch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from io import BytesIO
from typing import Any, Final, Literal

from PIL import Image, PngImagePlugin

from .output_measurement import Budget, measure_output
from .output_origin import names_a_preview
from .source_fit_image import MAX_SOURCE_PIXELS, PreparedSourceImage
from .source_fit_recipe import SourceExtensionRecipe

MAX_VERIFICATION_PIXELS: Final = 32 * 1024 * 1024
MAX_VERIFICATION_BYTES: Final = 64 * 1024 * 1024

Reason = Literal[
    "throwaway",
    "origin_unknown",
    "binding_unconfirmed",
    "source_unavailable",
    "output_unreadable",
    "output_unsupported",
    "canvas_mismatch",
    "over_budget",
]


class _CannotAssess(ValueError):
    def __init__(self, reason: Reason) -> None:
        self.reason = reason
        super().__init__(reason)


class PixelBudget:
    """Bound source plus every decoded output in this batch."""

    def __init__(self) -> None:
        self.pixels = MAX_VERIFICATION_PIXELS
        self.bytes = MAX_VERIFICATION_BYTES

    def spend(self, pixels: int, size: int) -> None:
        if pixels > self.pixels or size > self.bytes:
            raise _CannotAssess("over_budget")
        self.pixels -= pixels
        self.bytes -= size


def _not_assessed(reason: Reason) -> dict[str, Any]:
    return {"v": 1, "state": "not_assessed", "reason": reason}


class SourceFitPixelVerifier:
    """Compare one asset per call, allowing cancellation between thread hops."""

    def __init__(
        self,
        recipe: SourceExtensionRecipe,
        prepared: PreparedSourceImage | None,
        graph: object,
        engine: str,
    ) -> None:
        self.recipe = recipe
        self.prepared = prepared
        self.budget = PixelBudget()
        self._pixels: bytes | None = None
        self._source_failure: Reason | None = None
        self._binding_confirmed = engine == "comfyui"
        self.left = self.top = 0
        try:
            recipe.route(graph)
            margins = recipe.margins()
            self.left = margins["left"]
            self.top = margins["top"]
        except ValueError:
            self._binding_confirmed = False

    def check(self, content: bytes, measurement: object, origin: object) -> dict[str, Any]:
        if names_a_preview(origin):
            return _not_assessed("throwaway")
        if not isinstance(origin, Mapping) or origin.get("state") != "attributed":
            return _not_assessed("origin_unknown")
        if (
            not self._binding_confirmed
            or origin.get("node_id") != self.recipe.save_node_id
            or origin.get("output_type") != "output"
            or origin.get("collection") != "images"
        ):
            return _not_assessed("binding_unconfirmed")
        if not isinstance(measurement, Mapping) or measurement.get("state") != "measured":
            if isinstance(measurement, Mapping) and measurement.get("about") == "budget":
                return _not_assessed("over_budget")
            if isinstance(measurement, Mapping) and (
                (measurement.get("about"), measurement.get("reason"))
                in {("scope", "unsupported_container"), ("file", "interlaced_unsupported")}
            ):
                return _not_assessed("output_unsupported")
            return _not_assessed("output_unreadable")
        dimensions = (measurement.get("raster_width"), measurement.get("raster_height"))
        if dimensions != (self.recipe.canvas_width, self.recipe.canvas_height):
            return _not_assessed("canvas_mismatch")
        if measurement.get("animated") is not False:
            return _not_assessed("output_unsupported")
        try:
            source = self._source_pixels()
            self.budget.spend(self.recipe.canvas_width * self.recipe.canvas_height, len(content))
            return self._compare(content, source)
        except _CannotAssess as exc:
            return _not_assessed(exc.reason)

    def _source_pixels(self) -> bytes:
        if self._pixels is not None:
            return self._pixels
        if self._source_failure is not None:
            raise _CannotAssess(self._source_failure)
        try:
            self._pixels = self._decode_source()
            return self._pixels
        except _CannotAssess as exc:
            self._source_failure = exc.reason
            raise

    def _decode_source(self) -> bytes:
        prepared = self.prepared
        image = self.recipe.image
        if (
            prepared is None
            or prepared.source_artifact_id != image.source_artifact_id
            or prepared.source_sha256 != image.source_artifact_id.removeprefix("sha256:")
            or f"sha256:{prepared.sha256}" != image.prepared_artifact_id
            or (prepared.width, prepared.height) != (image.width, image.height)
            or type(prepared.content) is not bytes
        ):
            raise _CannotAssess("source_unavailable")
        self.budget.spend(image.width * image.height, len(prepared.content))
        if hashlib.sha256(prepared.content).hexdigest() != prepared.sha256:
            raise _CannotAssess("source_unavailable")
        measured = measure_output(
            prepared.content,
            Budget(decoded=MAX_SOURCE_PIXELS * 5),
        )
        if (
            measured.get("state") != "measured"
            or measured.get("animated") is not False
            or measured.get("exif_present") is not False
            or (measured.get("raster_width"), measured.get("raster_height"))
            != (image.width, image.height)
            or prepared.content[24:26] != bytes((8, 2))
        ):
            raise _CannotAssess("source_unavailable")
        try:
            with Image.open(BytesIO(prepared.content)) as source:
                if (
                    not isinstance(source, PngImagePlugin.PngImageFile)
                    or source.size != (image.width, image.height)
                    or source.mode != "RGB"
                    or source.info
                ):
                    raise _CannotAssess("source_unavailable")
                source.load()
                return source.tobytes()
        except _CannotAssess:
            raise
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            raise _CannotAssess("source_unavailable") from None

    def _compare(self, content: bytes, expected: bytes) -> dict[str, Any]:
        # The caller's measurement already walked these exact immutable bytes.
        # Recheck the decoded header before allocation; never use its metadata
        # or a display dimension as the source of the comparison rectangle.
        if len(content) < 26 or content[24] != 8:
            raise _CannotAssess("output_unsupported")
        try:
            with Image.open(BytesIO(content)) as output:
                if (
                    not isinstance(output, PngImagePlugin.PngImageFile)
                    or output.mode not in {"RGB", "RGBA", "L", "LA"}
                    or output.n_frames != 1
                    or any(
                        key in output.info
                        for key in ("exif", "icc_profile", "gamma", "chromaticity", "transparency")
                    )
                ):
                    raise _CannotAssess("output_unsupported")
                if output.size != (self.recipe.canvas_width, self.recipe.canvas_height):
                    raise _CannotAssess("output_unreadable")
                output.load()
                with output.crop(
                    (
                        self.left,
                        self.top,
                        self.left + self.recipe.image.width,
                        self.top + self.recipe.image.height,
                    )
                ) as kept:
                    opaque = True
                    if "A" in kept.getbands():
                        with kept.getchannel("A") as alpha:
                            opaque = alpha.getextrema() == (255, 255)
                    with kept.convert("RGB") as rgb:
                        preserved = opaque and rgb.tobytes() == expected
                return {"v": 1, "state": "preserved" if preserved else "changed"}
        except _CannotAssess:
            raise
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            raise _CannotAssess("output_unreadable") from None
