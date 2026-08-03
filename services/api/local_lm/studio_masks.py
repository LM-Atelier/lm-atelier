"""The mask contract: what a masked edit turn must record to be honest.

A studio selection is drawn at whatever resolution the browser held, which
is not necessarily the source image's resolution and never necessarily the
workflow's. Resizing a feathered mask is lossy in both directions, so the
request records exactly how the mask relates to the source - orientation,
both dimensions, the transform, the resampler, and the feather space -
rather than asserting the two are interchangeable. Coverage is always
reported in source-image coordinates so a percentage means one thing.

Nothing here reads pixels. It validates and normalizes the declaration; the
adapter resizes deterministically from these recorded terms.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ARTIFACT_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
MAX_FEATHER_PX = 128
MASK_SETTING_KEY = "mask"
MASK_SCHEMA_KIND = "mask"


class MaskContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MaskSelection:
    """One validated selection, bound to the exact artifact that holds it."""

    artifact_id: str
    feather_px: int
    invert: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "feather_px": self.feather_px,
            "invert": self.invert,
        }


@dataclass(frozen=True)
class MaskGeometry:
    """How the drawn mask maps onto the source image, recorded not assumed."""

    source_width: int
    source_height: int
    mask_width: int
    mask_height: int
    orientation: int
    resampler: str
    threshold: int

    @property
    def scale_x(self) -> float:
        return self.source_width / self.mask_width

    @property
    def scale_y(self) -> float:
        return self.source_height / self.mask_height

    @property
    def is_exact(self) -> bool:
        """True when no resampling is needed at all."""
        return self.source_width == self.mask_width and self.source_height == self.mask_height

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "mask_width": self.mask_width,
            "mask_height": self.mask_height,
            "orientation": self.orientation,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "resampler": self.resampler,
            "threshold": self.threshold,
            "exact": self.is_exact,
        }


def workflow_accepts_mask(input_schema: dict[str, Any] | None) -> bool:
    """Whether this workflow declares a mask input at all.

    A mask on a workflow with nowhere to put it is a refusal, never a
    silently ignored setting - the user would see an unmasked edit and
    believe their selection had been honored.
    """
    if not isinstance(input_schema, dict):
        return False
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        return False
    declared = properties.get(MASK_SETTING_KEY)
    if not isinstance(declared, dict):
        return False
    return declared.get("x-lm-atelier-kind") == MASK_SCHEMA_KIND


def parse_mask_setting(
    settings: dict[str, Any],
    input_schema: dict[str, Any] | None,
) -> MaskSelection | None:
    """Validate the turn's mask setting against what the workflow accepts."""
    raw = settings.get(MASK_SETTING_KEY)
    if raw is None:
        return None
    if not workflow_accepts_mask(input_schema):
        raise MaskContractError(
            "workflow-has-no-mask-input",
            "This workflow cannot apply a selection; choose one that supports inpainting.",
        )
    if not isinstance(raw, dict):
        raise MaskContractError("mask-setting-invalid", "The mask setting must be an object.")
    artifact_id = raw.get("artifact_id")
    if not isinstance(artifact_id, str) or not _ARTIFACT_ID.fullmatch(artifact_id):
        raise MaskContractError(
            "mask-artifact-invalid", "The mask must name a stored content-addressed artifact."
        )
    feather = raw.get("feather_px", 0)
    if isinstance(feather, bool) or not isinstance(feather, int):
        raise MaskContractError("mask-feather-invalid", "Mask feathering must be a whole number.")
    if feather < 0 or feather > MAX_FEATHER_PX:
        raise MaskContractError(
            "mask-feather-out-of-range",
            f"Mask feathering must be between 0 and {MAX_FEATHER_PX} pixels.",
        )
    invert = raw.get("invert", False)
    if not isinstance(invert, bool):
        raise MaskContractError("mask-invert-invalid", "Mask inversion must be true or false.")
    return MaskSelection(artifact_id=artifact_id, feather_px=feather, invert=invert)


def mask_geometry(
    *,
    source_width: int,
    source_height: int,
    mask_width: int,
    mask_height: int,
    orientation: int = 1,
    resampler: str = "bilinear",
    threshold: int = 128,
) -> MaskGeometry:
    """Record the mask-to-source relationship, refusing impossible ones."""
    for label, value in (
        ("source width", source_width),
        ("source height", source_height),
        ("mask width", mask_width),
        ("mask height", mask_height),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise MaskContractError("mask-geometry-invalid", f"The {label} must be positive.")
    if orientation not in range(1, 9):
        raise MaskContractError(
            "mask-orientation-invalid", "EXIF orientation must be between 1 and 8."
        )
    if threshold < 0 or threshold > 255:
        raise MaskContractError(
            "mask-threshold-invalid", "The mask threshold must be between 0 and 255."
        )
    # A mask whose aspect ratio does not match the source was drawn against a
    # different picture; stretching it would silently move the selection.
    source_ratio = source_width / source_height
    mask_ratio = mask_width / mask_height
    if abs(source_ratio - mask_ratio) > 0.01 * source_ratio:
        raise MaskContractError(
            "mask-aspect-mismatch",
            "The selection does not match this image; draw it again on the current image.",
        )
    return MaskGeometry(
        source_width=source_width,
        source_height=source_height,
        mask_width=mask_width,
        mask_height=mask_height,
        orientation=orientation,
        resampler=resampler,
        threshold=threshold,
    )


def mask_provenance(
    selection: MaskSelection,
    geometry: MaskGeometry,
    *,
    coverage: float,
) -> dict[str, Any]:
    """What the run records about the selection that produced its result."""
    if not 0.0 <= coverage <= 1.0:
        raise MaskContractError(
            "mask-coverage-invalid", "Mask coverage must be a fraction between 0 and 1."
        )
    return {
        **selection.as_dict(),
        "geometry": geometry.as_dict(),
        # Always in source-image coordinates, so a percentage means one thing
        # regardless of what resolution the selection was drawn at.
        "coverage": coverage,
    }
