"""Relight a picture from a chosen direction, then finish it against its source.

The edit itself runs on an instruction-edit workflow with a lighting adapter
LoRA and two pictures: the source, and a luminance map bright on the side the
light comes from. The adapter relights reliably only at full strength, and at
full strength it also darkens the whole scene. So a gentler light is not asked
of the model: the stored picture is the relit result mixed back toward the
source by the chosen intensity. A mix keeps the source's own detail and adds
nothing the model did not draw. Warmth is a white-balance grade applied last.

Which LoRA is the lighting adapter is declared here, by exact content identity.
A name, a trigger word or a repository alone never qualifies an asset, because
none of them says the file is the one whose behaviour was checked.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .studio_region_edit import (
    BLEND_MEDIA_TYPE,
    RESULT_RESAMPLER,
    encode_png,
    fit_result,
    load_source,
)

RELIGHT_SETTING_KEY = "relight"
RELIGHT_DIRECTIONS = ("left", "right", "top")
MIN_INTENSITY = 0.25
MAX_INTENSITY = 1.0
#: Warmer than about 4000 K the full white-balance shift turns a whole scene
#: amber, and cooler than 7500 K it turns it blue; neither reads as a light.
MIN_KELVIN = 4000
MAX_KELVIN = 7500
NEUTRAL_KELVIN = 6500
#: The adapter relights only at this strength; lower strengths drift into
#: recolouring and added objects instead of light.
ADAPTER_MODEL_STRENGTH = 1.0


class RelightContractError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LoraIdentity:
    """One exact LoRA file: where it came from and what its bytes hash to."""

    provider: str
    remote_id: str
    revision: str
    filename: str
    sha256: str


#: The Multi-Angle Lighting adapter for Qwen image editing, as checked.
LIGHTING_ADAPTER = LoraIdentity(
    provider="huggingface",
    remote_id="dx8152/Qwen-Edit-2509-Multi-Angle-Lighting",
    revision="a482aff27704ced5df6009c23ced59abc7348f4e",
    filename="多角度灯光-251121.safetensors",
    sha256="cffb85904608283d3920fee2f0e9e74af0712008fd29edb33042742f5fbce6ce",
)


def is_lighting_adapter(*, provider: str, remote_id: str, revision: str, manifest: object) -> bool:
    """Whether an installed LoRA is exactly the declared lighting adapter."""
    adapter = LIGHTING_ADAPTER
    if (provider, remote_id, revision) != (adapter.provider, adapter.remote_id, adapter.revision):
        return False
    if not isinstance(manifest, Mapping):
        return False
    expected = manifest.get("expected_sha256")
    return (
        manifest.get("sha256") == adapter.sha256
        and manifest.get("comfy_name") == adapter.filename
        and isinstance(expected, Mapping)
        and expected.get(adapter.filename) == adapter.sha256
    )


@dataclass(frozen=True)
class RelightSetting:
    direction: str
    intensity: float
    kelvin: int | None

    def as_dict(self) -> dict[str, Any]:
        return {"direction": self.direction, "intensity": self.intensity, "kelvin": self.kelvin}


def split_relight_setting(settings: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Separate the relight request from the settings a workflow declares.

    No workflow lists it among its fields, so validating it as one would refuse
    it as unknown; it is checked by `parse_relight_setting` instead.
    """
    relight = settings.get(RELIGHT_SETTING_KEY)
    rest = {key: value for key, value in settings.items() if key != RELIGHT_SETTING_KEY}
    return relight, rest


def parse_relight_setting(settings: Mapping[str, Any]) -> RelightSetting | None:
    raw = settings.get(RELIGHT_SETTING_KEY)
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - {"direction", "intensity", "kelvin"}:
        raise RelightContractError(
            "relight-setting-invalid",
            "The relight setting must name a direction, an intensity and an optional warmth.",
        )
    direction = raw.get("direction")
    if direction not in RELIGHT_DIRECTIONS:
        raise RelightContractError(
            "relight-direction-invalid",
            "Relight from the left, the right or the top.",
        )
    intensity = raw.get("intensity", MAX_INTENSITY)
    if (
        isinstance(intensity, bool)
        or not isinstance(intensity, int | float)
        or not math.isfinite(intensity)
        or not MIN_INTENSITY <= intensity <= MAX_INTENSITY
    ):
        raise RelightContractError(
            "relight-intensity-invalid",
            f"Relight intensity must be between {MIN_INTENSITY} and {MAX_INTENSITY}.",
        )
    kelvin = raw.get("kelvin")
    if kelvin is not None and (
        isinstance(kelvin, bool)
        or not isinstance(kelvin, int)
        or not MIN_KELVIN <= kelvin <= MAX_KELVIN
    ):
        raise RelightContractError(
            "relight-warmth-invalid",
            f"Relight warmth must be between {MIN_KELVIN} K and {MAX_KELVIN} K, or left out.",
        )
    return RelightSetting(direction=str(direction), intensity=float(intensity), kelvin=kelvin)


def require_relight_turn(
    setting: RelightSetting,
    *,
    operation: str,
    source_count: int,
    loras: object,
    adapter_asset_ids: Iterable[str],
) -> None:
    """Refuse a relight turn that is not a two-picture edit with the adapter at full strength."""
    if operation != "image_to_image" or source_count != 2:
        raise RelightContractError(
            "relight-needs-source-and-map",
            "Relighting needs the picture and its light map, and nothing else.",
        )
    adapters = set(adapter_asset_ids)
    applied = (
        [
            item
            for item in loras
            if isinstance(item, Mapping)
            and item.get("asset_id") in adapters
            and item.get("enabled", True) is True
        ]
        if isinstance(loras, list)
        else []
    )
    if not applied:
        raise RelightContractError(
            "relight-adapter-missing",
            "Relighting needs the lighting LoRA installed and applied.",
        )
    if any(item.get("model_strength", 1.0) != ADAPTER_MODEL_STRENGTH for item in applied):
        raise RelightContractError(
            "relight-adapter-strength",
            "The lighting LoRA relights only at full strength.",
        )


@dataclass(frozen=True)
class RelightFinish:
    """One relight request and the exact stored source it is finished against."""

    setting: RelightSetting
    source_artifact_id: str
    source: bytes


@dataclass(frozen=True)
class FinishedPicture:
    content: bytes
    media_type: str
    record: dict[str, Any]


def finish_relight(finish: RelightFinish, result: bytes) -> FinishedPicture:
    """Mix the relit result toward the source by the intensity, then grade its warmth."""
    source = load_source(finish.source)
    edited, fitted = fit_result(
        source,
        result,
        "The relit picture came back in a different shape, so it cannot be matched to the source.",
    )
    setting = finish.setting
    finished = (
        fitted
        if setting.intensity >= MAX_INTENSITY
        else Image.blend(source.image, fitted, setting.intensity)
    )
    if setting.kelvin is not None:
        finished = warmth_grade(finished, setting.kelvin)
    return FinishedPicture(
        content=encode_png(finished, source.icc_profile),
        media_type=BLEND_MEDIA_TYPE,
        record={
            **setting.as_dict(),
            "source_artifact_id": finish.source_artifact_id,
            "mix": "pixel",
            "grade": "white_balance" if setting.kelvin is not None else None,
            "result_sha256": hashlib.sha256(result).hexdigest(),
            "result_width": edited.width,
            "result_height": edited.height,
            "result_resampler": RESULT_RESAMPLER,
            "width": source.image.width,
            "height": source.image.height,
        },
    )


def warmth_grade(picture: Image.Image, kelvin: int) -> Image.Image:
    """White-balance the picture toward a colour temperature, keeping its brightness.

    Each channel is scaled by the blackbody colour at `kelvin` over the colour at
    6500 K, so 6500 K changes nothing, and the gains are normalised by their
    luma weight so a warmer light is not also a darker one. Transparency is kept.
    """
    target = _blackbody(kelvin)
    neutral = _blackbody(NEUTRAL_KELVIN)
    gains = [target[index] / neutral[index] for index in range(3)]
    luma = 0.2126 * gains[0] + 0.7152 * gains[1] + 0.0722 * gains[2]
    bands = list(picture.split())
    for index in range(3):
        gain = gains[index] / luma
        bands[index] = bands[index].point([min(255, round(value * gain)) for value in range(256)])
    return Image.merge(picture.mode, bands)


def _blackbody(kelvin: float) -> tuple[float, float, float]:
    """The colour of a blackbody at `kelvin`, from Tanner Helland's fit, 0-255."""
    t = kelvin / 100
    red = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    green = (
        99.4708025861 * math.log(t) - 161.1195681661
        if t <= 66
        else 288.1221695283 * (t - 60) ** -0.0755148492
    )
    blue = (
        255.0 if t >= 66 else 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    )
    return (_channel(red), _channel(green), _channel(blue))


def _channel(value: float) -> float:
    return min(255.0, max(1.0, value))
