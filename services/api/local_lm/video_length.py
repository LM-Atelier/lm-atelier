from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final

from .schemas import SettingField

VIDEO_LENGTH_SCHEMA_KEY: Final = "x-lm-atelier-video-length"
VIDEO_DURATION_SETTING_KEY: Final = "duration_seconds"
_CONTRACT_KEYS: Final = frozenset(
    {
        "version",
        "frames_parameter",
        "fps_parameter",
        "fps_numerator",
        "fps_denominator",
        "frame_alignment",
        "frame_offset",
    }
)
_MAX_COMPONENT: Final = 1_000_000
_MAX_FRAMES: Final = 2_147_483_647


@dataclass(frozen=True, slots=True)
class WorkflowVideoLength:
    frames_parameter: str
    fps_parameter: str
    fps: Fraction
    frame_alignment: int
    frame_offset: int
    minimum_frames: int
    maximum_frames: int
    default_frames: int

    @property
    def minimum_seconds(self) -> Fraction:
        return Fraction(self.minimum_frames, 1) / self.fps

    @property
    def maximum_seconds(self) -> Fraction:
        return Fraction(self.maximum_frames, 1) / self.fps

    @property
    def default_seconds(self) -> Fraction:
        return Fraction(self.default_frames, 1) / self.fps


def _integer(value: object, name: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(
            f"workflow video length {name} must be an integer from {minimum} to {maximum}"
        )
    return value


def _parameter(value: object, name: str) -> str:
    if type(value) is not str or not value or len(value) > 200:
        raise ValueError(f"workflow video length {name} must name a setting")
    return value


def _frame_value(schema: Mapping[str, Any], name: str) -> int:
    return _integer(schema.get(name), f"frames {name}", minimum=0, maximum=_MAX_FRAMES)


def workflow_video_length(
    input_schema: Mapping[str, Any] | None,
) -> WorkflowVideoLength | None:
    if not input_schema or VIDEO_LENGTH_SCHEMA_KEY not in input_schema:
        return None
    raw = input_schema[VIDEO_LENGTH_SCHEMA_KEY]
    if not isinstance(raw, Mapping) or set(raw) != _CONTRACT_KEYS:
        raise ValueError("workflow video length contract has unsupported or missing fields")
    if raw.get("version") != 1:
        raise ValueError("workflow video length version must be 1")
    frames_parameter = _parameter(raw.get("frames_parameter"), "frames_parameter")
    fps_parameter = _parameter(raw.get("fps_parameter"), "fps_parameter")
    if frames_parameter == fps_parameter or VIDEO_DURATION_SETTING_KEY in {
        frames_parameter,
        fps_parameter,
    }:
        raise ValueError("workflow video length parameters must be distinct")
    numerator = _integer(
        raw.get("fps_numerator"), "fps_numerator", minimum=1, maximum=_MAX_COMPONENT
    )
    denominator = _integer(
        raw.get("fps_denominator"), "fps_denominator", minimum=1, maximum=_MAX_COMPONENT
    )
    fps = Fraction(numerator, denominator)
    if (fps.numerator, fps.denominator) != (numerator, denominator):
        raise ValueError("workflow video length FPS must be reduced")
    alignment = _integer(
        raw.get("frame_alignment"), "frame_alignment", minimum=1, maximum=_MAX_FRAMES
    )
    offset = _integer(raw.get("frame_offset"), "frame_offset", minimum=0, maximum=_MAX_FRAMES)
    if offset >= alignment:
        raise ValueError("workflow video length frame_offset must be below frame_alignment")

    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("workflow video length requires declared workflow properties")
    frames_schema = properties.get(frames_parameter)
    fps_schema = properties.get(fps_parameter)
    if not isinstance(frames_schema, Mapping) or frames_schema.get("type") != "integer":
        raise ValueError("workflow video length frames_parameter must bind an integer property")
    if not isinstance(fps_schema, Mapping) or fps_schema.get("type") not in {"integer", "number"}:
        raise ValueError("workflow video length fps_parameter must bind a numeric property")
    if VIDEO_DURATION_SETTING_KEY in properties:
        raise ValueError("workflow video length reserves duration_seconds for its seconds control")
    minimum_frames = _frame_value(frames_schema, "minimum")
    maximum_frames = _frame_value(frames_schema, "maximum")
    default_frames = _frame_value(frames_schema, "default")
    if not minimum_frames <= default_frames <= maximum_frames:
        raise ValueError("workflow video length default frames must be inside the usable window")
    for name, value in (
        ("minimum", minimum_frames),
        ("maximum", maximum_frames),
        ("default", default_frames),
    ):
        if (value - offset) % alignment:
            raise ValueError(f"workflow video length frames {name} must follow its alignment")
    declared_fps = fps_schema.get("const", fps_schema.get("default"))
    if isinstance(declared_fps, bool) or not isinstance(declared_fps, (int, float)):
        raise ValueError("workflow video length FPS property must declare a default or const")
    if not math.isfinite(float(declared_fps)) or not math.isclose(
        float(declared_fps), float(fps), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("workflow video length FPS property must match the declared rational FPS")
    return WorkflowVideoLength(
        frames_parameter=frames_parameter,
        fps_parameter=fps_parameter,
        fps=fps,
        frame_alignment=alignment,
        frame_offset=offset,
        minimum_frames=minimum_frames,
        maximum_frames=maximum_frames,
        default_frames=default_frames,
    )


def video_duration_field(contract: WorkflowVideoLength) -> SettingField:
    return SettingField(
        key=VIDEO_DURATION_SETTING_KEY,
        label="Length (seconds)",
        type="number",
        default=float(contract.default_seconds),
        minimum=float(contract.minimum_seconds),
        maximum=float(contract.maximum_seconds),
        step=0.01,
        scope="workflow",
        visibility="basic",
        help=(
            "Choose a length in seconds. This workflow aligns the request to a supported "
            "frame count and shows the delivered length."
        ),
    )


def _nearest_frame_count(seconds: Fraction, contract: WorkflowVideoLength) -> int:
    target = seconds * contract.fps
    alignment = contract.frame_alignment
    offset = contract.frame_offset
    floor_value = target.numerator // target.denominator
    lower = offset + ((floor_value - offset) // alignment) * alignment
    upper = lower if target == lower else lower + alignment
    candidates = [
        value
        for value in (lower, upper)
        if contract.minimum_frames <= value <= contract.maximum_frames
    ]
    if not candidates:
        return (
            contract.minimum_frames if target < contract.minimum_frames else contract.maximum_frames
        )
    return min(candidates, key=lambda value: (abs(Fraction(value, 1) - target), value))


def resolve_video_length_settings(
    settings: Mapping[str, Any],
    input_schema: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, int | float] | None]:
    contract = workflow_video_length(input_schema)
    resolved = dict(settings)
    if contract is None or VIDEO_DURATION_SETTING_KEY not in resolved:
        return resolved, None
    requested = resolved[VIDEO_DURATION_SETTING_KEY]
    if isinstance(requested, bool) or not isinstance(requested, (int, float)):
        raise ValueError("duration_seconds must be a number")
    if not math.isfinite(float(requested)):
        raise ValueError("duration_seconds numbers must be finite")
    seconds = Fraction(str(requested))
    if seconds < contract.minimum_seconds and requested == float(contract.minimum_seconds):
        seconds = contract.minimum_seconds
    elif seconds > contract.maximum_seconds and requested == float(contract.maximum_seconds):
        seconds = contract.maximum_seconds
    if not contract.minimum_seconds <= seconds <= contract.maximum_seconds:
        raise ValueError("duration_seconds is outside this workflow's usable window")
    frames = _nearest_frame_count(seconds, contract)
    delivered = Fraction(frames, 1) / contract.fps
    resolved[contract.frames_parameter] = frames
    resolved[contract.fps_parameter] = (
        contract.fps.numerator if contract.fps.denominator == 1 else float(contract.fps)
    )
    return resolved, {
        "requested_seconds": float(seconds),
        "delivered_seconds": float(delivered),
        "frames": frames,
        "fps": float(contract.fps),
    }
