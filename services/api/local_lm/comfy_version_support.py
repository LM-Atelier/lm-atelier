from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from packaging.version import InvalidVersion, Version

VersionSupportReason = Literal[
    "ready",
    "version-missing",
    "version-invalid",
    "version-uncertified-build",
    "version-below-supported-range",
    "version-above-supported-range",
]
ComfyVersionComponent = Literal["comfyui", "frontend"]
ComfyRuntimeSupportReason = Literal[
    "ready",
    "comfyui-version-unsupported",
    "frontend-version-unsupported",
    "runtime-pair-not-certified",
]


@dataclass(frozen=True, slots=True)
class VersionSupport:
    supported: bool
    reason: VersionSupportReason
    actual: str | None
    floor: str
    ceiling: str


@dataclass(frozen=True, slots=True)
class VersionInterval:
    """An inclusive interval of audited, canonical stable releases."""

    floor: str
    ceiling: str

    def __post_init__(self) -> None:
        floor = _stable_canonical_version(self.floor)
        ceiling = _stable_canonical_version(self.ceiling)
        if floor is None or ceiling is None:
            raise ValueError("version support bounds must be canonical stable releases")
        if floor > ceiling:
            raise ValueError("version support floor must not exceed its ceiling")

    def evaluate(self, value: str | None) -> VersionSupport:
        if value is None:
            return self._result(False, "version-missing", None)
        parsed = _stable_canonical_version(value)
        if parsed is None:
            reason: VersionSupportReason = (
                "version-uncertified-build"
                if _is_valid_uncertified_build(value)
                else "version-invalid"
            )
            return self._result(False, reason, value)
        floor = Version(self.floor)
        ceiling = Version(self.ceiling)
        if parsed < floor:
            return self._result(False, "version-below-supported-range", value)
        if parsed > ceiling:
            return self._result(False, "version-above-supported-range", value)
        return self._result(True, "ready", value)

    def _result(
        self,
        supported: bool,
        reason: VersionSupportReason,
        actual: str | None,
    ) -> VersionSupport:
        return VersionSupport(supported, reason, actual, self.floor, self.ceiling)


@dataclass(frozen=True, slots=True)
class ComfyRuntimeSupport:
    supported: bool
    reason: ComfyRuntimeSupportReason
    component: ComfyVersionComponent | None
    comfyui: VersionSupport
    frontend: VersionSupport


@dataclass(frozen=True, slots=True)
class CertifiedComfyCompatibility:
    """One rectangle whose complete core-by-frontend matrix was certified."""

    comfyui: VersionInterval
    frontend: VersionInterval


@dataclass(frozen=True, slots=True)
class ComfyCompatibilityContract:
    """The audited runtime pair and frontend semantics accepted by the product."""

    certified: tuple[CertifiedComfyCompatibility, ...]

    def __post_init__(self) -> None:
        if not self.certified:
            raise ValueError("at least one certified Comfy compatibility region is required")

    def evaluate_runtime(
        self,
        *,
        comfyui_version: str | None,
        frontend_version: str | None,
    ) -> ComfyRuntimeSupport:
        evaluations = tuple(
            (
                region.comfyui.evaluate(comfyui_version),
                region.frontend.evaluate(frontend_version),
            )
            for region in self.certified
        )
        for comfyui, frontend in evaluations:
            if comfyui.supported and frontend.supported:
                return ComfyRuntimeSupport(True, "ready", None, comfyui, frontend)
        comfyui_matches = [value for value in evaluations if value[0].supported]
        frontend_matches = [value for value in evaluations if value[1].supported]
        if comfyui_matches and frontend_matches:
            return ComfyRuntimeSupport(
                False,
                "runtime-pair-not-certified",
                None,
                comfyui_matches[0][0],
                frontend_matches[0][1],
            )
        if comfyui_matches:
            comfyui, frontend = comfyui_matches[0]
            return ComfyRuntimeSupport(
                False,
                "frontend-version-unsupported",
                "frontend",
                comfyui,
                frontend,
            )
        comfyui = evaluations[0][0]
        frontend = frontend_matches[0][1] if frontend_matches else evaluations[0][1]
        return ComfyRuntimeSupport(
            False,
            "comfyui-version-unsupported",
            "comfyui",
            comfyui,
            frontend,
        )

    def evaluate_frontend_semantics(self, frontend_version: str | None) -> VersionSupport:
        evaluations = tuple(region.frontend.evaluate(frontend_version) for region in self.certified)
        return next(
            (evaluation for evaluation in evaluations if evaluation.supported),
            evaluations[0],
        )


def _stable_canonical_version(value: str) -> Version | None:
    if not value or value.strip() != value:
        return None
    try:
        parsed = Version(value)
    except InvalidVersion:
        return None
    if (
        parsed.epoch
        or parsed.pre is not None
        or parsed.post is not None
        or parsed.dev is not None
        or parsed.local is not None
        or str(parsed) != value
    ):
        return None
    return parsed


def _is_valid_uncertified_build(value: str) -> bool:
    if not value or value.strip() != value:
        return False
    try:
        parsed = Version(value)
    except InvalidVersion:
        return False
    return bool(
        parsed.epoch
        or parsed.pre is not None
        or parsed.post is not None
        or parsed.dev is not None
        or parsed.local is not None
    )


# These inclusive bounds describe the only runtime pair currently certified by
# the managed-runtime and native-editor matrix. Add or widen a region only after
# every core-by-frontend combination inside that rectangle passes the matrix.
COMFY_VERSION_SUPPORT = ComfyCompatibilityContract(
    certified=(
        CertifiedComfyCompatibility(
            comfyui=VersionInterval("0.28.0", "0.28.0"),
            frontend=VersionInterval("1.45.21", "1.45.21"),
        ),
    ),
)
