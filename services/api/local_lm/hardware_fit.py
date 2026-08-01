from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FitStatus = Literal["recommended", "likely", "tight", "unsupported", "unknown"]
FitBasis = Literal["unknown", "calculated", "declared", "measured", "tested", "certified"]
ReasonSeverity = Literal["info", "warning", "block"]

_LIKELY_UTILIZATION_LIMIT = 0.90
_RECOMMENDED_MEASURED_UTILIZATION_LIMIT = 0.80


@dataclass(frozen=True, slots=True)
class AcceleratorCapacity:
    backend: str
    name: str
    total_memory_bytes: int
    available_memory_bytes: int | None


@dataclass(frozen=True, slots=True)
class HardwareCapacity:
    platform: str
    architecture: str
    cpu_model: str
    cpu_capabilities: tuple[str, ...]
    system_memory_bytes: int
    system_memory_available_bytes: int | None
    accelerators: tuple[AcceleratorCapacity, ...]
    runtime_backends: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundedSetting:
    key: str
    label: str
    unit: str
    minimum: int
    maximum: int
    preferred_minimum: int
    preferred_maximum: int
    tight_minimum: int
    tight_maximum: int
    user_override: int | None = None


@dataclass(frozen=True, slots=True)
class FitEvidence:
    exact_match: bool
    claim: Literal["measured", "tested", "certified"] = "measured"
    peak_system_memory_bytes: int | None = None
    peak_accelerator_memory_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class FitRequirements:
    supported_platforms: tuple[str, ...] = ()
    supported_architectures: tuple[str, ...] = ()
    required_runtime_backends: tuple[str, ...] = ()
    required_accelerator_backends: tuple[str, ...] = ()
    minimum_system_memory_bytes: int | None = None
    recommended_system_memory_bytes: int | None = None
    estimated_system_memory_bytes: int | None = None
    minimum_accelerator_memory_bytes: int | None = None
    recommended_accelerator_memory_bytes: int | None = None
    estimated_accelerator_memory_bytes: int | None = None
    concurrent_system_memory_bytes: int = 0
    concurrent_accelerator_memory_bytes: int = 0
    settings: tuple[BoundedSetting, ...] = ()


@dataclass(frozen=True, slots=True)
class FitReason:
    code: str
    severity: ReasonSeverity
    message: str


@dataclass(frozen=True, slots=True)
class FitAlternative:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SettingRecommendation:
    key: str
    label: str
    unit: str
    minimum: int
    maximum: int
    advisory_only: bool
    preserves_user_override: bool


@dataclass(frozen=True, slots=True)
class HardwareFit:
    status: FitStatus
    basis: FitBasis
    evidence_label: Literal["tested", "certified"] | None
    reasons: tuple[FitReason, ...]
    alternatives: tuple[FitAlternative, ...]
    settings: tuple[SettingRecommendation, ...]


@dataclass(frozen=True, slots=True)
class _MemoryAssessment:
    status: FitStatus
    basis: FitBasis
    reasons: tuple[FitReason, ...]
    immediate_pressure: bool


def recommend_hardware_fit(
    capacity: HardwareCapacity,
    requirements: FitRequirements,
    *,
    evidence: FitEvidence | None = None,
) -> HardwareFit:
    """Return an advisory fit without weakening compatibility enforcement.

    The caller is responsible for deciding whether recorded evidence still
    matches the exact model, runtime, workflow, driver, and hardware identities.
    Evidence is used here only when ``exact_match`` is true.
    """

    _validate_capacity(capacity)
    _validate_requirements(requirements)
    reasons: list[FitReason] = []
    alternatives: list[FitAlternative] = []

    if requirements.supported_platforms and not _matches(
        capacity.platform, requirements.supported_platforms
    ):
        reasons.append(
            FitReason(
                "platform_unsupported",
                "block",
                "This runtime does not support the current operating system.",
            )
        )
    if requirements.supported_architectures and not _matches(
        capacity.architecture, requirements.supported_architectures
    ):
        reasons.append(
            FitReason(
                "architecture_unsupported",
                "block",
                "This runtime does not support the current processor architecture.",
            )
        )
    if requirements.required_runtime_backends and not _overlaps(
        capacity.runtime_backends, requirements.required_runtime_backends
    ):
        reasons.append(
            FitReason(
                "runtime_backend_missing",
                "block",
                "A required local runtime backend is not installed or available.",
            )
        )
        alternatives.append(
            FitAlternative(
                "install_supported_runtime",
                "Install a supported local runtime backend before using this model.",
            )
        )
    available_accelerator_backends = tuple(item.backend for item in capacity.accelerators)
    if requirements.required_accelerator_backends and not _overlaps(
        available_accelerator_backends,
        requirements.required_accelerator_backends,
    ):
        reasons.append(
            FitReason(
                "accelerator_backend_missing",
                "block",
                "No compatible accelerator backend was detected.",
            )
        )
        alternatives.append(
            FitAlternative(
                "choose_compatible_backend",
                "Use a build or model variant that supports an available accelerator.",
            )
        )

    matching_evidence = evidence if evidence and evidence.exact_match else None
    if evidence and not evidence.exact_match:
        reasons.append(
            FitReason(
                "evidence_stale",
                "info",
                "Earlier measurements do not match this exact setup and were not reused.",
            )
        )

    accelerator_capacity = _best_accelerator(
        capacity.accelerators,
        requirements.required_accelerator_backends,
    )
    system = _assess_memory(
        kind="system",
        capacity_bytes=capacity.system_memory_bytes,
        available_bytes=capacity.system_memory_available_bytes,
        minimum_bytes=requirements.minimum_system_memory_bytes,
        recommended_bytes=requirements.recommended_system_memory_bytes,
        estimated_bytes=requirements.estimated_system_memory_bytes,
        concurrent_bytes=requirements.concurrent_system_memory_bytes,
        measured_peak_bytes=(
            matching_evidence.peak_system_memory_bytes if matching_evidence else None
        ),
        evidence_claim=matching_evidence.claim if matching_evidence else None,
    )
    accelerator = _assess_memory(
        kind="accelerator",
        capacity_bytes=(accelerator_capacity.total_memory_bytes if accelerator_capacity else 0),
        available_bytes=(
            accelerator_capacity.available_memory_bytes if accelerator_capacity else None
        ),
        minimum_bytes=requirements.minimum_accelerator_memory_bytes,
        recommended_bytes=requirements.recommended_accelerator_memory_bytes,
        estimated_bytes=requirements.estimated_accelerator_memory_bytes,
        concurrent_bytes=requirements.concurrent_accelerator_memory_bytes,
        measured_peak_bytes=(
            matching_evidence.peak_accelerator_memory_bytes if matching_evidence else None
        ),
        evidence_claim=matching_evidence.claim if matching_evidence else None,
    )
    reasons.extend(system.reasons)
    reasons.extend(accelerator.reasons)

    hard_block = any(reason.severity == "block" for reason in reasons)
    has_memory_requirements = any(
        value is not None
        for value in (
            requirements.minimum_system_memory_bytes,
            requirements.recommended_system_memory_bytes,
            requirements.estimated_system_memory_bytes,
            requirements.minimum_accelerator_memory_bytes,
            requirements.recommended_accelerator_memory_bytes,
            requirements.estimated_accelerator_memory_bytes,
            matching_evidence.peak_system_memory_bytes if matching_evidence else None,
            matching_evidence.peak_accelerator_memory_bytes if matching_evidence else None,
        )
    )
    test_claim: Literal["tested", "certified"] | None = None
    if matching_evidence is not None:
        if matching_evidence.claim == "tested":
            test_claim = "tested"
        elif matching_evidence.claim == "certified":
            test_claim = "certified"
    exact_test_claim = test_claim is not None
    if hard_block:
        status: FitStatus = "unsupported"
    elif not has_memory_requirements:
        status = "recommended" if exact_test_claim else "unknown"
    else:
        status = _worst_status(system.status, accelerator.status)

    if system.immediate_pressure or accelerator.immediate_pressure:
        alternatives.append(
            FitAlternative(
                "free_current_memory",
                "Close other GPU-heavy apps or unload idle models before starting.",
            )
        )
    if status in {"tight", "unsupported"}:
        alternatives.append(
            FitAlternative(
                "choose_smaller_variant",
                "Choose a smaller or more compressed model variant.",
            )
        )

    setting_recommendations = _setting_recommendations(requirements.settings, status)
    if status == "tight" and setting_recommendations:
        alternatives.append(
            FitAlternative(
                "use_safer_settings",
                "Use the suggested lower-cost settings for the first run.",
            )
        )

    basis = _strongest_basis(system.basis, accelerator.basis)
    evidence_label: Literal["tested", "certified"] | None = None
    if test_claim is not None and not hard_block:
        evidence_label = test_claim
        basis = _strongest_basis(basis, test_claim)
    return HardwareFit(
        status=status,
        basis=basis,
        evidence_label=evidence_label,
        reasons=tuple(reasons),
        alternatives=_deduplicate_alternatives(alternatives),
        settings=setting_recommendations,
    )


def _assess_memory(
    *,
    kind: Literal["system", "accelerator"],
    capacity_bytes: int,
    available_bytes: int | None,
    minimum_bytes: int | None,
    recommended_bytes: int | None,
    estimated_bytes: int | None,
    concurrent_bytes: int,
    measured_peak_bytes: int | None,
    evidence_claim: Literal["measured", "tested", "certified"] | None,
) -> _MemoryAssessment:
    label = "system memory" if kind == "system" else "accelerator memory"
    reasons: list[FitReason] = []
    basis: FitBasis = "unknown"
    demanded = [
        value
        for value in (minimum_bytes, recommended_bytes, estimated_bytes, measured_peak_bytes)
        if value is not None
    ]
    if not demanded:
        return _MemoryAssessment("recommended", basis, (), False)

    if minimum_bytes is not None and minimum_bytes + concurrent_bytes > capacity_bytes:
        reasons.append(
            FitReason(
                f"{kind}_memory_below_minimum",
                "block",
                f"Available hardware is below the declared minimum {label} requirement.",
            )
        )
        return _MemoryAssessment("unsupported", "declared", tuple(reasons), False)

    load_bytes = max(demanded) + concurrent_bytes
    if capacity_bytes <= 0:
        reasons.append(
            FitReason(
                f"{kind}_memory_unknown",
                "warning",
                f"The required {label} capacity could not be confirmed.",
            )
        )
        return _MemoryAssessment("unknown", basis, tuple(reasons), False)

    utilization = load_bytes / capacity_bytes
    if measured_peak_bytes is not None and evidence_claim is not None:
        basis = evidence_claim
        status: FitStatus = (
            "recommended" if utilization <= _RECOMMENDED_MEASURED_UTILIZATION_LIMIT else "tight"
        )
        reasons.append(
            FitReason(
                f"{kind}_memory_measured",
                "info" if status == "recommended" else "warning",
                f"An exact matching run measured this setup's {label} use.",
            )
        )
    elif recommended_bytes is not None:
        basis = "declared"
        if recommended_bytes + concurrent_bytes <= capacity_bytes:
            status = "recommended"
            message = f"Hardware meets the declared recommended {label} capacity."
        else:
            status = "tight"
            message = f"Hardware meets the minimum but not the recommended {label} capacity."
        reasons.append(
            FitReason(
                f"{kind}_memory_declared",
                "info" if status == "recommended" else "warning",
                message,
            )
        )
    else:
        basis = "calculated"
        status = "likely" if utilization <= _LIKELY_UTILIZATION_LIMIT else "tight"
        reasons.append(
            FitReason(
                f"{kind}_memory_estimated",
                "info" if status == "likely" else "warning",
                (
                    f"The calculated {label} estimate leaves practical headroom."
                    if status == "likely"
                    else f"The calculated {label} estimate leaves little headroom."
                ),
            )
        )

    immediate_pressure = available_bytes is not None and load_bytes > available_bytes
    if immediate_pressure:
        reasons.append(
            FitReason(
                f"{kind}_memory_busy",
                "warning",
                f"The model fits total {label}, but currently free capacity is lower.",
            )
        )
    return _MemoryAssessment(status, basis, tuple(reasons), immediate_pressure)


def _setting_recommendations(
    settings: tuple[BoundedSetting, ...],
    status: FitStatus,
) -> tuple[SettingRecommendation, ...]:
    recommendations: list[SettingRecommendation] = []
    for setting in settings:
        _validate_setting(setting)
        if status == "tight":
            lower = setting.tight_minimum
            upper = setting.tight_maximum
        else:
            lower = setting.preferred_minimum
            upper = setting.preferred_maximum
        recommendations.append(
            SettingRecommendation(
                key=setting.key,
                label=setting.label,
                unit=setting.unit,
                minimum=lower,
                maximum=upper,
                advisory_only=True,
                preserves_user_override=setting.user_override is not None,
            )
        )
    return tuple(recommendations)


def _matches(value: str, expected: tuple[str, ...]) -> bool:
    return value.casefold() in {item.casefold() for item in expected}


def _overlaps(available: tuple[str, ...], required: tuple[str, ...]) -> bool:
    normalized = {item.casefold() for item in available}
    return bool(normalized & {item.casefold() for item in required})


def _best_accelerator(
    accelerators: tuple[AcceleratorCapacity, ...],
    required_backends: tuple[str, ...],
) -> AcceleratorCapacity | None:
    required = {item.casefold() for item in required_backends}
    compatible = [
        item for item in accelerators if not required or item.backend.casefold() in required
    ]
    return max(
        compatible,
        key=lambda item: (item.total_memory_bytes, item.available_memory_bytes or 0),
        default=None,
    )


def _worst_status(first: FitStatus, second: FitStatus) -> FitStatus:
    order: dict[FitStatus, int] = {
        "recommended": 0,
        "likely": 1,
        "tight": 2,
        "unknown": 3,
        "unsupported": 4,
    }
    return first if order[first] >= order[second] else second


def _strongest_basis(first: FitBasis, second: FitBasis) -> FitBasis:
    order: dict[FitBasis, int] = {
        "unknown": 0,
        "calculated": 1,
        "declared": 2,
        "measured": 3,
        "tested": 4,
        "certified": 5,
    }
    return first if order[first] >= order[second] else second


def _deduplicate_alternatives(
    alternatives: list[FitAlternative],
) -> tuple[FitAlternative, ...]:
    unique: dict[str, FitAlternative] = {}
    for alternative in alternatives:
        unique.setdefault(alternative.code, alternative)
    return tuple(unique.values())


def _validate_capacity(capacity: HardwareCapacity) -> None:
    values = (
        capacity.system_memory_bytes,
        capacity.system_memory_available_bytes,
    )
    if any(value is not None and value < 0 for value in values):
        raise ValueError("Hardware memory values must not be negative.")
    if any(
        item.total_memory_bytes < 0
        or (item.available_memory_bytes is not None and item.available_memory_bytes < 0)
        for item in capacity.accelerators
    ):
        raise ValueError("Accelerator memory values must not be negative.")


def _validate_requirements(requirements: FitRequirements) -> None:
    values = (
        requirements.minimum_system_memory_bytes,
        requirements.recommended_system_memory_bytes,
        requirements.estimated_system_memory_bytes,
        requirements.minimum_accelerator_memory_bytes,
        requirements.recommended_accelerator_memory_bytes,
        requirements.estimated_accelerator_memory_bytes,
        requirements.concurrent_system_memory_bytes,
        requirements.concurrent_accelerator_memory_bytes,
    )
    if any(value is not None and value < 0 for value in values):
        raise ValueError("Hardware requirements must not be negative.")
    pairs = (
        (
            requirements.minimum_system_memory_bytes,
            requirements.recommended_system_memory_bytes,
        ),
        (
            requirements.minimum_accelerator_memory_bytes,
            requirements.recommended_accelerator_memory_bytes,
        ),
    )
    if any(
        minimum is not None and recommended is not None and recommended < minimum
        for minimum, recommended in pairs
    ):
        raise ValueError("Recommended memory must not be below the declared minimum.")


def _validate_setting(setting: BoundedSetting) -> None:
    ranges = (
        (setting.minimum, setting.maximum),
        (setting.preferred_minimum, setting.preferred_maximum),
        (setting.tight_minimum, setting.tight_maximum),
    )
    if any(lower > upper for lower, upper in ranges):
        raise ValueError(f"Invalid bounded setting range: {setting.key}")
    if not (
        setting.minimum <= setting.preferred_minimum <= setting.preferred_maximum <= setting.maximum
        and setting.minimum <= setting.tight_minimum <= setting.tight_maximum <= setting.maximum
    ):
        raise ValueError(f"Suggested ranges escape declared bounds: {setting.key}")
