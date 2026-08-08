from __future__ import annotations

import pytest

from local_lm.comfy_version_support import (
    COMFY_VERSION_SUPPORT,
    CertifiedComfyCompatibility,
    ComfyCompatibilityContract,
    VersionInterval,
)


def test_audited_runtime_pair_is_supported() -> None:
    result = COMFY_VERSION_SUPPORT.evaluate_runtime(
        comfyui_version="0.28.0",
        frontend_version="1.45.21",
    )

    assert result.supported
    assert result.reason == "ready"
    assert result.component is None
    assert result.comfyui.reason == "ready"
    assert result.frontend.reason == "ready"


@pytest.mark.parametrize(
    ("comfyui_version", "frontend_version", "component", "reason"),
    [
        ("0.27.999", "1.45.21", "comfyui", "version-below-supported-range"),
        ("0.28.1", "1.45.21", "comfyui", "version-above-supported-range"),
        ("0.28.0", "1.45.20", "frontend", "version-below-supported-range"),
        ("0.28.0", "1.45.22", "frontend", "version-above-supported-range"),
    ],
)
def test_runtime_pair_fails_at_the_first_unsupported_component(
    comfyui_version: str,
    frontend_version: str,
    component: str,
    reason: str,
) -> None:
    result = COMFY_VERSION_SUPPORT.evaluate_runtime(
        comfyui_version=comfyui_version,
        frontend_version=frontend_version,
    )

    assert not result.supported
    assert result.component == component
    evaluated = result.comfyui if component == "comfyui" else result.frontend
    assert evaluated.reason == reason


def test_interval_bounds_are_inclusive() -> None:
    interval = VersionInterval("1.2.3", "1.2.5")

    assert interval.evaluate("1.2.3").supported
    assert interval.evaluate("1.2.4").supported
    assert interval.evaluate("1.2.5").supported
    assert interval.evaluate("1.2.2").reason == "version-below-supported-range"
    assert interval.evaluate("1.2.6").reason == "version-above-supported-range"


def test_canonical_stable_trailing_zero_spelling_keeps_pep440_equivalence() -> None:
    interval = VersionInterval("1.45.21", "1.45.21")

    result = interval.evaluate("1.45.21.0")

    assert result.supported
    assert result.actual == "1.45.21.0"


@pytest.mark.parametrize(
    "value",
    [
        "1!1.45.21",
        "1.45.21rc1",
        "1.45.21.post1",
        "1.45.21.dev1",
        "1.45.21+vendor",
    ],
)
def test_uncertified_release_kinds_are_distinct_from_invalid_versions(value: str) -> None:
    result = COMFY_VERSION_SUPPORT.evaluate_frontend_semantics(value)

    assert not result.supported
    assert result.reason == "version-uncertified-build"
    assert result.actual == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "",
        " 1.45.21",
        "1.45.21 ",
        "v1.45.21",
        "01.45.21",
        "not-a-version",
    ],
)
def test_missing_malformed_and_noncanonical_versions_fail_closed(
    value: str | None,
) -> None:
    result = COMFY_VERSION_SUPPORT.evaluate_frontend_semantics(value)

    assert not result.supported
    assert result.reason == ("version-missing" if value is None else "version-invalid")


@pytest.mark.parametrize(
    ("floor", "ceiling"),
    [
        ("1.2.4", "1.2.3"),
        ("1.2.3rc1", "1.2.3"),
        ("1.2.3", "1.2.3+vendor"),
        ("v1.2.3", "1.2.3"),
    ],
)
def test_invalid_or_reversed_support_bounds_are_refused(
    floor: str,
    ceiling: str,
) -> None:
    with pytest.raises(ValueError):
        VersionInterval(floor, ceiling)


def test_runtime_support_requires_membership_in_one_certified_pair_region() -> None:
    contract = ComfyCompatibilityContract(
        certified=(
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.28.0", "0.28.1"),
                frontend=VersionInterval("1.45.21", "1.45.22"),
            ),
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.29.0", "0.29.1"),
                frontend=VersionInterval("1.46.0", "1.46.1"),
            ),
        )
    )

    for comfyui_version in ("0.28.0", "0.28.1"):
        for frontend_version in ("1.45.21", "1.45.22"):
            assert contract.evaluate_runtime(
                comfyui_version=comfyui_version,
                frontend_version=frontend_version,
            ).supported
    for comfyui_version in ("0.29.0", "0.29.1"):
        for frontend_version in ("1.46.0", "1.46.1"):
            assert contract.evaluate_runtime(
                comfyui_version=comfyui_version,
                frontend_version=frontend_version,
            ).supported

    crossed = contract.evaluate_runtime(
        comfyui_version="0.28.1",
        frontend_version="1.46.0",
    )

    assert not crossed.supported
    assert crossed.reason == "runtime-pair-not-certified"
    assert crossed.component is None


def test_frontend_semantics_accepts_the_union_of_certified_regions() -> None:
    contract = ComfyCompatibilityContract(
        certified=(
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.28.0", "0.28.0"),
                frontend=VersionInterval("1.45.21", "1.45.21"),
            ),
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.29.0", "0.29.0"),
                frontend=VersionInterval("1.46.0", "1.46.0"),
            ),
        )
    )

    assert contract.evaluate_frontend_semantics("1.45.21").supported
    assert contract.evaluate_frontend_semantics("1.46.0").supported
    assert not contract.evaluate_frontend_semantics("1.45.22").supported


def test_runtime_refusal_retains_a_frontend_match_from_a_later_region() -> None:
    contract = ComfyCompatibilityContract(
        certified=(
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.28.0", "0.28.0"),
                frontend=VersionInterval("1.45.21", "1.45.21"),
            ),
            CertifiedComfyCompatibility(
                comfyui=VersionInterval("0.29.0", "0.29.0"),
                frontend=VersionInterval("1.46.0", "1.46.0"),
            ),
        )
    )

    result = contract.evaluate_runtime(
        comfyui_version="0.30.0",
        frontend_version="1.46.0",
    )

    assert not result.supported
    assert result.reason == "comfyui-version-unsupported"
    assert result.component == "comfyui"
    assert result.frontend.supported


def test_empty_certification_contract_is_refused() -> None:
    with pytest.raises(ValueError):
        ComfyCompatibilityContract(certified=())
