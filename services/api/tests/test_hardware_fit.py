from __future__ import annotations

import pytest

from local_lm.hardware_fit import (
    AcceleratorCapacity,
    BoundedSetting,
    FitEvidence,
    FitRequirements,
    HardwareCapacity,
    recommend_hardware_fit,
)

_GIB = 1024**3


def _capacity(
    *,
    system: int = 32 * _GIB,
    system_free: int | None = 24 * _GIB,
    accelerator: int = 16 * _GIB,
    accelerator_free: int | None = 14 * _GIB,
    accelerator_backends: tuple[str, ...] = ("cuda",),
    runtime_backends: tuple[str, ...] = ("llama.cpp", "comfyui"),
) -> HardwareCapacity:
    return HardwareCapacity(
        platform="windows",
        architecture="amd64",
        cpu_model="Example CPU",
        cpu_capabilities=("avx2",),
        system_memory_bytes=system,
        system_memory_available_bytes=system_free,
        accelerators=tuple(
            AcceleratorCapacity(
                backend=backend,
                name=f"Example {backend} GPU",
                total_memory_bytes=accelerator,
                available_memory_bytes=accelerator_free,
            )
            for backend in accelerator_backends
        ),
        runtime_backends=runtime_backends,
    )


def test_declared_recommended_capacity_is_recommended() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(
            supported_platforms=("Windows",),
            required_runtime_backends=("COMFYUI",),
            required_accelerator_backends=("CUDA",),
            minimum_system_memory_bytes=16 * _GIB,
            recommended_system_memory_bytes=24 * _GIB,
            minimum_accelerator_memory_bytes=8 * _GIB,
            recommended_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert result.basis == "declared"
    assert result.evidence_label is None


def test_calculated_fit_is_likely_and_never_claims_tested() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=10 * _GIB),
    )

    assert result.status == "likely"
    assert result.basis == "calculated"
    assert result.evidence_label is None


def test_calculated_fit_with_little_headroom_is_tight() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=15 * _GIB),
    )

    assert result.status == "tight"
    assert any(reason.code == "accelerator_memory_estimated" for reason in result.reasons)
    assert any(item.code == "choose_smaller_variant" for item in result.alternatives)


def test_declared_minimum_that_cannot_fit_is_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(minimum_accelerator_memory_bytes=20 * _GIB),
    )

    assert result.status == "unsupported"
    assert result.basis == "declared"
    assert any(reason.severity == "block" for reason in result.reasons)


def test_missing_estimated_accelerator_capacity_is_unknown_not_unsupported() -> None:
    result = recommend_hardware_fit(
        _capacity(
            accelerator=0,
            accelerator_free=None,
            accelerator_backends=(),
        ),
        FitRequirements(estimated_accelerator_memory_bytes=8 * _GIB),
    )

    assert result.status == "unknown"
    assert result.basis == "unknown"


def test_current_free_memory_is_a_warning_not_general_fit_downgrade() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator_free=4 * _GIB),
        FitRequirements(
            minimum_accelerator_memory_bytes=8 * _GIB,
            recommended_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert any(reason.code == "accelerator_memory_busy" for reason in result.reasons)
    assert any(item.code == "free_current_memory" for item in result.alternatives)


def test_stale_evidence_is_ignored() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(estimated_accelerator_memory_bytes=15 * _GIB),
        evidence=FitEvidence(
            exact_match=False,
            claim="certified",
            peak_accelerator_memory_bytes=6 * _GIB,
        ),
    )

    assert result.status == "tight"
    assert result.basis == "calculated"
    assert result.evidence_label is None
    assert any(reason.code == "evidence_stale" for reason in result.reasons)


def test_exact_test_evidence_can_be_labeled_tested() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=True,
            claim="tested",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )

    assert result.status == "recommended"
    assert result.basis == "tested"
    assert result.evidence_label == "tested"


def test_exact_certification_label_requires_matching_evidence() -> None:
    exact = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=True,
            claim="certified",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )
    stale = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(
            exact_match=False,
            claim="certified",
            peak_accelerator_memory_bytes=10 * _GIB,
        ),
    )

    assert exact.evidence_label == "certified"
    assert stale.evidence_label is None
    assert stale.status == "unknown"


def test_exact_recorded_matrix_can_be_labeled_without_a_memory_peak() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(exact_match=True, claim="certified"),
    )

    assert result.status == "recommended"
    assert result.basis == "certified"
    assert result.evidence_label == "certified"


def test_unmeasured_non_test_evidence_does_not_make_a_fit_claim() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(),
        evidence=FitEvidence(exact_match=True, claim="measured"),
    )

    assert result.status == "unknown"
    assert result.basis == "unknown"
    assert result.evidence_label is None


def test_required_backends_fail_closed_without_model_name_branches() -> None:
    result = recommend_hardware_fit(
        _capacity(accelerator_backends=("directml",), runtime_backends=("llama.cpp",)),
        FitRequirements(
            required_runtime_backends=("comfyui",),
            required_accelerator_backends=("cuda",),
        ),
    )

    assert result.status == "unsupported"
    assert {reason.code for reason in result.reasons} >= {
        "runtime_backend_missing",
        "accelerator_backend_missing",
    }


def test_memory_is_taken_from_the_compatible_device_not_another_backend() -> None:
    capacity = HardwareCapacity(
        platform="windows",
        architecture="amd64",
        cpu_model="Example CPU",
        cpu_capabilities=("avx2",),
        system_memory_bytes=32 * _GIB,
        system_memory_available_bytes=24 * _GIB,
        accelerators=(
            AcceleratorCapacity(
                backend="directml",
                name="Large DirectML GPU",
                total_memory_bytes=24 * _GIB,
                available_memory_bytes=22 * _GIB,
            ),
            AcceleratorCapacity(
                backend="cuda",
                name="Small CUDA GPU",
                total_memory_bytes=8 * _GIB,
                available_memory_bytes=8 * _GIB,
            ),
        ),
        runtime_backends=("comfyui",),
    )

    result = recommend_hardware_fit(
        capacity,
        FitRequirements(
            required_accelerator_backends=("cuda",),
            minimum_accelerator_memory_bytes=12 * _GIB,
        ),
    )

    assert result.status == "unsupported"
    assert any(reason.code == "accelerator_memory_below_minimum" for reason in result.reasons)


def test_tight_fit_uses_bounded_safer_ranges_without_overwriting_user_choice() -> None:
    result = recommend_hardware_fit(
        _capacity(),
        FitRequirements(
            estimated_accelerator_memory_bytes=15 * _GIB,
            settings=(
                BoundedSetting(
                    key="resolution",
                    label="Resolution",
                    unit="pixels",
                    minimum=512,
                    maximum=2048,
                    preferred_minimum=768,
                    preferred_maximum=1024,
                    tight_minimum=512,
                    tight_maximum=768,
                    user_override=1536,
                ),
            ),
        ),
    )

    assert result.status == "tight"
    assert result.settings[0].minimum == 512
    assert result.settings[0].maximum == 768
    assert result.settings[0].advisory_only is True
    assert result.settings[0].preserves_user_override is True


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError, match="escape declared bounds"):
        recommend_hardware_fit(
            _capacity(),
            FitRequirements(
                settings=(
                    BoundedSetting(
                        key="steps",
                        label="Steps",
                        unit="steps",
                        minimum=1,
                        maximum=10,
                        preferred_minimum=4,
                        preferred_maximum=12,
                        tight_minimum=1,
                        tight_maximum=4,
                    ),
                )
            ),
        )
