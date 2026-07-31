"""A capability proof should survive a machine that has not changed.

`hardware_capability_class` hashes the CPU model, total memory and the device
list - and that device list is only populated when `llama-server` and
`nvidia-smi` resolve on PATH. A driver update or a PATH change therefore threw
away every proof on hardware that was exactly as capable as before. These cover
the envelope that replaces that equality test.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from local_lm.capability_evidence import _hardware_still_sufficient
from local_lm.config import Settings
from local_lm.hardware import (
    HARDWARE_ENVELOPE_VERSION,
    hardware_envelope_satisfied,
)

# The class is computed from the real machine; these tests only care that it
# differs from the recorded one, which any fixed string guarantees.
_SETTINGS = Settings()

_GIB = 1024**3


def _envelope(**overrides: Any) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "version": HARDWARE_ENVELOPE_VERSION,
        "platform": "windows",
        "architecture": "amd64",
        "memory_bytes": 64 * _GIB,
        "accelerator_memory_bytes": 16 * _GIB,
        "backends": ["cuda"],
    }
    envelope.update(overrides)
    return envelope


def test_the_same_machine_still_satisfies_its_own_proof() -> None:
    assert hardware_envelope_satisfied(_envelope(), _envelope())


def test_a_changed_capability_class_no_longer_discards_the_proof() -> None:
    """The regression this exists to prevent.

    A driver update changes the reported device name, which changes the
    capability-class hash, while leaving the machine exactly as able to run the
    model. Previously the equality test alone rejected the evidence.
    """
    evidence = SimpleNamespace(
        hardware_class="windows-amd64-beforethedriverupdate",
        hardware_envelope_json=_envelope(),
    )

    assert _hardware_still_sufficient(evidence, _SETTINGS, _envelope())


def test_a_row_with_no_envelope_still_needs_an_exact_capability_class() -> None:
    """Rows written before envelopes existed must behave exactly as before."""
    legacy = SimpleNamespace(
        hardware_class="windows-amd64-somethingelse",
        hardware_envelope_json=None,
    )

    assert not _hardware_still_sufficient(legacy, _SETTINGS, _envelope())


def test_an_envelope_the_machine_cannot_meet_is_still_rejected() -> None:
    """The equality test is a fast accept, not a way around the envelope."""
    evidence = SimpleNamespace(
        hardware_class="windows-amd64-somethingelse",
        hardware_envelope_json=_envelope(accelerator_memory_bytes=48 * _GIB),
    )

    assert not _hardware_still_sufficient(evidence, _SETTINGS, _envelope())


def test_more_capable_hardware_inherits_the_proof() -> None:
    upgraded = _envelope(memory_bytes=128 * _GIB, accelerator_memory_bytes=24 * _GIB)

    assert hardware_envelope_satisfied(_envelope(), upgraded)


def test_less_memory_does_not_inherit_the_proof() -> None:
    assert not hardware_envelope_satisfied(_envelope(), _envelope(memory_bytes=32 * _GIB))


def test_a_smaller_accelerator_does_not_inherit_the_proof() -> None:
    smaller = _envelope(accelerator_memory_bytes=8 * _GIB)

    assert not hardware_envelope_satisfied(_envelope(), smaller)


def test_losing_the_accelerator_entirely_does_not_inherit_the_proof() -> None:
    cpu_only = _envelope(accelerator_memory_bytes=0, backends=[])

    assert not hardware_envelope_satisfied(_envelope(), cpu_only)


def test_a_proof_that_needed_no_accelerator_still_applies_to_a_machine_with_one() -> None:
    cpu_proof = _envelope(accelerator_memory_bytes=0, backends=[])

    assert hardware_envelope_satisfied(cpu_proof, _envelope())


def test_a_different_backend_does_not_inherit_the_proof() -> None:
    """CUDA-proven does not mean ROCm-proven, whatever the memory says."""
    other = _envelope(backends=["rocm"])

    assert not hardware_envelope_satisfied(_envelope(), other)


def test_another_platform_or_architecture_never_inherits() -> None:
    assert not hardware_envelope_satisfied(_envelope(), _envelope(platform="linux"))
    assert not hardware_envelope_satisfied(_envelope(), _envelope(architecture="arm64"))


def test_a_missing_or_unreadable_envelope_never_satisfies() -> None:
    """Absent evidence must never read as satisfied evidence."""
    assert not hardware_envelope_satisfied(None, _envelope())
    assert not hardware_envelope_satisfied({}, _envelope())
    # A future envelope version is not comparable and must not be guessed at.
    assert not hardware_envelope_satisfied(
        _envelope(version=HARDWARE_ENVELOPE_VERSION + 1), _envelope()
    )
