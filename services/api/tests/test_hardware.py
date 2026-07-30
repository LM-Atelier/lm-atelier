from __future__ import annotations

import local_lm.hardware as hardware
from local_lm.config import Settings


def _count_collect_calls(monkeypatch, settings: Settings) -> list[int]:  # type: ignore[no-untyped-def]
    calls = [0]
    original = hardware.collect_system_info

    def counted(value: Settings):  # type: ignore[no-untyped-def]
        calls[0] += 1
        return original(value)

    monkeypatch.setattr(hardware, "collect_system_info", counted)
    return calls


def test_capability_class_is_memoized_across_repeated_calls(
    settings: Settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """Readiness polls this once per install; device enumeration spawns processes."""
    hardware.reset_hardware_capability_class_cache()
    calls = _count_collect_calls(monkeypatch, settings)

    first = hardware.hardware_capability_class(settings)
    repeated = [hardware.hardware_capability_class(settings) for _ in range(20)]

    assert calls[0] == 1
    assert set(repeated) == {first}


def test_capability_class_is_recomputed_once_the_cache_expires(
    settings: Settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    hardware.reset_hardware_capability_class_cache()
    calls = _count_collect_calls(monkeypatch, settings)
    clock = [1000.0]
    monkeypatch.setattr(hardware, "monotonic", lambda: clock[0])

    hardware.hardware_capability_class(settings)
    clock[0] += hardware._CAPABILITY_CLASS_TTL_SECONDS + 1
    hardware.hardware_capability_class(settings)

    assert calls[0] == 2


def test_resetting_the_cache_forces_a_fresh_measurement(
    settings: Settings,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    hardware.reset_hardware_capability_class_cache()
    calls = _count_collect_calls(monkeypatch, settings)

    hardware.hardware_capability_class(settings)
    hardware.reset_hardware_capability_class_cache()
    hardware.hardware_capability_class(settings)

    assert calls[0] == 2
