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


def test_every_device_kind_produced_here_is_in_the_declared_vocabulary() -> None:
    """DeviceKind is declared in schemas.py but produced only in this module.

    There is no stored column behind DeviceInfo.kind and therefore no database
    constraint to derive the vocabulary from, so nothing else would notice a
    fourth kind being introduced here. The API would keep advertising three and
    serialising a fourth, which is a serialization failure on read rather than
    an odd string passed through - the same trap the vocabulary gate exists for.

    This reads the literals out of the producing source rather than restating
    them, because a restated copy is exactly what the declaration removed.

    The comparison is EQUALITY, not containment. An earlier version asserted
    only that everything produced was declared, which let the public server and
    browser protocols jointly advertise a device kind the sole producer never
    emits - an invented member passed untouched. That is the same false-member
    drift the contract test rejects one layer later with its declared-minus-served
    assertion, so this refuses in both directions and names which side is wrong.
    """
    import ast
    import inspect
    from typing import get_args

    from local_lm.schemas import DeviceKind

    tree = ast.parse(inspect.getsource(hardware))
    produced: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "DeviceInfo":
            continue
        for keyword in node.keywords:
            if keyword.arg == "kind":
                assert isinstance(keyword.value, ast.Constant), (
                    f"DeviceInfo at line {node.lineno} computes its kind, so this "
                    "test can no longer prove the vocabulary from the source"
                )
                produced.add(keyword.value.value)

    assert produced, "no DeviceInfo construction found; this test is measuring nothing"

    declared = set(get_args(DeviceKind))
    invented = sorted(declared - produced)
    missing = sorted(produced - declared)
    assert not missing and not invented, (
        "DeviceKind and hardware.py disagree about the device vocabulary. "
        f"Produced here but not declared: {missing or 'none'}. "
        f"Declared but never produced: {invented or 'none'}."
    )
