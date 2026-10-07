"""The saved workspace lock setting, and the PIN verifier kept inside it."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from local_lm import workspace_lock_policy
from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import AppSetting
from local_lm.portable_archive_v1 import MIN_MEMORY_KIB, KeyDerivation
from local_lm.workspace_lock_policy import (
    POLICY_KEY,
    PinCheckUnavailable,
    PinVerifier,
    SavedWorkspaceLock,
    WorkspaceLockPolicyStale,
    WorkspaceLockSettingInvalid,
    clear_policy,
    make_pin_verifier,
    pin_matches,
    read_policy,
    write_policy,
)

PIN = "4826"


@pytest.fixture
def derivations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace Argon2id with a quick stand-in, and count what it derives."""

    counted: list[int] = []

    class QuickArgon2id:
        def __init__(
            self, *, salt: bytes, length: int, iterations: int, lanes: int, memory_cost: int
        ) -> None:
            self._salt = salt
            self._length = length

        def derive(self, key_material: bytes) -> bytes:
            counted.append(len(key_material))
            return hashlib.sha256(self._salt + key_material).digest()[: self._length]

    monkeypatch.setattr(workspace_lock_policy, "Argon2id", QuickArgon2id)
    return counted


def _store(value: Any) -> None:
    with SessionLocal() as session:
        session.merge(AppSetting(key=POLICY_KEY, value_json=value))
        session.commit()


def _read() -> SavedWorkspaceLock:
    with SessionLocal() as session:
        return read_policy(session)


def _write(expected_revision: int, *, enabled: bool) -> SavedWorkspaceLock:
    with SessionLocal() as session:
        written = write_policy(
            session, expected_revision=expected_revision, enabled=enabled, pin=None
        )
        session.commit()
        return written


def test_with_nothing_saved_the_lock_is_off_with_no_pin(settings: Settings) -> None:
    assert _read() == SavedWorkspaceLock(enabled=False, revision=0, pin=None)


def test_the_first_choice_creates_the_setting_and_a_second_from_zero_is_stale(
    settings: Settings,
) -> None:
    assert _write(0, enabled=True) == SavedWorkspaceLock(enabled=True, revision=1, pin=None)

    with SessionLocal() as session:
        with pytest.raises(WorkspaceLockPolicyStale) as stale:
            write_policy(session, expected_revision=0, enabled=False, pin=None)
        session.rollback()

    assert stale.value.current_revision == 1
    assert _read() == SavedWorkspaceLock(enabled=True, revision=1, pin=None)


def test_a_change_applies_only_at_the_revision_it_was_based_on(settings: Settings) -> None:
    _write(0, enabled=True)
    assert _write(1, enabled=False) == SavedWorkspaceLock(enabled=False, revision=2, pin=None)

    with SessionLocal() as session:
        with pytest.raises(WorkspaceLockPolicyStale) as stale:
            write_policy(session, expected_revision=1, enabled=True, pin=None)
        session.rollback()
        stored = session.get(AppSetting, POLICY_KEY)
        assert stored is not None
        assert stored.value_json == {"enabled": False, "revision": 2, "pin": None}

    assert stale.value.current_revision == 2


def test_the_reader_sees_a_change_saved_after_it_last_read(settings: Settings) -> None:
    _write(0, enabled=True)

    with SessionLocal() as reader:
        # Holding the row keeps it in the session, as a caller that had read it would.
        held = reader.get(AppSetting, POLICY_KEY)
        assert held is not None and read_policy(reader).revision == 1
        _write(1, enabled=False)
        assert read_policy(reader) == SavedWorkspaceLock(enabled=False, revision=2, pin=None)


def test_clearing_the_setting_leaves_the_lock_off(settings: Settings) -> None:
    _write(0, enabled=True)

    with SessionLocal() as session:
        clear_policy(session)
        session.commit()

    assert _read() == SavedWorkspaceLock(enabled=False, revision=0, pin=None)


@pytest.mark.parametrize(
    "document",
    [
        {"enabled": True, "revision": 1},
        {"enabled": True, "revision": 1, "pin": None, "note": "kept"},
        {"enabled": 1, "revision": 1, "pin": None},
        {"enabled": "true", "revision": 1, "pin": None},
        {"enabled": True, "revision": 0, "pin": None},
        {"enabled": True, "revision": True, "pin": None},
        {"enabled": True, "revision": "1", "pin": None},
        {"enabled": True, "revision": 1, "pin": "set"},
        [True, 1, None],
        "enabled",
    ],
)
def test_a_document_this_version_did_not_write_is_refused(
    settings: Settings, document: Any
) -> None:
    _store(document)

    with pytest.raises(WorkspaceLockSettingInvalid):
        _read()


_REMOVED = object()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("salt", "ab" * 15),
        ("salt", "AB" * 16),
        ("salt", "zz" * 16),
        ("hash", "cd" * 31),
        ("hash", "CD" * 32),
        ("memory_kib", 8),
        ("memory_kib", 4 * 1024 * 1024),
        ("iterations", 0),
        ("iterations", "3"),
        ("lanes", 99),
        ("algorithm", "argon2i-v19"),
        ("lanes", _REMOVED),
        ("note", "kept"),
    ],
)
def test_a_malformed_verifier_makes_the_setting_unreadable(
    settings: Settings, derivations: list[int], field: str, value: object
) -> None:
    verifier = make_pin_verifier(PIN).model_dump(mode="json")
    _store({"enabled": True, "revision": 1, "pin": verifier})
    assert _read().pin == PinVerifier.model_validate(verifier)

    if value is _REMOVED:
        del verifier[field]
    else:
        verifier[field] = value
    _store({"enabled": True, "revision": 1, "pin": verifier})

    with pytest.raises(WorkspaceLockSettingInvalid):
        _read()


def test_a_real_verifier_matches_its_own_pin_and_no_other(monkeypatch: pytest.MonkeyPatch) -> None:
    floor = KeyDerivation(iterations=1, lanes=1, memory_kib=MIN_MEMORY_KIB)
    monkeypatch.setattr(workspace_lock_policy, "PIN_KEY_DERIVATION", floor)

    verifier = make_pin_verifier(PIN)

    assert verifier.algorithm == "argon2id-v19"
    assert (verifier.iterations, verifier.lanes, verifier.memory_kib) == (1, 1, 65536)
    assert (len(verifier.salt), len(verifier.hash)) == (32, 64)
    assert pin_matches(verifier, PIN)
    assert not pin_matches(verifier, "4827")
    assert make_pin_verifier(PIN).salt != verifier.salt


def test_text_that_could_never_be_the_pin_costs_no_derivation(derivations: list[int]) -> None:
    verifier = make_pin_verifier(PIN)
    assert len(derivations) == 1

    for candidate in (None, "", "482", "4" * 65, "\ud800826"):
        assert not pin_matches(verifier, candidate)

    assert len(derivations) == 1
    assert pin_matches(verifier, PIN)
    assert len(derivations) == 2


@pytest.mark.parametrize("pin", ["", "482", "4" * 65, "\ud800826"])
def test_a_new_pin_must_be_four_to_sixty_four_characters(derivations: list[int], pin: str) -> None:
    with pytest.raises(ValueError, match="4 to 64 characters"):
        make_pin_verifier(pin)

    assert derivations == []


def test_a_pin_is_taken_exactly_as_typed(derivations: list[int]) -> None:
    verifier = make_pin_verifier(" 4826 ")

    assert pin_matches(verifier, " 4826 ")
    assert not pin_matches(verifier, PIN)


def test_a_derivation_that_cannot_get_its_memory_is_not_a_wrong_pin(
    monkeypatch: pytest.MonkeyPatch, derivations: list[int]
) -> None:
    verifier = make_pin_verifier(PIN)

    class NoMemory:
        def __init__(self, **_: object) -> None:
            pass

        def derive(self, key_material: bytes) -> bytes:
            raise MemoryError

    monkeypatch.setattr(workspace_lock_policy, "Argon2id", NoMemory)

    with pytest.raises(PinCheckUnavailable):
        pin_matches(verifier, PIN)
    with pytest.raises(PinCheckUnavailable):
        make_pin_verifier(PIN)
