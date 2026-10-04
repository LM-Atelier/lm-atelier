"""The saved workspace lock setting, and the verifier that checks its PIN.

The setting is one document in the application settings table: whether the lock
is on, the revision every change is checked against, and, once a PIN is set, a
salted Argon2id verifier of it. The PIN itself is never stored, and no answer
the API gives says more about the verifier than whether there is one.

A verifier records the cost it was derived at, so one made today still checks
after the default changes. A saved cost is used only inside the bounds portable
archives accept. Anything else makes the document unreadable rather than
derivable, because an edited document could otherwise ask for a derivation
cheap enough to make guessing easy, or one large enough to exhaust memory.
"""

from __future__ import annotations

import hmac
import os
from typing import Any, Final, Literal, Self

from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)
from sqlalchemy import delete, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from sqlalchemy.sql.dml import Insert, Update

from .domain import utcnow
from .models import AppSetting
from .portable_archive_v1 import DEFAULT_KEY_DERIVATION, KEY_BYTES, SALT_BYTES, KeyDerivation

POLICY_KEY = "workspace_lock"
PIN_ALGORITHM: Final = "argon2id-v19"
MIN_PIN_CHARACTERS = 4
MAX_PIN_CHARACTERS = 64
#: The cost of every new verifier. It is read at each use, so a test can lower
#: it to the floor the bounds allow.
PIN_KEY_DERIVATION = DEFAULT_KEY_DERIVATION

_DOCUMENT_KEYS = frozenset({"enabled", "revision", "pin"})
_LOWER_HEX = r"^[0-9a-f]+$"
#: A verifier's digest written as lowercase hex.
_DIGEST_HEX_CHARACTERS = 2 * KEY_BYTES
_UNREADABLE = "The saved workspace lock setting cannot be read."


class WorkspaceLockSettingInvalid(Exception):
    """The saved setting is not one this version writes, so it cannot be trusted."""


class PinCheckUnavailable(Exception):
    """The derivation could not get its memory, which says nothing about the PIN."""


class WorkspaceLockPolicyStale(Exception):
    """The setting changed after the writer read it."""

    def __init__(self, current_revision: int) -> None:
        super().__init__(current_revision)
        self.current_revision = current_revision


class PinVerifier(BaseModel):
    """A salted Argon2id digest of the PIN, and the cost it was derived at."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["argon2id-v19"]
    iterations: StrictInt
    lanes: StrictInt
    memory_kib: StrictInt
    salt: StrictStr = Field(
        pattern=_LOWER_HEX, min_length=2 * SALT_BYTES, max_length=2 * SALT_BYTES
    )
    hash: StrictStr = Field(
        pattern=_LOWER_HEX, min_length=_DIGEST_HEX_CHARACTERS, max_length=_DIGEST_HEX_CHARACTERS
    )

    @property
    def derivation(self) -> KeyDerivation:
        return KeyDerivation(
            iterations=self.iterations, lanes=self.lanes, memory_kib=self.memory_kib
        )

    @model_validator(mode="after")
    def cost_within_bounds(self) -> Self:
        if not self.derivation.within_limits():
            raise ValueError("The PIN verifier's cost is outside the accepted bounds.")
        return self


class SavedWorkspaceLock(BaseModel):
    """The saved setting. With nothing saved it is off, with no PIN, at revision 0."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: StrictBool = False
    revision: StrictInt = Field(default=0, ge=0)
    pin: PinVerifier | None = None


def read_policy(session: Session) -> SavedWorkspaceLock:
    """The saved setting, read from the database rather than from this session's memory."""

    row = session.get(AppSetting, POLICY_KEY, populate_existing=True)
    if row is None:
        return SavedWorkspaceLock()
    value: Any = row.value_json
    if not isinstance(value, dict) or set(value) != _DOCUMENT_KEYS:
        raise WorkspaceLockSettingInvalid(_UNREADABLE)
    try:
        policy = SavedWorkspaceLock.model_validate(value)
    except ValidationError:
        raise WorkspaceLockSettingInvalid(_UNREADABLE) from None
    if policy.revision < 1:
        raise WorkspaceLockSettingInvalid(_UNREADABLE)
    return policy


def write_policy(
    session: Session, *, expected_revision: int, enabled: bool, pin: PinVerifier | None
) -> SavedWorkspaceLock:
    """Save a setting if it is still at `expected_revision`. The calling route commits."""

    policy = SavedWorkspaceLock(enabled=enabled, revision=expected_revision + 1, pin=pin)
    document = policy.model_dump(mode="json")
    now = utcnow()
    statement: Insert | Update
    if expected_revision == 0:
        # The first choice creates the row. One that already exists means
        # another writer chose first, which the statement itself refuses.
        statement = (
            sqlite_insert(AppSetting)
            .values(key=POLICY_KEY, value_json=document, created_at=now, updated_at=now)
            .on_conflict_do_nothing(index_elements=["key"])
        )
    else:
        # The revision check and the write are one statement, so two writers
        # that read the same revision cannot both succeed.
        statement = (
            update(AppSetting)
            .where(
                AppSetting.key == POLICY_KEY,
                AppSetting.value_json["revision"].as_integer() == expected_revision,
            )
            .values(value_json=document, updated_at=now)
            .execution_options(synchronize_session=False)
        )
    if getattr(session.execute(statement), "rowcount", 0) != 1:
        raise WorkspaceLockPolicyStale(_saved_revision(session))
    return policy


def _saved_revision(session: Session) -> int:
    row = session.get(AppSetting, POLICY_KEY, populate_existing=True)
    value: Any = row.value_json if row is not None else None
    revision = value.get("revision") if isinstance(value, dict) else None
    if isinstance(revision, int) and not isinstance(revision, bool) and revision >= 0:
        return revision
    return 0


def clear_policy(session: Session) -> None:
    """Forget the saved setting and its PIN, which leaves the lock off. The caller commits."""

    session.execute(delete(AppSetting).where(AppSetting.key == POLICY_KEY))


def pin_within_bounds(pin: str) -> bool:
    """Whether `pin` can be a PIN: 4 to 64 characters, taken exactly as typed."""

    return MIN_PIN_CHARACTERS <= len(pin) <= MAX_PIN_CHARACTERS and _pin_bytes(pin) is not None


def _pin_bytes(pin: str) -> bytes | None:
    # JSON can carry a lone surrogate, which has no UTF-8 form.
    try:
        return pin.encode("utf-8")
    except UnicodeEncodeError:
        return None


def _derive(material: bytes, salt: bytes, derivation: KeyDerivation) -> bytes:
    if not derivation.within_limits():
        raise ValueError("The PIN key derivation is outside the accepted bounds.")
    try:
        return Argon2id(
            salt=salt,
            length=KEY_BYTES,
            iterations=derivation.iterations,
            lanes=derivation.lanes,
            memory_cost=derivation.memory_kib,
        ).derive(material)
    except MemoryError:
        raise PinCheckUnavailable("The PIN could not be checked.") from None


def make_pin_verifier(pin: str) -> PinVerifier:
    """Derive a verifier for a new PIN, under a fresh salt and at the current cost."""

    material = _pin_bytes(pin)
    if material is None or not pin_within_bounds(pin):
        raise ValueError("A PIN is 4 to 64 characters.")
    derivation = PIN_KEY_DERIVATION
    salt = os.urandom(SALT_BYTES)
    return PinVerifier(
        algorithm=PIN_ALGORITHM,
        iterations=derivation.iterations,
        lanes=derivation.lanes,
        memory_kib=derivation.memory_kib,
        salt=salt.hex(),
        hash=_derive(material, salt, derivation).hex(),
    )


def pin_matches(verifier: PinVerifier, pin: str | None) -> bool:
    """Whether `pin` is the PIN `verifier` was made from, compared in constant time.

    A missing PIN, or text that could never be a PIN, does not match and costs
    no derivation.
    """

    material = None if pin is None or not pin_within_bounds(pin) else _pin_bytes(pin)
    if material is None:
        return False
    derived = _derive(material, bytes.fromhex(verifier.salt), verifier.derivation)
    return hmac.compare_digest(derived, bytes.fromhex(verifier.hash))
