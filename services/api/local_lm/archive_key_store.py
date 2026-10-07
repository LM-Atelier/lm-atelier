"""The one archive key a restore waiting for the next start needs, held in the system vault.

An encrypted backup restored on the next start has to be opened again then,
before anyone could be asked for its passphrase, and the passphrase is never
kept. What waits instead is the backup's own archive key: the random key that
encrypts that one file and nothing else. It is held in the operating system's
credential vault, under a service of its own that no other credential shares,
beside the identities of the restore it belongs to, and it is removed when that
restore is applied, fails or is withdrawn.

Each data directory has one account of its own, named from the directory,
because one vault serves every LM Atelier data directory on the computer. A
restore asked for again in the same directory replaces its key there, and a
crash between storing it and arming the restore leaves at most one stale entry
for that directory's next restore to replace; another directory's key is never
read, replaced or removed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import keyring
from keyring.errors import KeyringError

from .portable_archive_v1 import KEY_BYTES, KEY_ID_BYTES

logger = logging.getLogger(__name__)

SERVICE = "lm-atelier-archive-keys"
_ACCOUNT_PREFIX = "restore-on-next-start-"
_ENTRY_VERSION = "v1"
_OPERATION = re.compile(r"[0-9a-f]{32}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEY_ID = re.compile(f"[0-9a-f]{{{2 * KEY_ID_BYTES}}}")
_KEY = re.compile(f"[0-9a-f]{{{2 * KEY_BYTES}}}")


class ArchiveKeyVaultUnavailable(RuntimeError):
    """The system vault cannot hold the key, so nothing that needs it was scheduled."""


class ArchiveKeyMissing(Exception):
    """The vault holds no key for this restore, or cannot be read."""


@dataclass(frozen=True)
class RestoreKeyBinding:
    """Which restore a stored key belongs to: none of these identities is secret.

    ``operation`` names the restore, ``sha256`` is the digest of the encrypted
    file it opens, and ``key_id`` is the key slot that file was opened with.
    """

    operation: str
    sha256: str
    key_id: str

    def __post_init__(self) -> None:
        if not (
            _OPERATION.fullmatch(self.operation)
            and _SHA256.fullmatch(self.sha256)
            and _KEY_ID.fullmatch(self.key_id)
        ):
            raise ValueError("restore key binding is malformed")

    def as_marker(self) -> dict[str, str]:
        return {"operation": self.operation, "sha256": self.sha256, "key_id": self.key_id}

    @classmethod
    def from_marker(cls, value: object) -> RestoreKeyBinding:
        if not isinstance(value, dict) or set(value) != {"operation", "sha256", "key_id"}:
            raise ValueError("restore key binding is malformed")
        operation, sha256, key_id = value["operation"], value["sha256"], value["key_id"]
        if not all(isinstance(part, str) for part in (operation, sha256, key_id)):
            raise ValueError("restore key binding is malformed")
        return cls(operation, sha256, key_id)


def vault_account(data_dir: Path) -> str:
    """The vault account of the restores of one data directory, and of no other."""

    resolved = os.path.normcase(str(Path(data_dir).resolve()))
    return _ACCOUNT_PREFIX + hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:32]


class ArchiveKeyStore:
    def __init__(self, data_dir: Path) -> None:
        self.account = vault_account(data_dir)

    def available(self) -> bool:
        """Whether a usable vault is present, decided without reading anything stored."""

        if keyring is None:
            return False
        try:
            backend = keyring.get_keyring()
            return float(getattr(backend, "priority", 0)) > 0
        except (KeyringError, TypeError, ValueError):
            return False

    def put(self, binding: RestoreKeyBinding, key: bytes) -> None:
        """Hold ``key`` for the restore ``binding`` names, replacing this directory's key."""

        if len(key) != KEY_BYTES:
            raise ValueError("archive key is the wrong size")
        if keyring is None:
            raise ArchiveKeyVaultUnavailable
        entry = " ".join((_ENTRY_VERSION, binding.operation, binding.sha256, binding.key_id))
        try:
            keyring.set_password(SERVICE, self.account, f"{entry} {key.hex()}")
        except KeyringError as exc:
            raise ArchiveKeyVaultUnavailable from exc

    def key_for(self, binding: RestoreKeyBinding) -> bytes:
        """The key held for exactly this restore, or ArchiveKeyMissing.

        A key held for any other restore, or for another file, is not returned.
        Reading does not remove it; :meth:`discard` does, whatever the outcome.
        """

        if keyring is None:
            raise ArchiveKeyMissing
        try:
            stored = keyring.get_password(SERVICE, self.account)
        except KeyringError as exc:
            raise ArchiveKeyMissing from exc
        key = _held_key(stored, binding)
        if key is None:
            raise ArchiveKeyMissing
        return key

    def discard(self, binding: RestoreKeyBinding) -> None:
        """Remove this directory's key if it is the one held for ``binding``.

        A key held for any other restore stays where it is: it is not this
        restore's to remove, and the directory's next restore replaces it. A
        vault that refuses is logged, not raised.
        """

        if keyring is None:
            return
        try:
            stored = keyring.get_password(SERVICE, self.account)
            if stored is None or _held_key(stored, binding) is None:
                return
            keyring.delete_password(SERVICE, self.account)
        except KeyringError:
            logger.warning("The key held for a restore could not be removed from the vault.")


def _held_key(stored: object, binding: RestoreKeyBinding) -> bytes | None:
    parts = stored.split(" ") if isinstance(stored, str) else []
    expected = [_ENTRY_VERSION, binding.operation, binding.sha256, binding.key_id]
    if len(parts) != 5 or parts[:4] != expected or not _KEY.fullmatch(parts[4]):
        return None
    return bytes.fromhex(parts[4])
