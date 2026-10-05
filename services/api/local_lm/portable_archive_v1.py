"""The passphrase-encrypted envelope around a portable archive, version 1.

An encrypted archive is a short clear header followed by the archive's bytes,
encrypted. The header names the format, the kind of archive and up to two key
slots; it says nothing about what the archive holds. Each slot carries the
Argon2id settings that turn a passphrase into a key, and the archive's own
random key, encrypted under it.

Both the slots and the body use Cobblestone-256 from the cryptography package,
an implementation of the published C2SP chunked-encryption construction: each
16 KiB chunk is authenticated in order, the last chunk marks the end, and a key
commitment is checked before any chunk, so a wrong key is refused before a
single byte comes out. Nothing here builds a construction of its own.

Plaintext is released chunk by chunk as each one authenticates, and only the
final check proves that nothing was cut off. A caller writes it somewhere it
can discard, and keeps it only when :func:`open_archive` returns.
"""

from __future__ import annotations

import os
import struct
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import BinaryIO, Literal

from cryptography.cobblestone import Cobblestone256Decryptor, Cobblestone256Encryptor
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id

MAGIC = b"LMAARCH\x00"
FORMAT_VERSION = 1
# Suite 1: Argon2id opens each slot; Cobblestone-256 encrypts the slots and the body.
SUITE = 1

KEY_BYTES = 32
KEY_ID_BYTES = 16
SALT_BYTES = 16
# Cobblestone adds a 24-byte salt and a 32-byte commitment, and one 16-byte tag
# for the single short chunk that holds a 32-byte key.
WRAPPED_KEY_BYTES = 24 + 32 + KEY_BYTES + 16
MAX_SLOTS = 2
MAX_PASSPHRASE_BYTES = 1024
MIN_MEMORY_KIB = 64 * 1024
MAX_MEMORY_KIB = 1024 * 1024
MAX_ITERATIONS = 10
MAX_LANES = 8
READ_BYTES = 64 * 1024

_PREFIX = struct.Struct(">8sBBBB")
_SLOT = struct.Struct(f">B{KEY_ID_BYTES}s{SALT_BYTES}sIBI{WRAPPED_KEY_BYTES}s")
_SLOT_CONTEXT = b"lm-atelier portable archive v1 key slot\x00"
_PAYLOAD_CONTEXT = b"lm-atelier portable archive v1 payload\x00"

RefusalCode = Literal[
    "archive-format-unsupported",
    "archive-kind-mismatch",
    "archive-limits-exceeded",
    "archive-passphrase-invalid",
    "archive-key-derivation-failed",
    "invalid_passphrase_or_corrupt",
]


class ArchiveRefused(Exception):
    """An archive this version will not write or open, named by a fixed code and nothing else."""

    def __init__(self, code: RefusalCode) -> None:
        super().__init__(code)
        self.code = code


class ArchiveKind(IntEnum):
    PROJECT = 1
    OUTPUT_RECIPE = 2
    BACKUP = 3


class Recipient(IntEnum):
    """Who holds a slot's passphrase: the person exporting, or the stored backup key."""

    PASSPHRASE = 1
    STORED_BACKUP_KEY = 2


@dataclass(frozen=True)
class KeyDerivation:
    """Argon2id settings for one slot. Memory is in KiB, as Argon2id itself counts it."""

    iterations: int
    lanes: int
    memory_kib: int

    def within_limits(self) -> bool:
        return (
            1 <= self.iterations <= MAX_ITERATIONS
            and 1 <= self.lanes <= MAX_LANES
            and MIN_MEMORY_KIB <= self.memory_kib <= MAX_MEMORY_KIB
        )


DEFAULT_KEY_DERIVATION = KeyDerivation(iterations=3, lanes=4, memory_kib=256 * 1024)


@dataclass(frozen=True)
class KeySlot:
    recipient: Recipient
    key_id: bytes
    salt: bytes
    derivation: KeyDerivation
    wrapped_key: bytes

    def public_fields(self) -> bytes:
        """The slot as written, without its encrypted key."""
        return self.packed()[: _SLOT.size - WRAPPED_KEY_BYTES]

    def packed(self) -> bytes:
        return _SLOT.pack(
            self.recipient,
            self.key_id,
            self.salt,
            self.derivation.iterations,
            self.derivation.lanes,
            self.derivation.memory_kib,
            self.wrapped_key,
        )


@dataclass(frozen=True)
class ArchiveHeader:
    kind: ArchiveKind
    slots: tuple[KeySlot, ...]

    def prefix(self) -> bytes:
        return _PREFIX.pack(MAGIC, FORMAT_VERSION, SUITE, self.kind, len(self.slots))

    def packed(self) -> bytes:
        return self.prefix() + b"".join(slot.packed() for slot in self.slots)


# Replaced only by tests that need to see what was drawn.
_random_bytes: Callable[[int], bytes] = os.urandom


def write_archive(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    kind: ArchiveKind,
    passphrase: bytes,
    derivation: KeyDerivation = DEFAULT_KEY_DERIVATION,
) -> int:
    """Encrypt everything ``source`` holds into ``destination`` under ``passphrase``.

    Returns the number of bytes written. Every archive gets a new random key,
    slot salt and key id, so the same input never encrypts the same way twice.
    """

    if not 1 <= len(passphrase) <= MAX_PASSPHRASE_BYTES:
        raise ArchiveRefused("archive-passphrase-invalid")
    if not derivation.within_limits():
        raise ArchiveRefused("archive-limits-exceeded")
    data_key = _random_bytes(KEY_BYTES)
    unwrapped = KeySlot(
        Recipient.PASSPHRASE,
        _random_bytes(KEY_ID_BYTES),
        _random_bytes(SALT_BYTES),
        derivation,
        bytes(WRAPPED_KEY_BYTES),
    )
    prefix = ArchiveHeader(kind, (unwrapped,)).prefix()
    slot_key = _slot_key(passphrase, unwrapped)
    wrapping = Cobblestone256Encryptor(slot_key, _slot_context(prefix, unwrapped, 0))
    wrapped_key = wrapping.update(data_key) + wrapping.finalize()
    if len(wrapped_key) != WRAPPED_KEY_BYTES:
        raise RuntimeError("The wrapped archive key is not the size this format records.")
    slot = KeySlot(unwrapped.recipient, unwrapped.key_id, unwrapped.salt, derivation, wrapped_key)
    header = ArchiveHeader(kind, (slot,)).packed()
    destination.write(header)
    written = len(header)
    encryptor = Cobblestone256Encryptor(data_key, _PAYLOAD_CONTEXT + header)
    while chunk := source.read(READ_BYTES):
        written += destination.write(encryptor.update(chunk))
    written += destination.write(encryptor.finalize())
    return written


def read_header(source: BinaryIO) -> ArchiveHeader:
    """Read and check an archive's clear header, deriving nothing.

    Every refusal here is decided from public bytes, so it reveals nothing a
    holder of the file could not read for themselves.
    """

    prefix = source.read(_PREFIX.size)
    if len(prefix) != _PREFIX.size:
        raise ArchiveRefused("archive-format-unsupported")
    magic, version, suite, kind, slot_count = _PREFIX.unpack(prefix)
    if magic != MAGIC or version != FORMAT_VERSION or suite != SUITE:
        raise ArchiveRefused("archive-format-unsupported")
    if kind not in {member.value for member in ArchiveKind}:
        raise ArchiveRefused("archive-format-unsupported")
    if not 1 <= slot_count <= MAX_SLOTS:
        raise ArchiveRefused("archive-limits-exceeded")
    slots = []
    for _ in range(slot_count):
        packed = source.read(_SLOT.size)
        if len(packed) != _SLOT.size:
            raise ArchiveRefused("archive-format-unsupported")
        recipient, key_id, salt, iterations, lanes, memory_kib, wrapped_key = _SLOT.unpack(packed)
        if recipient not in {member.value for member in Recipient}:
            raise ArchiveRefused("archive-format-unsupported")
        derivation = KeyDerivation(iterations, lanes, memory_kib)
        if not derivation.within_limits():
            raise ArchiveRefused("archive-limits-exceeded")
        slots.append(KeySlot(Recipient(recipient), key_id, salt, derivation, wrapped_key))
    return ArchiveHeader(ArchiveKind(kind), tuple(slots))


def open_archive(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    kind: ArchiveKind,
    passphrase: bytes,
) -> int:
    """Decrypt ``source`` into ``destination``, returning the number of bytes written.

    Each slot is tried in turn, at most two derivations in all. Wrong
    passphrase, tampering, truncation and trailing bytes all end in the one
    ``invalid_passphrase_or_corrupt`` refusal. ``destination`` may already
    hold authenticated chunks when a later one is refused, so the caller
    discards it unless this returns.
    """

    header = read_header(source)
    if header.kind != kind:
        raise ArchiveRefused("archive-kind-mismatch")
    _slot, data_key = unwrap_key(header, passphrase)
    return open_with_key(source, destination, header, data_key)


def unwrap_key(header: ArchiveHeader, passphrase: bytes) -> tuple[KeySlot, bytes]:
    """The archive's own key, and the slot that held it, from ``passphrase``.

    Each slot is tried in turn, at most two derivations in all. A passphrase
    that opens none of them is refused as ``invalid_passphrase_or_corrupt``,
    the same refusal a damaged slot gets.
    """

    if not 1 <= len(passphrase) <= MAX_PASSPHRASE_BYTES:
        raise ArchiveRefused("invalid_passphrase_or_corrupt")
    prefix = header.prefix()
    for index, slot in enumerate(header.slots):
        unwrapping = Cobblestone256Decryptor(
            _slot_key(passphrase, slot), _slot_context(prefix, slot, index)
        )
        try:
            data_key = unwrapping.update(slot.wrapped_key) + unwrapping.finalize()
        except InvalidTag:
            continue
        if len(data_key) == KEY_BYTES:
            return slot, data_key
        break
    raise ArchiveRefused("invalid_passphrase_or_corrupt")


def open_with_key(
    source: BinaryIO, destination: BinaryIO, header: ArchiveHeader, data_key: bytes
) -> int:
    """Decrypt the body that follows ``header`` in ``source`` with the archive's own key.

    ``source`` must stand just past the header, as :func:`read_header` leaves
    it. Tampering, truncation, trailing bytes and a key of another archive all
    end in ``invalid_passphrase_or_corrupt``, and as with :func:`open_archive`
    the caller discards ``destination`` unless this returns.
    """

    if len(data_key) != KEY_BYTES:
        raise ArchiveRefused("invalid_passphrase_or_corrupt")
    decryptor = Cobblestone256Decryptor(data_key, _PAYLOAD_CONTEXT + header.packed())
    written = 0
    try:
        while chunk := source.read(READ_BYTES):
            written += destination.write(decryptor.update(chunk))
        written += destination.write(decryptor.finalize())
    except InvalidTag:
        raise ArchiveRefused("invalid_passphrase_or_corrupt") from None
    return written


def self_test() -> bool:
    """Whether this build can derive and encrypt as the format needs, by known answers.

    Run by the packaged application's runtime self-test, so a build that cannot
    load the primitives fails before it ships rather than when someone exports.
    """

    try:
        # RFC 9106 section 5.3.
        derived = Argon2id(
            salt=bytes([2]) * 16,
            length=32,
            iterations=3,
            lanes=4,
            memory_cost=32,
            ad=bytes([4]) * 12,
            secret=bytes([3]) * 8,
        ).derive(bytes([1]) * 32)
        key = bytes(range(KEY_BYTES))
        sealing = Cobblestone256Encryptor(key, b"self test")
        sealed = sealing.update(b"portable archive") + sealing.finalize()
        opening = Cobblestone256Decryptor(key, b"self test")
        opened = opening.update(sealed) + opening.finalize()
    except Exception:
        return False
    return (
        derived.hex() == "0d640df58d78766c08c037a34a8b53c9d01ef0452d75b65eb52520e96b01e659"
        and opened == b"portable archive"
    )


def _slot_key(passphrase: bytes, slot: KeySlot) -> bytes:
    try:
        return Argon2id(
            salt=slot.salt,
            length=KEY_BYTES,
            iterations=slot.derivation.iterations,
            lanes=slot.derivation.lanes,
            memory_cost=slot.derivation.memory_kib,
        ).derive(passphrase)
    except MemoryError:
        # Says nothing about the passphrase, so it is not reported as one.
        raise ArchiveRefused("archive-key-derivation-failed") from None


def _slot_context(prefix: bytes, slot: KeySlot, index: int) -> bytes:
    return _SLOT_CONTEXT + prefix + slot.public_fields() + bytes([index])
