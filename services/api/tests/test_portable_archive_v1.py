"""The passphrase-encrypted envelope: what it writes, what it opens, and everything it refuses."""

from __future__ import annotations

import io
import struct
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.argon2 import Argon2id
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

from local_lm import portable_archive_v1 as archive
from local_lm.portable_archive_v1 import ArchiveKind, ArchiveRefused, KeyDerivation

PASSPHRASE = b"correct horse battery staple"
# The cheapest derivation a reader accepts, so these tests stay quick.
CHEAP = KeyDerivation(iterations=1, lanes=1, memory_kib=archive.MIN_MEMORY_KIB)
CHUNK = 16 * 1024
SEALED_CHUNK = CHUNK + 16
PREFIX_BYTES = 12
SLOT_BYTES = 1 + 16 + 16 + 4 + 1 + 4 + archive.WRAPPED_KEY_BYTES
HEADER_BYTES = PREFIX_BYTES + SLOT_BYTES
# Written by this format's first version and kept as it was written: every
# later version must still open it to exactly this text.
GOLDEN = bytes.fromhex(
    "4c4d414152434800010101010187cf241d773c71db1cad0d5da2ef7108826c5dfc7bbb5c"
    "7143ff3107ee738a6600000001010001000097b181596b58f61a7d10c3d9b8b20e66994b"
    "785f784b972fd97fe40e589b2cdb7abcbceffb14d723240744b629b8c2163ca1803f75cb"
    "c04d616021e86e3bbf9d9fc08fcb83e9217a954d7d22febc9d86c9994efbfe244c695826"
    "d3b1f5f059e55d0f306a2cca7322fa02478d78f87eb108efe4ec46bc5743fd9b6b581708"
    "eb0158dd2fc437299acd4887424a0b2b0e2dc7e5399379d75e4613e79f332b6a94d2c064"
    "843bb0d13a8906c0a45c06ab7b59f07bc98a73fed66cddd1e37da18d86ee5fc06720c352"
    "fd94a0cf7aadfa73964ce6a35fb490"
)
GOLDEN_TEXT = b"A portable archive, kept as written.\n"


def _sealed(payload: bytes, *, kind: ArchiveKind = ArchiveKind.PROJECT) -> bytes:
    out = io.BytesIO()
    archive.write_archive(
        io.BytesIO(payload), out, kind=kind, passphrase=PASSPHRASE, derivation=CHEAP
    )
    return out.getvalue()


def _opened(
    sealed: bytes, *, kind: ArchiveKind = ArchiveKind.PROJECT, passphrase: bytes = PASSPHRASE
) -> bytes:
    out = io.BytesIO()
    archive.open_archive(io.BytesIO(sealed), out, kind=kind, passphrase=passphrase)
    return out.getvalue()


def _refusal(sealed: bytes, *, kind: ArchiveKind = ArchiveKind.PROJECT) -> str:
    with pytest.raises(ArchiveRefused) as refused:
        _opened(sealed, kind=kind)
    return refused.value.code


@pytest.mark.parametrize("size", [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK + 5])
def test_an_archive_opens_to_exactly_what_was_sealed(size: int) -> None:
    payload = bytes(index % 251 for index in range(size))

    sealed = _sealed(payload)

    assert sealed[: len(archive.MAGIC)] == archive.MAGIC
    assert _opened(sealed) == payload


def test_the_same_input_never_seals_the_same_way_twice() -> None:
    first, second = _sealed(b"same input"), _sealed(b"same input")

    assert first != second
    assert (
        first[PREFIX_BYTES + 1 : PREFIX_BYTES + 33] != second[PREFIX_BYTES + 1 : PREFIX_BYTES + 33]
    )


def test_an_archive_written_by_the_first_version_still_opens() -> None:
    assert _opened(GOLDEN) == GOLDEN_TEXT


def test_the_header_names_the_kind_and_settings_and_nothing_of_the_contents() -> None:
    sealed = _sealed(b"private words that stay private", kind=ArchiveKind.BACKUP)

    header = archive.read_header(io.BytesIO(sealed))

    assert header.kind is ArchiveKind.BACKUP
    assert [slot.derivation for slot in header.slots] == [CHEAP]
    assert b"private" not in sealed


def test_a_wrong_passphrase_is_refused_and_nothing_is_written() -> None:
    out = io.BytesIO()
    with pytest.raises(ArchiveRefused) as refused:
        archive.open_archive(
            io.BytesIO(_sealed(b"contents")), out, kind=ArchiveKind.PROJECT, passphrase=b"wrong"
        )

    assert refused.value.code == "invalid_passphrase_or_corrupt"
    assert out.getvalue() == b""


def test_an_archive_of_another_kind_is_refused_before_any_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = _sealed(b"contents")
    derived = _counting_derivations(monkeypatch)

    code = _refusal(sealed, kind=ArchiveKind.BACKUP)

    assert code == "archive-kind-mismatch"
    assert derived == []


@pytest.mark.parametrize(
    ("field", "offset", "replacement", "code"),
    [
        ("magic", 0, b"X", "archive-format-unsupported"),
        ("version", 8, b"\x02", "archive-format-unsupported"),
        ("suite", 9, b"\x02", "archive-format-unsupported"),
        ("kind", 10, b"\x09", "archive-format-unsupported"),
        ("no slots", 11, b"\x00", "archive-limits-exceeded"),
        ("too many slots", 11, b"\x03", "archive-limits-exceeded"),
        ("recipient", 12, b"\x07", "archive-format-unsupported"),
        (
            "iterations",
            45,
            struct.pack(">I", archive.MAX_ITERATIONS + 1),
            "archive-limits-exceeded",
        ),
        ("no iterations", 45, struct.pack(">I", 0), "archive-limits-exceeded"),
        ("lanes", 49, bytes([archive.MAX_LANES + 1]), "archive-limits-exceeded"),
        (
            "memory floor",
            50,
            struct.pack(">I", archive.MIN_MEMORY_KIB - 1),
            "archive-limits-exceeded",
        ),
        (
            "memory ceiling",
            50,
            struct.pack(">I", archive.MAX_MEMORY_KIB + 1),
            "archive-limits-exceeded",
        ),
    ],
)
def test_a_header_outside_this_format_is_refused_before_any_derivation(
    monkeypatch: pytest.MonkeyPatch, field: str, offset: int, replacement: bytes, code: str
) -> None:
    sealed = bytearray(_sealed(b"contents"))
    sealed[offset : offset + len(replacement)] = replacement
    derived = _counting_derivations(monkeypatch)

    assert _refusal(bytes(sealed)) == code
    assert derived == []


@pytest.mark.parametrize(
    ("field", "offset", "bit"),
    [
        ("key id", 13, 0x01),
        ("salt", 29, 0x01),
        ("iterations within the limits", 48, 0x02),
        ("memory within the limits", 53, 0x01),
        ("wrapped key salt", 54, 0x01),
        ("wrapped key commitment", 54 + 24, 0x01),
        ("wrapped key", 54 + 56, 0x01),
        ("wrapped key tag", HEADER_BYTES - 1, 0x01),
    ],
)
def test_a_changed_slot_is_refused_as_the_one_refusal(field: str, offset: int, bit: int) -> None:
    sealed = bytearray(_sealed(b"contents"))
    sealed[offset] ^= bit

    assert _refusal(bytes(sealed)) == "invalid_passphrase_or_corrupt"


def test_the_body_is_bound_to_its_own_header() -> None:
    first, second = _sealed(b"first archive"), _sealed(b"second archive")

    assert _refusal(first[:HEADER_BYTES] + second[HEADER_BYTES:]) == "invalid_passphrase_or_corrupt"


def test_a_body_moved_under_another_header_with_the_same_key_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both archives draw the same archive key, so only the binding of each body
    # to its own header tells them apart.
    real = archive._random_bytes
    monkeypatch.setattr(
        archive,
        "_random_bytes",
        lambda size: bytes(range(size)) if size == archive.KEY_BYTES else real(size),
    )
    first, second = _sealed(b"first archive"), _sealed(b"second archive")

    assert _opened(second) == b"second archive"
    assert _refusal(first[:HEADER_BYTES] + second[HEADER_BYTES:]) == "invalid_passphrase_or_corrupt"


def _chunks(sealed: bytes) -> tuple[bytes, list[bytes]]:
    body = sealed[HEADER_BYTES:]
    start, rest = body[:56], body[56:]
    return sealed[:HEADER_BYTES] + start, [
        rest[index : index + SEALED_CHUNK] for index in range(0, len(rest), SEALED_CHUNK)
    ]


@pytest.mark.parametrize(
    "change",
    ["swapped", "duplicated", "dropped", "flipped", "last chunk dropped", "trailing bytes"],
)
def test_a_changed_body_is_refused_as_the_one_refusal(change: str) -> None:
    head, chunks = _chunks(_sealed(bytes(3 * CHUNK + 100)))
    assert len(chunks) == 4
    if change == "swapped":
        chunks[0], chunks[1] = chunks[1], chunks[0]
    elif change == "duplicated":
        chunks.insert(1, chunks[1])
    elif change == "dropped":
        del chunks[1]
    elif change == "flipped":
        chunks[2] = chunks[2][:-1] + bytes([chunks[2][-1] ^ 1])
    elif change == "last chunk dropped":
        del chunks[-1]
    else:
        chunks.append(b"\x00" * 17)

    assert _refusal(head + b"".join(chunks)) == "invalid_passphrase_or_corrupt"


@pytest.mark.parametrize("cut", [1, 17, SEALED_CHUNK])
def test_a_cut_off_archive_is_refused_even_after_releasing_whole_chunks(cut: int) -> None:
    sealed = _sealed(bytes(2 * CHUNK + 10))
    out = io.BytesIO()

    with pytest.raises(ArchiveRefused) as refused:
        archive.open_archive(
            io.BytesIO(sealed[:-cut]), out, kind=ArchiveKind.PROJECT, passphrase=PASSPHRASE
        )

    assert refused.value.code == "invalid_passphrase_or_corrupt"
    # Whole chunks may already be out, which is why a caller keeps nothing
    # unless opening returns.
    assert len(out.getvalue()) < 2 * CHUNK + 10


def test_a_header_too_short_to_read_is_not_this_format() -> None:
    assert _refusal(_sealed(b"contents")[: HEADER_BYTES - 1]) == "archive-format-unsupported"


def test_a_derivation_that_cannot_get_its_memory_is_not_called_a_wrong_passphrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sealed = _sealed(b"contents")

    class _NoMemory:
        def __init__(self, **kwargs: object) -> None:
            pass

        def derive(self, key_material: bytes) -> bytes:
            raise MemoryError("not enough memory")

    monkeypatch.setattr(archive, "Argon2id", _NoMemory)

    assert _refusal(sealed) == "archive-key-derivation-failed"


@pytest.mark.parametrize("passphrase", [b"", b"x" * (archive.MAX_PASSPHRASE_BYTES + 1)])
def test_a_passphrase_outside_its_bounds_seals_nothing(passphrase: bytes) -> None:
    out = io.BytesIO()
    with pytest.raises(ArchiveRefused) as refused:
        archive.write_archive(
            io.BytesIO(b"contents"), out, kind=ArchiveKind.PROJECT, passphrase=passphrase
        )

    assert refused.value.code == "archive-passphrase-invalid"
    assert out.getvalue() == b""


def test_settings_outside_the_limits_seal_nothing() -> None:
    out = io.BytesIO()
    with pytest.raises(ArchiveRefused) as refused:
        archive.write_archive(
            io.BytesIO(b"contents"),
            out,
            kind=ArchiveKind.PROJECT,
            passphrase=PASSPHRASE,
            derivation=KeyDerivation(iterations=1, lanes=1, memory_kib=8),
        )

    assert refused.value.code == "archive-limits-exceeded"
    assert out.getvalue() == b""


def test_an_archive_decodes_by_the_published_construction_alone() -> None:
    """Opened here with plain Argon2id, HKDF-SHA-512 and AES-256-GCM, without Cobblestone.

    Following c2sp.org/chunked-encryption step by step, so the archive is shown
    to be that construction and not merely whatever the library writes.
    """

    payload = bytes(index % 7 for index in range(2 * CHUNK + 33))
    sealed = _sealed(payload)
    header, body = sealed[:HEADER_BYTES], sealed[HEADER_BYTES:]
    _recipient, _key_id, salt, iterations, lanes, memory_kib, wrapped = struct.unpack(
        f">B16s16sIBI{archive.WRAPPED_KEY_BYTES}s", header[PREFIX_BYTES:]
    )
    slot_key = Argon2id(
        salt=salt, length=32, iterations=iterations, lanes=lanes, memory_cost=memory_kib
    ).derive(PASSPHRASE)
    slot_context = (
        b"lm-atelier portable archive v1 key slot\x00"
        + header[:PREFIX_BYTES]
        + header[PREFIX_BYTES : HEADER_BYTES - archive.WRAPPED_KEY_BYTES]
        + b"\x00"
    )
    data_key = _c2sp_open(slot_key, slot_context, wrapped)

    assert (
        _c2sp_open(data_key, b"lm-atelier portable archive v1 payload\x00" + header, body)
        == payload
    )


def test_the_key_derivation_matches_rfc_9106() -> None:
    derived = Argon2id(
        salt=bytes([2]) * 16,
        length=32,
        iterations=3,
        lanes=4,
        memory_cost=32,
        ad=bytes([4]) * 12,
        secret=bytes([3]) * 8,
    ).derive(bytes([1]) * 32)

    assert derived.hex() == "0d640df58d78766c08c037a34a8b53c9d01ef0452d75b65eb52520e96b01e659"


def test_the_runtime_self_test_passes() -> None:
    assert archive.self_test() is True


def test_the_runtime_self_test_fails_on_a_wrong_known_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _WrongAnswer:
        def __init__(self, **kwargs: object) -> None:
            pass

        def derive(self, key_material: bytes) -> bytes:
            return bytes(32)

    monkeypatch.setattr(archive, "Argon2id", _WrongAnswer)

    assert archive.self_test() is False


def _c2sp_open(key: bytes, context: bytes, sealed: bytes) -> bytes:
    salt, commitment, body = sealed[:24], sealed[24:56], sealed[56:]
    derived = HKDFExpand(
        algorithm=hashes.SHA512(),
        length=32 + 12 + 32,
        info=b"c2sp.org/chunked-encryption@v1+AEAD_AES_256_GCM\x00" + salt + context,
    ).derive(key)
    chunk_key, base_nonce, expected_commitment = derived[:32], derived[32:44], derived[44:]
    assert commitment == expected_commitment
    aead, plain, number = AESGCM(chunk_key), b"", 0
    while True:
        chunk = body[number * SEALED_CHUNK : (number + 1) * SEALED_CHUNK]
        nonce = (int.from_bytes(base_nonce, "big") ^ number).to_bytes(12, "big")
        plain += aead.decrypt(nonce, chunk, None)
        if len(chunk) < SEALED_CHUNK:
            return plain
        number += 1


def _counting_derivations(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    derived: list[int] = []
    real = archive.Argon2id

    def counting(**kwargs: Any) -> Argon2id:
        derived.append(1)
        return real(**kwargs)

    monkeypatch.setattr(archive, "Argon2id", counting)
    return derived
