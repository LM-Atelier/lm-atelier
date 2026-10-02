"""Read uncompressed text from a PNG without expanding compressed chunks."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR = b"IHDR"
_TEXT = b"tEXt"
_IEND = b"IEND"
_COMPRESSED = frozenset({b"zTXt", b"iTXt"})
_MAX_TEXT_CHUNKS = 256
_MAX_TEXT_BYTES = 2 * 1024 * 1024
_MAX_STRING_BYTES = 64 * 1024
_MAX_KEYWORD_BYTES = 79
_MAX_CHUNKS = 100_000
_MESSAGE = "png text metadata is not valid"


@dataclass(frozen=True)
class PngTextClaim:
    keyword: str
    text: str


def read_png_text_metadata(payload: bytes) -> tuple[PngTextClaim, ...]:
    """Return tEXt claims, or refuse a PNG that is not a bounded text stream.

    Compressed text is refused before it is expanded. A chunk that fails its
    checksum, runs past the buffer, or exceeds the text ceilings is refused
    the same way. Nothing from the chunk is returned.
    """

    if not isinstance(payload, bytes) or not payload.startswith(_SIGNATURE):
        raise ValueError(_MESSAGE)
    claims: list[PngTextClaim] = []
    text_bytes = 0
    offset = len(_SIGNATURE)
    chunks = 0
    saw_end = False
    while offset < len(payload):
        chunks += 1
        if chunks > _MAX_CHUNKS:
            raise ValueError(_MESSAGE)
        kind, data = _chunk(payload, offset)
        if chunks == 1 and kind != _IHDR:
            raise ValueError(_MESSAGE)
        if kind in _COMPRESSED:
            raise ValueError(_MESSAGE)
        if kind == _TEXT:
            claim, size = _text_claim(data)
            text_bytes += size
            if len(claims) >= _MAX_TEXT_CHUNKS or text_bytes > _MAX_TEXT_BYTES:
                raise ValueError(_MESSAGE)
            claims.append(claim)
        offset += 12 + len(data)
        if kind == _IEND:
            saw_end = True
            break
    if not saw_end:
        raise ValueError(_MESSAGE)
    return tuple(claims)


def _chunk(payload: bytes, offset: int) -> tuple[bytes, bytes]:
    if offset + 8 > len(payload):
        raise ValueError(_MESSAGE)
    length = struct.unpack(">I", payload[offset : offset + 4])[0]
    end = offset + 8 + length
    if length > len(payload) or end + 4 > len(payload):
        raise ValueError(_MESSAGE)
    kind = payload[offset + 4 : offset + 8]
    data = payload[offset + 8 : end]
    recorded = struct.unpack(">I", payload[end : end + 4])[0]
    if (zlib.crc32(kind + data) & 0xFFFFFFFF) != recorded:
        raise ValueError(_MESSAGE)
    if not kind.isascii() or not kind.isalnum():
        raise ValueError(_MESSAGE)
    return kind, data


def _text_claim(data: bytes) -> tuple[PngTextClaim, int]:
    if len(data) > _MAX_STRING_BYTES + _MAX_KEYWORD_BYTES + 1:
        raise ValueError(_MESSAGE)
    separator = data.find(b"\x00")
    if separator <= 0 or separator > _MAX_KEYWORD_BYTES:
        raise ValueError(_MESSAGE)
    keyword = _plain(data[:separator], keyword=True)
    text = _plain(data[separator + 1 :], keyword=False)
    if len(text.encode("latin-1")) > _MAX_STRING_BYTES:
        raise ValueError(_MESSAGE)
    return PngTextClaim(keyword, text), len(text.encode("latin-1"))


def _plain(raw: bytes, *, keyword: bool) -> str:
    text = raw.decode("latin-1")
    if keyword and (not text or text != text.strip() or "  " in text):
        raise ValueError(_MESSAGE)
    for character in text:
        if not _allowed(character, keyword=keyword):
            raise ValueError(_MESSAGE)
    return text


def _allowed(character: str, *, keyword: bool) -> bool:
    code = ord(character)
    if not keyword and character in "\t\n\r":
        return True
    return 32 <= code <= 126 or 161 <= code <= 255
