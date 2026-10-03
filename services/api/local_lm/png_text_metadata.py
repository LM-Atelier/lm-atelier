"""Read uncompressed text from a PNG without expanding compressed chunks."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IHDR = b"IHDR"
_TEXT = b"tEXt"
_INTERNATIONAL = b"iTXt"
_COMPRESSED = b"zTXt"
_IEND = b"IEND"
_MAX_TEXT_CHUNKS = 256
_MAX_TEXT_BYTES = 2 * 1024 * 1024
#: The longest single text chunk read. A graph a workflow tool stores is one
#: chunk and often longer than any one value in it, so the limit on single
#: values (MAX_STRING_BYTES) is applied to what is read from a chunk instead.
_MAX_CHUNK_TEXT_BYTES = 1024 * 1024
MAX_STRING_BYTES = 64 * 1024
_MAX_KEYWORD_BYTES = 79
_MAX_CHUNKS = 100_000
_MESSAGE = "png text metadata is not valid"


@dataclass(frozen=True)
class PngTextClaim:
    keyword: str
    text: str


@dataclass(frozen=True)
class SkippedPngText:
    """A text chunk left unread, and why; its keyword when it could be read safely."""

    keyword: str | None
    reason: str


@dataclass(frozen=True)
class PngText:
    claims: tuple[PngTextClaim, ...]
    skipped: tuple[SkippedPngText, ...]


@dataclass(frozen=True)
class _Unread:
    """Why a chunk's text is left unread."""

    reason: str


def read_png_text_metadata(payload: bytes) -> tuple[PngTextClaim, ...]:
    """Return the text claims of a PNG, or refuse one that is not a bounded text stream."""

    return read_png_text(payload).claims


def read_png_text(payload: bytes) -> PngText:
    """Return text claims and the text chunks left unread, or refuse a damaged PNG.

    Latin-1 text (tEXt) and uncompressed international text (iTXt, UTF-8) are
    read. Compressed text (zTXt, and iTXt marked compressed) is never expanded:
    it is listed as skipped. So is a chunk past the single-chunk ceiling, one
    holding control characters, and international text that is not UTF-8, so
    one such chunk does not hide the rest. A chunk that fails its checksum,
    runs past the buffer, or names its keyword with characters a keyword
    cannot have refuses the whole file, and so does more text, or more text
    chunks, than the ceilings allow. Nothing from a refused file is returned.
    """

    if not isinstance(payload, bytes) or not payload.startswith(_SIGNATURE):
        raise ValueError(_MESSAGE)
    claims: list[PngTextClaim] = []
    skipped: list[SkippedPngText] = []
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
        is_text = kind in {_TEXT, _INTERNATIONAL, _COMPRESSED}
        if is_text and len(claims) + len(skipped) >= _MAX_TEXT_CHUNKS:
            raise ValueError(_MESSAGE)
        raw: tuple[str, bytes, str] | None = None
        if kind == _COMPRESSED or (kind == _INTERNATIONAL and _marked_compressed(data)):
            skipped.append(SkippedPngText(_keyword_of(data), "compressed"))
        elif kind == _INTERNATIONAL:
            raw = (*_international_keyword_and_text(data), "utf-8")
        elif kind == _TEXT:
            raw = (*_keyword_and_text(data), "latin-1")
        if raw is not None:
            keyword, text, encoding = raw
            if len(text) > _MAX_CHUNK_TEXT_BYTES:
                skipped.append(SkippedPngText(keyword, "too_long"))
            else:
                text_bytes += len(text)
                if text_bytes > _MAX_TEXT_BYTES:
                    raise ValueError(_MESSAGE)
                read = _text(text, encoding)
                if isinstance(read, _Unread):
                    skipped.append(SkippedPngText(keyword, read.reason))
                else:
                    claims.append(PngTextClaim(keyword, read))
        offset += 12 + len(data)
        if kind == _IEND:
            saw_end = True
            break
    if not saw_end:
        raise ValueError(_MESSAGE)
    return PngText(tuple(claims), tuple(skipped))


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


def _keyword_and_text(data: bytes) -> tuple[str, bytes]:
    separator = data.find(b"\x00")
    if separator <= 0 or separator > _MAX_KEYWORD_BYTES:
        raise ValueError(_MESSAGE)
    return _keyword(data[:separator]), data[separator + 1 :]


def _marked_compressed(data: bytes) -> bool:
    """Whether an iTXt chunk says its text is compressed; a chunk too short to say is damaged."""

    separator = data.find(b"\x00")
    if separator < 0 or separator + 1 >= len(data):
        raise ValueError(_MESSAGE)
    return data[separator + 1] != 0


def _international_keyword_and_text(data: bytes) -> tuple[str, bytes]:
    """An uncompressed iTXt chunk's keyword and text, past its language and translated keyword."""

    keyword, rest = _keyword_and_text(data)
    # The compression flag and method, then two NUL-terminated fields.
    language_end = rest.find(b"\x00", 2)
    translated_end = rest.find(b"\x00", language_end + 1) if language_end >= 0 else -1
    if len(rest) < 2 or translated_end < 0:
        raise ValueError(_MESSAGE)
    return keyword, rest[translated_end + 1 :]


def _keyword_of(data: bytes) -> str | None:
    """A compressed chunk's keyword when it is a plain one; it never refuses the file."""

    separator = data.find(b"\x00")
    if separator <= 0 or separator > _MAX_KEYWORD_BYTES:
        return None
    try:
        return _keyword(data[:separator])
    except ValueError:
        return None


def _keyword(raw: bytes) -> str:
    keyword = raw.decode("latin-1")
    if not keyword or keyword != keyword.strip() or "  " in keyword:
        raise ValueError(_MESSAGE)
    if not all(
        32 <= ord(character) <= 126 or 161 <= ord(character) <= 255 for character in keyword
    ):
        raise ValueError(_MESSAGE)
    return keyword


def _text(raw: bytes, encoding: str) -> str | _Unread:
    """A chunk's text, or why it is left unread: text a viewer could not show as written."""

    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        return _Unread("not_utf8")
    for character in text:
        code = ord(character)
        # Control characters other than tab and the line ends, in either encoding.
        if (code < 32 and character not in "\t\n\r") or 127 <= code <= 159:
            return _Unread("unsafe_text")
    return text
