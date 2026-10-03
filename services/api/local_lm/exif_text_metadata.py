"""Read the settings text a JPEG or WebP picture carries in EXIF, without decoding the picture.

Two kinds of EXIF text are read:

- the Exif ``UserComment``, where most image tools write the same settings text
  a PNG carries under ``parameters``;
- an IFD0 text tag holding ``<name>:<JSON>``, the way a workflow tool writes its
  prompt graph (``prompt``) and its workflow (``workflow``) into a WebP.

Nothing else in the EXIF block is read, and the picture itself is never
decoded. An XMP packet is listed as skipped and never parsed. A JPEG segment or
WebP chunk that runs past the file, more of them than the ceiling allows, or an
EXIF directory that points outside its block refuses the whole file. A single
value that cannot be read safely is skipped with a reason, so it does not hide
the rest.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import Final

_MESSAGE = "exif text metadata is not valid"
_JPEG_START = b"\xff\xd8"
_JPEG_EXIF = b"Exif\x00\x00"
_JPEG_XMP = (b"http://ns.adobe.com/xap/1.0/\x00", b"http://ns.adobe.com/xmp/extension/\x00")
_RIFF = b"RIFF"
_WEBP = b"WEBP"
#: More segments than any JPEG writes before its image data.
_MAX_SEGMENTS: Final = 4096
#: An animated WebP stores one chunk per frame, and its EXIF comes after them.
_MAX_CHUNKS: Final = 100_000
#: The largest EXIF block read; a larger one is skipped, not refused.
MAX_EXIF_BYTES: Final = 1024 * 1024
_MAX_IFD_ENTRIES: Final = 1024
#: How many text values, and how much text, one EXIF block may give: more than
#: any writer stores, and the PNG reader's own ceilings.
_MAX_TEXT_VALUES: Final = 256
_MAX_TEXT_BYTES: Final = 2 * 1024 * 1024
#: How much of a text tag is looked at to see whether it names JSON.
_NAME_WINDOW: Final = 64
_EXIF_POINTER: Final = 0x8769
_USER_COMMENT: Final = 0x9286
_BYTE: Final = 1
_ASCII: Final = 2
_UNDEFINED: Final = 7
_LONG: Final = 4
_IFD: Final = 13
#: Bytes per value of each EXIF type a read entry can have.
_TYPE_SIZES: Final = {
    1: 1,
    2: 1,
    3: 2,
    4: 4,
    5: 8,
    6: 1,
    7: 1,
    8: 2,
    9: 4,
    10: 8,
    11: 4,
    12: 8,
    13: 4,
}
#: A text tag a workflow tool writes: its name, a colon, then JSON.
_NAMED_JSON = re.compile(rb"([A-Za-z][A-Za-z0-9_]{0,31}):\s*[\[{]")
#: A comment named ASCII, or named nothing: written padded with nulls or spaces.
_COMMENT_TEXT = re.compile(rb"(?:ASCII)?[\x00 ]*")
_COMMENT_UNICODE = b"UNICODE\x00"


@dataclass(frozen=True)
class ExifTextClaim:
    keyword: str
    text: str


@dataclass(frozen=True)
class SkippedExifText:
    """A value left unread, and why; its name when it could be read safely."""

    keyword: str | None
    reason: str


@dataclass(frozen=True)
class ExifText:
    claims: tuple[ExifTextClaim, ...]
    skipped: tuple[SkippedExifText, ...]


@dataclass(frozen=True)
class _Entry:
    tag: int
    kind: int
    count: int
    field: bytes
    #: Where the field sits in the block, for a value short enough to be held in it.
    at: int


def is_jpeg(payload: bytes) -> bool:
    return payload.startswith(_JPEG_START + b"\xff")


def is_webp(payload: bytes) -> bool:
    return len(payload) >= 12 and payload[:4] == _RIFF and payload[8:12] == _WEBP


def read_jpeg_text(payload: bytes) -> ExifText:
    """Return the EXIF text of a JPEG, read from the segments before its image data.

    Refuses a file whose segments run past it, or that has more segments than
    the ceiling, before its image data starts.
    """

    if not is_jpeg(payload):
        raise ValueError(_MESSAGE)
    blocks: list[bytes] = []
    xmp = False
    offset = len(_JPEG_START)
    for _ in range(_MAX_SEGMENTS):
        if offset >= len(payload) or payload[offset] != 0xFF:
            raise ValueError(_MESSAGE)
        while offset < len(payload) and payload[offset] == 0xFF:
            # A marker may be preceded by any number of fill bytes.
            offset += 1
        if offset >= len(payload):
            raise ValueError(_MESSAGE)
        marker = payload[offset]
        offset += 1
        if marker in (0xD9, 0xDA):
            # The image data, or the end of the file: no metadata follows that is read.
            return _text_of(blocks, xmp=xmp)
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(payload):
            raise ValueError(_MESSAGE)
        length = struct.unpack(">H", payload[offset : offset + 2])[0]
        end = offset + length
        if length < 2 or end > len(payload):
            raise ValueError(_MESSAGE)
        data = payload[offset + 2 : end]
        if marker == 0xE1 and data.startswith(_JPEG_EXIF):
            blocks.append(data[len(_JPEG_EXIF) :])
        elif marker == 0xE1 and data.startswith(_JPEG_XMP):
            xmp = True
        offset = end
    raise ValueError(_MESSAGE)


def read_webp_text(payload: bytes) -> ExifText:
    """Return the EXIF text of a WebP, from its ``EXIF`` chunk.

    Refuses a file whose RIFF size or any chunk runs past it, or that has more
    chunks than the ceiling.
    """

    if not is_webp(payload):
        raise ValueError(_MESSAGE)
    size = struct.unpack("<I", payload[4:8])[0]
    end = 8 + size
    if size < 4 or end > len(payload):
        raise ValueError(_MESSAGE)
    blocks: list[bytes] = []
    xmp = False
    offset = 12
    chunks = 0
    while offset < end:
        chunks += 1
        if chunks > _MAX_CHUNKS or offset + 8 > end:
            raise ValueError(_MESSAGE)
        kind = payload[offset : offset + 4]
        length = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]
        start = offset + 8
        if start + length > end:
            raise ValueError(_MESSAGE)
        if kind == b"EXIF":
            block = payload[start : start + length]
            # Some writers keep the JPEG segment's header in the chunk.
            blocks.append(block[len(_JPEG_EXIF) :] if block.startswith(_JPEG_EXIF) else block)
        elif kind == b"XMP ":
            xmp = True
        # Chunks are padded to an even length.
        offset = start + length + (length & 1)
    return _text_of(blocks, xmp=xmp)


def _text_of(blocks: list[bytes], *, xmp: bool) -> ExifText:
    claims: list[ExifTextClaim] = []
    skipped: list[SkippedExifText] = []
    if xmp:
        skipped.append(SkippedExifText("XMP", "xmp_not_read"))
    for index, block in enumerate(blocks):
        if index:
            skipped.append(SkippedExifText("EXIF", "repeated"))
        elif len(block) > MAX_EXIF_BYTES:
            skipped.append(SkippedExifText("EXIF", "too_long"))
        else:
            _read_block(block, claims, skipped)
    return ExifText(tuple(claims), tuple(skipped))


class _Values:
    """What one EXIF block has given so far; past either ceiling the file is refused.

    Entries can share one value's bytes, so a block's own size does not bound
    how much text its entries name.
    """

    def __init__(self) -> None:
        self.count = 0
        self.size = 0

    def take(self, size: int) -> None:
        self.count += 1
        self.size += size
        if self.count > _MAX_TEXT_VALUES or self.size > _MAX_TEXT_BYTES:
            raise ValueError(_MESSAGE)


def _read_block(block: bytes, claims: list[ExifTextClaim], skipped: list[SkippedExifText]) -> None:
    if len(block) < 8:
        raise ValueError(_MESSAGE)
    if block[:4] == b"II*\x00":
        order = "<"
    elif block[:4] == b"MM\x00*":
        order = ">"
    else:
        raise ValueError(_MESSAGE)
    first = struct.unpack(order + "I", block[4:8])[0]
    entries = _directory(block, first, order)
    values = _Values()
    for entry in entries:
        if entry.kind != _ASCII or entry.tag == _USER_COMMENT:
            continue
        start, size = _span(block, entry, order)
        # Only the start is looked at, so a camera's text is never copied or counted.
        named = _NAMED_JSON.match(block, start, start + min(size, _NAME_WINDOW))
        if named is None:
            # A camera's make, a date or a program name: not settings, and not listed.
            continue
        values.take(size)
        name = named.group(1).decode("ascii")
        text = block[start + len(name) + 1 : start + size].partition(b"\x00")[0]
        # A workflow tool's loader reads these names in either case.
        _add(claims, skipped, name.lower(), text, "utf-8")
    nested: list[_Entry] = []
    pointer = next((entry for entry in entries if entry.tag == _EXIF_POINTER), None)
    if pointer is not None:
        if pointer.kind not in (_LONG, _IFD) or pointer.count != 1:
            raise ValueError(_MESSAGE)
        offset = struct.unpack(order + "I", pointer.field)[0]
        if offset == first:
            raise ValueError(_MESSAGE)
        nested = _directory(block, offset, order)
    # The comment belongs in the Exif directory; a few writers put it in the first.
    for entry in [*nested, *entries]:
        # Writers store the comment as undefined bytes, plain bytes or text.
        if entry.tag == _USER_COMMENT and entry.kind in (_BYTE, _ASCII, _UNDEFINED):
            start, size = _span(block, entry, order)
            values.take(size)
            _user_comment(block[start : start + size], entry.kind, order, claims, skipped)


def _directory(block: bytes, offset: int, order: str) -> list[_Entry]:
    """One EXIF directory's entries; a directory outside the block refuses the file."""

    if offset < 8 or offset + 2 > len(block):
        raise ValueError(_MESSAGE)
    count = struct.unpack(order + "H", block[offset : offset + 2])[0]
    table = offset + 2
    if count > _MAX_IFD_ENTRIES or table + 12 * count > len(block):
        raise ValueError(_MESSAGE)
    entries: list[_Entry] = []
    for start in range(table, table + 12 * count, 12):
        tag, kind, number = struct.unpack(order + "HHI", block[start : start + 8])
        entries.append(_Entry(tag, kind, number, block[start + 8 : start + 12], start + 8))
    return entries


def _span(block: bytes, entry: _Entry, order: str) -> tuple[int, int]:
    """Where an entry's value is in the block, and its size; a value outside it refuses."""

    size = _TYPE_SIZES[entry.kind] * entry.count
    if size <= 4:
        return entry.at, size
    offset = struct.unpack(order + "I", entry.field)[0]
    if offset < 8 or offset + size > len(block):
        raise ValueError(_MESSAGE)
    return offset, size


def _user_comment(
    raw: bytes,
    kind: int,
    order: str,
    claims: list[ExifTextClaim],
    skipped: list[SkippedExifText],
) -> None:
    prefix, body = raw[:8], raw[8:]
    if prefix == _COMMENT_UNICODE:
        if body[:2] in (b"\xfe\xff", b"\xff\xfe"):
            encoding = "utf-16-be" if body[:2] == b"\xfe\xff" else "utf-16-le"
            body = body[2:]
        else:
            # Without a byte order mark, the text follows the EXIF block's own order.
            encoding = "utf-16-be" if order == ">" else "utf-16-le"
    elif len(raw) >= 8 and _COMMENT_TEXT.fullmatch(prefix):
        # Cameras often pad the name with spaces rather than nulls; the text ends at a null.
        encoding, body = "utf-8", body.partition(b"\x00")[0]
    elif kind == _ASCII:
        # A comment stored as text with no encoding named is the text itself.
        encoding, body = "utf-8", raw.partition(b"\x00")[0]
    elif not raw.strip(b"\x00 "):
        return
    else:
        skipped.append(SkippedExifText("UserComment", "unsupported_encoding"))
        return
    _add(claims, skipped, "UserComment", body, encoding)


def _add(
    claims: list[ExifTextClaim],
    skipped: list[SkippedExifText],
    keyword: str,
    raw: bytes,
    encoding: str,
) -> None:
    try:
        text = raw.decode(encoding)
    except UnicodeDecodeError:
        skipped.append(SkippedExifText(keyword, "not_text"))
        return
    text = text.rstrip("\x00")
    if not text.strip():
        # An empty comment is how many cameras fill the field; there is nothing to list.
        return
    for character in text:
        code = ord(character)
        # Control characters other than tab and the line ends.
        if (code < 32 and character not in "\t\n\r") or 127 <= code <= 159:
            skipped.append(SkippedExifText(keyword, "unsafe_text"))
            return
    claims.append(ExifTextClaim(keyword, text))
