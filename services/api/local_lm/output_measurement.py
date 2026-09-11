"""What a produced file actually measures, as fact, with no verdict attached.

The tool freezes the size a turn asked for and replays it exactly. Nothing has
ever read the file that came back, so nothing could notice a run that produced
something else. This records what the bytes are and stops there: it never sees
the requested size, so it cannot report an agreement it did not check.

Reading the header is not enough and decoding is too much. A truncated PNG still
declares its original geometry and an ordinary decoder agrees, so a header read
would report the size that was asked for over bytes that render as nothing. A
full decode costs what the file's BYTES cost rather than what its pixels cost,
and a file of millions of tiny chunks spends seconds without ever claiming a
large picture. So the pixel stream is decompressed and counted, never kept, and
the answer is the geometry the stream actually filled.

The cost is bounded by three declared numbers rather than by anything the file
says about itself. A declared geometry bounds neither the bytes fed to the
decompressor nor the number of times around a loop, and those are the two terms
that run away. The budget is shared across one generation, so many hostile
outputs cost what one does.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, Final

RECORD_VERSION: Final = 1

#: One step is one chunk header, one slice fed to the decompressor, or one drain
#: of its output. Bounding the loop is what stops a file of millions of empty
#: chunks, which carries no picture and costs seconds.
MAX_STEPS: Final = 131_072
#: Compressed bytes handed to the decompressor across one generation.
MAX_INFLATE_INPUT: Final = 64 * 1024 * 1024
#: Raster bytes the decompressor may produce across one generation. A file may
#: declare far more than this; the declaration is checked against what is left
#: before a decompressor is built, so an absurd claim costs one comparison.
MAX_DECODED: Final = 288 * 1024 * 1024

_SLICE: Final = 64 * 1024
_PNG_MAGIC: Final = b"\x89PNG\r\n\x1a\n"

#: Channels per pixel for each PNG colour type, and the bit depths it allows.
#: Checked here rather than trusted: three illegal pairs - a palette at sixteen
#: bits, greyscale at three, colour at four - are accepted by a framer that only
#: walks structure, and a measurement recorded for bytes nothing renders is the
#: same false success this module exists to prevent.
_COLOUR_TYPES: Final[dict[int, tuple[int, frozenset[int]]]] = {
    0: (1, frozenset({1, 2, 4, 8, 16})),
    2: (3, frozenset({8, 16})),
    3: (1, frozenset({1, 2, 4, 8})),
    4: (2, frozenset({8, 16})),
    6: (4, frozenset({8, 16})),
}


class Budget:
    """What one generation may spend measuring every file it produced."""

    def __init__(
        self,
        steps: int = MAX_STEPS,
        inflate_input: int = MAX_INFLATE_INPUT,
        decoded: int = MAX_DECODED,
    ) -> None:
        self.steps = steps
        self.inflate_input = inflate_input
        self.decoded = decoded


class _Stop(Exception):
    """A measurement that ended for a named reason rather than an answer."""

    def __init__(self, about: str, reason: str) -> None:
        super().__init__(reason)
        self.about = about
        self.reason = reason


def _spend_step(budget: Budget) -> None:
    budget.steps -= 1
    if budget.steps < 0:
        raise _Stop("budget", "over_step_budget")


def _unmeasured(about: str, reason: str) -> dict[str, Any]:
    return {"v": RECORD_VERSION, "state": "unmeasured", "about": about, "reason": reason}


def _scanline_bytes(width: int, height: int, depth: int, channels: int) -> int:
    """The exact raster a complete pixel stream decompresses to.

    Every row carries one filter byte and then its pixels packed to a byte
    boundary. Comparing against this exactly is what separates a complete
    picture from a stream that ended early but decompressed cleanly.
    """
    bits = width * channels * depth
    return height * (1 + (bits + 7) // 8)


def _account(produced: bytes, budget: Budget, decoded: int, expected: int) -> int:
    budget.decoded -= len(produced)
    if budget.decoded < 0:
        raise _Stop("budget", "over_decode_budget")
    decoded += len(produced)
    if decoded > expected:
        # Stop the moment the stream passes what the header promised, rather
        # than decompressing the rest to find out how far over it went.
        raise _Stop("file", "raster_overrun")
    return decoded


def _feed(
    decompressor: Any,
    body: memoryview,
    budget: Budget,
    decoded: int,
    expected: int,
) -> int:
    """Push one chunk through in slices, keeping the count and nothing else.

    Slices are small and fixed so the tail the decompressor holds back is never
    more than one of them. Feeding a whole chunk at once lets that tail grow
    with the chunk, and re-presenting it then turns the walk quadratic.
    """
    at = 0
    while at < len(body):
        _spend_step(budget)
        piece = body[at : at + _SLICE]
        at += len(piece)
        budget.inflate_input -= len(piece)
        if budget.inflate_input < 0:
            raise _Stop("budget", "over_input_budget")
        try:
            produced = decompressor.decompress(piece, _SLICE)
        except zlib.error:
            raise _Stop("file", "pixel_stream_unreadable") from None
        decoded = _account(produced, budget, decoded, expected)
        while decompressor.unconsumed_tail:
            _spend_step(budget)
            try:
                produced = decompressor.decompress(decompressor.unconsumed_tail, _SLICE)
            except zlib.error:
                raise _Stop("file", "pixel_stream_unreadable") from None
            decoded = _account(produced, budget, decoded, expected)
    return decoded


def _drain(decompressor: Any, budget: Budget, decoded: int, expected: int) -> int:
    """Finish the stream without letting a tail of empty blocks run forever."""
    while not decompressor.eof and decompressor.unconsumed_tail:
        _spend_step(budget)
        try:
            produced = decompressor.decompress(decompressor.unconsumed_tail, _SLICE)
        except zlib.error:
            raise _Stop("file", "pixel_stream_unreadable") from None
        decoded = _account(produced, budget, decoded, expected)
    return decoded


def _measure_png(payload: bytes, budget: Budget) -> dict[str, Any]:
    view = memoryview(payload)
    offset = len(_PNG_MAGIC)
    header: tuple[int, int] | None = None
    decompressor: Any = None
    decoded = 0
    expected = 0
    saw_end = False
    exif_present = False
    animated = False

    while offset < len(view):
        _spend_step(budget)
        if len(view) - offset < 12:
            raise _Stop("file", "container_incomplete")
        (length,) = struct.unpack_from(">I", view, offset)
        kind = bytes(view[offset + 4 : offset + 8])
        body_at = offset + 8
        if length > len(view) - body_at - 4:
            raise _Stop("file", "container_incomplete")
        body = view[body_at : body_at + length]
        (declared_crc,) = struct.unpack_from(">I", view, body_at + length)
        if zlib.crc32(view[offset + 4 : body_at + length]) & 0xFFFFFFFF != declared_crc:
            raise _Stop("file", "container_incomplete")
        offset = body_at + length + 4

        if kind == b"IHDR":
            if header is not None or length != 13:
                raise _Stop("file", "header_unreadable")
            width, height, depth, colour, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if width == 0 or height == 0 or colour not in _COLOUR_TYPES:
                raise _Stop("file", "header_unreadable")
            channels, depths = _COLOUR_TYPES[colour]
            if depth not in depths:
                raise _Stop("file", "header_unreadable")
            if interlace:
                # An interlaced stream decompresses to seven smaller passes, so
                # the exact-length check below would not describe it. Declining
                # to measure is honest; guessing is the failure this avoids.
                raise _Stop("file", "interlaced_unsupported")
            expected = _scanline_bytes(width, height, depth, channels)
            if expected > budget.decoded:
                # Asked before a decompressor exists, so a file declaring a
                # picture nobody could hold costs one comparison.
                raise _Stop("budget", "over_decode_budget")
            header = (width, height)
        elif kind == b"IDAT":
            if header is None:
                raise _Stop("file", "header_unreadable")
            if decompressor is None:
                decompressor = zlib.decompressobj()
            decoded = _feed(decompressor, body, budget, decoded, expected)
        elif kind == b"eXIf":
            exif_present = True
        elif kind == b"acTL":
            animated = True
        elif kind == b"IEND":
            saw_end = True
            break

    if not saw_end or offset != len(view):
        # IEND ends the file. Trailing bytes mean something was appended to a
        # complete picture, and the picture is then no longer the file.
        raise _Stop("file", "container_incomplete")
    if header is None:
        raise _Stop("file", "header_unreadable")
    if decompressor is None:
        raise _Stop("file", "no_pixel_stream")
    decoded = _drain(decompressor, budget, decoded, expected)
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise _Stop("file", "pixel_stream_incomplete")
    if decoded < expected:
        raise _Stop("file", "raster_short")

    width, height = header
    return {
        "v": RECORD_VERSION,
        "state": "measured",
        "method": "png_idat_consumed",
        "raster_width": width,
        "raster_height": height,
        "exif_present": exif_present,
        "animated": animated,
    }


def measure_output(payload: bytes, budget: Budget) -> dict[str, Any]:
    """Measure one produced file, or name why it was not measured.

    The answer is a fact about these bytes and nothing else. It never reads the
    run, the workflow, the operation or the size that was asked for, so it
    cannot compare a number with something derived from that number - which is
    how every earlier attempt at this ended up agreeing with itself.
    """
    try:
        if not payload.startswith(_PNG_MAGIC):
            return _unmeasured("scope", "unsupported_container")
        return _measure_png(payload, budget)
    except _Stop as stop:
        return _unmeasured(stop.about, stop.reason)


def record_to_keep(previous: object, fresh: dict[str, Any]) -> dict[str, Any]:
    """Which of two records about the same bytes survives.

    A store addressed by a digest of its own content holds one row for one set
    of bytes, so a later run that produces an identical file writes where an
    earlier one already did. The answer is a pure function of those bytes and
    normally agrees with itself - but a budget is shared across a generation
    and can be spent, so the second look may be `unmeasured/budget` where the
    first was a measurement.

    That is a fact about our ceiling on a busy run, not about the file, and it
    must not overwrite evidence we already hold. Only a measurement replaces a
    measurement; anything else defers to one.
    """
    if not isinstance(previous, dict):
        return fresh
    held: dict[str, Any] = previous
    if held.get("state") == "measured" and fresh.get("state") != "measured":
        return held
    return fresh
