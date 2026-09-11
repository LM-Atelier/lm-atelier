"""What a produced file measures, recorded as fact and never as a verdict."""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from PIL import Image, UnidentifiedImageError

from local_lm.output_measurement import (
    MAX_DECODED,
    Budget,
    measure_output,
)

_MAGIC = b"\x89PNG\r\n\x1a\n"
#: Channels per pixel for each colour type, so a fixture can build the raster a
#: declared header really implies rather than a guess at it.
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


def _chunk(kind: bytes, body: bytes) -> bytes:
    return (
        struct.pack(">I", len(body))
        + kind
        + body
        + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    )


def _ihdr(width: int, height: int, depth: int = 8, colour: int = 2, interlace: int = 0) -> bytes:
    return _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, interlace))


_IEND = _chunk(b"IEND", b"")


def _raster(width: int, height: int, depth: int, colour: int) -> bytes:
    """The exact bytes a complete pixel stream for this header decodes to."""
    row = (width * _CHANNELS[colour] * depth + 7) // 8
    return b"".join(b"\x00" + bytes(row) for _ in range(height))


def _built(width: int, height: int, depth: int = 8, colour: int = 2, **kinds: bytes) -> bytes:
    """A structurally complete PNG whose pixel stream matches its header."""
    payload = _MAGIC + _ihdr(width, height, depth, colour)
    for kind, body in kinds.items():
        payload += _chunk(kind.encode("ascii"), body)
    return payload + _chunk(b"IDAT", zlib.compress(_raster(width, height, depth, colour))) + _IEND


def _real(width: int, height: int) -> bytes:
    """A PNG an ordinary encoder wrote, so the happy path is not self-built."""
    pixels = bytes(
        (row * 7 + column * 13) % 256 for row in range(height) for column in range(width * 3)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), pixels).save(buffer, "PNG")
    return buffer.getvalue()


def test_a_produced_picture_is_measured_at_the_size_its_pixels_fill() -> None:
    """The happy path, over bytes a real encoder wrote rather than a fixture."""
    record = measure_output(_real(1216, 704), Budget())

    assert record["state"] == "measured"
    assert record["method"] == "png_idat_consumed"
    assert (record["raster_width"], record["raster_height"]) == (1216, 704)
    assert record["exif_present"] is False
    assert record["animated"] is False


def test_the_record_says_nothing_about_what_was_asked_for() -> None:
    """The whole point: this is evidence, so it must not echo the request.

    Every earlier attempt at this feature compared a number against something
    derived from the same number and could never fail. The guard is structural -
    the function takes bytes and a budget, and there is no argument through
    which a requested size could reach it.
    """
    record = measure_output(_real(64, 32), Budget())

    assert set(record) == {
        "v",
        "state",
        "method",
        "raster_width",
        "raster_height",
        "exif_present",
        "animated",
    }
    assert "requested" not in repr(record)


def test_a_stream_that_ends_early_is_not_the_size_its_header_claims() -> None:
    """The defeat that killed an earlier design, and the reason for the exact count.

    This file is structurally perfect and its pixel stream decompresses without
    error; it simply stops half way down. An ordinary decoder reports the
    header's geometry for it, so a design that trusted either would have
    recorded "produced 1024x1024, as requested" for bytes that render as
    nothing.
    """
    half = _raster(1024, 512, 8, 2)
    payload = _MAGIC + _ihdr(1024, 1024) + _chunk(b"IDAT", zlib.compress(half)) + _IEND

    assert measure_output(payload, Budget()) == {
        "v": 1,
        "state": "unmeasured",
        "about": "file",
        "reason": "raster_short",
    }
    # The control: an ordinary decoder agrees with the header instead.
    assert Image.open(io.BytesIO(payload)).size == (1024, 1024)


def test_every_pixel_present_is_not_the_same_as_a_finished_stream() -> None:
    """The case that is complete by length and unfinished by integrity.

    Dropping the pixel stream's trailing checksum leaves a file whose raster
    decompresses to exactly the size the header declares - so a length
    comparison alone calls it measured - while the stream never reaches its end
    and nothing ever verified the bytes. Recording a proven geometry for it
    would be the same false success as trusting a truncated file's header, one
    layer further in.
    """
    raster = _raster(64, 64, 8, 2)
    unterminated = zlib.compress(raster)[:-4]
    payload = _MAGIC + _ihdr(64, 64) + _chunk(b"IDAT", unterminated) + _IEND

    assert measure_output(payload, Budget())["reason"] == "pixel_stream_incomplete"
    # The control: the same raster with its checksum intact is measured.
    whole = _MAGIC + _ihdr(64, 64) + _chunk(b"IDAT", zlib.compress(raster)) + _IEND
    assert measure_output(whole, Budget())["state"] == "measured"


def test_a_truncated_file_and_an_appended_one_are_both_incomplete() -> None:
    """IEND ends the file. Bytes after it mean the picture is not the file."""
    whole = _real(64, 64)

    assert measure_output(whole[: len(whole) // 2], Budget())["reason"] == "container_incomplete"
    assert measure_output(whole + b"appended", Budget())["reason"] == "container_incomplete"


def test_a_header_no_decoder_would_accept_is_not_measured() -> None:
    """Three illegal pairs a structural walk accepts, measured both ways.

    `reference_review._png_complete_and_consumed` returns True for each of these
    when the raster matches the declared shape, because it checks structure and
    length rather than whether the combination is legal. It is safe where it
    lives only because a decoder opens first. Lifting that walk to a path with
    no decoder would record a proven measurement for bytes nothing renders.
    """
    for depth, colour in ((16, 3), (3, 0), (4, 2)):
        payload = _built(8, 8, depth, colour)

        assert measure_output(payload, Budget())["reason"] == "header_unreadable"
        with pytest.raises(UnidentifiedImageError):
            Image.open(io.BytesIO(payload)).load()


def test_an_interlaced_picture_says_so_rather_than_guessing() -> None:
    """Seven smaller passes do not add up to the exact length checked here."""
    payload = _MAGIC + _ihdr(64, 64, 8, 2, interlace=1)
    payload += _chunk(b"IDAT", zlib.compress(bytes(64))) + _IEND

    assert measure_output(payload, Budget())["reason"] == "interlaced_unsupported"


def test_a_container_with_no_method_is_named_rather_than_attempted() -> None:
    """v1 measures PNG. Everything else says so in microseconds."""
    for payload in (b"\xff\xd8\xff\xe0" + bytes(64), b"RIFF" + bytes(64), b"", b"<svg/>"):
        record = measure_output(payload, Budget())

        assert record["about"] == "scope"
        assert record["reason"] == "unsupported_container"


def test_the_two_extra_facts_are_recorded_because_a_later_reader_needs_them() -> None:
    """An orientation tag means a viewer may draw the transposed pair, and an
    animation control means the raster proven is only the default image."""
    record = measure_output(_built(8, 8, eXIf=b"\x00" * 8, acTL=b"\x00" * 8), Budget())

    assert record["state"] == "measured"
    assert record["exif_present"] is True
    assert record["animated"] is True


def test_a_picture_nobody_could_hold_costs_one_comparison() -> None:
    """The declared size is judged before a decompressor exists.

    A header may claim any geometry at all. Checking the claim against what is
    left of the budget first means an absurd one is refused without decoding a
    byte, rather than being discovered part way through.
    """
    payload = _MAGIC + _ihdr(30000, 30000) + _chunk(b"IDAT", zlib.compress(b"x")) + _IEND
    budget = Budget()

    record = measure_output(payload, budget)

    assert record["about"] == "budget"
    assert record["reason"] == "over_decode_budget"
    assert budget.decoded == MAX_DECODED, "nothing was decoded to find that out"


def test_a_file_of_many_tiny_chunks_spends_steps_rather_than_seconds() -> None:
    """The cost that no declared geometry bounds.

    This file declares an eight by eight picture, so any ceiling derived from
    its geometry is tiny, and it still walks two hundred thousand chunks. The
    step budget is the only thing that stops it.
    """
    payload = _MAGIC + _ihdr(8, 8)
    payload += b"".join(_chunk(b"tEXt", b"k\x00" + bytes(13)) for _ in range(200_000))
    payload += _IEND

    record = measure_output(payload, Budget())

    assert record["about"] == "budget"
    assert record["reason"] == "over_step_budget"


def test_one_budget_is_spent_across_a_whole_generation() -> None:
    """Many outputs cost what one does, because the budget is shared.

    A per-file ceiling multiplies by however many files a run produced. This is
    the same object for every measurement in one generation, so the ceiling is
    the generation's rather than each file's.
    """
    budget = Budget()
    first = measure_output(_real(256, 256), budget)
    spent = MAX_DECODED - budget.decoded
    second = measure_output(_real(256, 256), budget)

    assert first["state"] == "measured" and second["state"] == "measured"
    assert MAX_DECODED - budget.decoded == spent * 2, "the second measurement spent the same budget"
    assert budget.decoded < MAX_DECODED


def test_an_exhausted_budget_says_so_about_itself_and_not_about_the_file() -> None:
    """A starved measurement must never read as evidence about the bytes.

    `about` carries that difference: a file reason is evidence about the file, a
    budget reason is evidence about nothing but our own ceiling. A reader that
    cannot tell them apart will eventually treat our exhaustion as a defect in
    somebody's picture.
    """
    record = measure_output(_real(256, 256), Budget(steps=2))

    assert record["about"] == "budget"
    assert record["reason"] == "over_step_budget"
    assert "raster_width" not in record
