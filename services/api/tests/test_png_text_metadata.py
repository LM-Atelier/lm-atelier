"""PNG text metadata is read from the chunk stream; unsafe text is skipped or refused."""

from __future__ import annotations

import struct
import zlib

import pytest

from local_lm.png_text_metadata import (
    PngTextClaim,
    SkippedPngText,
    read_png_text,
    read_png_text_metadata,
)

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_STRING_BYTES = 64 * 1024
_MAX_CHUNK_TEXT_BYTES = 1024 * 1024


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _png(*chunks: bytes) -> bytes:
    return _SIGNATURE + b"".join(chunks)


def _ihdr() -> bytes:
    return _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))


def _iend() -> bytes:
    return _chunk(b"IEND", b"")


def _text(keyword: str, text: str) -> bytes:
    return _chunk(b"tEXt", keyword.encode("latin-1") + b"\x00" + text.encode("latin-1"))


def test_a_neutral_png_comment_is_returned() -> None:
    payload = _png(_ihdr(), _text("Comment", "A neutral fixture."), _iend())

    claims = read_png_text_metadata(payload)

    assert [(claim.keyword, claim.text) for claim in claims] == [("Comment", "A neutral fixture.")]


def test_a_non_png_is_refused() -> None:
    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(b"not a png")


def test_an_overlong_png_text_chunk_is_skipped_and_the_rest_still_read() -> None:
    # A graph is one chunk, so a chunk may be longer than any one value in it.
    long_enough = _text("Comment", "x" * (_MAX_STRING_BYTES + 1))
    too_long = _text("workflow", "x" * (_MAX_CHUNK_TEXT_BYTES + 1))
    payload = _png(_ihdr(), long_enough, too_long, _text("Title", "A neutral fixture."), _iend())

    text = read_png_text(payload)

    assert [(claim.keyword, len(claim.text)) for claim in text.claims] == [
        ("Comment", _MAX_STRING_BYTES + 1),
        ("Title", 18),
    ]
    assert text.skipped == (SkippedPngText("workflow", "too_long"),)


def test_a_bad_png_text_crc_is_refused() -> None:
    payload = bytearray(_png(_ihdr(), _text("Comment", "A neutral fixture."), _iend()))
    payload[-5] ^= 0x01

    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(bytes(payload))


def test_png_text_stops_at_two_mebibytes() -> None:
    text = "x" * _MAX_STRING_BYTES
    accepted = _png(_ihdr(), *(_text("Comment", text) for _ in range(32)), _iend())
    refused = _png(
        _ihdr(),
        *(_text("Comment", text) for _ in range(32)),
        _text("Comment", "y"),
        _iend(),
    )

    assert len(read_png_text_metadata(accepted)) == 32
    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(refused)


def test_png_text_stops_at_two_hundred_fifty_six_chunks() -> None:
    accepted = _png(_ihdr(), *(_text("Comment", "x") for _ in range(256)), _iend())
    refused = _png(_ihdr(), *(_text("Comment", "x") for _ in range(257)), _iend())

    assert len(read_png_text_metadata(accepted)) == 256
    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(refused)


def test_compressed_png_text_is_skipped_without_being_expanded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed = zlib.compress(b"A neutral fixture.")
    payload = _png(
        _ihdr(),
        _chunk(b"zTXt", b"Comment\x00\x00" + compressed),
        _chunk(b"iTXt", b"Title\x00\x01\x00\x00\x00" + compressed),
        _chunk(b"zTXt", b"\x00\x00" + compressed),
        _text("Software", "A neutral fixture."),
        _iend(),
    )

    def never(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("compressed text was expanded")

    monkeypatch.setattr(zlib, "decompress", never)
    text = read_png_text(payload)

    assert [(claim.keyword, claim.text) for claim in text.claims] == [
        ("Software", "A neutral fixture.")
    ]
    assert text.skipped == (
        SkippedPngText("Comment", "compressed"),
        SkippedPngText("Title", "compressed"),
        SkippedPngText(None, "compressed"),
    )
    assert read_png_text_metadata(payload) == text.claims


def test_latin_1_text_with_a_no_break_space_is_read() -> None:
    payload = _png(_ihdr(), _text("Comment", "a ceramic cup\xa0on a table"), _iend())

    assert read_png_text(payload).claims == (
        PngTextClaim("Comment", "a ceramic cup\xa0on a table"),
    )


@pytest.mark.parametrize("character", ["\x00", "\x07", "\x1b", "\x7f", "\x85"])
def test_text_holding_a_control_character_is_skipped_and_the_rest_read(character: str) -> None:
    payload = _png(
        _ihdr(),
        _text("Comment", f"neutral{character}note"),
        _text("Title", "A neutral fixture."),
        _iend(),
    )

    text = read_png_text(payload)

    assert text.claims == (PngTextClaim("Title", "A neutral fixture."),)
    assert text.skipped == (SkippedPngText("Comment", "unsafe_text"),)


def _international(keyword: str, text: bytes, *, flag: int = 0) -> bytes:
    header = keyword.encode("latin-1") + b"\x00" + bytes([flag, 0]) + b"en\x00Title\x00"
    return _chunk(b"iTXt", header + text)


def test_uncompressed_international_text_is_read_as_utf_8() -> None:
    payload = _png(
        _ihdr(),
        _international("Comment", "a ceramic cup \u2014 on a table".encode()),
        _iend(),
    )

    assert read_png_text(payload).claims == (
        PngTextClaim("Comment", "a ceramic cup \u2014 on a table"),
    )


def test_international_text_that_is_not_utf_8_or_holds_controls_is_skipped() -> None:
    payload = _png(
        _ihdr(),
        _international("Comment", b"a ceramic cup \xff"),
        _international("Title", "neutral\u0085note".encode()),
        _international("Software", b"A neutral fixture."),
        _iend(),
    )

    text = read_png_text(payload)

    assert text.claims == (PngTextClaim("Software", "A neutral fixture."),)
    assert text.skipped == (
        SkippedPngText("Comment", "not_utf8"),
        SkippedPngText("Title", "unsafe_text"),
    )


@pytest.mark.parametrize(
    "data",
    [b"Comment", b"Comment\x00", b"Comment\x00\x00\x00en", b"Comment\x00\x00\x00en\x00Title"],
    ids=["no-separator", "no-flag", "no-language-end", "no-translated-end"],
)
def test_international_text_cut_short_refuses_the_file(data: bytes) -> None:
    payload = _png(_ihdr(), _chunk(b"iTXt", data), _iend())

    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text(payload)


def test_skipped_chunks_count_toward_the_chunk_ceiling() -> None:
    compressed = _chunk(b"zTXt", b"Comment\x00\x00" + zlib.compress(b"x"))
    accepted = _png(_ihdr(), *(compressed for _ in range(255)), _text("Title", "x"), _iend())
    refused = _png(_ihdr(), *(compressed for _ in range(256)), _text("Title", "x"), _iend())

    assert len(read_png_text(accepted).skipped) == 255
    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text(refused)


def test_a_damaged_chunk_after_a_skipped_one_still_refuses_the_file() -> None:
    payload = bytearray(
        _png(
            _ihdr(),
            _chunk(b"zTXt", b"Comment\x00\x00" + zlib.compress(b"x")),
            _text("Title", "A neutral fixture."),
            _iend(),
        )
    )
    # The last byte of the Title chunk's checksum, just before IEND's 12 bytes.
    payload[-13] ^= 0x01

    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text(bytes(payload))
