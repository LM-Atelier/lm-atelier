"""PNG text metadata is read from the chunk stream and refused when it is unsafe."""

from __future__ import annotations

import struct
import zlib

import pytest

from local_lm.png_text_metadata import read_png_text_metadata

_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_STRING_BYTES = 64 * 1024


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


def test_an_overlong_png_text_chunk_is_refused() -> None:
    payload = _png(_ihdr(), _text("Comment", "x" * (_MAX_STRING_BYTES + 1)), _iend())

    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(payload)


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


def test_compressed_png_text_is_not_expanded() -> None:
    compressed = zlib.compress(b"A neutral fixture.")
    payload = _png(_ihdr(), _chunk(b"zTXt", b"Comment\x00\x00" + compressed), _iend())

    with pytest.raises(ValueError, match="png text metadata is not valid"):
        read_png_text_metadata(payload)
