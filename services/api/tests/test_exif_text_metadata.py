"""A JPEG's or WebP's EXIF text is read within fixed bounds, and its picture is never decoded."""

from __future__ import annotations

import json
import struct
import time
from io import BytesIO

import pytest
from PIL import Image

from local_lm.exif_text_metadata import (
    MAX_EXIF_BYTES,
    ExifText,
    read_jpeg_text,
    read_webp_text,
)

_SETTINGS = "a ceramic cup on a wooden table\nSteps: 20, Sampler: Euler a, Seed: 12345"
_GRAPH = json.dumps({"3": {"class_type": "KSampler", "inputs": {"seed": 5, "steps": 20}}})
_WORKFLOW = json.dumps({"nodes": [], "links": []})
_SIZES = {2: 1, 4: 4, 7: 1}
_UNICODE = b"UNICODE\x00"


def _directory(entries: list[tuple[int, int, bytes]], start: int, order: str) -> bytes:
    """One EXIF directory at ``start``: its entry table, then the values too long to inline."""

    data_at = start + 2 + 12 * len(entries) + 4
    rows, data = b"", b""
    for tag, kind, value in entries:
        count = len(value) // _SIZES[kind]
        if len(value) <= 4:
            field = value.ljust(4, b"\x00")
        else:
            field = struct.pack(order + "I", data_at + len(data))
            data += value + b"\x00" * (len(value) % 2)
        rows += struct.pack(order + "HHI", tag, kind, count) + field
    return struct.pack(order + "H", len(entries)) + rows + b"\x00" * 4 + data


def exif_block(
    ifd0: list[tuple[int, int, bytes]],
    exif: list[tuple[int, int, bytes]] | None = None,
    *,
    order: str = "<",
) -> bytes:
    """A TIFF-structured EXIF block, with an Exif directory when ``exif`` is given."""

    head = (b"II*\x00" if order == "<" else b"MM\x00*") + struct.pack(order + "I", 8)
    if exif is None:
        return head + _directory(ifd0, 8, order)
    placeholder = [*ifd0, (0x8769, 4, b"\x00" * 4)]
    nested = 8 + len(_directory(placeholder, 8, order))
    first = _directory([*ifd0, (0x8769, 4, struct.pack(order + "I", nested))], 8, order)
    return head + first + _directory(exif, nested, order)


def comment(text: str, *, order: str = "<") -> list[tuple[int, int, bytes]]:
    encoding = "utf-16-le" if order == "<" else "utf-16-be"
    return [(0x9286, 7, _UNICODE + text.encode(encoding))]


def ascii_tag(tag: int, text: str) -> tuple[int, int, bytes]:
    return (tag, 2, text.encode("utf-8") + b"\x00")


def jpeg(*segments: tuple[int, bytes]) -> bytes:
    """A JPEG's header segments, then the start of its image data, which is never read."""

    out = b"\xff\xd8"
    for marker, data in segments:
        out += bytes([0xFF, marker]) + struct.pack(">H", len(data) + 2) + data
    return out + b"\xff\xda\x00\x02" + b"\x00" * 8 + b"\xff\xd9"


def jpeg_with_exif(block: bytes) -> bytes:
    return jpeg(
        (0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"), (0xE1, b"Exif\x00\x00" + block)
    )


def webp(*chunks: tuple[bytes, bytes]) -> bytes:
    body = b"WEBP"
    for kind, data in [(b"VP8L", b"\x2f" + b"\x00" * 4), *chunks]:
        body += kind + struct.pack("<I", len(data)) + data + b"\x00" * (len(data) % 2)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _read(text: ExifText) -> tuple[list[tuple[str, str]], list[tuple[str | None, str]]]:
    return (
        [(claim.keyword, claim.text) for claim in text.claims],
        [(item.keyword, item.reason) for item in text.skipped],
    )


@pytest.mark.parametrize("order", ["<", ">"])
def test_a_jpeg_user_comment_is_read_in_the_exif_blocks_own_byte_order(order: str) -> None:
    payload = jpeg_with_exif(exif_block([], comment(_SETTINGS, order=order), order=order))

    assert _read(read_jpeg_text(payload)) == ([("UserComment", _SETTINGS)], [])


@pytest.mark.parametrize(
    "raw",
    [
        b"ASCII\x00\x00\x00" + _SETTINGS.encode("ascii"),
        b"\x00" * 8 + _SETTINGS.encode("utf-8"),
        # A byte order mark overrides the block's order.
        _UNICODE + b"\xfe\xff" + _SETTINGS.encode("utf-16-be"),
        _UNICODE + b"\xff\xfe" + _SETTINGS.encode("utf-16-le"),
    ],
)
def test_a_user_comment_is_read_in_each_encoding_it_names(raw: bytes) -> None:
    payload = jpeg_with_exif(exif_block([], [(0x9286, 7, raw)]))

    assert _read(read_jpeg_text(payload)) == ([("UserComment", _SETTINGS)], [])


def test_a_webp_carries_a_workflow_tools_graphs_in_its_named_text_tags() -> None:
    block = exif_block(
        [
            ascii_tag(0x010F, "workflow:" + _WORKFLOW),
            ascii_tag(0x0110, "prompt:" + _GRAPH),
            # A camera's or program's own text is not settings and is not listed.
            ascii_tag(0x0131, "neutral-editor 1.0"),
            ascii_tag(0x0132, "2026:10:03 12:00:00"),
        ]
    )

    for payload in (webp((b"EXIF", block)), webp((b"EXIF", b"Exif\x00\x00" + block))):
        assert _read(read_webp_text(payload)) == (
            [("workflow", _WORKFLOW), ("prompt", _GRAPH)],
            [],
        )


def test_files_written_by_an_image_library_are_read() -> None:
    exif = Image.Exif()
    exif[0x0110] = "prompt:" + _GRAPH
    # The library writes the comment as plain bytes, in its own byte order.
    order = "utf-16-be" if exif.tobytes()[6:8] == b"MM" else "utf-16-le"
    exif[0x8769] = {0x9286: _UNICODE + _SETTINGS.encode(order)}
    picture = Image.new("RGB", (8, 8), (40, 90, 160))
    expected = [("prompt", _GRAPH), ("UserComment", _SETTINGS)]
    for kind, read in (("JPEG", read_jpeg_text), ("WEBP", read_webp_text)):
        buffer = BytesIO()
        picture.save(buffer, kind, exif=exif.tobytes())
        assert _read(read(buffer.getvalue())) == (expected, []), kind


def test_a_comment_written_as_text_in_the_first_directory_is_read() -> None:
    # As one image tool writes it: the comment as plain text, with no Exif directory.
    exif = Image.Exif()
    exif[0x9286] = _SETTINGS
    exif[0x0131] = "neutral-tool 2.5"
    picture = Image.new("RGB", (8, 8), (40, 90, 160))
    for kind, read in (("JPEG", read_jpeg_text), ("WEBP", read_webp_text)):
        buffer = BytesIO()
        picture.save(buffer, kind, exif=exif.tobytes())
        assert _read(read(buffer.getvalue())) == ([("UserComment", _SETTINGS)], []), kind


def test_the_exif_directorys_comment_comes_before_one_in_the_first_directory() -> None:
    block = exif_block([ascii_tag(0x9286, "Taken from the harbour wall")], comment(_SETTINGS))

    assert _read(read_jpeg_text(jpeg_with_exif(block))) == (
        [("UserComment", _SETTINGS), ("UserComment", "Taken from the harbour wall")],
        [],
    )


def test_named_graph_tags_are_read_in_either_case() -> None:
    block = exif_block(
        [ascii_tag(0x010F, "Prompt:" + _GRAPH), ascii_tag(0x010E, "Workflow:" + _WORKFLOW)]
    )

    assert _read(read_webp_text(webp((b"EXIF", block)))) == (
        [("prompt", _GRAPH), ("workflow", _WORKFLOW)],
        [],
    )


def _shared(count: int, value: bytes, *, tag: int = 0x010E, kind: int = 2) -> bytes:
    """An EXIF block whose ``count`` first-directory entries all name one value's bytes."""

    data_at = 8 + 2 + 12 * count + 4
    row = struct.pack("<HHII", tag, kind, len(value), data_at)
    return b"II*\x00" + struct.pack("<IH", 8, count) + row * count + b"\x00" * 4 + value


@pytest.mark.parametrize(
    "block",
    [
        # More values than any picture holds.
        _shared(257, b'prompt:{"a": 1}\x00'),
        _shared(257, b"ASCII\x00\x00\x00" + _SETTINGS.encode(), tag=0x9286, kind=7),
        # Few values, but more text than the block itself could hold.
        _shared(30, b"prompt:{" + b" " * 100_000 + b"}\x00"),
    ],
    ids=["many-values", "many-comments", "much-text"],
)
def test_values_sharing_their_bytes_cannot_multiply_what_is_read(block: bytes) -> None:
    started = time.monotonic()
    with pytest.raises(ValueError):
        read_webp_text(webp((b"EXIF", block)))
    assert time.monotonic() - started < 2


def test_a_cameras_long_text_named_many_times_is_neither_copied_nor_counted() -> None:
    block = _shared(1000, b"Holiday notes " * 37_000 + b"\x00")

    started = time.monotonic()
    assert _read(read_webp_text(webp((b"EXIF", block)))) == ([], [])
    assert time.monotonic() - started < 2


def test_xmp_is_listed_and_never_parsed() -> None:
    xmp = b"<x:xmpmeta><rdf:Description parameters='Steps: 20'/></x:xmpmeta>"
    from_jpeg = jpeg((0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + xmp))
    from_webp = webp((b"XMP ", xmp))

    assert _read(read_jpeg_text(from_jpeg)) == ([], [("XMP", "xmp_not_read")])
    assert _read(read_webp_text(from_webp)) == ([], [("XMP", "xmp_not_read")])


@pytest.mark.parametrize(
    ("raw", "skipped"),
    [
        (b"JIS\x00\x00\x00\x00\x00" + b"\x1b$B", [("UserComment", "unsupported_encoding")]),
        (b"OTHER\x00\x00\x00text", [("UserComment", "unsupported_encoding")]),
        (_UNICODE + "\ud800".encode("utf-16-le", "surrogatepass"), [("UserComment", "not_text")]),
        (b"ASCII\x00\x00\x00" + b"\xff\xfe", [("UserComment", "not_text")]),
        (b"ASCII\x00\x00\x00" + b"settings\x1b[2Jhidden", [("UserComment", "unsafe_text")]),
        (b"ASCII\x00\x00\x00" + "settings\u0085".encode(), [("UserComment", "unsafe_text")]),
        # How many cameras fill the field: nothing is there to list.
        (b"ASCII\x00\x00\x00" + b"\x00" * 16, []),
        (b"\x00" * 8 + b"   ", []),
        (b"   ", []),
        # The name padded with spaces, as cameras often write it.
        (b"ASCII   " + b" " * 36, []),
        (b" " * 8 + b" " * 12, []),
    ],
)
def test_a_value_that_cannot_be_read_safely_is_skipped_not_refused(
    raw: bytes, skipped: list[tuple[str, str]]
) -> None:
    payload = jpeg_with_exif(
        exif_block([ascii_tag(0x0110, "prompt:" + _GRAPH)], [(0x9286, 7, raw)])
    )

    assert _read(read_jpeg_text(payload)) == ([("prompt", _GRAPH)], skipped)


def test_a_comments_text_ends_at_its_first_null() -> None:
    raw = b"ASCII   " + b"Holiday\x00" + b" " * 20
    payload = jpeg_with_exif(exif_block([], [(0x9286, 7, raw)]))

    assert _read(read_jpeg_text(payload)) == ([("UserComment", "Holiday")], [])


def test_only_the_first_exif_block_is_read_and_an_oversized_one_is_skipped() -> None:
    first = exif_block([ascii_tag(0x0110, "prompt:" + _GRAPH)])
    second = exif_block([ascii_tag(0x0110, "workflow:" + _WORKFLOW)])
    repeated = jpeg((0xE1, b"Exif\x00\x00" + first), (0xE1, b"Exif\x00\x00" + second))
    oversized = webp(
        (b"EXIF", exif_block([ascii_tag(0x0110, "prompt:[" + " " * MAX_EXIF_BYTES + "]")]))
    )

    assert _read(read_jpeg_text(repeated)) == ([("prompt", _GRAPH)], [("EXIF", "repeated")])
    assert _read(read_webp_text(oversized)) == ([], [("EXIF", "too_long")])


def _damaged() -> dict[str, bytes]:
    good = exif_block([], comment(_SETTINGS))
    # The header, the first directory (its count, one pointer entry, the next
    # offset), then the Exif directory's count: its one entry starts here.
    entry_at = 8 + 2 + 12 + 4 + 2
    return {
        "segment past the end": b"\xff\xd8\xff\xe1\xff\xf0Exif\x00\x00",
        "segment shorter than its length field": b"\xff\xd8\xff\xe1\x00\x01" + b"\x00" * 8,
        "no image data": jpeg_with_exif(good)[:-14],
        "no marker": b"\xff\xd8\xff\xe0\x00\x04\x00\x00\x12\x34",
        "riff size past the end": webp((b"EXIF", good))[:-4],
        "chunk past the riff": b"RIFF"
        + struct.pack("<I", 16)
        + b"WEBPEXIF"
        + struct.pack("<I", 99)
        + b"\x00" * 4,
        "not tiff": jpeg_with_exif(b"XX*\x00\x08\x00\x00\x00" + good[8:]),
        "block shorter than a header": jpeg_with_exif(b"II*\x00"),
        "first directory outside": jpeg_with_exif(b"II*\x00" + struct.pack("<I", 4096)),
        "directory table past the block": jpeg_with_exif(b"II*\x00\x08\x00\x00\x00\x05\x00"),
        "too many entries": jpeg_with_exif(
            b"II*\x00\x08\x00\x00\x00" + struct.pack("<H", 1025) + b"\x00" * 12 * 1025
        ),
        "exif directory outside": jpeg_with_exif(
            exif_block([(0x8769, 4, struct.pack("<I", 1 << 30))])
        ),
        "exif pointer of two values": jpeg_with_exif(
            exif_block([(0x8769, 4, struct.pack("<II", 26, 26))])
        ),
        "comment past the block": jpeg_with_exif(
            good[:entry_at] + struct.pack("<HHI", 0x9286, 7, 1 << 31) + good[entry_at + 8 :]
        ),
    }


@pytest.mark.parametrize("case", sorted(_damaged()))
def test_a_damaged_container_or_exif_block_refuses_the_whole_file(case: str) -> None:
    payload = _damaged()[case]
    reader = read_webp_text if payload.startswith(b"RIFF") else read_jpeg_text

    with pytest.raises(ValueError):
        reader(payload)


def test_the_ceilings_hold_in_fixed_time() -> None:
    restarts = b"\xff\xd8" + b"\xff\xd0" * 5000 + b"\xff\xda\x00\x02"
    tiny = b"JUNK" + struct.pack("<I", 0)
    chunks = b"WEBP" + tiny * 100_001
    many = b"RIFF" + struct.pack("<I", len(chunks)) + chunks

    started = time.monotonic()
    with pytest.raises(ValueError):
        read_jpeg_text(restarts)
    with pytest.raises(ValueError):
        read_webp_text(many)
    assert time.monotonic() - started < 2


def test_a_file_of_another_kind_is_refused_by_each_reader() -> None:
    with pytest.raises(ValueError):
        read_jpeg_text(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError):
        read_webp_text(b"RIFF\x04\x00\x00\x00WAVE")
