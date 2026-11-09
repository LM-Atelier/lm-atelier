"""Pictures Pillow recognises and then fails on, each in a way of its own.

Each is a few pixels of flat colour around one malformed part. Pillow refuses
them with different exceptions, and not only while opening them: some fail
the load, some the first read of their orientation, one only when its
orientation data is written back, and one only when it is converted to gain
an alpha. A reader that guards one call, or names the exceptions it expects,
lets the rest through.

Also here are two that read cleanly in modes a reader may not expect: grey
with premultiplied alpha, and 32-bit grey.
"""

from __future__ import annotations

import struct
import zlib

WIDTH, HEIGHT = 4, 3


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png(
    before: tuple[tuple[bytes, bytes], ...] = (), after: tuple[tuple[bytes, bytes], ...] = ()
) -> bytes:
    """An RGB PNG of WIDTH by HEIGHT with extra chunks before and after its pixels."""
    rows = b"".join(b"\x00" + bytes((90, 120, 150) * WIDTH) for _ in range(HEIGHT))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 2, 0, 0, 0))
        + b"".join(_chunk(kind, data) for kind, data in before)
        + _chunk(b"IDAT", zlib.compress(rows))
        + b"".join(_chunk(kind, data) for kind, data in after)
        + _chunk(b"IEND", b"")
    )


def with_chunk(png: bytes, kind: bytes, data: bytes, *, after_pixels: bool = False) -> bytes:
    """`png` with one more chunk, just before its first image data or just before its end."""
    at = png.index(b"IEND" if after_pixels else b"IDAT") - 4
    return png[:at] + _chunk(kind, data) + png[at:]


def _exif(*entries: bytes) -> bytes:
    """A big-endian TIFF block with one directory holding `entries`, as EXIF carries it."""
    return (
        b"MM\x00*"
        + struct.pack(">I", 8)
        + struct.pack(">H", len(entries))
        + b"".join(entries)
        + struct.pack(">I", 0)
    )


#: The picture turned a quarter, as a camera held on its side records it.
_TURNED = struct.pack(">HHI", 0x0112, 3, 1) + struct.pack(">HH", 6, 0)
#: A chunk body two bytes long, shorter than a gamma or physical-size chunk can be.
SHORT_BODY = b"\x00\x01"
#: An eXIf body naming itself as EXIF once more than a reader strips before its TIFF header.
DOUBLED_EXIF_BODY = b"Exif\x00\x00Exif\x00\x00" + _exif(_TURNED)
#: An eXIf body turning the picture, with its resolution stored as a single byte: it
#: reads, and cannot be written back once the turn is taken out of it.
RETYPED_RESOLUTION_BODY = _exif(_TURNED, struct.pack(">HHI", 0x011A, 1, 1) + bytes((72, 0, 0, 0)))

SHORT_GAMMA = _png(before=((b"gAMA", SHORT_BODY),))
SHORT_PHYSICAL_SIZE = _png(before=((b"pHYs", SHORT_BODY),))
#: The short gamma chunk after the pixels, where only the load reads it.
SHORT_GAMMA_AFTER_PIXELS = _png(after=((b"gAMA", SHORT_BODY),))
#: A colour profile chunk with nothing in it, after the pixels.
EMPTY_PROFILE_AFTER_PIXELS = _png(after=((b"iCCP", b""),))
DOUBLED_EXIF_PREFIX = _png(before=((b"eXIf", DOUBLED_EXIF_BODY),))
RETYPED_RESOLUTION = _png(before=((b"eXIf", RETYPED_RESOLUTION_BODY),))


def _tiff(entries: list[tuple[int, int, int, bytes | None]], pixels: bytes) -> bytes:
    """A little-endian TIFF.

    An entry whose value is None points at the pixels, and one whose value is
    longer than four bytes is stored after the directory, ahead of the pixels.
    """
    after_directory = 8 + 2 + 12 * len(entries) + 4
    stored = b"".join(value for *_, value in entries if value is not None and len(value) > 4)
    pixel_offset = after_directory + len(stored)
    directory = struct.pack("<H", len(entries))
    placed = after_directory
    for tag, kind, count, value in entries:
        if value is None:
            field = struct.pack("<I", pixel_offset)
        elif len(value) > 4:
            field = struct.pack("<I", placed)
            placed += len(value)
        else:
            field = value
        directory += struct.pack("<HHI", tag, kind, count) + field
    return b"II*\x00" + struct.pack("<I", 8) + directory + struct.pack("<I", 0) + stored + pixels


def _short(value: int) -> bytes:
    return struct.pack("<HH", value, 0)


def _long(value: int) -> bytes:
    return struct.pack("<I", value)


def _palette_tiff() -> bytes:
    """A palette TIFF whose colour map, stored in 32-bit entries, holds a value past 16 bits."""
    colours = [0] * 768
    colours[0] = 70_000
    # The colour map follows the pixels, so it sits at the pixels' offset plus their length.
    pixels = bytes(WIDTH * HEIGHT) + struct.pack("<768I", *colours)
    entries: list[tuple[int, int, int, bytes | None]] = [
        (256, 3, 1, _short(WIDTH)),
        (257, 3, 1, _short(HEIGHT)),
        (258, 3, 1, _short(8)),
        (259, 3, 1, _short(1)),
        (262, 3, 1, _short(3)),
        (273, 4, 1, None),
        (277, 3, 1, _short(1)),
        (278, 3, 1, _short(HEIGHT)),
        (279, 4, 1, _long(WIDTH * HEIGHT)),
        (320, 4, 768, _long(8 + 2 + 12 * 10 + 4 + WIDTH * HEIGHT)),
    ]
    return _tiff(entries, pixels)


WIDE_PALETTE_TIFF = _palette_tiff()

#: Pictures that fail while they are opened or decoded.
UNDECODABLE: dict[str, bytes] = {
    "short gamma": SHORT_GAMMA,
    "short physical size": SHORT_PHYSICAL_SIZE,
    "short gamma after the pixels": SHORT_GAMMA_AFTER_PIXELS,
    "empty profile after the pixels": EMPTY_PROFILE_AFTER_PIXELS,
    "colour map past sixteen bits": WIDE_PALETTE_TIFF,
}
#: Pictures that decode, and fail only once their orientation is read or taken out.
UNTURNABLE: dict[str, bytes] = {
    "doubled EXIF prefix": DOUBLED_EXIF_PREFIX,
    "orientation that cannot be written back": RETYPED_RESOLUTION,
}


def _palette_png() -> bytes:
    """A palette PNG of two colours whose transparency names three hundred."""
    rows = b"".join(b"\x00" + bytes(x % 2 for x in range(WIDTH)) for _ in range(HEIGHT))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", WIDTH, HEIGHT, 8, 3, 0, 0, 0))
        + _chunk(b"PLTE", bytes((90, 120, 150, 10, 20, 30)))
        + _chunk(b"tRNS", bytes(range(256)) + bytes(44))
        + _chunk(b"IDAT", zlib.compress(rows))
        + _chunk(b"IEND", b"")
    )


#: A picture that decodes and turns, and fails only when it is converted to gain an alpha.
TRANSPARENCY_PAST_THE_PALETTE = _palette_png()
#: Pictures that decode, and fail only once they are converted to gain an alpha.
UNCONVERTIBLE: dict[str, bytes] = {"transparency past the palette": TRANSPARENCY_PAST_THE_PALETTE}
#: Every picture above: each recognised, then failed on.
UNREADABLE: dict[str, bytes] = {**UNDECODABLE, **UNTURNABLE, **UNCONVERTIBLE}
#: The pictures above whose header or orientation cannot be read, for a reader that
#: reads only those: a PNG decodes whole to read an orientation it has not met yet.
UNREADABLE_SHAPE: dict[str, bytes] = {
    name: UNREADABLE[name]
    for name in (
        "short gamma",
        "short physical size",
        "short gamma after the pixels",
        "empty profile after the pixels",
        "doubled EXIF prefix",
    )
}


def _grey_tiff(
    width: int, height: int, bits: int, extra_samples: int | None, sample_format: int = 1
) -> bytes:
    samples = 1 if extra_samples is None else 2
    pixel_bytes = bits // 8 * samples
    pixels = bytes(((x * 61) % 256) for x in range(width * height * pixel_bytes))
    entries: list[tuple[int, int, int, bytes | None]] = [
        (256, 4, 1, _long(width)),
        (257, 4, 1, _long(height)),
        (258, 3, samples, _short(bits) if samples == 1 else struct.pack("<HH", bits, bits)),
        (259, 3, 1, _short(1)),
        (262, 3, 1, _short(1)),
        (273, 4, 1, None),
        (277, 3, 1, _short(samples)),
        (278, 4, 1, _long(height)),
        (279, 4, 1, _long(len(pixels))),
        (284, 3, 1, _short(1)),
    ]
    if extra_samples is not None:
        entries.append((338, 3, 1, _short(extra_samples)))
    if sample_format != 1:
        entries.append((339, 3, 1, _short(sample_format)))
    return _tiff(entries, pixels)


def premultiplied_grey_tiff(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Grey with premultiplied ("associated") alpha, which Pillow reads as La."""
    return _grey_tiff(width, height, 8, extra_samples=1)


def thirty_two_bit_grey_tiff(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Grey in signed 32-bit samples, which Pillow reads as I."""
    return _grey_tiff(width, height, 32, extra_samples=None, sample_format=2)


def grey_with_an_unnamed_sample_tiff(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Grey with a second sample it does not name, which Pillow 13 reads as L and drops."""
    return _grey_tiff(width, height, 8, extra_samples=0)


def runtime_unreadable(width: int = WIDTH, height: int = HEIGHT) -> dict[str, bytes]:
    """TIFF layouts Pillow 13 reads and Pillow 12.3, the managed media runtime's, refuses."""
    return {
        "premultiplied grey": premultiplied_grey_tiff(width, height),
        "grey with an unnamed sample": grey_with_an_unnamed_sample_tiff(width, height),
    }


PREMULTIPLIED_GREY_TIFF = premultiplied_grey_tiff()
THIRTY_TWO_BIT_GREY_TIFF = thirty_two_bit_grey_tiff()


def _cielab_tiff() -> bytes:
    pixels = bytes((128, 128, 128) * (WIDTH * HEIGHT))
    entries: list[tuple[int, int, int, bytes | None]] = [
        (256, 3, 1, _short(WIDTH)),
        (257, 3, 1, _short(HEIGHT)),
        (258, 3, 3, struct.pack("<HHH", 8, 8, 8)),
        (259, 3, 1, _short(1)),
        (262, 3, 1, _short(8)),
        (273, 4, 1, None),
        (277, 3, 1, _short(3)),
        (278, 3, 1, _short(HEIGHT)),
        (279, 4, 1, _long(len(pixels))),
        (284, 3, 1, _short(1)),
    ]
    return _tiff(entries, pixels)


#: A CIELab TIFF: it reads, and converts to RGB, but not to the grey a selection is read as.
CIELAB_TIFF = _cielab_tiff()
