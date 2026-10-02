"""Exporting a picture in another format: upright, profiled, and flattened only for JPEG."""

from __future__ import annotations

import io
import struct
import zlib
from typing import Any
from urllib.parse import unquote

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageCms, PngImagePlugin

from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.picture_export import (
    ExportFormat,
    PictureExportError,
    export_file_name,
    export_picture,
)


def _pixel(picture: Image.Image, xy: tuple[int, int]) -> tuple[int, ...]:
    """One pixel's channels: every picture read this way has more than one."""
    value = picture.getpixel(xy)
    assert isinstance(value, tuple)
    return value


RED, WHITE = (200, 0, 0), (255, 255, 255)


def _encoded(image: Image.Image, file_format: str = "PNG", **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=file_format, **options)
    return buffer.getvalue()


def _open(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def _wide() -> Image.Image:
    """Four by two, red in the top-left corner and white elsewhere."""

    picture = Image.new("RGB", (4, 2), WHITE)
    picture.putpixel((0, 0), RED)
    return picture


def test_a_picture_comes_out_upright_without_its_orientation_tag() -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, shown turned a quarter clockwise
    stored = _encoded(_wide(), exif=exif.tobytes())

    exported = _open(export_picture(stored, "png"))

    assert exported.format == "PNG"
    assert exported.size == (2, 4)
    assert exported.getexif().get(0x0112) in (None, 1)


@pytest.mark.parametrize("file_format", ["png", "jpeg", "webp"])
def test_an_export_leaves_behind_the_text_and_camera_details_the_file_carried(
    file_format: ExportFormat,
) -> None:
    # A generated picture can carry the words and settings it was made with in
    # its text chunks; a file from a camera, who and what took it.
    words = PngImagePlugin.PngInfo()
    words.add_text("parameters", "neutral words kept with the file")
    details = Image.Exif()
    details[0x010F] = "Neutral Camera"  # the maker
    details[0x0131] = "Neutral Editor"  # the software
    stored = _encoded(_wide(), pnginfo=words, exif=details.tobytes())
    original = _open(stored)
    assert isinstance(original, PngImagePlugin.PngImageFile)
    assert original.text == {"parameters": "neutral words kept with the file"}

    exported = _open(export_picture(stored, file_format))

    assert "parameters" not in exported.info
    assert getattr(exported, "text", {}) == {}
    assert exported.getexif().get(0x010F) is None
    assert exported.getexif().get(0x0131) is None


def test_jpeg_lays_transparency_on_white() -> None:
    picture = Image.new("RGBA", (4, 4), (0, 0, 200, 255))
    for x in range(2):
        for y in range(4):
            picture.putpixel((x, y), (200, 0, 0, 0))

    exported = _open(export_picture(_encoded(picture), "jpeg", quality=95))

    assert exported.format == "JPEG"
    assert exported.mode == "RGB"
    # The hidden red under the transparent half does not show; white does.
    red, green, blue = _pixel(exported, (0, 0))
    assert min(red, green, blue) > 240


def test_png_and_webp_keep_transparency() -> None:
    picture = Image.new("RGBA", (4, 4), (0, 0, 200, 255))
    picture.putpixel((0, 0), (0, 0, 0, 0))

    cases: list[tuple[ExportFormat, str]] = [("png", "PNG"), ("webp", "WEBP")]
    for file_format, expected in cases:
        exported = _open(export_picture(_encoded(picture), file_format))

        assert exported.format == expected
        assert exported.mode == "RGBA"
        assert _pixel(exported, (0, 0))[3] == 0


def test_a_color_profile_is_kept_only_where_it_still_describes_the_pixels() -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    kept = _open(export_picture(_encoded(_wide(), icc_profile=profile), "jpeg"))
    grey = Image.new("L", (2, 2), 90)
    dropped = _open(export_picture(_encoded(grey, icc_profile=profile), "jpeg"))

    assert kept.info.get("icc_profile") == profile
    assert "icc_profile" not in dropped.info


def test_bytes_that_are_not_a_picture_are_refused() -> None:
    with pytest.raises(PictureExportError) as refused:
        export_picture(b"not a picture", "png")

    assert refused.value.code == "picture-export-unreadable"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _malformed_pictures() -> list[bytes]:
    """Files Pillow recognises and then fails on in ways other than an OSError."""

    short_header = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", b"\x00\x00\x00\x04\x00\x00\x00\x04\x08")
        + _chunk(b"IEND", b"")
    )
    orientation = Image.Exif()
    orientation[0x0112] = 6
    with_exif = _encoded(Image.new("RGB", (4, 4)), "WEBP", exif=orientation.tobytes())
    at = with_exif.find(b"MM\x00*")
    assert at > 0
    broken_orientation = with_exif[:at] + b"$" + with_exif[at + 1 :]
    return [short_header, broken_orientation]


@pytest.mark.parametrize("file_format", ["png", "jpeg", "webp"])
def test_a_malformed_picture_is_refused_rather_than_failing(file_format: ExportFormat) -> None:
    for payload in _malformed_pictures():
        with pytest.raises(PictureExportError) as refused:
            export_picture(payload, file_format)
        assert refused.value.code == "picture-export-unreadable"


def test_sixteen_bit_grey_stays_whole_in_png_and_is_scaled_in_eight_bit_formats() -> None:
    ramp = Image.new("I;16", (4, 1))
    ramp.putdata([0, 1000, 32896, 65535])
    mid_grey = Image.new("I;16", (8, 8), 32896)

    png = _open(export_picture(_encoded(ramp), "png"))

    assert png.mode == "I;16"
    assert png.tobytes() == ramp.tobytes()
    for file_format in ("jpeg", "webp"):
        flat = _open(export_picture(_encoded(mid_grey), file_format)).convert("L")
        # Clipped to eight bits this would be white; scaled it is the middle.
        assert abs(flat.getpixel((4, 4)) - 128) <= 3, file_format


def test_the_file_is_named_after_the_picture() -> None:
    assert export_file_name("holiday (cropped).png", "jpeg") == "holiday (cropped).jpg"
    assert export_file_name(None, "webp") == "picture.webp"


async def _upload(client: AsyncClient, name: str, content: bytes, media_type: str) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, media_type)})
    assert response.status_code == 201, response.text
    identifier: str = response.json()["id"]
    return identifier


async def test_the_export_route_answers_with_an_attachment_in_the_chosen_format(
    client: AsyncClient,
) -> None:
    picture_id = await _upload(client, "wide picture.png", _encoded(_wide()), "image/png")

    response = await client.get(
        f"/api/artifacts/{picture_id}/export", params={"format": "jpeg", "quality": 80}
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-content-type-options"] == "nosniff"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment; filename*=utf-8''")
    assert unquote(disposition.split("''", 1)[1]) == "wide picture.jpg"
    assert _open(response.content).size == (4, 2)


async def test_the_export_route_refuses_what_it_cannot_export(client: AsyncClient) -> None:
    picture_id = await _upload(client, "wide.png", _encoded(_wide()), "image/png")
    notes_id = await _upload(client, "notes.txt", b"plain words", "text/plain")

    strange_format = await client.get(
        f"/api/artifacts/{picture_id}/export", params={"format": "gif"}
    )
    too_good = await client.get(
        f"/api/artifacts/{picture_id}/export", params={"format": "jpeg", "quality": 101}
    )
    not_a_picture = await client.get(f"/api/artifacts/{notes_id}/export", params={"format": "png"})
    missing = await client.get(
        "/api/artifacts/sha256:" + "0" * 64 + "/export", params={"format": "png"}
    )

    assert strange_format.status_code == 422
    assert too_good.status_code == 422
    assert not_a_picture.status_code == 422
    assert not_a_picture.json()["code"] == "artifact-not-a-picture"
    assert missing.status_code == 404


async def test_the_export_route_refuses_a_malformed_picture_with_a_code(
    app: FastAPI, client: AsyncClient
) -> None:
    """An engine's output is stored as it arrived, so a malformed one can reach here."""

    with SessionLocal() as session:
        stored = [
            app.state.services.artifacts.ingest_bytes(
                session, payload, kind=ArtifactKind.IMAGE, media_type="image/png"
            ).id
            for payload in _malformed_pictures()
        ]
        session.commit()

    for picture_id in stored:
        response = await client.get(f"/api/artifacts/{picture_id}/export", params={"format": "png"})
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "picture-export-unreadable"
