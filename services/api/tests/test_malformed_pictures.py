"""A malformed picture is refused as unreadable wherever it is read, never as a server error.

Pillow fails on a malformed file in more ways than one and at more points than
opening it. Each case sends each file in `malformed_pictures` that its reader
can meet through a real route, so a reader that guards only some of those
ways, or only some of those points, shows here.
"""

from __future__ import annotations

import io

import pytest
from httpx2 import AsyncClient
from malformed_pictures import (
    CIELAB_TIFF,
    HEIGHT,
    PREMULTIPLIED_GREY_TIFF,
    UNCONVERTIBLE,
    UNDECODABLE,
    UNREADABLE,
    UNTURNABLE,
    WIDTH,
)
from PIL import Image


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _open(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def _picture() -> bytes:
    return _png(Image.new("RGB", (WIDTH, HEIGHT), (40, 80, 120)))


def _left_half_marked() -> bytes:
    mask = Image.new("L", (WIDTH, HEIGHT), 0)
    mask.paste(255, (0, 0, WIDTH // 2, HEIGHT))
    return _png(mask)


def _drawn_words() -> bytes:
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    overlay.putpixel((1, 1), (250, 250, 250, 255))
    return _png(overlay)


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    media_type = "image/tiff" if content.startswith(b"II*\x00") else "image/png"
    response = await client.post("/api/artifacts", files={"file": (name, content, media_type)})
    assert response.status_code == 201, response.text
    identifier: str = response.json()["id"]
    return identifier


async def _session_over(client: AsyncClient, artifact_id: str) -> str:
    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    assert opened.status_code == 200, opened.text
    session_id: str = opened.json()["id"]
    return session_id


async def _edit(
    client: AsyncClient, source: bytes, **marks: bytes
) -> tuple[int, dict[str, object]]:
    """Rotate `source`, or blur it, caption it or place a subject on it through what is given."""
    source_id = await _upload(client, "picture.png", source)
    session_id = await _session_over(client, source_id)
    request: dict[str, object] = {"source_artifact_id": source_id, "operation": "rotate_clockwise"}
    if "mask" in marks:
        mask_id = await _upload(client, "studio-selection.png", marks["mask"])
        request |= {"operation": "blur", "blur": {"mask_artifact_id": mask_id, "radius": 1}}
    if "words" in marks:
        words_id = await _upload(client, "studio-words.png", marks["words"])
        request |= {"operation": "caption", "caption": {"overlay_artifact_id": words_id}}
    if "subject" in marks:
        subject_id = await _upload(client, "studio-subject.png", marks["subject"])
        request |= {"operation": "subject", "subject": {"overlay_artifact_id": subject_id}}
    response = await client.post(f"/api/studio/sessions/{session_id}/local-edits", json=request)
    return response.status_code, response.json()


@pytest.mark.parametrize("name", sorted(UNREADABLE))
async def test_the_studio_refuses_to_edit_a_malformed_picture(
    client: AsyncClient, name: str
) -> None:
    status, body = await _edit(client, UNREADABLE[name])

    assert status == 422, body
    assert body["code"] == "studio-edit-unreadable"


#: A marked area and drawn words are never turned, and are converted to gain an alpha.
_UNUSABLE_MARKS = {**UNDECODABLE, **UNCONVERTIBLE}


@pytest.mark.parametrize("name", sorted(_UNUSABLE_MARKS))
async def test_the_studio_refuses_a_malformed_marked_area_or_drawing(
    client: AsyncClient, name: str
) -> None:
    blurred_status, blurred = await _edit(client, _picture(), mask=_UNUSABLE_MARKS[name])
    captioned_status, captioned = await _edit(client, _picture(), words=_UNUSABLE_MARKS[name])

    assert blurred_status == 422, blurred
    assert blurred["code"] == "region-image-unreadable"
    assert captioned_status == 422, captioned
    assert captioned["code"] == "studio-caption-unreadable"


@pytest.mark.parametrize("name", sorted(UNTURNABLE))
async def test_a_marked_area_is_never_read_for_an_orientation_it_does_not_use(
    client: AsyncClient, name: str
) -> None:
    """A selection is drawn upright on the picture, so its own orientation data is ignored."""

    status, body = await _edit(client, _picture(), mask=_carrying_marks(UNTURNABLE[name]))

    assert status == 200, body


def _carrying_marks(malformed: bytes) -> bytes:
    """`malformed` with its pixels replaced by a left-half selection, chunks and all kept."""
    selection = _open(_left_half_marked()).convert("RGB")
    picture = _open(_without_exif(malformed))
    assert picture.size == selection.size
    marked = _png(selection)
    # Graft the marked pixels into the malformed file: everything before its
    # image data is its own, everything from there on is the selection's.
    return malformed[: malformed.index(b"IDAT") - 4] + marked[marked.index(b"IDAT") - 4 :]


def _without_exif(payload: bytes) -> bytes:
    at = payload.index(b"eXIf") - 4
    length = int.from_bytes(payload[at : at + 4], "big")
    return payload[:at] + payload[at + 12 + length :]


async def test_the_studio_edits_grey_with_premultiplied_alpha(client: AsyncClient) -> None:
    status, body = await _edit(client, PREMULTIPLIED_GREY_TIFF)

    assert status == 200, body
    turned = await _result(client, body)
    assert turned.mode == "RGBA"
    assert turned.size == (HEIGHT, WIDTH)


async def test_grey_with_premultiplied_alpha_marks_an_area_and_is_laid_over(
    client: AsyncClient,
) -> None:
    blurred_status, blurred = await _edit(client, _picture(), mask=PREMULTIPLIED_GREY_TIFF)
    captioned_status, captioned = await _edit(client, _picture(), words=PREMULTIPLIED_GREY_TIFF)
    placed_status, placed = await _edit(client, _picture(), subject=PREMULTIPLIED_GREY_TIFF)

    assert blurred_status == 200, blurred
    assert captioned_status == 200, captioned
    assert placed_status == 200, placed
    assert (await _result(client, captioned)).size == (WIDTH, HEIGHT)
    assert (await _result(client, placed)).size == (WIDTH, HEIGHT)


async def test_a_marked_area_that_reads_but_not_as_grey_is_refused(client: AsyncClient) -> None:
    """A CIELab picture converts to RGB, and not to the grey a marked area is read as."""

    status, body = await _edit(client, _picture(), mask=CIELAB_TIFF)

    assert status == 422, body
    assert body["code"] == "region-image-unreadable"


async def _result(client: AsyncClient, body: dict[str, object]) -> Image.Image:
    messages = body["messages"]
    assert isinstance(messages, list)
    image_part = messages[-1]["parts"][0]
    response = await client.get(f"/api/artifacts/{image_part['artifact_id']}/content")
    assert response.status_code == 200
    return _open(response.content)


@pytest.mark.parametrize("name", sorted(UNREADABLE))
async def test_a_malformed_picture_is_refused_for_export(client: AsyncClient, name: str) -> None:
    picture_id = await _upload(client, "picture.png", UNREADABLE[name])

    response = await client.get(f"/api/artifacts/{picture_id}/export", params={"format": "png"})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "picture-export-unreadable"


async def test_grey_with_premultiplied_alpha_exports_with_its_transparency(
    client: AsyncClient,
) -> None:
    picture_id = await _upload(client, "grey.tif", PREMULTIPLIED_GREY_TIFF)

    response = await client.get(f"/api/artifacts/{picture_id}/export", params={"format": "png"})

    assert response.status_code == 200, response.text
    exported = _open(response.content)
    assert exported.mode == "RGBA"
    assert exported.size == (WIDTH, HEIGHT)
