"""Turning, mirroring, cropping and resizing a studio picture, each recorded as a step."""

from __future__ import annotations

import io
from typing import Any

import pytest
from httpx2 import AsyncClient
from PIL import Image, ImageCms

from local_lm.studio_local_edits import CropBox, LocalEditError, PictureSize, render_local_edit

RED, GREEN, BLUE, WHITE = (200, 0, 0), (0, 200, 0), (0, 0, 200), (250, 250, 250)


def _png(image: Image.Image, **options: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **options)
    return buffer.getvalue()


def _tiles() -> Image.Image:
    """Three by two, with a different colour in each corner that matters."""

    picture = Image.new("RGB", (3, 2), WHITE)
    picture.putpixel((0, 0), RED)
    picture.putpixel((2, 0), GREEN)
    picture.putpixel((0, 1), BLUE)
    return picture


def _open(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


@pytest.mark.parametrize(
    ("operation", "size", "red_at"),
    [
        # Top-left goes top-right when turned clockwise, and bottom-left the other way.
        ("rotate_clockwise", (2, 3), (1, 0)),
        ("rotate_counterclockwise", (2, 3), (0, 2)),
        ("flip_horizontal", (3, 2), (2, 0)),
        ("flip_vertical", (3, 2), (0, 1)),
    ],
)
def test_each_turn_and_flip_moves_every_pixel_exactly(
    operation: Any, size: tuple[int, int], red_at: tuple[int, int]
) -> None:
    result = _open(render_local_edit(_png(_tiles()), operation))

    assert result.format == "PNG"
    assert result.size == size
    assert result.getpixel(red_at) == RED
    # Nothing is resampled: the same colours, the same number of each.
    colours = result.convert("RGB").getcolors()
    assert colours is not None and sorted(colours) == sorted(_tiles().getcolors() or [])


def test_a_crop_keeps_exactly_the_box() -> None:
    result = _open(
        render_local_edit(_png(_tiles()), "crop", CropBox(left=1, top=0, width=2, height=1))
    )

    assert result.size == (2, 1)
    assert [result.getpixel((x, 0)) for x in range(2)] == [WHITE, GREEN]


@pytest.mark.parametrize(
    "box",
    [
        CropBox(left=0, top=0, width=4, height=1),
        CropBox(left=2, top=1, width=2, height=1),
        CropBox(left=0, top=0, width=0, height=1),
        CropBox(left=-1, top=0, width=1, height=1),
    ],
)
def test_a_crop_outside_the_picture_is_refused(box: CropBox) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "crop", box)

    assert refused.value.code == "studio-crop-outside-picture"


@pytest.mark.parametrize("size", [PictureSize(width=6, height=4), PictureSize(width=1, height=1)])
def test_a_resize_makes_exactly_the_size_asked_for(size: PictureSize) -> None:
    result = _open(render_local_edit(_png(_tiles()), "resize", size=size))

    assert result.format == "PNG"
    assert result.size == (size.width, size.height)
    assert result.mode == "RGB"


def test_a_resize_keeps_a_hidden_colour_from_bleeding_into_the_edge() -> None:
    """Under a transparent pixel there is still a colour, and nobody can see it."""
    picture = Image.new("RGBA", (2, 1))
    picture.putpixel((0, 0), (255, 0, 0, 0))
    picture.putpixel((1, 0), (0, 0, 255, 255))

    result = _open(render_local_edit(_png(picture), "resize", size=PictureSize(width=8, height=1)))

    assert result.mode == "RGBA"
    seen = [result.getpixel((x, 0)) for x in range(8)]
    assert any(0 < pixel[3] < 255 for pixel in seen)
    # Every pixel that shows at all shows blue: none of the hidden red.
    assert all(pixel[0] == 0 for pixel in seen if pixel[3] > 0)


@pytest.mark.parametrize(
    ("size", "code"),
    [
        (None, "studio-resize-missing"),
        (PictureSize(width=3, height=2), "studio-resize-unchanged"),
        (PictureSize(width=0, height=2), "studio-resize-empty"),
        (PictureSize(width=20_000, height=20_000), "studio-edit-too-large"),
    ],
)
def test_a_resize_that_cannot_be_made_is_refused(size: PictureSize | None, code: str) -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(_png(_tiles()), "resize", size=size)

    assert refused.value.code == code


def test_a_picture_is_edited_as_it_is_seen_upright() -> None:
    """A camera's orientation tag is how the person saw it, so it is what they turned."""
    exif = Image.Exif()
    exif[0x0112] = 6  # stored sideways, shown turned a quarter clockwise
    stored = _png(_tiles(), exif=exif.tobytes())

    result = _open(render_local_edit(stored, "flip_horizontal"))

    assert result.size == (2, 3)
    # The tag is spent: the pixels are now the way up they were seen.
    assert result.getexif().get(0x0112) in (None, 1)


def test_transparency_survives_a_turn() -> None:
    picture = Image.new("RGBA", (2, 1), (0, 0, 0, 0))
    picture.putpixel((0, 0), (10, 20, 30, 255))

    result = _open(render_local_edit(_png(picture), "flip_horizontal"))

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0))[3] == 0
    assert result.getpixel((1, 0)) == (10, 20, 30, 255)


def test_a_colour_profile_is_kept_only_where_it_still_describes_the_pixels() -> None:
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

    kept = _open(render_local_edit(_png(_tiles(), icc_profile=profile), "flip_vertical"))
    grey = Image.new("L", (2, 2), 90)
    # A grey picture's profile would misdescribe the RGB it is turned into.
    dropped = _open(render_local_edit(_png(grey, icc_profile=profile), "flip_vertical"))

    assert kept.info.get("icc_profile") == profile
    assert dropped.mode == "RGB"
    assert "icc_profile" not in dropped.info


def test_bytes_that_are_not_a_picture_are_refused() -> None:
    with pytest.raises(LocalEditError) as refused:
        render_local_edit(b"not a picture", "rotate_clockwise")

    assert refused.value.code == "studio-edit-unreadable"


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    response = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert response.status_code == 201, response.text
    identifier: str = response.json()["id"]
    return identifier


async def _session_over(client: AsyncClient, artifact_id: str) -> str:
    opened = await client.post("/api/studio/sessions", json={"source_artifact_id": artifact_id})
    assert opened.status_code == 200, opened.text
    session_id: str = opened.json()["id"]
    return session_id


async def _content(client: AsyncClient, artifact_id: str) -> Image.Image:
    response = await client.get(f"/api/artifacts/{artifact_id}/content")
    assert response.status_code == 200
    return _open(response.content)


async def test_a_turn_becomes_the_next_step_after_the_picture_it_changed(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "rotate_clockwise"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    request, answer = body["messages"]
    assert request["role"] == "user"
    assert [part["type"] for part in request["parts"]] == ["text", "image"]
    assert request["parts"][0]["text"] == "Rotate right"
    # First among the request's pictures is the one changed: compare shows it.
    assert request["parts"][1]["artifact_id"] == source_id
    assert answer["role"] == "assistant"
    assert answer["status"] == "complete"
    assert answer["parent_id"] == request["id"]
    assert body["active_head_message_id"] == answer["id"]
    image, metadata = answer["parts"]
    assert image["type"] == "image"
    assert metadata["type"] == "generation_metadata"
    # Says what was done to which picture, and names no model, because none ran.
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {"operation": "rotate_clockwise", "source_artifact_id": source_id}
    }
    turned = await _content(client, image["artifact_id"])
    assert turned.size == (2, 3)
    assert turned.getpixel((1, 0)) == RED

    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.status_code == 200
    assert detail.json()["kind"] == "image"
    assert detail.json()["original_name"] == "tiles (rotated right).png"
    assert detail.json()["generation_identity"] is None
    library = await client.get("/api/artifact-library", params={"kind": "image"})
    assert image["artifact_id"] in {row["artifact_id"] for row in library.json()["items"]}


async def test_a_result_already_in_the_session_can_be_turned_again(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)
    first = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "flip_horizontal"},
    )
    flipped = first.json()["messages"][-1]["parts"][0]["artifact_id"]

    second = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": flipped,
            "operation": "crop",
            "crop": {"left": 0, "top": 0, "width": 1, "height": 1},
        },
    )

    assert second.status_code == 200, second.text
    messages = second.json()["messages"]
    assert len(messages) == 4
    assert messages[2]["parent_id"] == messages[1]["id"]
    assert messages[2]["parts"][1]["artifact_id"] == flipped
    cropped = await _content(client, messages[3]["parts"][0]["artifact_id"])
    # The flip put green top-left; the crop kept only that pixel.
    assert cropped.size == (1, 1)
    assert cropped.getpixel((0, 0)) == GREEN


async def test_a_resize_is_recorded_with_its_size_and_method(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    response = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "resize",
            "size": {"width": 30, "height": 20},
        },
    )

    assert response.status_code == 200, response.text
    request, answer = response.json()["messages"]
    assert request["parts"][0]["text"] == "Resize"
    image, metadata = answer["parts"]
    assert metadata["metadata_json"]["provenance"] == {
        "local_edit": {
            "operation": "resize",
            "source_artifact_id": source_id,
            "size": {"width": 30, "height": 20},
            "resampler": "lanczos",
        }
    }
    resized = await _content(client, image["artifact_id"])
    assert resized.size == (30, 20)
    detail = await client.get(f"/api/artifacts/{image['artifact_id']}")
    assert detail.json()["original_name"] == "tiles (resized).png"


async def test_only_a_picture_in_the_session_can_be_edited_through_it(
    client: AsyncClient,
) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    elsewhere = await _upload(client, "other.png", _png(Image.new("RGB", (2, 2), BLUE)))
    session_id = await _session_over(client, source_id)

    refused = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": elsewhere, "operation": "rotate_clockwise"},
    )

    assert refused.status_code == 422
    assert refused.json()["code"] == "studio-edit-source-not-in-session"
    after = await client.get(f"/api/studio/sessions/{session_id}")
    assert after.json()["messages"] == []


async def test_a_refused_edit_leaves_the_session_as_it_was(client: AsyncClient) -> None:
    source_id = await _upload(client, "tiles.png", _png(_tiles()))
    session_id = await _session_over(client, source_id)

    outside = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "crop",
            "crop": {"left": 2, "top": 0, "width": 2, "height": 1},
        },
    )
    missing_box = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "crop"},
    )
    stray_box = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "flip_vertical",
            "crop": {"left": 0, "top": 0, "width": 1, "height": 1},
        },
    )
    missing_size = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={"source_artifact_id": source_id, "operation": "resize"},
    )
    stray_size = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "rotate_clockwise",
            "size": {"width": 4, "height": 4},
        },
    )
    unchanged = await client.post(
        f"/api/studio/sessions/{session_id}/local-edits",
        json={
            "source_artifact_id": source_id,
            "operation": "resize",
            "size": {"width": 3, "height": 2},
        },
    )
    absent = await client.post(
        "/api/studio/sessions/absent/local-edits",
        json={"source_artifact_id": source_id, "operation": "flip_vertical"},
    )

    assert outside.status_code == 422
    assert outside.json()["code"] == "studio-crop-outside-picture"
    assert missing_box.status_code == 422
    assert stray_box.status_code == 422
    assert missing_size.status_code == 422
    assert stray_size.status_code == 422
    assert unchanged.status_code == 422
    assert unchanged.json()["code"] == "studio-resize-unchanged"
    assert absent.status_code == 404
    assert absent.json()["code"] == "studio-session-not-found"
    after = await client.get(f"/api/studio/sessions/{session_id}")
    assert after.json()["messages"] == []
