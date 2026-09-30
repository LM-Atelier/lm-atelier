"""A revision's own size in the exact shape of a source picture, as it is shown."""

from __future__ import annotations

from io import BytesIO

from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from test_workflow_output_geometry_api import (
    _dimension,
    _generation_row_counts,
    _schema,
    _trusted_revision,
)
from test_workflow_output_geometry_api import geometry_runtime as geometry_runtime
from test_workflow_output_geometry_h3_video import _prove

from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.workflow_output_geometry import match_source_output_geometry


def picture(size: tuple[int, int], *, orientation: int | None = None, form: str = "PNG") -> bytes:
    output = BytesIO()
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    Image.new("RGB", size, (90, 120, 150)).save(output, format=form, exif=exif)
    return output.getvalue()


async def uploaded(client: AsyncClient, content: bytes, media_type: str = "image/png") -> str:
    name = "frame.jpg" if media_type == "image/jpeg" else "frame.png"
    response = await client.post("/api/artifacts", files={"file": (name, content, media_type)})
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def matched(revision_id: str) -> str:
    return f"/api/workflow-revisions/{revision_id}/output-geometry/match-source"


async def test_a_source_s_shape_resolves_to_the_revision_s_own_size_in_that_shape(
    client: AsyncClient,
) -> None:
    workflow_id, revision_id = await _trusted_revision(client, "Match source geometry")
    source = await uploaded(client, picture((300, 200)))
    capability = await client.get(f"/api/workflow-revisions/{revision_id}/output-geometry")
    workflows = (await client.get("/api/workflows")).json()
    rows = _generation_row_counts()

    response = await client.post(matched(revision_id), json={"source_artifact_id": source})

    assert response.status_code == 200, response.text
    # Worked by hand: 3:2 on a 64 grid is 192c by 128c, and c = 6 gives the area
    # nearest the 1024 x 768 default.
    assert response.json() == {
        "version": 1,
        "workflow_id": workflow_id,
        "revision_id": revision_id,
        "artifact_sha256": capability.json()["artifact_sha256"],
        "operation": "text_to_image",
        "engine": "comfyui",
        "mode": "image",
        "size_mode": "exact",
        "preset_id": None,
        "width": 1152,
        "height": 768,
        "graph_binding_verified": True,
        "request_authorized": False,
    }
    # Asking changes nothing: no workflow, message, plan, run, job or artifact.
    assert (await client.get("/api/workflows")).json() == workflows
    assert _generation_row_counts() == rows


async def test_a_picture_turned_on_its_side_is_matched_as_it_is_shown(client: AsyncClient) -> None:
    _, revision_id = await _trusted_revision(client, "Turned source geometry")
    # Stored 300 by 200 and turned a quarter, so it is shown 200 by 300.
    source = await uploaded(client, picture((300, 200), orientation=6))

    response = await client.post(matched(revision_id), json={"source_artifact_id": source})

    assert response.status_code == 200, response.text
    assert (response.json()["width"], response.json()["height"]) == (768, 1152)


async def test_a_photograph_is_matched_as_well_as_a_png(client: AsyncClient) -> None:
    _, revision_id = await _trusted_revision(client, "Photograph source geometry")
    source = await uploaded(client, picture((1600, 900), form="JPEG"), "image/jpeg")

    response = await client.post(matched(revision_id), json={"source_artifact_id": source})

    assert response.status_code == 200, response.text
    # 16:9 names the same pair the wide preset resolves to.
    assert (response.json()["width"], response.json()["height"]) == (1024, 576)


async def test_a_shape_the_revision_cannot_make_exactly_is_refused_not_approximated(
    client: AsyncClient,
) -> None:
    _, revision_id = await _trusted_revision(client, "Inexact source geometry")
    source = await uploaded(client, picture((1001, 999)))

    response = await client.post(matched(revision_id), json={"source_artifact_id": source})

    assert response.status_code == 422, response.text
    assert response.json()["code"] == "workflow-geometry-request-invalid"


async def test_a_revision_locked_to_one_size_matches_only_its_own_shape(
    client: AsyncClient,
) -> None:
    square = _dimension(1024, 1024, 1024)
    _, revision_id = await _trusted_revision(
        client, "Square source geometry", input_schema=_schema(width=square, height=dict(square))
    )
    squared = await uploaded(client, picture((500, 500)))
    wide = await uploaded(client, picture((300, 200)))

    same = await client.post(matched(revision_id), json={"source_artifact_id": squared})
    other = await client.post(matched(revision_id), json={"source_artifact_id": wide})

    assert same.status_code == 200, same.text
    assert (same.json()["width"], same.json()["height"]) == (1024, 1024)
    assert other.status_code == 422, other.text


async def test_a_missing_or_unreadable_source_is_refused(app: FastAPI, client: AsyncClient) -> None:
    _, revision_id = await _trusted_revision(client, "Unreadable source geometry")
    with SessionLocal() as session:
        broken = app.state.services.artifacts.ingest_bytes(
            session, b"not a picture", kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        session.commit()
        broken_id = broken.id

    missing = await client.post(
        matched(revision_id), json={"source_artifact_id": "sha256:" + "0" * 64}
    )
    unreadable = await client.post(matched(revision_id), json={"source_artifact_id": broken_id})
    unknown = await client.post(
        matched("does-not-exist"), json={"source_artifact_id": "sha256:" + "0" * 64}
    )

    assert missing.status_code == 404, missing.text
    assert unreadable.status_code == 422, unreadable.text
    assert unreadable.json()["code"] == "source-shape-unavailable"
    assert unknown.status_code == 404, unknown.text


def test_a_video_start_frame_is_matched_on_the_root_s_own_grid() -> None:
    result = _prove(operation="image_to_video")

    upright = match_source_output_geometry(result, 1080, 1920)
    portrait = match_source_output_geometry(result, 1080, 1350)

    # A 9:16 frame gives the pair the 9:16 preset gives.
    assert upright is not None
    assert (upright.geometry.mode, upright.geometry.size_mode) == ("video", "exact")
    assert (upright.geometry.width, upright.geometry.height) == (864, 1536)
    # Worked by hand: 4:5 on a 32 grid is 128c by 160c, and c = 7 gives the area
    # nearest the 1344 x 768 default.
    assert portrait is not None
    assert (portrait.geometry.width, portrait.geometry.height) == (896, 1120)
    assert match_source_output_geometry(result, 0, 1920) is None
    assert match_source_output_geometry(result, 1080.0, 1920) is None
