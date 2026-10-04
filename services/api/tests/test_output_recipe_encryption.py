"""Saving a generation record encrypted under a passphrase, and opening one again."""

from __future__ import annotations

import base64
import io
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from run_waits import wait_for_terminal_status

from local_lm import output_recipe_api, output_recipe_check, project_archive_encryption
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Run
from local_lm.portable_archive_v1 import MAGIC, ArchiveKind, write_archive

PASSPHRASE = "correct horse battery staple"
PROMPT = "a ceramic cup on a wooden table"


async def _generated(client: AsyncClient) -> str:
    chat = (await client.post("/api/chats", json={"title": "Encrypted record"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": PROMPT, "mode": "image"}
    )
    assert turn.status_code == 202, turn.text
    run_id = cast(str, turn.json()["run"]["id"])

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    await wait_for_terminal_status(read, what=f"run {run_id}", expected="complete")
    return run_id


def _picture() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (40, 90, 160)).save(buffer, format="PNG")
    return buffer.getvalue()


def _make_output(app: FastAPI, run_id: str) -> str:
    """Make the run's first output a stored picture, as an engine that saves files would."""

    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session, _picture(), kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        stored = session.get(Run, run_id)
        assert stored is not None
        outputs = [dict(item) for item in stored.provenance_json["outputs"]]
        outputs[0] = {**outputs[0], "artifact_id": artifact.id}
        stored.provenance_json = {**stored.provenance_json, "outputs": outputs}
        session.commit()
        return cast(str, artifact.id)


async def _shown(app: FastAPI, client: AsyncClient) -> tuple[str, str, Any]:
    run_id = await _generated(client)
    artifact_id = _make_output(app, run_id)
    shown = await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe", params={"prompts": "include"}
    )
    assert shown.status_code == 200, shown.text
    return run_id, artifact_id, shown


def _header(passphrase: str) -> dict[str, str]:
    return {
        "content-type": "application/octet-stream",
        "x-archive-passphrase": base64.b64encode(passphrase.encode()).decode(),
    }


async def _open(client: AsyncClient, content: bytes, passphrase: str = PASSPHRASE) -> Any:
    return await client.post(
        "/api/output-recipes/open", content=content, headers=_header(passphrase)
    )


async def test_an_encrypted_record_opens_to_the_record_that_was_shown(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id, artifact_id, shown = await _shown(app, client)
    digest = shown.headers["x-output-recipe-digest"]

    sealed = await client.post(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe/encrypted",
        json={"passphrase": PASSPHRASE, "prompts": "include", "digest": digest},
    )

    assert sealed.status_code == 200, sealed.text
    assert sealed.headers["content-type"] == "application/octet-stream"
    assert sealed.headers["cache-control"] == "private, no-store"
    assert sealed.headers["content-disposition"].endswith(".json.encrypted")
    assert sealed.content.startswith(MAGIC)
    assert PROMPT.encode() not in sealed.content and PASSPHRASE.encode() not in sealed.content
    opened = await _open(client, sealed.content)
    assert opened.status_code == 200, opened.text
    assert opened.headers["cache-control"] == "private, no-store"
    assert opened.content == shown.content
    checked = await client.post(
        "/api/output-recipes/check",
        content=opened.content,
        headers={"content-type": "application/octet-stream"},
    )
    assert checked.status_code == 200 and checked.json()["digest"] == digest


async def test_an_encrypted_bundle_opens_to_the_bundle_the_plain_download_makes(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id, artifact_id, shown = await _shown(app, client)
    digest = shown.headers["x-output-recipe-digest"]
    plain = await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe-bundle",
        params={"prompts": "include", "digest": digest},
    )

    sealed = await client.post(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe-bundle/encrypted",
        json={"passphrase": PASSPHRASE, "prompts": "include", "digest": digest},
    )

    assert sealed.status_code == 200, sealed.text
    assert sealed.headers["content-disposition"].endswith(".zip.encrypted")
    assert sealed.content.startswith(MAGIC) and PROMPT.encode() not in sealed.content
    assert (await _open(client, sealed.content)).content == plain.content


async def test_a_record_that_changed_since_it_was_shown_is_not_encrypted(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id, artifact_id, shown = await _shown(app, client)
    # The digest of the record with the prompt, asked for without it.
    body = {
        "passphrase": PASSPHRASE,
        "prompts": "omit",
        "digest": shown.headers["x-output-recipe-digest"],
    }

    for path in ("recipe/encrypted", "recipe-bundle/encrypted"):
        refused = await client.post(f"/api/runs/{run_id}/outputs/{artifact_id}/{path}", json=body)
        assert (refused.status_code, refused.json()["code"]) == (409, "output-recipe-changed")
        assert PASSPHRASE not in refused.text


async def test_an_encrypted_record_that_does_not_open_again_is_not_sent(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, artifact_id, shown = await _shown(app, client)
    real = project_archive_encryption.write_archive

    def damaging(source: Any, destination: Any, **kwargs: Any) -> int:
        # A byte past the end, which no honest archive has.
        return real(source, destination, **kwargs) + destination.write(b"\x00")

    monkeypatch.setattr(project_archive_encryption, "write_archive", damaging)
    refused = await client.post(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe/encrypted",
        json={
            "passphrase": PASSPHRASE,
            "prompts": "include",
            "digest": shown.headers["x-output-recipe-digest"],
        },
    )

    assert (refused.status_code, refused.json()["code"]) == (
        500,
        "output-recipe-encryption-unverified",
    )
    assert refused.content.find(MAGIC) == -1


async def test_a_file_that_cannot_be_opened_gives_nothing_back(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id, artifact_id, shown = await _shown(app, client)
    sealed = (
        await client.post(
            f"/api/runs/{run_id}/outputs/{artifact_id}/recipe/encrypted",
            json={
                "passphrase": PASSPHRASE,
                "prompts": "include",
                "digest": shown.headers["x-output-recipe-digest"],
            },
        )
    ).content
    damaged = bytearray(sealed)
    damaged[-1] ^= 0x01

    for content, passphrase in (
        (sealed, "a wrong passphrase"),
        (bytes(damaged), PASSPHRASE),
        (sealed[:-20], PASSPHRASE),
    ):
        refused = await _open(client, content, passphrase)
        assert (refused.status_code, refused.json()["code"]) == (
            422,
            "archive-passphrase-or-archive-invalid",
        )
        assert PROMPT not in refused.text and passphrase not in refused.text

    missing = await client.post(
        "/api/output-recipes/open",
        content=sealed,
        headers={"content-type": "application/octet-stream"},
    )
    assert (missing.status_code, missing.json()["code"]) == (422, "archive-passphrase-required")
    malformed = await client.post(
        "/api/output-recipes/open",
        content=sealed,
        headers={"content-type": "application/octet-stream", "x-archive-passphrase": "not base64!"},
    )
    assert (malformed.status_code, malformed.json()["code"]) == (422, "archive-passphrase-invalid")


async def test_a_project_archive_is_not_opened_as_a_record(client: AsyncClient) -> None:
    project = io.BytesIO()
    write_archive(
        io.BytesIO(b"PK neutral project bytes"),
        project,
        kind=ArchiveKind.PROJECT,
        passphrase=PASSPHRASE.encode(),
    )

    refused = await _open(client, project.getvalue())

    assert (refused.status_code, refused.json()["code"]) == (422, "archive-kind-mismatch")
    assert refused.json()["detail"] == "This file is not an encrypted generation record."


async def test_a_file_larger_than_a_record_can_be_is_refused(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    sealed = io.BytesIO()
    write_archive(
        io.BytesIO(b"x" * 64),
        sealed,
        kind=ArchiveKind.OUTPUT_RECIPE,
        passphrase=PASSPHRASE.encode(),
    )

    monkeypatch.setattr(output_recipe_api, "MAX_CHECK_BYTES", 32)
    opens_too_large = await _open(client, sealed.getvalue())
    monkeypatch.setattr(output_recipe_api, "_MAX_ENCRYPTED_BYTES", 16)
    sent_too_large = await _open(client, sealed.getvalue())

    for refused in (opens_too_large, sent_too_large):
        assert (refused.status_code, refused.json()["code"]) == (413, "output-recipe-too-large")


async def test_an_opened_file_that_is_not_a_bundle_is_held_to_a_records_bound(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    record, bundle = io.BytesIO(), io.BytesIO()
    passphrase = PASSPHRASE.encode()
    write_archive(
        io.BytesIO(b"{" + b" " * 62 + b"}"),
        record,
        kind=ArchiveKind.OUTPUT_RECIPE,
        passphrase=passphrase,
    )
    write_archive(
        io.BytesIO(b"PK\x03\x04" + b"x" * 60),
        bundle,
        kind=ArchiveKind.OUTPUT_RECIPE,
        passphrase=passphrase,
    )
    monkeypatch.setattr(output_recipe_check, "MAX_RECORD_BYTES", 32)

    as_record = await _open(client, record.getvalue())
    as_bundle = await _open(client, bundle.getvalue())

    assert (as_record.status_code, as_record.json()["code"]) == (413, "output-recipe-too-large")
    assert as_bundle.status_code == 200 and as_bundle.content.startswith(b"PK\x03\x04")
