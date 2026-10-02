"""Saving a generated picture together with its record, through the real route."""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
import zlib
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, ImageCms, PngImagePlugin
from run_waits import wait_for_terminal_status

from local_lm import output_recipe_bundle
from local_lm.db import SessionLocal
from local_lm.domain import ArtifactKind
from local_lm.models import Run
from local_lm.output_recipe_bundle import (
    MANIFEST_FILE,
    PICTURE_FILE,
    RECORD_FILE,
    OutputRecipeBundleFormatError,
    open_output_recipe_bundle,
    seal_bundle_manifest,
)
from local_lm.output_recipe_v1 import open_output_recipe

#: Text a saved picture carries in its own file, standing in for the graph an
#: engine writes there.
_EMBEDDED_MARKER = "embedded-graph-marker-5d1c"
_ORDER = (MANIFEST_FILE, RECORD_FILE, PICTURE_FILE)


async def _generated(client: AsyncClient, prompt: str = "a ceramic cup on a wooden table") -> str:
    chat = (await client.post("/api/chats", json={"title": "Bundle"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": prompt, "mode": "image"}
    )
    assert turn.status_code == 202, turn.text
    run_id = cast(str, turn.json()["run"]["id"])

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    await wait_for_terminal_status(read, what=f"run {run_id}", expected="complete")
    return run_id


def _gradient() -> Image.Image:
    picture = Image.new("RGB", (8, 6))
    picture.putdata([(x * 30, y * 40, 90) for y in range(6) for x in range(8)])
    return picture


def _png(picture: Image.Image, **options: Any) -> bytes:
    buffer = io.BytesIO()
    picture.save(buffer, format=options.pop("format", "PNG"), **options)
    return buffer.getvalue()


def _engine_png() -> bytes:
    words = PngImagePlugin.PngInfo()
    words.add_text("workflow", json.dumps({"note": _EMBEDDED_MARKER}))
    words.add_text("Comment", _EMBEDDED_MARKER)
    return _png(_gradient(), pnginfo=words)


def _make_output(app: FastAPI, run_id: str, content: bytes, **entry: Any) -> str:
    """Make the run's first output a stored picture, as an engine that saves files would."""

    with SessionLocal() as session:
        artifact = app.state.services.artifacts.ingest_bytes(
            session, content, kind=ArtifactKind.IMAGE, media_type="image/png"
        )
        stored = session.get(Run, run_id)
        assert stored is not None
        outputs = [dict(item) for item in stored.provenance_json["outputs"]]
        outputs[0] = {**outputs[0], "artifact_id": artifact.id, **entry}
        stored.provenance_json = {**stored.provenance_json, "outputs": outputs}
        session.commit()
        return cast(str, artifact.id)


async def _record(client: AsyncClient, run_id: str, artifact_id: str, prompts: str) -> Any:
    return await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe", params={"prompts": prompts}
    )


async def _bundle(
    client: AsyncClient, run_id: str, artifact_id: str, prompts: str, digest: str
) -> Any:
    return await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe-bundle",
        params={"prompts": prompts, "digest": digest},
    )


async def _bundled(app: FastAPI, client: AsyncClient, stored: bytes) -> Any:
    """The bundle of a fresh run whose output is `stored`, with the prompt left out."""

    run_id = await _generated(client)
    artifact_id = _make_output(app, run_id, stored)
    digest = (await _record(client, run_id, artifact_id, "omit")).headers["x-output-recipe-digest"]
    return await _bundle(client, run_id, artifact_id, "omit", digest)


def _chunk_types(png: bytes) -> list[str]:
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    types, offset = [], 8
    while offset < len(png):
        (length,) = struct.unpack(">I", png[offset : offset + 4])
        types.append(png[offset + 4 : offset + 8].decode("ascii"))
        offset += 12 + length
    return types


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png_with_header(header: bytes) -> bytes:
    """A PNG of a given header, one row of pixel data and an end."""

    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(b"\x00" * 8))
        + _chunk(b"IEND", b"")
    )


def _entries(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _picture_in(content: bytes) -> Image.Image:
    copy = Image.open(io.BytesIO(_entries(content)[PICTURE_FILE]))
    copy.load()
    return copy


def _rezip(
    entries: dict[str, bytes],
    *,
    names: tuple[str, ...] = _ORDER,
    compression: int = zipfile.ZIP_STORED,
    date_time: tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0),
    extra: bytes = b"",
    mode: int = 0o100644,
    comment: bytes = b"",
) -> bytes:
    """The three entries in order under `names`; a fourth name gets a small file."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index, name in enumerate(names):
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.create_system = 3
            info.external_attr = mode << 16
            info.compress_type = compression
            info.extra = extra
            data = entries[_ORDER[index]] if index < len(_ORDER) else b"extra"
            archive.writestr(info, data)
        archive.comment = comment
    return buffer.getvalue()


async def test_a_picture_saves_with_its_record_and_a_copy_holding_only_its_pixels(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id = await _generated(client)
    stored = _engine_png()
    artifact_id = _make_output(app, run_id, stored)
    shown = await _record(client, run_id, artifact_id, "include")
    assert shown.status_code == 200, shown.text
    digest = shown.headers["x-output-recipe-digest"]

    response = await _bundle(client, run_id, artifact_id, "include", digest)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-output-recipe-digest"] == digest
    bundle = open_output_recipe_bundle(response.content)
    entries = _entries(response.content)
    assert list(entries) == list(_ORDER)
    # The record inside is the record that was shown, byte for byte.
    assert entries[RECORD_FILE] == shown.content
    assert bundle["record"]["digest"] == digest
    # The copy has the stored picture's pixels and none of what its file carried.
    stored_sha256 = hashlib.sha256(stored).hexdigest()
    assert bundle["manifest"]["picture"]["copy_of"] == stored_sha256
    assert bundle["manifest"]["picture"]["sha256"] != stored_sha256
    assert bundle["manifest"]["picture"]["metadata"] == "none"
    assert _chunk_types(entries[PICTURE_FILE]) == ["IHDR", "IDAT", "IEND"]
    assert _EMBEDDED_MARKER.encode() not in response.content
    copy = _picture_in(response.content)
    assert copy.mode == "RGB"
    assert copy.tobytes() == _gradient().tobytes()
    assert run_id.encode() not in response.content
    # Every entry is written the same way on every computer.
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.compress_type == zipfile.ZIP_STORED
    again = await _bundle(client, run_id, artifact_id, "include", digest)
    assert again.content == response.content


async def test_a_color_profile_is_applied_to_the_pixels_and_never_copied(
    app: FastAPI, client: AsyncClient
) -> None:
    """A profile can carry text of its own, so the copy is converted to sRGB instead."""

    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    readable = await _bundled(app, client, _png(_gradient(), icc_profile=profile))
    unreadable = await _bundled(app, client, _png(_gradient(), icc_profile=b"not a profile"))

    for response in (readable, unreadable):
        assert response.status_code == 200, response.text
        assert _chunk_types(_entries(response.content)[PICTURE_FILE]) == ["IHDR", "IDAT", "IEND"]
        assert profile not in response.content
    # sRGB to sRGB leaves the pixels where they were; an unreadable profile is
    # left out and the pixels kept exactly.
    applied = _picture_in(readable.content).tobytes()
    assert max(abs(a - b) for a, b in zip(applied, _gradient().tobytes(), strict=True)) <= 1
    assert _picture_in(unreadable.content).tobytes() == _gradient().tobytes()


async def test_a_copy_keeps_sixteen_bit_grey_and_transparency(
    app: FastAPI, client: AsyncClient
) -> None:
    grey = Image.new("I;16", (8, 1))
    grey.putdata([0, 1000, 8000, 16000, 32000, 48000, 60000, 65535])
    palette = Image.new("P", (2, 1))
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    palette.putdata([0, 1])

    deep = await _bundled(app, client, _png(grey))
    clear = await _bundled(app, client, _png(palette, transparency=0))

    assert deep.status_code == 200, deep.text
    assert _picture_in(deep.content).mode == "I;16"
    assert _picture_in(deep.content).tobytes() == grey.tobytes()
    assert clear.status_code == 200, clear.text
    copy = _picture_in(clear.content)
    assert copy.mode == "RGBA"
    assert copy.getpixel((0, 0)) == (255, 0, 0, 0)
    assert copy.getpixel((1, 0)) == (0, 255, 0, 255)


async def test_a_picture_that_cannot_be_copied_whole_is_refused_with_a_code(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed file is refused as unreadable rather than failing the request."""

    short_header = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", b"\x00\x00\x00\x04\x00\x00\x00\x04\x08")
        + _chunk(b"IEND", b"")
    )
    exif = Image.Exif()
    exif[0x0112] = 6
    with_exif = _png(Image.new("RGB", (4, 4)), format="WEBP", exif=exif.tobytes())
    at = with_exif.find(b"MM\x00*")
    assert at > 0
    broken_exif = with_exif[:at] + b"$" + with_exif[at + 1 :]
    frames = [Image.new("RGB", (2, 2), (value, 0, 0)) for value in (0, 90, 180)]
    animated = _png(frames[0], format="GIF", save_all=True, append_images=frames[1:])

    for stored in (short_header, broken_exif, animated):
        response = await _bundled(app, client, stored)
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "output-recipe-bundle-unreadable"

    monkeypatch.setattr("local_lm.studio_region_edit.MAX_BLEND_PIXELS", 10)
    too_large = await _bundled(app, client, _png(_gradient()))
    assert too_large.status_code == 422, too_large.text
    assert too_large.json()["code"] == "output-recipe-bundle-too-large"


async def test_a_record_that_changed_since_it_was_shown_is_not_bundled(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id = await _generated(client)
    artifact_id = _make_output(app, run_id, _engine_png())
    without_prompt = (await _record(client, run_id, artifact_id, "omit")).headers[
        "x-output-recipe-digest"
    ]

    switched = await _bundle(client, run_id, artifact_id, "include", without_prompt)
    unknown = await _bundle(client, run_id, artifact_id, "omit", "sha256:" + "0" * 64)

    assert switched.status_code == 409
    assert switched.json()["code"] == "output-recipe-changed"
    assert unknown.status_code == 409
    assert unknown.json()["code"] == "output-recipe-changed"
    assert (await _bundle(client, run_id, artifact_id, "omit", without_prompt)).status_code == 200


async def test_the_record_shown_and_the_prompt_choice_are_both_required(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id = await _generated(client)
    artifact_id = _make_output(app, run_id, _engine_png())
    path = f"/api/runs/{run_id}/outputs/{artifact_id}/recipe-bundle"

    no_digest = await client.get(path, params={"prompts": "omit"})
    no_choice = await client.get(path, params={"digest": "sha256:" + "0" * 64})
    malformed = await client.get(path, params={"prompts": "omit", "digest": "0" * 64})

    assert no_digest.status_code == 422
    assert no_choice.status_code == 422
    assert malformed.status_code == 422


async def test_only_a_picture_that_can_be_decoded_is_bundled(
    app: FastAPI, client: AsyncClient
) -> None:
    drawn = await _generated(client)
    drawn_output = (await client.get(f"/api/runs/{drawn}")).json()["provenance_json"]["outputs"][0][
        "artifact_id"
    ]
    filmed = await _generated(client, "a glass vase by a window")
    filmed_output = _make_output(app, filmed, _engine_png(), kind="video")

    for run_id, artifact_id, code in (
        # The test engine draws its pictures as SVG, which has no pixels to copy.
        (drawn, drawn_output, "output-recipe-bundle-unreadable"),
        (filmed, filmed_output, "output-recipe-bundle-picture-only"),
    ):
        digest = (await _record(client, run_id, artifact_id, "omit")).headers[
            "x-output-recipe-digest"
        ]
        response = await _bundle(client, run_id, artifact_id, "omit", digest)
        assert response.status_code == 422, (code, response.text)
        assert response.json()["code"] == code


async def test_a_bundle_that_is_not_exactly_as_written_is_refused(
    app: FastAPI, client: AsyncClient
) -> None:
    written = (await _bundled(app, client, _engine_png())).content
    entries = _entries(written)
    manifest = json.loads(entries[MANIFEST_FILE])

    def resealed(*, version: object = 1, **sections: dict[str, Any]) -> bytes:
        unsigned = {key: value for key, value in manifest.items() if key != "digest"}
        unsigned["version"] = version
        for section, values in sections.items():
            unsigned[section] = {**unsigned[section], **values}
        return seal_bundle_manifest(unsigned)

    def with_picture(picture: bytes) -> bytes:
        """The bundle around another picture, its manifest resealed to match it."""

        return _rezip(
            {
                **entries,
                PICTURE_FILE: picture,
                MANIFEST_FILE: resealed(picture={"sha256": hashlib.sha256(picture).hexdigest()}),
            }
        )

    words = PngImagePlugin.PngInfo()
    words.add_text("Comment", _EMBEDDED_MARKER)
    with_words = _png(_gradient(), pnginfo=words)
    encrypted = bytearray(written)
    central = written.find(b"PK\x01\x02")
    encrypted[central + 8] |= 1
    reordered = (RECORD_FILE, MANIFEST_FILE, PICTURE_FILE)
    altered = {
        "an extra entry": _rezip(entries, names=(*_ORDER, "notes.txt")),
        "a missing entry": _rezip(entries, names=_ORDER[:2]),
        # Each entry keeps its own bytes; only the order changes.
        "entries in another order": _rezip(
            {
                MANIFEST_FILE: entries[RECORD_FILE],
                RECORD_FILE: entries[MANIFEST_FILE],
                PICTURE_FILE: entries[PICTURE_FILE],
            },
            names=reordered,
        ),
        "a name with a path": _rezip(entries, names=(MANIFEST_FILE, RECORD_FILE, "x/output.png")),
        "a link entry": _rezip(entries, mode=0o120777),
        "a compressed entry": _rezip(entries, compression=zipfile.ZIP_DEFLATED),
        "another time": _rezip(entries, date_time=(2024, 5, 6, 7, 8, 10)),
        "an extra field": _rezip(entries, extra=b"\x99\x99\x00\x00"),
        "an archive comment": _rezip(entries, comment=b"note"),
        "bytes before the archive": b"<html></html>" + written,
        "an encrypted entry": bytes(encrypted),
        "another picture": _rezip({**entries, PICTURE_FILE: _png(Image.new("RGB", (2, 2)))}),
        "a manifest out of canonical form": _rezip(
            {**entries, MANIFEST_FILE: json.dumps(manifest, indent=1).encode()}
        ),
        "a manifest whose digest is another": _rezip(
            {
                **entries,
                MANIFEST_FILE: json.dumps(
                    {**manifest, "digest": "sha256:" + "0" * 64},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            }
        ),
        "a version that only equals one": _rezip(
            {**entries, MANIFEST_FILE: resealed(version=True)}
        ),
        "a manifest holding NaN": _rezip({**entries, MANIFEST_FILE: b'{"x":NaN}'}),
        "a picture said to be of another output": _rezip(
            {**entries, MANIFEST_FILE: resealed(picture={"copy_of": "2" * 64})}
        ),
        "a picture carrying text": _rezip(
            {
                **entries,
                PICTURE_FILE: with_words,
                MANIFEST_FILE: resealed(picture={"sha256": hashlib.sha256(with_words).hexdigest()}),
            }
        ),
        "a picture header claiming no width": with_picture(
            _png_with_header(struct.pack(">IIBBBBB", 0, 6, 8, 2, 0, 0, 0))
        ),
        "a picture header of the wrong length": with_picture(
            _png_with_header(struct.pack(">IIB", 8, 6, 8))
        ),
        "a picture header past the decode limit": with_picture(
            _png_with_header(struct.pack(">IIBBBBB", 0xFFFFFFFF, 0xFFFFFFFF, 8, 2, 0, 0, 0))
        ),
        "a picture in a layout the copy never uses": with_picture(
            _png_with_header(struct.pack(">IIBBBBB", 8, 6, 8, 3, 0, 0, 0))
        ),
        "a record that does not open": _rezip({**entries, RECORD_FILE: b"{}"}),
        "a record with a number too long to read": _rezip(
            {**entries, RECORD_FILE: b'{"version":' + b"7" * 5000 + b"}"}
        ),
        "not a ZIP at all": entries[RECORD_FILE],
    }

    assert _rezip(entries) == written
    assert open_output_recipe_bundle(written)["record"]["digest"] == manifest["record"]["digest"]
    assert open_output_recipe(entries[RECORD_FILE])["digest"] == manifest["record"]["digest"]
    refused = []
    for name, content in altered.items():
        try:
            open_output_recipe_bundle(content)
        except OutputRecipeBundleFormatError:
            refused.append(name)
    assert refused == list(altered)


async def _check(client: AsyncClient, content: Any) -> Any:
    return await client.post(
        "/api/output-recipes/check",
        content=content,
        headers={"content-type": "application/octet-stream"},
    )


async def test_a_bundle_checks_as_its_record_does_and_names_its_picture(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id = await _generated(client)
    stored = _engine_png()
    artifact_id = _make_output(app, run_id, stored)
    shown = await _record(client, run_id, artifact_id, "omit")
    digest = shown.headers["x-output-recipe-digest"]
    bundle = await _bundle(client, run_id, artifact_id, "omit", digest)

    as_record = await _check(client, shown.content)
    as_bundle = await _check(client, bundle.content)

    assert as_record.status_code == 200, as_record.text
    assert as_bundle.status_code == 200, as_bundle.text
    record_report = as_record.json()
    bundle_report = as_bundle.json()
    assert record_report.pop("picture") is None
    assert bundle_report.pop("picture") == {
        "sha256": json.loads(_entries(bundle.content)[MANIFEST_FILE])["picture"]["sha256"],
        "copy_of": hashlib.sha256(stored).hexdigest(),
        "width": 8,
        "height": 6,
    }
    assert bundle_report == record_report
    assert bundle_report["digest"] == digest


async def test_a_bundle_that_is_not_as_written_is_refused_by_the_check(
    app: FastAPI, client: AsyncClient
) -> None:
    written = (await _bundled(app, client, _engine_png())).content

    for content in (written + b"trailing", _rezip(_entries(written), comment=b"note")):
        response = await _check(client, content)
        assert response.status_code == 422, response.text
        assert response.json()["code"] == "output-recipe-bundle-unreadable"
        assert "note" not in response.text


async def test_the_check_reads_no_more_than_the_kind_of_file_allows(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bundle is read to a bundle's bound; anything else only to a record's."""

    monkeypatch.setattr("local_lm.output_recipe_check.MAX_CHECK_BYTES", 4096)
    monkeypatch.setattr("local_lm.output_recipe_check.MAX_RECORD_BYTES", 1024)
    sent = {"declared": 0, "bundle": 0, "record": 0}

    def streamed(kind: str, start: bytes) -> Any:
        async def parts() -> Any:
            for index in range(64):
                sent[kind] += 1
                yield (start if index == 0 else b"") + b"x" * 512

        return parts()

    declared = await client.post(
        "/api/output-recipes/check",
        content=streamed("declared", b"PK\x03\x04"),
        headers={"content-type": "application/octet-stream", "content-length": str(64 * 512 + 4)},
    )
    bundle = await _check(client, streamed("bundle", b"PK\x03\x04"))
    record = await _check(client, streamed("record", b"{"))

    for response in (declared, bundle, record):
        assert response.status_code == 413, response.text
        assert response.json()["code"] == "output-recipe-too-large"
    # Refused on what it declared, before a part was read.
    assert sent["declared"] == 0
    # Stopped once past each bound: 4096 bytes for a bundle, 1024 for anything else.
    assert sent["bundle"] == 8
    assert sent["record"] == 2


def test_a_directory_listing_many_entries_is_refused_before_it_is_parsed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsing every listed entry would cost far more than the file's own size."""

    record = b"PK\x01\x02" + b"\x00" * 42
    count = 2000
    crafted = (
        b"PK\x03\x04"
        + b"\x00" * 26
        + record * count
        + struct.pack("<4sHHHHIIH", b"PK\x05\x06", 0, 0, count, count, len(record) * count, 30, 0)
    )

    def parsed(*_arguments: object, **_options: object) -> None:
        raise AssertionError("the archive was parsed")

    monkeypatch.setattr(output_recipe_bundle.zipfile, "ZipFile", parsed)

    with pytest.raises(OutputRecipeBundleFormatError):
        open_output_recipe_bundle(crafted)
