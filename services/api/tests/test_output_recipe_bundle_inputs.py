"""A record's input pictures, carried in its bundle only when asked for."""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image, PngImagePlugin
from test_output_recipe_bundle import (
    _EMBEDDED_MARKER,
    _check,
    _chunk_types,
    _engine_png,
    _entries,
    _generated,
    _make_output,
    _record,
)
from test_output_recipe_replay import _edit_chat, _finished, _profile, _workflow
from test_output_recipe_replay_masks import MASKED_SCHEMA

from local_lm import output_recipe_bundle
from local_lm.db import SessionLocal
from local_lm.output_recipe import OutputRecipeUnavailable
from local_lm.output_recipe_bundle import (
    MANIFEST_FILE,
    MAX_BUNDLE_INPUTS,
    PICTURE_FILE,
    RECORD_FILE,
    OutputRecipeBundleFormatError,
    input_file,
    open_output_recipe_bundle,
    seal_bundle_manifest,
)
from local_lm.output_recipe_v1 import open_output_recipe, seal_output_recipe


def _marked_png(colour: tuple[int, int, int]) -> bytes:
    """A small picture whose file carries text an export must never copy."""

    words = PngImagePlugin.PngInfo()
    words.add_text("Comment", _EMBEDDED_MARKER)
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), colour).save(buffer, format="PNG", pnginfo=words)
    return buffer.getvalue()


async def _uploaded(client: AsyncClient, content: bytes) -> str:
    uploaded = await client.post(
        "/api/artifacts", files={"file": ("still.png", content, "image/png")}
    )
    assert uploaded.status_code == 201, uploaded.text
    return cast(str, uploaded.json()["id"])


async def _edited(app: FastAPI, client: AsyncClient) -> tuple[str, str, str, str]:
    """An edit of an uploaded picture inside a selection, with a picture as its output.

    Returns the run, its output, the source picture and the selection.
    """

    source = await _uploaded(client, _marked_png((40, 90, 160)))
    mask = await _uploaded(client, _marked_png((255, 255, 255)))
    revision_id = await _workflow(client, "image_to_image", MASKED_SCHEMA)
    turn = await client.post(
        f"/api/chats/{await _edit_chat(client)}/turns",
        json={
            "text": "make the cup blue",
            "mode": "image",
            "profile_id": _profile(),
            "workflow_revision_id": revision_id,
            "input_artifact_ids": [source],
            "settings": {"seed": 1234, "mask": {"artifact_id": mask}},
        },
    )
    assert turn.status_code == 202, turn.text
    run = await _finished(client, turn.json()["run"]["id"])
    output = _make_output(app, run["id"], _engine_png())
    return run["id"], output, source, mask


async def _bundle(
    client: AsyncClient, run_id: str, artifact_id: str, digest: str, inputs: str | None
) -> Any:
    params = {"prompts": "omit", "digest": digest}
    if inputs is not None:
        params["inputs"] = inputs
    return await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe-bundle", params=params
    )


async def _with_inputs(app: FastAPI, client: AsyncClient) -> tuple[bytes, str, str]:
    run_id, output, source, mask = await _edited(app, client)
    digest = (await _record(client, run_id, output, "omit")).headers["x-output-recipe-digest"]
    response = await _bundle(client, run_id, output, digest, "include")
    assert response.status_code == 200, response.text
    return cast(bytes, response.content), source, mask


def _resealed(content: bytes, change: Any) -> bytes:
    """The bundle's files after `change`, with the manifest sealed again to match."""

    files = _entries(content)
    manifest = json.loads(files[MANIFEST_FILE])
    manifest.pop("digest")
    change(files, manifest)
    files[MANIFEST_FILE] = seal_bundle_manifest(manifest)
    return output_recipe_bundle._zip(files)


async def test_an_edit_bundle_carries_its_input_pictures_only_when_asked(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id, output, source, mask = await _edited(app, client)
    digest = (await _record(client, run_id, output, "omit")).headers["x-output-recipe-digest"]

    without = await _bundle(client, run_id, output, digest, None)
    omitted = await _bundle(client, run_id, output, digest, "omit")
    included = await _bundle(client, run_id, output, digest, "include")

    assert without.status_code == omitted.status_code == included.status_code == 200
    # Left out, the bundle is exactly as before.
    assert without.content == omitted.content
    assert list(_entries(omitted.content)) == [MANIFEST_FILE, RECORD_FILE, PICTURE_FILE]
    assert json.loads(_entries(omitted.content)[MANIFEST_FILE])["version"] == 1
    entries = _entries(included.content)
    assert list(entries) == [MANIFEST_FILE, RECORD_FILE, PICTURE_FILE, "input-1.png", "input-2.png"]
    # The record inside is the same record, and the output the same copy.
    assert entries[RECORD_FILE] == _entries(omitted.content)[RECORD_FILE]
    assert entries[PICTURE_FILE] == _entries(omitted.content)[PICTURE_FILE]
    bundle = open_output_recipe_bundle(included.content)
    manifest = bundle["manifest"]
    assert manifest["version"] == 2
    assert [
        (line["file"], line["position"], line["copy_of"], line["metadata"])
        for line in manifest["inputs"]
    ] == [
        ("input-1.png", 0, source.removeprefix("sha256:"), "none"),
        ("input-2.png", 1, mask.removeprefix("sha256:"), "none"),
    ]
    assert [(item["role"], item["width"], item["height"]) for item in bundle["inputs"]] == [
        ("source", 8, 6),
        ("mask", 8, 6),
    ]
    for line in manifest["inputs"]:
        copy = entries[line["file"]]
        # A copy holds the stored picture's pixels and nothing else from its file.
        assert line["sha256"] == hashlib.sha256(copy).hexdigest() != line["copy_of"]
        assert _chunk_types(copy) == ["IHDR", "IDAT", "IEND"]
    # Each copy holds its own picture: the source, then the selection.
    for name, colour in (("input-1.png", (40, 90, 160)), ("input-2.png", (255, 255, 255))):
        pixels = Image.open(io.BytesIO(entries[name]))
        assert pixels.convert("RGB").getpixel((0, 0)) == colour, name
    assert _EMBEDDED_MARKER.encode() not in included.content
    assert run_id.encode() not in included.content
    again = await _bundle(client, run_id, output, digest, "include")
    assert again.content == included.content


async def test_a_picture_made_without_inputs_bundles_the_same_either_way(
    app: FastAPI, client: AsyncClient
) -> None:
    run_id = await _generated(client)
    output = _make_output(app, run_id, _engine_png())
    digest = (await _record(client, run_id, output, "omit")).headers["x-output-recipe-digest"]

    omitted = await _bundle(client, run_id, output, digest, "omit")
    included = await _bundle(client, run_id, output, digest, "include")

    assert omitted.status_code == included.status_code == 200
    assert included.content == omitted.content


async def test_an_input_that_cannot_be_read_refuses_the_bundle_with_a_code(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id, output, source, _mask = await _edited(app, client)
    digest = (await _record(client, run_id, output, "omit")).headers["x-output-recipe-digest"]
    store = type(app.state.services.artifacts)
    read = store.verified_bytes

    def unreadable(self: Any, artifact: Any, *, maximum_bytes: int) -> bytes:
        if artifact.id == source:
            raise OSError("gone")
        return cast(bytes, read(self, artifact, maximum_bytes=maximum_bytes))

    monkeypatch.setattr(store, "verified_bytes", unreadable)

    refused = await _bundle(client, run_id, output, digest, "include")
    still = await _bundle(client, run_id, output, digest, "omit")

    assert refused.status_code == 410, refused.text
    assert refused.json()["code"] == "output-recipe-bundle-input-unreadable"
    assert still.status_code == 200, still.text


def test_inputs_are_bounded_and_must_all_be_here(app: FastAPI) -> None:
    artifacts = app.state.services.artifacts
    absent = {"sha256": "0" * 64, "role": "input", "size_bytes": None, "media_type": None}
    with SessionLocal() as session:
        with pytest.raises(OutputRecipeUnavailable) as too_many:
            output_recipe_bundle._input_copies(
                session, artifacts, [absent] * (MAX_BUNDLE_INPUTS + 1), 10**9
            )
        with pytest.raises(OutputRecipeUnavailable) as missing:
            output_recipe_bundle._input_copies(session, artifacts, [absent], 10**9)

    assert (too_many.value.status, too_many.value.code) == (
        422,
        "output-recipe-bundle-too-many-inputs",
    )
    assert (missing.value.status, missing.value.code) == (
        409,
        "output-recipe-bundle-input-missing",
    )


async def test_inputs_that_do_not_fit_the_bundle_refuse_it(
    app: FastAPI, client: AsyncClient
) -> None:
    source = await _uploaded(client, _marked_png((40, 90, 160)))
    recorded = {
        "sha256": source.removeprefix("sha256:"),
        "role": "source",
        "size_bytes": None,
        "media_type": None,
    }
    with SessionLocal() as session, pytest.raises(OutputRecipeUnavailable) as refused:
        output_recipe_bundle._input_copies(session, app.state.services.artifacts, [recorded], 10)

    assert (refused.value.status, refused.value.code) == (422, "output-recipe-bundle-too-large")


async def test_an_input_that_cannot_be_copied_is_named_as_an_input(
    app: FastAPI, client: AsyncClient
) -> None:
    frames = io.BytesIO()
    Image.new("RGB", (8, 6), (1, 2, 3)).save(
        frames,
        format="GIF",
        save_all=True,
        append_images=[Image.new("RGB", (8, 6), (4, 5, 6))],
    )
    animated = await client.post(
        "/api/artifacts", files={"file": ("moving.gif", frames.getvalue(), "image/gif")}
    )
    assert animated.status_code == 201, animated.text
    recorded = {
        "sha256": animated.json()["id"].removeprefix("sha256:"),
        "role": "source",
        "size_bytes": None,
        "media_type": None,
    }
    with SessionLocal() as session, pytest.raises(OutputRecipeUnavailable) as refused:
        output_recipe_bundle._input_copies(session, app.state.services.artifacts, [recorded], 10**9)

    assert (refused.value.status, refused.value.code) == (
        422,
        "output-recipe-bundle-input-uncopyable",
    )


async def test_a_bundle_whose_inputs_are_not_exactly_as_written_is_refused(
    app: FastAPI, client: AsyncClient
) -> None:
    content, _source, _mask = await _with_inputs(app, client)
    other = io.BytesIO()
    Image.new("RGB", (8, 6), (1, 2, 3)).save(other, format="PNG")

    def drop_last(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        files.pop("input-2.png")
        manifest["inputs"].pop()

    def replace_copy(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        # The manifest still names the copy that was written.
        files["input-1.png"] = other.getvalue()

    def wrong_original(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["inputs"][0]["copy_of"] = "f" * 64

    def swap_positions(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        first, second = manifest["inputs"]
        manifest["inputs"] = [
            {**second, "file": input_file(0), "position": 0},
            {**first, "file": input_file(1), "position": 1},
        ]
        files["input-1.png"], files["input-2.png"] = files["input-2.png"], files["input-1.png"]

    def as_version_one(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["version"] = 1

    def without_listing(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["version"] = 1
        manifest.pop("inputs")

    def renamed_file(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["inputs"][1]["file"] = "input-9.png"

    def shifted_position(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["inputs"][1]["position"] = 0

    def boolean_position(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        manifest["inputs"][0]["position"] = False

    def empty_listing(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        files.pop("input-1.png")
        files.pop("input-2.png")
        manifest["inputs"] = []

    for change in (
        drop_last,
        replace_copy,
        wrong_original,
        swap_positions,
        as_version_one,
        without_listing,
        renamed_file,
        shifted_position,
        boolean_position,
        empty_listing,
    ):
        with pytest.raises(OutputRecipeBundleFormatError):
            open_output_recipe_bundle(_resealed(content, change))
    # A copy that is not only pixels is refused even with its hash in the manifest.
    words = PngImagePlugin.PngInfo()
    words.add_text("Comment", "kept")
    marked = io.BytesIO()
    Image.new("RGB", (8, 6), (40, 90, 160)).save(marked, format="PNG", pnginfo=words)

    def with_text(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
        files["input-1.png"] = marked.getvalue()
        manifest["inputs"][0]["sha256"] = hashlib.sha256(marked.getvalue()).hexdigest()

    with pytest.raises(OutputRecipeBundleFormatError):
        open_output_recipe_bundle(_resealed(content, with_text))


async def test_one_picture_listed_twice_is_carried_as_the_same_copy_both_times(
    app: FastAPI, client: AsyncClient
) -> None:
    content, _source, _mask = await _with_inputs(app, client)

    def listed_twice(same_copy: bool) -> Any:
        def change(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
            # A record naming the source at both positions, as one whose selection
            # was also one of its pictures does.
            record = open_output_recipe(files[RECORD_FILE])
            del record["digest"]
            record["inputs"][1]["sha256"] = record["inputs"][0]["sha256"]
            files[RECORD_FILE] = seal_output_recipe(record)
            manifest["record"]["sha256"] = hashlib.sha256(files[RECORD_FILE]).hexdigest()
            manifest["record"]["digest"] = open_output_recipe(files[RECORD_FILE])["digest"]
            manifest["inputs"][1]["copy_of"] = manifest["inputs"][0]["copy_of"]
            if same_copy:
                files["input-2.png"] = files["input-1.png"]
                manifest["inputs"][1]["sha256"] = manifest["inputs"][0]["sha256"]

        return change

    accepted = open_output_recipe_bundle(_resealed(content, listed_twice(same_copy=True)))
    with pytest.raises(OutputRecipeBundleFormatError):
        open_output_recipe_bundle(_resealed(content, listed_twice(same_copy=False)))

    assert [item["copy_of"] for item in accepted["inputs"]] == [
        accepted["inputs"][0]["copy_of"]
    ] * 2


async def test_a_bundle_with_its_inputs_checks_as_its_record_does(
    app: FastAPI, client: AsyncClient
) -> None:
    content, _source, _mask = await _with_inputs(app, client)

    as_bundle = await _check(client, content)
    as_record = await _check(client, _entries(content)[RECORD_FILE])

    assert as_bundle.status_code == as_record.status_code == 200, as_bundle.text
    bundle_report = as_bundle.json()
    record_report = as_record.json()
    assert bundle_report.pop("picture")["width"] == 8
    assert record_report.pop("picture") is None
    assert bundle_report.pop("bundled_inputs") == [0, 1]
    assert record_report.pop("bundled_inputs") == []
    assert bundle_report == record_report
