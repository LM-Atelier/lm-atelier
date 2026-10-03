"""A picture a record's bundle carries, standing in for one that is not here."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select
from test_output_recipe_adaptation import _adapt, _codes
from test_output_recipe_bundle import _check, _entries, _record
from test_output_recipe_bundle_inputs import _edited, _resealed
from test_output_recipe_replay import _edit_chat, _finished

from local_lm.db import SessionLocal
from local_lm.models import Artifact, ArtifactLibraryEntry
from local_lm.output_recipe_bundle import RECORD_FILE
from local_lm.output_recipe_v1 import open_output_recipe, seal_output_recipe

#: A recorded picture that no computer running these tests holds.
ELSEWHERE = "7" * 64


def _source_elsewhere(files: dict[str, bytes], manifest: dict[str, Any]) -> None:
    """As the bundle of the same edit made where its source picture was another file."""

    record = open_output_recipe(files[RECORD_FILE])
    del record["digest"]
    record["inputs"][0]["sha256"] = ELSEWHERE
    files[RECORD_FILE] = seal_output_recipe(record)
    manifest["record"]["sha256"] = hashlib.sha256(files[RECORD_FILE]).hexdigest()
    manifest["record"]["digest"] = open_output_recipe(files[RECORD_FILE])["digest"]
    if "inputs" in manifest:
        manifest["inputs"][0]["copy_of"] = ELSEWHERE


async def _bundles(app: FastAPI, client: AsyncClient) -> tuple[bytes, bytes, str]:
    """A bundle carrying its inputs and one without, both of a record whose source is not here.

    Returns them with the hash of the source's copy.
    """

    run_id, output, _source, _mask = await _edited(app, client)
    # With its prompt, as a record must be to be generated again at all.
    digest = (await _record(client, run_id, output, "include")).headers["x-output-recipe-digest"]
    path = f"/api/runs/{run_id}/outputs/{output}/recipe-bundle"
    params = {"prompts": "include", "digest": digest}
    carried = (await client.get(path, params={**params, "inputs": "include"})).content
    bare = (await client.get(path, params={**params, "inputs": "omit"})).content
    copy = hashlib.sha256(_entries(carried)["input-1.png"]).hexdigest()
    return _resealed(carried, _source_elsewhere), _resealed(bare, _source_elsewhere), copy


def _stored() -> tuple[int, int]:
    with SessionLocal() as session:
        return (
            session.scalar(select(func.count()).select_from(Artifact)) or 0,
            session.scalar(select(func.count()).select_from(ArtifactLibraryEntry)) or 0,
        )


async def test_the_check_names_the_inputs_a_bundle_carries(
    app: FastAPI, client: AsyncClient
) -> None:
    carried, bare, _copy = await _bundles(app, client)

    with_inputs = await _check(client, carried)
    without = await _check(client, bare)
    record_only = await _check(client, _entries(carried)[RECORD_FILE])

    assert with_inputs.status_code == without.status_code == record_only.status_code == 200
    assert with_inputs.json()["bundled_inputs"] == [0, 1]
    assert without.json()["bundled_inputs"] == []
    assert record_only.json()["bundled_inputs"] == []


async def test_a_bundled_copy_stands_in_for_a_source_that_is_not_here(
    app: FastAPI, client: AsyncClient
) -> None:
    carried, _bare, copy = await _bundles(app, client)
    before = _stored()

    unchosen = await _adapt(client, await _edit_chat(client), carried, [])
    response = await _adapt(client, await _edit_chat(client), carried, [("input", "0:bundle")])

    assert unchosen.status_code == 409, unchosen.text
    assert _codes(unchosen) == ["replay-input-missing"]
    assert response.status_code == 202, response.text
    adapted = await _finished(client, response.json()["run"]["id"])
    stand_in = f"sha256:{copy}"
    assert adapted["provenance_json"]["input_artifact_ids"] == [stand_in]
    adaptation = adapted["provenance_json"]["adaptation"]
    assert adaptation["choices"] == [{"requirement": "input", "position": 0, "chosen": stand_in}]
    assert "inputs" in adaptation["differs"]
    # Kept as an input, as a picture attached to a turn is, and not listed in the library.
    with SessionLocal() as session:
        stored = session.get(Artifact, stand_in)
        assert stored is not None and stored.kind == "input"
        assert stored.metadata_json.get("uploaded") is True
        assert stored.original_name == "record-input-1.png"
        assert (
            session.scalar(
                select(ArtifactLibraryEntry).where(ArtifactLibraryEntry.artifact_id == stand_in)
            )
            is None
        )
    assert _stored() == (before[0] + 1, before[1])


async def test_a_bundled_copy_is_refused_where_there_is_none(
    app: FastAPI, client: AsyncClient
) -> None:
    carried, bare, _copy = await _bundles(app, client)
    record = _entries(carried)[RECORD_FILE]
    before = _stored()

    for content, choice in (
        (record, "0:bundle"),
        (bare, "0:bundle"),
        (carried, "2:bundle"),
        (carried, "0:bundled"),
    ):
        refused = await _adapt(client, await _edit_chat(client), content, [("input", choice)])
        assert refused.status_code == 422, (choice, refused.text)
        assert refused.json()["code"] == "adaptation-choice-invalid"

    assert _stored() == before
