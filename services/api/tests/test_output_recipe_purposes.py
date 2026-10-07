"""A generation record keeps the purpose each picture was chosen for, and gives it back."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from test_image_role_execution import selected_images, selected_workflow
from test_ordered_image_purposes import enable_purposes
from test_output_recipe_adaptation import _adapt
from test_output_recipe_bundle import _check, _engine_png, _entries, _make_output
from test_output_recipe_encryption import PASSPHRASE, _open
from test_output_recipe_replay import _finished as replay_finished
from test_output_recipe_replay import _new_chat as replay_new_chat
from test_output_recipe_replay import _profile as replay_profile
from test_output_recipe_replay import _record_of as replay_record_of
from test_output_recipe_replay import _recorded as replay_recorded
from test_output_recipe_replay import _replay as replay
from test_output_recipe_replay_plan import _plan, _profile, _recorded, _resealed

from local_lm import output_recipe as output_recipe_module
from local_lm.accepted_turn_context import accepted_context
from local_lm.db import SessionLocal
from local_lm.model_planner import workflow_artifact_contract
from local_lm.models import Run, WorkflowDefinition, WorkflowRevision
from local_lm.output_recipe import _Draft, _inputs
from local_lm.output_recipe_bundle import RECORD_FILE, open_output_recipe_bundle
from local_lm.output_recipe_replay import (
    AdaptationChoices,
    edit_prompt_preamble,
    plan_output_recipe_adaptation,
    replay_turn_request,
)
from local_lm.output_recipe_v1 import (
    OutputRecipeFormatError,
    open_output_recipe,
    seal_output_recipe,
)

PICTURE = b"\x89PNG\r\n\x1a\n" + b"neutral picture bytes"


def _entry(sha256: str, role: str) -> dict[str, Any]:
    return {"sha256": sha256, "role": role, "size_bytes": 10, "media_type": "image/png"}


async def _pictures(client: AsyncClient, count: int) -> list[str]:
    """Pictures in the Media Library, by their content hashes."""

    hashes: list[str] = []
    for index in range(count):
        uploaded = await client.post(
            "/api/artifacts",
            params={"kind": "image"},
            files={"file": (f"picture-{index}.png", PICTURE + bytes([index]), "image/png")},
        )
        assert uploaded.status_code == 201, uploaded.text
        hashes.append(str(uploaded.json()["id"]).removeprefix("sha256:"))
    return hashes


def _as_version_2(content: bytes, inputs: list[dict[str, Any]], **values: Any) -> bytes:
    def change(record: dict[str, Any]) -> None:
        record["version"] = 2
        record["inputs"] = inputs
        record.update(values)

    return _resealed(content, change)


async def test_a_record_reads_pictures_named_by_purpose_and_refuses_an_ambiguous_one(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    a, b, c = "5" * 64, "6" * 64, "7" * 64

    sealed = _as_version_2(
        content, [_entry(a, "reference"), _entry(b, "edit_source"), _entry(c, "mask")]
    )
    assert [item["role"] for item in open_output_recipe(sealed)["inputs"]] == [
        "reference",
        "edit_source",
        "mask",
    ]
    # Words alone may still name purposes: none, or only pictures read beside them.
    assert open_output_recipe(_as_version_2(content, []))["version"] == 2

    ambiguous = {
        "two pictures to change": [_entry(a, "edit_source"), _entry(b, "edit_source")],
        "a selection before a picture": [_entry(c, "mask"), _entry(a, "edit_source")],
        "two selections": [_entry(a, "edit_source"), _entry(b, "mask"), _entry(c, "mask")],
        "a place in place of a purpose": [_entry(a, "source"), _entry(b, "reference")],
    }
    for name, inputs in ambiguous.items():
        with pytest.raises(OutputRecipeFormatError):
            _as_version_2(content, inputs)
        assert name
    # And a purpose is not a place: version 1 keeps its own words.
    with pytest.raises(OutputRecipeFormatError):
        _resealed(content, lambda record: record.update(inputs=[_entry(a, "edit_source")]))
    with pytest.raises(OutputRecipeFormatError, match="version 1 or 2"):
        _resealed(content, lambda record: record.update(version=3))


def _snapshot(ids: list[str], roles: list[str] | None) -> Any:
    return SimpleNamespace(input_artifact_ids=ids, input_image_roles=roles)


@pytest.mark.parametrize(
    ("operation", "roles", "expected"),
    [
        ("image_to_image", ["reference", "edit_source"], ["reference", "edit_source", "mask"]),
        ("text_to_image", ["reference", "reference"], ["reference", "reference", "mask"]),
        ("image_to_image", None, ["source", "input", "mask"]),
    ],
)
def test_the_writer_names_each_picture_by_its_accepted_purpose(
    app: FastAPI, operation: str, roles: list[str] | None, expected: list[str]
) -> None:
    ids = ["sha256:" + "5" * 64, "sha256:" + "6" * 64]
    settings = {"mask": {"artifact_id": "sha256:" + "7" * 64}}
    run = SimpleNamespace(operation=operation, provenance_json={})
    with SessionLocal() as session:
        entries = _inputs(session, cast(Any, run), _snapshot(ids, roles), settings, _Draft())

    assert [entry["sha256"][:1] for entry in entries] == ["5", "6", "7"]
    assert [entry["role"] for entry in entries] == expected


async def test_a_replay_sends_the_recorded_purposes_in_the_recorded_order(
    app: FastAPI, client: AsyncClient
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    first, second = await _pictures(client, 2)
    sealed = _as_version_2(content, [_entry(first, "reference"), _entry(second, "reference")])

    plan = await _plan(client, sealed)

    assert plan["ready"] is True, plan["refusals"]
    resolved = plan["resolved"]
    assert resolved["input_artifact_ids"] == [f"sha256:{first}", f"sha256:{second}"]
    assert resolved["input_image_roles"] == ["reference", "reference"]
    with SessionLocal() as session:
        turn = await replay_turn_request(
            session, app.state.services.engines, open_output_recipe(sealed), resolved
        )
    assert turn.input_artifact_ids == resolved["input_artifact_ids"]
    assert turn.input_image_roles == ["reference", "reference"]

    # A version 1 record names no purposes, and its replay sends none.
    legacy = await _plan(client, content)
    assert "input_image_roles" not in legacy["resolved"]
    with SessionLocal() as session:
        legacy_turn = await replay_turn_request(
            session, app.state.services.engines, open_output_recipe(content), legacy["resolved"]
        )
    assert legacy_turn.input_image_roles is None


def _unsupported(plan: dict[str, Any]) -> list[str]:
    return [
        reason
        for item in plan["refusals"]
        if item["code"] == "replay-record-unsupported"
        for reason in item["reasons"]
    ]


async def test_purposes_a_turn_of_that_kind_cannot_take_are_refused_before_planning(
    client: AsyncClient,
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    first, second = await _pictures(client, 2)
    preamble = edit_prompt_preamble()

    def as_a_change(inputs: list[dict[str, Any]], operation: str = "image_to_image") -> bytes:
        changed = _as_version_2(content, inputs, operation=operation)
        return _resealed(
            changed,
            lambda value: value["prompt"].update(positive=preamble + value["prompt"]["positive"]),
        )

    words_changing_a_picture = _as_version_2(content, [_entry(first, "edit_source")])
    assert "inputs_for_operation" in _unsupported(await _plan(client, words_changing_a_picture))
    # A change that names no picture to change would start from whatever picture a turn found.
    for operation in ("image_to_image", "image_to_video"):
        for inputs in ([], [_entry(first, "reference"), _entry(second, "reference")]):
            plan = await _plan(client, as_a_change(inputs, operation))
            assert "inputs_for_operation" in _unsupported(plan), (operation, inputs)

    # A change named by purpose need not list the picture it changes first.
    reordered = as_a_change([_entry(first, "reference"), _entry(second, "edit_source")])
    assert "inputs_for_operation" not in _unsupported(await _plan(client, reordered))


async def test_a_picture_standing_in_keeps_the_purpose_of_its_place(
    app: FastAPI, client: AsyncClient
) -> None:
    profile_id, _ = _profile()
    _, content = await _recorded(client, profile_id)
    first, second, stand_in = await _pictures(client, 3)
    sealed = _as_version_2(content, [_entry(first, "reference"), _entry(second, "reference")])

    with SessionLocal() as session:
        plan = plan_output_recipe_adaptation(
            session,
            open_output_recipe(sealed),
            AdaptationChoices(
                workflow_revision_id=None,
                profile_id=None,
                loras={},
                inputs={1: f"sha256:{stand_in}"},
            ),
            media_engine=app.state.services.engines.settings.media_engine,
        )

    assert plan["ready"] is True, plan["refusals"]
    resolved = plan["resolved"]
    assert resolved["input_artifact_ids"] == [f"sha256:{first}", f"sha256:{stand_in}"]
    assert resolved["input_image_roles"] == ["reference", "reference"]


async def test_a_run_accepted_with_explicitly_no_pictures_is_recorded_as_version_2(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Purposes given and empty are still given: the record says so and its plan keeps it.

    A replayed run is one whose inputs are frozen when it is accepted. The test
    engines cannot take explicit purposes, so its frozen inputs are read back
    once as they are and once with an empty purpose list, as a run accepted
    with one would be read.
    """

    profile_id = replay_profile()
    _, content = await replay_recorded(client, profile_id)
    response = await replay(client, await replay_new_chat(client), content)
    assert response.status_code == 202, response.text
    again = await replay_finished(client, response.json()["run"]["id"])

    # Frozen without purposes, it stays a version 1 record.
    assert open_output_recipe(await replay_record_of(client, again))["version"] == 1

    accepted = accepted_context

    def with_no_pictures(session: Any, run: Any) -> Any:
        context = accepted(session, run)
        assert context is not None
        return context.model_copy(update={"input_image_roles": []})

    monkeypatch.setattr(output_recipe_module, "accepted_context", with_no_pictures)
    named = await replay_record_of(client, again)

    record = open_output_recipe(named)
    assert (record["version"], record["inputs"]) == (2, [])
    plan = await _plan(client, named)
    assert plan["resolved"]["input_image_roles"] == []
    assert seal_output_recipe({k: v for k, v in record.items() if k != "digest"}) == named


async def _made_with_purposes(
    app: FastAPI, client: AsyncClient, roles: list[str], operation: str
) -> tuple[str, str, list[str]]:
    """A finished picture whose turn named each of its three pictures' purposes.

    Returns the run, its stored output and the pictures in the order given.
    """

    ids = await selected_images(client)
    slots = (
        ("edit_source", "reference", "reference") if "edit_source" in roles else ("reference",) * 3
    )
    revision_id = selected_workflow(slots)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        definition = session.get(WorkflowDefinition, revision.workflow_id)
        assert definition is not None
        definition.operation = operation
        # Identified for what it now runs, as a workflow saved for that operation is.
        revision.artifact_sha256 = workflow_artifact_contract(
            operation=operation,
            engine=revision.engine,
            api_graph=revision.api_graph_json,
            input_schema=revision.input_schema_json,
            dependencies=revision.dependencies_json,
        )
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Garden record"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "Paint a garden",
            "mode": "image",
            "profile_id": replay_profile(),
            "input_artifact_ids": ids,
            "input_image_roles": roles,
            "workflow_revision_id": revision_id,
        },
    )
    assert turn.status_code == 202, turn.text
    run_id = str(turn.json()["run"]["id"])

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    await wait_for_terminal_status(read, what=f"run {run_id}", expected="complete")
    return run_id, _make_output(app, run_id, _engine_png()), ids


@pytest.mark.parametrize(
    ("roles", "operation"),
    [
        (["reference", "edit_source", "reference"], "image_to_image"),
        (["reference", "reference", "reference"], "text_to_image"),
    ],
    ids=["a-canvas-second", "references-only"],
)
async def test_purposes_survive_a_bundle_and_its_replay_exactly(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    roles: list[str],
    operation: str,
) -> None:
    await enable_purposes(app, monkeypatch)
    run_id, output, ids = await _made_with_purposes(app, client, roles, operation)
    shown = await client.get(
        f"/api/runs/{run_id}/outputs/{output}/recipe", params={"prompts": "include"}
    )
    assert shown.status_code == 200, shown.text
    record = open_output_recipe(shown.content)
    assert (record["version"], record["operation"]) == (2, operation)
    assert [(item["sha256"], item["role"]) for item in record["inputs"]] == [
        (identifier.removeprefix("sha256:"), role)
        for identifier, role in zip(ids, roles, strict=True)
    ]
    digest = shown.headers["x-output-recipe-digest"]
    path = f"/api/runs/{run_id}/outputs/{output}/recipe-bundle"
    asked = {"prompts": "include", "digest": digest, "inputs": "include"}

    bundle = await client.get(path, params=asked)
    sealed = await client.post(f"{path}/encrypted", json={**asked, "passphrase": PASSPHRASE})

    assert bundle.status_code == sealed.status_code == 200, (bundle.text, sealed.text)
    # The record travels byte for byte, so its digest, order and purposes are the ones shown.
    assert _entries(bundle.content)[RECORD_FILE] == shown.content
    opened = open_output_recipe_bundle(bundle.content)
    assert [(item["position"], item["role"]) for item in opened["inputs"]] == list(enumerate(roles))
    assert (await _open(client, sealed.content)).content == bundle.content
    checked = await _check(client, bundle.content)
    assert checked.status_code == 200, checked.text
    assert checked.json()["digest"] == digest

    replayed = await _adapt(
        client,
        (await client.post("/api/chats", json={"title": "Again"})).json()["id"],
        bundle.content,
        [],
    )

    assert replayed.status_code == 202, replayed.text
    again = replayed.json()["run"]["id"]
    with SessionLocal() as session:
        run = session.get(Run, again)
        assert run is not None
        frozen = accepted_context(session, run)
        assert frozen is not None
        assert (list(frozen.input_artifact_ids), frozen.input_image_roles) == (ids, roles)
    await replay_finished(client, again)
