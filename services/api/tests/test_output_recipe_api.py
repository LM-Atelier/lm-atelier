"""Downloading the portable record of one generated picture, through the real route."""

from __future__ import annotations

from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

from local_lm import output_recipe
from local_lm.db import SessionLocal
from local_lm.models import Run
from local_lm.output_recipe_v1 import open_output_recipe

pytestmark = pytest.mark.asyncio

_PROMPT = "a ceramic cup on a wooden table"


async def _generated(client: AsyncClient, prompt: str = _PROMPT) -> tuple[dict[str, Any], str]:
    """A finished picture: its run as the API reports it, and the picture's id."""

    chat = (await client.post("/api/chats", json={"title": "Record"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": prompt, "mode": "image"}
    )
    assert turn.status_code == 202, turn.text
    run_id = turn.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    run = dict(await wait_for_terminal_status(read, what=f"run {run_id}", expected="complete"))
    return run, run["provenance_json"]["outputs"][0]["artifact_id"]


async def _record(client: AsyncClient, run_id: str, artifact_id: str, prompts: str) -> Any:
    return await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe", params={"prompts": prompts}
    )


def _local_identifiers(run: dict[str, Any]) -> list[str]:
    return [
        value
        for value in (
            run["id"],
            run["chat_id"],
            run["user_message_id"],
            run["assistant_message_id"],
            run.get("work_plan_id"),
            run.get("work_step_id"),
            run.get("profile_id"),
            run.get("workflow_revision_id"),
        )
        if isinstance(value, str) and value
    ]


async def test_a_generated_picture_downloads_its_record(client: AsyncClient) -> None:
    run, artifact_id = await _generated(client)

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.headers["content-disposition"].startswith("attachment;")
    assert response.headers["cache-control"] == "private, no-store"
    record = open_output_recipe(response.content)
    assert response.headers["x-output-recipe-digest"] == record["digest"]
    assert f"sha256:{record['output']['sha256']}" == artifact_id
    assert record["operation"] == "text_to_image"
    assert record["prompt"] == {
        "included": True,
        "positive": _PROMPT,
        "negative": None,
        "omitted_reason": None,
    }
    assert record["seed"]["value"] == run["settings_json"]["seed"]
    # The mock engine stands in for a real one, and nothing froze this turn.
    assert {"mock_engine", "frozen_snapshot_absent"} <= set(record["reproducibility"]["missing"])
    assert record["reproducibility"]["status"] == "incomplete"
    for identifier in _local_identifiers(run):
        assert identifier.encode() not in response.content, identifier


async def test_an_omitted_prompt_leaves_no_trace_in_the_record(client: AsyncClient) -> None:
    """Free text follows the prompt out; a setting's fixed choice stays."""

    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.settings_json = {**stored.settings_json, "house_style": "art nouveau"}
        session.commit()

    response = await _record(client, run["id"], artifact_id, "omit")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"] == {
        "included": False,
        "positive": None,
        "negative": None,
        "omitted_reason": "chosen",
    }
    assert _PROMPT.encode() not in response.content
    assert b"art nouveau" not in response.content
    assert {"prompt", "settings.house_style"} <= set(record["removed"])
    assert "prompt_omitted" in record["reproducibility"]["missing"]
    assert record["settings"]["unbound"]["sampler"] == run["settings_json"]["sampler"]


async def test_the_prompt_must_be_chosen_explicitly(client: AsyncClient) -> None:
    run, artifact_id = await _generated(client)

    response = await client.get(f"/api/runs/{run['id']}/outputs/{artifact_id}/recipe")

    assert response.status_code == 422


async def test_an_output_is_named_by_the_run_that_made_it(client: AsyncClient) -> None:
    """The same bytes can come from two runs, so the run is part of the name."""

    first, first_output = await _generated(client)
    # A different prompt, so the mock engine makes different bytes.
    second, second_output = await _generated(client, "a glass vase by a window")
    assert second_output != first_output

    elsewhere = await _record(client, second["id"], first_output, "include")
    unknown = await _record(client, "run_" + "0" * 32, first_output, "include")

    assert elsewhere.status_code == 404
    assert elsewhere.json()["code"] == "output-recipe-output-not-found"
    assert unknown.status_code == 404
    assert unknown.json()["code"] == "output-recipe-run-not-found"
    assert (await _record(client, first["id"], first_output, "include")).status_code == 200


async def test_a_prompt_taken_back_from_the_chat_is_not_written(client: AsyncClient) -> None:
    run, artifact_id = await _generated(client)
    message_id = run["user_message_id"]
    impact = await client.get(f"/api/messages/{message_id}/removal-impact")
    assert impact.status_code == 200, impact.text
    removed = await client.post(
        f"/api/messages/{message_id}/remove-content",
        json={
            "expected_message_id": message_id,
            "expected_revision_id": impact.json()["message_revision_id"],
            "operation_key": "take-back-the-prompt",
        },
    )
    assert removed.status_code == 200, removed.text

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "removed_from_chat"
    assert _PROMPT.encode() not in response.content


async def test_a_prompt_naming_a_local_identifier_is_left_out(client: AsyncClient) -> None:
    """Dispatch can write another step's id into a prompt; that prompt stays home."""

    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.standalone_prompt = f"Output from step_{'1' * 32}:\nblue tiles"
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "contains_local_reference"
    assert b"step_" not in response.content


@pytest.mark.parametrize(
    "path",
    [
        "C:\\Users\\someone\\overlay.png",
        "/home/someone/overlay.png",
        "~/someone/overlay.png",
        "../someone/overlay.png",
        "\\Users\\someone\\overlay.png",
        "file:///home/someone/overlay.png",
    ],
)
async def test_a_setting_holding_a_path_or_a_structure_is_left_out(
    client: AsyncClient, path: str
) -> None:
    """A path in a text setting stays home in every spelling; words with a slash stay in."""

    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.settings_json = {
            **stored.settings_json,
            "overlay": path,
            "pairing": "a cat / a dog",
            "region": {"x": 1, "y": 2},
        }
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert {"settings.overlay", "settings.region"} <= set(record["removed"])
    assert b"someone" not in response.content
    assert record["settings"]["unbound"]["pairing"] == "a cat / a dog"
    assert "settings_removed" in record["reproducibility"]["missing"]


async def test_a_prompt_holding_a_path_is_left_out(client: AsyncClient) -> None:
    run, artifact_id = await _generated(client, "a cup like the one in /home/someone/cup.png")

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "contains_local_reference"
    assert b"someone" not in response.content


async def test_a_record_that_would_carry_a_local_identifier_is_refused(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last check reads the finished bytes, whatever a section let through."""

    run, artifact_id = await _generated(client)

    def leaky_model(stored: Run, draft: object) -> dict[str, Any]:
        return {
            "files": {},
            "provider": stored.id,
            "remote_id": None,
            "revision": None,
            "content_rating": None,
        }

    monkeypatch.setattr(output_recipe, "_model", leaky_model)

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 409
    assert response.json()["code"] == "output-recipe-unsafe"


async def _finished(client: AsyncClient, run_id: str) -> dict[str, Any]:
    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    return dict(await wait_for_terminal_status(read, what=run_id, expected="complete"))


async def test_a_follow_up_prompt_is_left_out_once_the_earlier_turn_is_taken_back(
    client: AsyncClient,
) -> None:
    """A follow-up's prompt repeats the earlier prompt, which the person removed."""

    chat = (await client.post("/api/chats", json={"title": "Follow-up"})).json()
    first = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": _PROMPT, "mode": "image"}
    )
    first_message = first.json()["run"]["user_message_id"]
    await _finished(client, first.json()["run"]["id"])
    follow_up = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": "make it blue", "mode": "image"}
    )
    assert follow_up.status_code == 202, follow_up.text
    run = await _finished(client, follow_up.json()["run"]["id"])
    artifact_id = run["provenance_json"]["outputs"][0]["artifact_id"]
    impact = await client.get(f"/api/messages/{first_message}/removal-impact")
    removed = await client.post(
        f"/api/messages/{first_message}/remove-content",
        json={
            "expected_message_id": first_message,
            "expected_revision_id": impact.json()["message_revision_id"],
            "operation_key": "take-back-the-first-turn",
        },
    )
    assert removed.status_code == 200, removed.text

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "removed_from_chat"
    assert b"ceramic" not in response.content


async def test_a_step_fed_by_another_output_names_it_and_says_so(client: AsyncClient) -> None:
    """Dispatch hands an earlier step's picture to a later one; the record lists it."""

    _, earlier_output = await _generated(client, "a glass vase by a window")
    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.provenance_json = {
            **stored.provenance_json,
            "resolved_dependency_artifact_ids": [earlier_output],
        }
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert [item["sha256"] for item in record["inputs"]] == [earlier_output.split(":", 1)[1]]
    assert "depends_on_other_outputs" in record["reproducibility"]["missing"]


async def test_a_picture_finished_after_generation_is_never_called_recorded(
    client: AsyncClient,
) -> None:
    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        outputs = [dict(item) for item in stored.provenance_json["outputs"]]
        outputs[0]["region_edit"] = {"version": 1}
        stored.provenance_json = {**stored.provenance_json, "outputs": outputs}
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert "finished_after_generation" in record["reproducibility"]["missing"]


async def test_how_a_selection_or_workflow_lora_was_applied_is_named_as_left_out(
    client: AsyncClient,
) -> None:
    _, selection = await _generated(client, "a plain grey backdrop")
    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.settings_json = {
            **stored.settings_json,
            "mask": {"artifact_id": selection, "invert": True, "apply": "blend"},
            "workflow_lora_overrides": {"version": 1, "rows": [{"strength": 0.5}]},
        }
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert {"settings.mask", "settings.workflow_lora_overrides"} <= set(record["removed"])
    assert "settings_removed" in record["reproducibility"]["missing"]
    assert record["inputs"][-1]["sha256"] == selection.split(":", 1)[1]
    assert record["inputs"][-1]["role"] == "mask"


async def test_malformed_recorded_assets_leave_the_prompt_out_rather_than_fail(
    client: AsyncClient,
) -> None:
    """Older and imported runs can hold provenance in shapes dispatch never wrote."""

    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.provenance_json = {**stored.provenance_json, "auxiliary_assets": ["unexpected"]}
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "unavailable"


async def test_a_prompt_too_long_for_a_record_is_left_out_with_that_reason(
    client: AsyncClient,
) -> None:
    run, artifact_id = await _generated(client)
    with SessionLocal() as session:
        stored = session.get(Run, run["id"])
        assert stored is not None
        stored.standalone_prompt = "a ceramic cup " * 12_000
        session.commit()

    response = await _record(client, run["id"], artifact_id, "include")

    assert response.status_code == 200, response.text
    record = open_output_recipe(response.content)
    assert record["prompt"]["omitted_reason"] == "too_long"
