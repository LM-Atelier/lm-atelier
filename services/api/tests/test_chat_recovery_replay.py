"""Recorded generations respect deletion before and during turn preparation."""

from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from test_output_recipe_replay import (
    EXECUTION,
    _counts,
    _finished,
    _new_chat,
    _profile,
    _record_of,
    _recorded,
    _replay,
)

from local_lm import api as api_module
from local_lm.output_recipe_v1 import open_output_recipe
from local_lm.schemas import TurnRequest


async def _change(
    client: AsyncClient, preview_path: str, command_path: str, key: str
) -> dict[str, Any]:
    preview = await client.get(preview_path)
    assert preview.status_code == 200, preview.text
    impact = preview.json()
    changed = await client.post(
        command_path,
        json={
            "expected_revision": impact["revision"],
            "impact_sha256": impact["impact_sha256"],
            "operation_key": key,
        },
    )
    assert changed.status_code == 200, changed.text
    return cast(dict[str, Any], changed.json())


@pytest.mark.parametrize("when", ["before-preparation", "during-preparation"])
async def test_a_deleted_destination_refuses_replay_until_the_same_chat_is_restored(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    _original_run, content = await _recorded(client, _profile())
    target = await _new_chat(client)
    preparation_count = 0
    deletion: dict[str, Any] = {}
    prepare = api_module.replay_turn_request

    async def trash() -> dict[str, Any]:
        return await _change(
            client,
            f"/api/chats/{target}/deletion-impact",
            f"/api/chats/{target}/trash",
            "trash-record-destination",
        )

    async def prepare_then_trash(*args: Any, **kwargs: Any) -> TurnRequest:
        nonlocal preparation_count, deletion
        preparation_count += 1
        payload = await prepare(*args, **kwargs)
        if when == "during-preparation" and preparation_count == 1:
            deletion = await trash()
        return payload

    if when == "before-preparation":
        deletion = await trash()
    monkeypatch.setattr(api_module, "replay_turn_request", prepare_then_trash)
    before = _counts()

    refused = await _replay(client, target, content)

    assert refused.status_code == 404, refused.text
    assert refused.json()["code"] == "chat-not-found"
    assert preparation_count == (0 if when == "before-preparation" else 1)
    assert _counts() == before
    assert deletion["subject_id"] == target
    assert (await client.get(f"/api/chats/{target}")).status_code == 404
    deletion_id = deletion["deletion_id"]
    restored = await _change(
        client,
        f"/api/recovery-items/{deletion_id}/impact",
        f"/api/recovery-items/{deletion_id}/restore",
        "restore-record-destination",
    )
    assert restored["subject_id"] == target

    accepted = await _replay(client, target, content)

    assert accepted.status_code == 202, accepted.text
    again = await _finished(client, accepted.json()["run"]["id"])
    assert again["chat_id"] == target
    first = open_output_recipe(content)
    second = open_output_recipe(await _record_of(client, again))
    assert {section: second[section] for section in EXECUTION} == {
        section: first[section] for section in EXECUTION
    }
