"""A render refused for what it asks of a plan names the limit it passed.

A turn, a Prompt Library batch, a regenerate, an edit or a branch can ask for
more than one plan may hold: more outputs, more waiting work, more work or
storage, more steps or more words in a step. Each answers with the status and
code it always did, and adds `errors` items naming the request field that
asked for it, or `plan` for a limit of the whole plan, the fixed limit, and
where a count came from when the request did not give it. The detail is the
one it always was, except where that detail used to quote the request: a step
with more words than a step may hold, and words asking for no outputs or for
more than a plan may carry, are now refused in fixed words. An ordered plan
names the role-specific choices it cannot take the same way, and a batch names
each saved setting only some of its workflows take. No item repeats a value
or any words the request sent, and nothing is queued.

Words may ask for more outputs than one request may make on paths that answer
before that limit is checked. Each answers such a count as it answers any
other: a canvas preview shows its canvas, a regenerate of a Prompt Library
draft makes its one output, and a turn from a Prompt Library source refuses
any count but one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from sqlalchemy import select
from test_generation_experiment_preflight import _profile, _revision
from test_prompt_library_api import _composer_source
from test_source_fit_acceptance import prepared_turn
from test_workflow_output_geometry_api import _generation_row_counts
from workflow_fixtures import seed_workflow_trust

from local_lm.api import _sent_setting_keys
from local_lm.db import SessionLocal
from local_lm.models import Chat, Project, Run
from local_lm.schemas import TurnRequest

TEXT = "Create an image of a ceramic cup"
STORY_THEN_PICTURE = "Write a short story about a paper boat, then create an image based on it"
STORY_PICTURE_VIDEO = (
    "Write a short story about a paper boat, then create an image based on it, "
    "then animate the image into a video"
)
OUTPUTS_PAST_TWO = "A media request can create at most 2 outputs."
COUNT_PAST_TWO = {"code": "plan-too-many-outputs", "field": "output_count", "limit": 2}
#: What an answer would carry if it passed on the text of a failed validation.
VALIDATION_WORDS = ("pydantic", "input_value", "Input should be", "validation error")


def _errors(response: Any, code: str = "turn-invalid", status: int = 422) -> list[dict[str, Any]]:
    """The items a refusal names, each checked to carry a sentence, without the sentences."""

    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code, body
    items: list[dict[str, Any]] = body["errors"]
    for item in items:
        message = item.pop("message")
        assert isinstance(message, str) and message.endswith("."), item
    return items


def _detail(response: Any) -> str:
    detail: str = response.json()["detail"]
    return detail


async def _chat(client: AsyncClient, **fields: Any) -> str:
    chat = await client.post("/api/chats", json={"title": "Plan limits", **fields})
    assert chat.status_code == 201, chat.text
    chat_id: str = chat.json()["id"]
    return chat_id


async def _picture(client: AsyncClient, chat_id: str, **fields: Any) -> Any:
    return await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": TEXT, "mode": "image", **fields}
    )


async def _refused_turn(client: AsyncClient, chat_id: str, **fields: Any) -> Any:
    """A picture turn that is refused, having queued nothing."""

    before = _generation_row_counts()
    response = await _picture(client, chat_id, **fields)
    assert _generation_row_counts() == before, "nothing was queued"
    return response


async def _refused_ordered(
    client: AsyncClient, text: str = STORY_THEN_PICTURE, **fields: Any
) -> Any:
    """An Auto turn that compiles to an ordered plan, refused before anything is written."""

    chat_id = await _chat(client)
    before = _generation_row_counts()
    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": text, "mode": "auto", "confirm_media": True, **fields},
    )
    assert _generation_row_counts() == before, "nothing was queued"
    return response


def _cap_outputs(app: FastAPI, monkeypatch: pytest.MonkeyPatch, most: int) -> None:
    monkeypatch.setattr(app.state.services.engines.settings, "max_media_outputs_per_plan", most)


async def _finished(client: AsyncClient, run_id: str) -> None:
    """Wait for a run to end, so its response can be regenerated."""

    async def read() -> dict[str, Any]:
        run = await client.get(f"/api/runs/{run_id}")
        assert run.status_code == 200, run.text
        return cast(dict[str, Any], run.json())

    await wait_for_terminal_status(read, what=f"run {run_id}")


async def _completed(client: AsyncClient, chat_id: str, text: str = TEXT) -> Any:
    """A picture turn made to the end, so it can be regenerated."""

    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={"text": text, "mode": "image"}
    )
    assert response.status_code == 202, response.text
    await _finished(client, response.json()["run"]["id"])
    return response.json()


# The output-count cap.


async def test_a_count_past_the_cap_is_named_as_the_requests(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cap_outputs(app, monkeypatch, 2)

    response = await _refused_turn(client, await _chat(client), output_count=3)

    assert _errors(response) == [COUNT_PAST_TWO]
    assert _detail(response) == OUTPUTS_PAST_TWO


@pytest.mark.parametrize("saved", [True, False], ids=["saved for the chat", "sent"])
async def test_a_batch_size_past_the_cap_is_named_with_where_it_is_saved(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, saved: bool
) -> None:
    _cap_outputs(app, monkeypatch, 2)
    chat_id = await _chat(client)
    if saved:
        stored = await client.patch(
            f"/api/chats/{chat_id}", json={"generation_settings_json": {"image": {"batch_size": 3}}}
        )
        assert stored.status_code == 200, stored.text

    response = await _refused_turn(
        client, chat_id, **({} if saved else {"settings": {"batch_size": 3}})
    )

    named: dict[str, Any] = {"code": "setting-above-maximum", "field": "batch_size", "limit": 2}
    assert _errors(response) == [{**named, "source": "chat"} if saved else named]
    assert _detail(response) == OUTPUTS_PAST_TWO


async def test_a_count_the_words_asked_for_is_named_as_planned(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cap_outputs(app, monkeypatch, 2)
    chat_id = await _chat(client)
    before = _generation_row_counts()

    response = await client.post(
        f"/api/chats/{chat_id}/turns",
        json={"text": "Create three images of a ceramic cup", "mode": "image"},
    )

    assert _generation_row_counts() == before
    assert _errors(response) == [{**COUNT_PAST_TWO, "source": "planned"}]
    assert _detail(response) == OUTPUTS_PAST_TWO


async def _batch(client: AsyncClient, chat_id: str, subjects: list[str]) -> dict[str, Any]:
    """A Prompt Library batch of one draft for each subject, all on one workflow."""

    created = await client.post(
        "/api/workflows",
        json={
            "name": "Batch limits workflow",
            "operation": "text_to_image",
            "engine": "mock",
            "api_graph": {"node": {"class_type": "BatchLimitsWorkflow"}},
            "input_schema": {"type": "object", "properties": {}},
        },
    )
    assert created.status_code == 201, created.text
    revision_id = created.json()["current_revision_id"]
    seed_workflow_trust(revision_id)
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "batch-limits-template",
            "name": "Batch limits",
            "description": "One subject slot",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": {
                    "mode": "fixed",
                    "workflow_revision_id": revision_id,
                    "lora_policy": {"mode": "none"},
                },
            },
        },
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    created_batch = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "batch-limits-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": len(subjects),
            "selection_seed": 3,
            "inputs": {"subject": subjects},
        },
    )
    assert created_batch.status_code in (200, 201), created_batch.text
    batch: dict[str, Any] = created_batch.json()
    return batch


async def _queue(client: AsyncClient, batch: dict[str, Any]) -> Any:
    return await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json={
            "idempotency_key": "batch-limits-queue",
            "expected_plan_version": batch["plan_version"],
            "expected_plan_sha256": batch["plan_sha256"],
        },
    )


async def test_a_prompt_library_batch_past_the_cap_names_its_count(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    batch = await _batch(client, await _chat(client), ["gray square", "ceramic cup", "paper boat"])
    _cap_outputs(app, monkeypatch, 2)
    before = _generation_row_counts()

    queued = await _queue(client, batch)

    assert _generation_row_counts() == before
    assert _errors(queued, "generation-settings-invalid") == [COUNT_PAST_TWO]
    assert _detail(queued) == OUTPUTS_PAST_TWO


@pytest.mark.parametrize("sent", [False, True], ids=["carried over", "sent"])
async def test_an_edit_says_when_its_count_was_carried_over(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, sent: bool
) -> None:
    _cap_outputs(app, monkeypatch, 3)
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        source = await _picture(client, chat_id, output_count=3)
        assert source.status_code == 202, source.text
        _cap_outputs(app, monkeypatch, 2)
        before = _generation_row_counts()
        edited = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/edits",
            json={
                "text": TEXT,
                "mode": "image",
                "idempotency_key": f"plan-limit-edit-{sent}",
                **({"output_count": 3} if sent else {}),
            },
        )
        assert _generation_row_counts() == before

    assert _errors(edited) == [
        COUNT_PAST_TWO if sent else {**COUNT_PAST_TWO, "source": "inherited"}
    ]
    assert _detail(edited) == OUTPUTS_PAST_TWO


async def test_a_branch_names_the_count_it_sends(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        source = await _picture(client, chat_id)
        assert source.status_code == 202, source.text
        _cap_outputs(app, monkeypatch, 2)
        before = _generation_row_counts()
        branched = await client.post(
            f"/api/messages/{source.json()['user_message']['id']}/branch",
            json={"text": TEXT, "mode": "image", "output_count": 3},
        )
        assert _generation_row_counts() == before

    assert _errors(branched) == [COUNT_PAST_TWO]


#: What words asking for no outputs at all are told, and the items that name it.
ASKED_FOR_NONE = (
    "A media request makes at least one output.",
    [{"code": "plan-too-few-outputs", "field": "output_count", "limit": 1, "source": "planned"}],
)
PAST_FOUR = {
    "code": "plan-too-many-outputs",
    "field": "output_count",
    "limit": 4,
    "source": "planned",
}
#: What words asking for more outputs than four are told, with four as the cap.
ASKED_PAST_FOUR = ("A media request can create at most 4 outputs.", [PAST_FOUR])
#: Ninety-nine is also more than the 32 a conversation may have waiting.
ASKED_PAST_FOUR_AND_THE_WAITING_LIMIT = (
    "A media request can create at most 4 outputs.",
    [
        PAST_FOUR,
        {
            "code": "plan-too-many-pending",
            "field": "output_count",
            "limit": 32,
            "source": "planned",
        },
    ],
)
#: Counts in words outside what one request may make, each with what it is told.
COUNTS_IN_WORDS = {
    "none": ("Create 0 images of a ceramic cup", ASKED_FOR_NONE),
    "seventeen": ("Create 17 images of a ceramic cup", ASKED_PAST_FOUR),
    "ninety-nine": ("Create 99 images of a ceramic cup", ASKED_PAST_FOUR_AND_THE_WAITING_LIMIT),
    "sixteen spelled out": ("Create sixteen images of a ceramic cup", ASKED_PAST_FOUR),
}


async def _asked_through(app: FastAPI, client: AsyncClient, route: str, words: str) -> Any:
    """A picture asked for in these words as a new turn, or as a branch or an edit of one."""

    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        if route == "turn":
            before = _generation_row_counts()
            response = await _picture(client, chat_id, text=words)
        else:
            source = await _picture(client, chat_id)
            assert source.status_code == 202, source.text
            message_id = source.json()["user_message"]["id"]
            before = _generation_row_counts()
            response = await client.post(
                f"/api/messages/{message_id}/{'branch' if route == 'branch' else 'edits'}",
                json={
                    "text": words,
                    "mode": "image",
                    **({"idempotency_key": "plan-limit-words"} if route == "edit" else {}),
                },
            )
        assert _generation_row_counts() == before, "nothing was queued"
    return response


@pytest.mark.parametrize("route", ["turn", "branch", "edit"])
@pytest.mark.parametrize("asked", sorted(COUNTS_IN_WORDS))
async def test_a_count_in_words_no_request_may_make_is_named_as_planned(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, route: str, asked: str
) -> None:
    _cap_outputs(app, monkeypatch, 4)
    words, (detail, items) = COUNTS_IN_WORDS[asked]

    response = await _asked_through(app, client, route, words)

    assert not any(word in response.text for word in VALIDATION_WORDS), response.text
    assert _errors(response) == items
    assert _detail(response) == detail


@pytest.mark.parametrize(
    ("mode", "words", "told"),
    [
        ("auto", "Draw 17 pictures of a ceramic cup", ASKED_PAST_FOUR),
        ("auto", "Draw 0 pictures of a ceramic cup", ASKED_FOR_NONE),
        ("video", "Make 17 videos of a gray table", ASKED_PAST_FOUR),
        ("video", "Make 0 videos of a gray table", ASKED_FOR_NONE),
    ],
    ids=["auto past the cap", "auto none", "video past the cap", "video none"],
)
async def test_a_count_in_words_is_named_the_same_in_auto_and_video(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    words: str,
    told: tuple[str, list[dict[str, Any]]],
) -> None:
    _cap_outputs(app, monkeypatch, 4)
    chat_id = await _chat(client)
    before = _generation_row_counts()

    response = await client.post(f"/api/chats/{chat_id}/turns", json={"text": words, "mode": mode})

    assert _generation_row_counts() == before
    detail, items = told
    assert not any(word in response.text for word in VALIDATION_WORDS), response.text
    assert _errors(response) == items
    assert _detail(response) == detail


#: A picture workflow whose own batch size makes three outputs when nothing chooses one.
OWN_BATCH_SIZE = {
    "type": "object",
    "properties": {"batch_size": {"type": "integer", "default": 3, "minimum": 1, "maximum": 16}},
}


@pytest.mark.parametrize("waiting", [False, True], ids=["alone", "with too much waiting work"])
async def test_a_workflows_own_batch_size_past_the_cap_is_named_as_the_whole_plans(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch, waiting: bool
) -> None:
    _cap_outputs(app, monkeypatch, 2)
    if waiting:
        monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 2)
    revision_id = _revision("Own batch size", input_schema_json=OWN_BATCH_SIZE)
    profile_id = _profile("Own batch size")

    response = await _refused_turn(
        client, await _chat(client), workflow_revision_id=revision_id, profile_id=profile_id
    )

    whole_plan = [{"code": "plan-too-many-outputs", "field": "plan", "limit": 2}]
    if waiting:
        whole_plan.append({"code": "plan-too-many-pending", "field": "plan", "limit": 2})
    assert _errors(response) == whole_plan
    assert _detail(response) == OUTPUTS_PAST_TWO


# Waiting work.


async def test_a_count_past_the_waiting_work_a_chat_may_hold_names_the_fixed_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 2)

    response = await _refused_turn(client, await _chat(client), output_count=3)

    assert _errors(response) == [
        {"code": "plan-too-many-pending", "field": "output_count", "limit": 2}
    ]
    assert (
        _detail(response) == "This request would exceed the limit of 2 pending items in one chat."
    )


async def test_a_count_past_both_limits_names_each_with_the_caps_detail(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _cap_outputs(app, monkeypatch, 2)
    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 2)

    response = await _refused_turn(client, await _chat(client), output_count=3)

    assert _errors(response) == [
        COUNT_PAST_TWO,
        {"code": "plan-too-many-pending", "field": "output_count", "limit": 2},
    ]
    assert _detail(response) == OUTPUTS_PAST_TWO


async def test_a_chat_already_full_names_the_whole_plan(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 1)
    chat_id = await _chat(client)
    async with app.state.services.scheduler.lease("primary"):
        waiting = await _picture(client, chat_id)
        assert waiting.status_code == 202, waiting.text
        response = await _refused_turn(client, chat_id)

    assert _errors(response) == [{"code": "plan-too-many-pending", "field": "plan", "limit": 1}]
    assert _detail(response) == (
        "This chat already has 1 pending items. "
        "Cancel one or wait for work to finish before sending another."
    )


async def test_a_regenerate_in_a_full_chat_names_the_whole_plan(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat_id = await _chat(client)
    original = await _completed(client, chat_id)
    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 1)
    async with app.state.services.scheduler.lease("primary"):
        waiting = await _picture(client, chat_id)
        assert waiting.status_code == 202, waiting.text
        before = _generation_row_counts()
        regenerated = await client.post(
            f"/api/messages/{original['assistant_message']['id']}/regenerate",
            json={"settings": {}},
        )
        assert _generation_row_counts() == before

    assert _errors(regenerated) == [{"code": "plan-too-many-pending", "field": "plan", "limit": 1}]


@pytest.mark.parametrize(
    ("most", "subjects", "named"),
    [
        (1, ["gray square"], {"field": "plan"}),
        (2, ["gray square", "ceramic cup"], {"field": "output_count"}),
    ],
    ids=["into a chat already full", "past the room the chat has"],
)
async def test_a_batch_names_its_count_only_when_the_count_is_what_does_not_fit(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    most: int,
    subjects: list[str],
    named: dict[str, str],
) -> None:
    """No count fits a chat that is already full, so that is named as the whole plan's."""

    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", most)
    chat_id = await _chat(client)
    batch = await _batch(client, chat_id, subjects)
    async with app.state.services.scheduler.lease("primary"):
        waiting = await _picture(client, chat_id)
        assert waiting.status_code == 202, waiting.text
        before = _generation_row_counts()
        queued = await _queue(client, batch)
        assert _generation_row_counts() == before

    assert _errors(queued, "generation-settings-invalid") == [
        {"code": "plan-too-many-pending", **named, "limit": most}
    ]
    assert _detail(queued) == (
        f"This request would exceed the limit of {most} pending items in one chat."
    )


# One turn's whole-plan limits.


@pytest.mark.parametrize(
    ("limits", "items", "detail"),
    [
        (
            {"max_media_plan_work_units": 1},
            [{"code": "plan-too-much-work", "field": "plan"}],
            "This media request is too large to queue safely. "
            "Reduce the output count, resolution, frames, or steps.",
        ),
        (
            {"max_media_plan_estimated_bytes": 1},
            [{"code": "plan-storage-too-large", "field": "plan", "limit": 1}],
            "This media request has an unsafe storage estimate. "
            "Reduce the output count, resolution, or duration.",
        ),
        (
            {"max_media_plan_work_units": 1, "max_media_plan_estimated_bytes": 1},
            [
                {"code": "plan-too-much-work", "field": "plan"},
                {"code": "plan-storage-too-large", "field": "plan", "limit": 1},
            ],
            "This media request is too large to queue safely. "
            "Reduce the output count, resolution, frames, or steps.",
        ),
    ],
    ids=["work", "storage", "both"],
)
async def test_a_turn_past_a_whole_plan_limit_names_each_it_passes(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    limits: dict[str, int],
    items: list[dict[str, Any]],
    detail: str,
) -> None:
    for limit, value in limits.items():
        monkeypatch.setattr(app.state.services.engines.settings, limit, value)

    response = await _refused_turn(client, await _chat(client))

    assert _errors(response) == items
    assert _detail(response) == detail


async def test_a_turn_without_room_to_write_names_the_storage_it_needs(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_lm.orchestrator.shutil.disk_usage", lambda _path: SimpleNamespace(free=0)
    )

    response = await _refused_turn(client, await _chat(client))

    assert _errors(response) == [{"code": "plan-storage-unavailable", "field": "plan"}]
    assert _detail(response) == "There is not enough free storage to queue this media request."


# What an ordered plan cannot take from the request.


async def test_an_ordered_plan_names_the_count_it_cannot_multiply(client: AsyncClient) -> None:
    response = await _refused_ordered(client, output_count=2)

    assert _errors(response) == [{"code": "plan-one-output-per-step", "field": "output_count"}]
    assert _detail(response) == (
        "A heterogeneous ordered plan cannot multiply all steps. "
        "Request variations in a separate media turn."
    )


@pytest.mark.parametrize("role", ["audio", "Not A Role"])
async def test_an_ordered_plan_never_names_back_a_role_it_does_not_have(
    client: AsyncClient, role: str
) -> None:
    response = await _refused_ordered(client, ordered_settings={role: {}})

    assert role not in response.text
    assert _errors(response) == [
        {
            "code": "setting-role-unknown",
            "field": "ordered_settings",
            "allowed": ["chat", "image", "video"],
        }
    ]
    assert _detail(response) == "Ordered settings contain an unsupported role."


async def test_an_ordered_plan_names_each_shared_setting_it_needs_a_role_for(
    client: AsyncClient,
) -> None:
    response = await _refused_ordered(
        client, settings={"max_tokens": 64, "Not A Name": "a-sent-marker"}
    )

    assert "a-sent-marker" not in response.text and "Not A Name" not in response.text
    assert _errors(response) == [
        {"code": "setting-needs-a-role", "field": "max_tokens"},
        {"code": "setting-needs-a-role", "field": "*"},
    ]
    assert _detail(response) == (
        "Ordered plans accept role-specific ordered_settings, not shared settings."
    )


async def test_an_ordered_steps_refused_setting_names_its_step(client: AsyncClient) -> None:
    response = await _refused_ordered(client, ordered_settings={"image": {"steps": 999}})

    assert _errors(response) == [
        {"code": "setting-above-maximum", "field": "steps", "limit": 200, "step": 2}
    ]


# An ordered plan's steps and words.


async def test_a_plan_of_too_many_steps_names_the_most_it_may_have(client: AsyncClient) -> None:
    text = "; ".join(["create an image of a gray square"] * 9)

    response = await _refused_ordered(client, text)

    assert _errors(response) == [{"code": "plan-too-many-steps", "field": "plan", "limit": 8}]
    assert _detail(response) == "Ordered plans can contain at most 8 steps."


async def test_a_plan_with_too_many_words_in_all_names_the_most_it_may_hold(
    client: AsyncClient,
) -> None:
    clause = "create an image of a " + "gray " * 3_400
    text = " then ".join([clause] * 3)

    response = await _refused_ordered(client, text)

    assert "gray gray" not in response.text
    assert _errors(response) == [{"code": "plan-prompt-too-long", "field": "plan", "limit": 50_000}]
    assert _detail(response) == "The ordered plan prompt budget is too large."


async def test_a_step_with_too_many_words_is_named_by_its_place_without_its_words(
    client: AsyncClient,
) -> None:
    clause = "create an image of a" + " gray" * 3_996 + "s"
    assert len(clause) == 20_001

    response = await _refused_ordered(client, f"{clause} then describe it")

    # A quoted value is cut to its first and last two dozen characters or so,
    # so any run of twenty would show one.
    assert not any(
        clause[start : start + 20] in response.text for start in range(len(clause) - 19)
    ), "no part of the step's words is said back"
    assert _errors(response) == [
        {"code": "plan-step-prompt-too-long", "field": "plan", "limit": 20_000, "step": 1}
    ]
    assert _detail(response) == "Each step of an ordered plan can be at most 20,000 characters."


# An ordered plan's whole-plan limits.


async def test_an_ordered_plan_past_the_waiting_work_names_the_fixed_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One free place passes the chat's own check; the plan's two steps do not fit in it.
    monkeypatch.setattr("local_lm.orchestrator.MAX_PENDING_WORK_PER_CHAT", 1)

    response = await _refused_ordered(client)

    assert _errors(response) == [{"code": "plan-too-many-pending", "field": "plan", "limit": 1}]
    assert _detail(response) == (
        "This ordered request would exceed the limit of 1 pending items in one chat."
    )


@pytest.mark.parametrize(
    ("limits", "items", "detail"),
    [
        (
            {"max_media_plan_work_units": 0},
            [{"code": "plan-too-much-work", "field": "plan"}],
            "This ordered plan is too large to queue safely. "
            "Reduce its media steps, resolution, frames, or generation steps.",
        ),
        (
            {"max_media_plan_estimated_bytes": 0},
            [{"code": "plan-storage-too-large", "field": "plan", "limit": 0}],
            "This ordered plan has an unsafe storage estimate.",
        ),
        (
            {"max_media_plan_duration_seconds": 0},
            [{"code": "plan-video-too-long", "field": "plan", "limit": 0}],
            "This ordered plan requests too much total video duration.",
        ),
        (
            {
                "max_media_plan_work_units": 0,
                "max_media_plan_estimated_bytes": 0,
                "max_media_plan_duration_seconds": 0,
            },
            [
                {"code": "plan-too-much-work", "field": "plan"},
                {"code": "plan-storage-too-large", "field": "plan", "limit": 0},
                {"code": "plan-video-too-long", "field": "plan", "limit": 0},
            ],
            "This ordered plan is too large to queue safely. "
            "Reduce its media steps, resolution, frames, or generation steps.",
        ),
    ],
    ids=["work", "storage", "video", "all three"],
)
async def test_an_ordered_plan_past_a_whole_plan_limit_names_each_it_passes(
    app: FastAPI,
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    limits: dict[str, int],
    items: list[dict[str, Any]],
    detail: str,
) -> None:
    for limit, value in limits.items():
        monkeypatch.setattr(app.state.services.engines.settings, limit, value)

    response = await _refused_ordered(client, STORY_PICTURE_VIDEO)

    assert _errors(response) == items
    assert _detail(response) == detail


async def test_an_ordered_plan_without_room_to_write_names_the_storage_it_needs(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "local_lm.orchestrator.shutil.disk_usage", lambda _path: SimpleNamespace(free=0)
    )

    response = await _refused_ordered(client)

    assert _errors(response) == [{"code": "plan-storage-unavailable", "field": "plan"}]
    assert _detail(response) == "There is not enough free storage for this ordered plan."


# A batch's saved setting that only some of its workflows take.


async def test_a_pooled_batch_names_each_saved_setting_only_some_workflows_take(
    client: AsyncClient,
) -> None:
    workflow_ids: list[str] = []
    for index, properties in enumerate(
        ({"render_style": {"type": "string", "default": "soft"}}, {}), start=1
    ):
        response = await client.post(
            "/api/workflows",
            json={
                "name": f"Pooled settings workflow {index}",
                "operation": "text_to_image",
                "engine": "mock",
                "api_graph": {"node": {"class_type": f"PooledSettingsWorkflow{index}"}},
                "input_schema": {"type": "object", "properties": properties},
            },
        )
        assert response.status_code == 201, response.text
        seed_workflow_trust(response.json()["current_revision_id"])
        workflow_ids.append(response.json()["current_revision_id"])
    template = await client.post(
        "/api/prompt-templates",
        json={
            "idempotency_key": "pooled-settings-template",
            "name": "Pooled settings",
            "description": "One subject slot",
            "contract": {
                "schema_version": 1,
                "operation": "text_to_image",
                "body": "A {{subject}}.",
                "slots": [{"name": "subject", "mode": "input", "variation_scope": "item"}],
                "resource_policy": {
                    "mode": "pool",
                    "strategy": "round_robin",
                    "options": [
                        {"workflow_revision_id": workflow_id, "lora_policy": {"mode": "none"}}
                        for workflow_id in workflow_ids
                    ],
                },
            },
        },
    )
    assert template.status_code == 201, template.text
    revision = template.json()["revision"]
    project = await client.post("/api/projects", json={"name": "Pooled settings project"})
    assert project.status_code == 201, project.text
    project_id = project.json()["id"]
    chat_id = await _chat(client, project_id=project_id)
    created_batch = await client.post(
        f"/api/chats/{chat_id}/prompt-batches",
        json={
            "idempotency_key": "pooled-settings-preview",
            "template_revision_id": revision["id"],
            "contract_sha256": revision["contract_sha256"],
            "item_count": 2,
            "selection_seed": 0,
            "inputs": {"subject": ["gray square", "ceramic cup"]},
        },
    )
    assert created_batch.status_code in (200, 201), created_batch.text
    batch = created_batch.json()
    with SessionLocal() as session:
        for saved_on in (session.get(Chat, chat_id), session.get(Project, project_id)):
            assert saved_on is not None
            saved_on.generation_settings_json = {"image": {"render_style": "a-saved-marker"}}
        session.commit()
    before = _generation_row_counts()

    queued = await client.post(
        f"/api/prompt-batches/{batch['id']}/queue",
        json={
            "idempotency_key": "pooled-settings-queue",
            "expected_plan_version": batch["plan_version"],
            "expected_plan_sha256": batch["plan_sha256"],
        },
    )

    assert _generation_row_counts() == before
    assert "a-saved-marker" not in queued.text
    taken_by_some = {"code": "setting-not-taken-by-every-workflow", "field": "render_style"}
    assert _errors(queued, "generation-settings-invalid") == [
        {**taken_by_some, "source": "chat"},
        {**taken_by_some, "source": "project"},
    ]
    assert _detail(queued) == (
        "The saved chat setting render_style is not compatible with every workflow selected "
        "for this prompt batch."
    )


# Paths that answer before the output cap is checked.

#: The canvas the preview tests ask for, as the source canvas fixtures do.
CANVAS = {"mode": "extend", "width": 6, "height": 7}


def _media_worker_ready(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    """The media worker reads as running and ready, as a reviewed workflow's preview needs."""

    services = app.state.services
    statuses = services.processes.statuses
    monkeypatch.setattr(
        services.processes,
        "statuses",
        lambda: [
            worker.model_copy(
                update={"managed": True, "running": True, "state": "ready", "pid": 43210}
            )
            if worker.name == "media"
            else worker
            for worker in statuses()
        ],
    )


async def test_a_canvas_preview_answers_a_large_count_in_words_as_any_other(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _media_worker_ready(app, monkeypatch)
    chat_id, payload = await prepared_turn(app, client, monkeypatch)
    answered: dict[int, list[tuple[int, Any]]] = {}
    for count in (17, 12):
        turn = await client.post(
            f"/api/chats/{chat_id}/source-fit/preview",
            json={
                **payload,
                "text": f"Extend the neutral color grid into {count} versions",
                "source_fit": CANVAS,
            },
        )
        answered[count] = [(turn.status_code, turn.json())]
    async with app.state.services.scheduler.lease("primary"):
        accepted = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**payload, "source_fit": {"mode": "extend", "width": 4, "height": 5}},
        )
        assert accepted.status_code == 202, accepted.text
        for count in (17, 12):
            edit = await client.post(
                f"/api/messages/{accepted.json()['user_message']['id']}/edits/source-fit/preview",
                json={
                    "text": f"Extend the neutral color grid into {count} versions",
                    "mode": "image",
                    "idempotency_key": f"canvas-words-{count}",
                    "source_fit": CANVAS,
                },
            )
            answered[count].append((edit.status_code, edit.json()))

    assert [status for status, _ in answered[12]] == [200, 200], answered[12]
    assert answered[17] == answered[12]


async def test_a_regenerated_prompt_library_draft_answers_a_large_count_in_words_as_any_other(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    batch = await _batch(
        client, chat_id, ["gray square in 17 pictures", "gray square in 12 pictures"]
    )
    queued = await _queue(client, batch)
    assert queued.status_code == 202, queued.text
    with SessionLocal() as session:
        drafts = {
            (17 if "17 pictures" in run.standalone_prompt else 12): (
                run.id,
                run.assistant_message_id,
            )
            for run in session.scalars(select(Run).where(Run.chat_id == chat_id))
        }
    answered: dict[int, Any] = {}
    for count, (run_id, message_id) in sorted(drafts.items()):
        await _finished(client, run_id)
        regenerated = await client.post(
            f"/api/messages/{message_id}/regenerate", json={"settings": {}}
        )
        body = regenerated.json()
        answered[count] = (
            (202, body["run"]["provenance_json"]["media_output"]["count"])
            if regenerated.status_code == 202
            else (regenerated.status_code, body)
        )

    assert sorted(drafts) == [12, 17]
    assert answered[12] == (202, 1), "a regenerate makes its one output"
    assert answered[17] == answered[12]


async def test_a_prompt_library_source_answers_a_large_count_in_words_as_any_other(
    client: AsyncClient,
) -> None:
    chat_id = await _chat(client)
    batch = await _batch(client, chat_id, ["gray square"])
    source = _composer_source(batch, batch["items"][0])
    before = _generation_row_counts()
    answered: dict[int, tuple[int, str, str]] = {}
    for count in (17, 12):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={
                "text": f"Create {count} pictures of a gray square",
                "mode": "image",
                "prompt_source": source,
            },
        )
        body = response.json()
        answered[count] = (response.status_code, body["code"], body["detail"])

    assert _generation_row_counts() == before
    assert answered[12][:2] == (422, "prompt-source-invalid"), answered[12]
    assert answered[17] == answered[12]


# What a request is said to have sent.


def test_a_request_is_said_to_send_its_count_and_its_roles() -> None:
    request = TurnRequest(
        text=TEXT,
        output_count=2,
        settings={"seed": 4},
        ordered_settings={"image": {"steps": 9}, "chat": {}},
    )

    assert _sent_setting_keys(request) == {"seed", "steps", "image", "chat", "output_count"}
    assert "output_count" not in _sent_setting_keys(TurnRequest(text=TEXT))
