"""A remix preview judges a picture's settings against a workflow chosen here, writing nothing."""

from __future__ import annotations

import dataclasses
import json
from io import BytesIO
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from sqlalchemy import func, select

from local_lm import api as api_module
from local_lm.db import SessionLocal
from local_lm.models import (
    Artifact,
    Chat,
    GenerationPreset,
    Job,
    Message,
    ModelProfile,
    Run,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowRevision,
    WorkflowUseCasePreset,
    WorkPlan,
)
from local_lm.picture_remix import preview_remix

PROMPT = "a ceramic cup on a wooden table"
NEGATIVE = "blurry, low contrast"
SETTINGS = (
    f"{PROMPT}\n"
    f"Negative prompt: {NEGATIVE}\n"
    "Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 12345, "
    "Size: 512x768, Model: neutral-model, Denoising strength: 0.5, Batch size: 2"
)
COUNTED = (Chat, Message, WorkPlan, Run, Job, GenerationPreset, ModelProfile)


def _png(*texts: tuple[str, str]) -> bytes:
    info = PngInfo()
    for keyword, text in texts:
        info.add_text(keyword, text)
    output = BytesIO()
    Image.new("RGB", (8, 8), (90, 120, 150)).save(output, format="PNG", pnginfo=info)
    return output.getvalue()


def _profile(name: str, *, role: str = "image", **saved: Any) -> str:
    with SessionLocal() as session:
        profile = ModelProfile(name=name, role=role, engine="mock", request_settings_json=saved)
        session.add(profile)
        session.commit()
        return profile.id


def _revision(name: str, *, input_schema: dict[str, Any] | None = None) -> str:
    with SessionLocal() as session:
        family = WorkflowFamily(name=name)
        definition = WorkflowDefinition(
            family=family, variant_key="create", name=name, operation="text_to_image"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            api_graph_json={},
            input_schema_json=input_schema or {},
            dependencies_json={},
            trusted=True,
        )
        session.add_all([family, definition, revision])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


async def _uploaded(client: AsyncClient, content: bytes) -> str:
    response = await client.post(
        "/api/artifacts", files={"file": ("picture.png", content, "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _preview(
    client: AsyncClient,
    artifact_id: str,
    revision_id: str,
    profile_id: str,
    apply: list[str] | None = None,
) -> Any:
    return await client.post(
        f"/api/artifacts/{artifact_id}/remix-preview",
        json={
            "workflow_revision_id": revision_id,
            "profile_id": profile_id,
            "apply": apply or [],
        },
    )


def _states(body: dict[str, Any]) -> dict[str, tuple[str | None, str, str | None]]:
    return {
        claim["key"]: (claim["setting"], claim["state"], claim["reason"])
        for claim in body["claims"]
    }


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: int(session.scalar(select(func.count()).select_from(model)) or 0)
            for model in COUNTED
        }


async def test_each_claim_is_judged_against_the_chosen_workflow(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    response = await _preview(client, artifact_id, revision_id, profile_id)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ready"] is True and body["refusals"] == []
    assert _states(body) == {
        "prompt": (None, "supported", None),
        "negative_prompt": ("negative_prompt", "supported", None),
        "steps": ("steps", "supported", None),
        # Names mean something only against a vocabulary; this workflow has none.
        "sampler": ("sampler", "unresolved", "no_vocabulary"),
        "scheduler": ("scheduler", "unresolved", "no_vocabulary"),
        "guidance": ("cfg", "supported", None),
        "seed": ("seed", "supported", None),
        "denoise": (None, "ignored", "edit_only"),
        "batch": (None, "ignored", "one_picture"),
        "width": ("width", "supported", None),
        "height": ("height", "supported", None),
    }
    assert {"name": "Model", "reason": "names_a_file"} in body["ignored"]
    assert [claim["applied"] for claim in body["claims"] if claim["key"] == "prompt"] == [True]
    assert not any(claim["applied"] for claim in body["claims"] if claim["key"] != "prompt")
    resolved = body["resolved"]
    assert resolved["text"] == PROMPT
    # Nothing applied: no negative words, one picture, and a seed drawn when queued.
    assert resolved["settings"]["negative_prompt"] == ""
    assert resolved["settings"]["batch_size"] == 1
    assert "seed" not in resolved["settings"]
    assert body["review_digest"].startswith("sha256:")
    assert body["workflow_revision_id"] == revision_id and body["profile_id"] == profile_id


async def test_applied_claims_are_what_the_remix_would_run_with(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    apply = ["negative_prompt", "steps", "guidance", "seed", "width", "height"]

    plain = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    applied = (await _preview(client, artifact_id, revision_id, profile_id, apply)).json()

    settings = applied["resolved"]["settings"]
    assert settings["negative_prompt"] == NEGATIVE
    assert (settings["steps"], settings["cfg"], settings["seed"]) == (20, 7.0, 12345)
    assert (settings["width"], settings["height"]) == (512, 768)
    assert {claim["key"] for claim in applied["claims"] if claim["applied"]} == {"prompt", *apply}
    assert applied["review_digest"] != plain["review_digest"]
    again = (await _preview(client, artifact_id, revision_id, profile_id, apply)).json()
    assert again["review_digest"] == applied["review_digest"]


async def test_only_a_supported_claim_can_be_applied_and_a_size_only_whole(
    client: AsyncClient,
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    for apply in (["sampler"], ["width"], ["seed", "height"]):
        response = await _preview(client, artifact_id, revision_id, profile_id, apply)
        assert response.status_code == 422, apply
        assert response.json()["code"] == "remix-choice-invalid"
    unknown = await _preview(client, artifact_id, revision_id, profile_id, ["prompt"])
    assert unknown.status_code == 422


async def test_values_the_workflow_refuses_are_incompatible_and_never_wrapped(
    client: AsyncClient,
) -> None:
    text = f"{PROMPT}\nSteps: 500, CFG scale: 45, Seed: 6342567893452345234, Size: 500x768"
    artifact_id = await _uploaded(client, _png(("parameters", text)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    states = _states(body)
    assert states["steps"] == ("steps", "incompatible", "value_refused")
    assert states["guidance"] == ("cfg", "incompatible", "value_refused")
    assert states["seed"] == ("seed", "incompatible", "value_refused")
    # Off the size step, and no proof of what sizes the workflow makes.
    assert states["width"] == ("width", "unresolved", "size_unproven")
    seed = next(claim for claim in body["claims"] if claim["key"] == "seed")
    assert seed["value"] == "6342567893452345234"


def _graph(sampler_name: str, latent_inputs: dict[str, Any]) -> str:
    return json.dumps(
        {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 5,
                    "steps": 12,
                    "cfg": 6.5,
                    "sampler_name": sampler_name,
                    "positive": ["6", 0],
                    "latent_image": ["5", 0],
                },
            },
            "5": {"class_type": "EmptyLatentImage", "inputs": latent_inputs},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT}},
        }
    )


async def test_a_name_is_supported_only_when_the_workflow_knows_it_exactly(
    client: AsyncClient,
) -> None:
    schema = {
        "type": "object",
        "properties": {"sampler": {"type": "string", "enum": ["euler", "dpmpp_2m"]}},
    }
    profile_id, revision_id = _profile("Ceramic model"), _revision("Named", input_schema=schema)
    known = await _uploaded(client, _png(("prompt", _graph("euler", {"width": 512}))))
    unknown = await _uploaded(client, _png(("parameters", SETTINGS)))

    known_body = (await _preview(client, known, revision_id, profile_id)).json()
    unknown_body = (await _preview(client, unknown, revision_id, profile_id)).json()

    assert _states(known_body)["sampler"] == ("sampler", "supported", None)
    # Only half a size was given, so no size is applied from it.
    assert _states(known_body)["width"] == ("width", "unresolved", "size_incomplete")
    assert _states(unknown_body)["sampler"] == ("sampler", "incompatible", "not_a_choice")


async def test_a_setting_the_workflow_puts_in_two_places_is_unresolved(
    client: AsyncClient,
) -> None:
    schema = {
        "type": "object",
        "properties": {
            "steps": {"type": "integer", "minimum": 1, "maximum": 200},
            "workflow_steps_0a1b2c": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "x-lm-atelier-graph-settings": {
            "version": 1,
            "bindings": [
                {"parameter": "steps", "node_id": "3", "input_name": "steps"},
                {"parameter": "workflow_steps_0a1b2c", "node_id": "9", "input_name": "steps"},
            ],
        },
    }
    profile_id = _profile("Ceramic model")
    revision_id = _revision("Two passes", input_schema=schema)
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert body["ready"] is True, body
    assert _states(body)["steps"] == ("steps", "unresolved", "several_controls")


async def test_a_preview_writes_nothing_and_is_never_cached(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    with SessionLocal() as session:
        row = session.get(Artifact, artifact_id)
        assert row is not None
        metadata_before = dict(row.metadata_json)
    before = _counts()

    response = await _preview(client, artifact_id, revision_id, profile_id, ["steps"])

    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert _counts() == before
    with SessionLocal() as session:
        row = session.get(Artifact, artifact_id)
        assert row is not None
        assert row.metadata_json == metadata_before


async def test_a_name_in_the_file_chooses_nothing(client: AsyncClient) -> None:
    # A model with exactly the name the file gives is never picked by that name.
    _profile("neutral-model")
    chosen = _profile("Ceramic model")
    revision_id = _revision("Ceramic workflow")
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    body = (await _preview(client, artifact_id, revision_id, chosen)).json()

    assert body["profile_id"] == chosen
    assert {"name": "Model", "reason": "names_a_file"} in body["ignored"]


async def test_saved_defaults_do_not_fill_what_was_not_applied(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    with SessionLocal() as session:
        session.add(
            GenerationPreset(
                name="Everyday image",
                role="image",
                is_default=True,
                settings_json={"steps": 3, "negative_prompt": "saved words"},
            )
        )
        session.commit()

    settings = (await _preview(client, artifact_id, revision_id, profile_id)).json()["resolved"][
        "settings"
    ]

    assert settings["negative_prompt"] == ""
    assert settings["steps"] != 3


async def test_a_model_s_saved_settings_do_not_fill_what_was_not_applied(
    client: AsyncClient,
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id = _profile("Ceramic model", negative_prompt="saved words", batch_size=4, steps=9)
    revision_id = _revision("Ceramic workflow")

    settings = (await _preview(client, artifact_id, revision_id, profile_id)).json()["resolved"][
        "settings"
    ]

    # Said outright, so a model's saved words and picture count never stand in.
    assert settings["negative_prompt"] == ""
    assert settings["batch_size"] == 1
    # A setting the picture's file was not asked for keeps the model's own value.
    assert settings["steps"] == 9


async def test_a_choice_that_cannot_run_is_an_answer(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    video_profile = _profile("Moving model", role="video")
    no_prompt = await _uploaded(client, _png(("parameters", "Steps: 20, Seed: 7")))
    not_png = await _uploaded(client, b"GIF89a" + bytes(32))

    cases = [
        (artifact_id, revision_id, "profile_missing", "remix-model-unusable"),
        (artifact_id, revision_id, video_profile, "remix-model-unusable"),
        (artifact_id, "revision_missing", profile_id, "remix-workflow-unusable"),
        (no_prompt, revision_id, profile_id, "remix-prompt-missing"),
        (not_png, revision_id, profile_id, "remix-prompt-missing"),
    ]
    for picture, revision, profile, code in cases:
        response = await _preview(client, picture, revision, profile)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["ready"] is False
        assert [item["code"] for item in body["refusals"]] == [code]
        assert body["resolved"] is None and body["review_digest"] is None
        assert body["claims"] == []


async def test_an_unknown_picture_is_not_found(client: AsyncClient) -> None:
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    response = await _preview(client, "sha256:" + "0" * 64, revision_id, profile_id)

    assert response.status_code == 404
    assert response.json()["code"] == "artifact-not-found"


async def test_a_turn_with_the_same_choices_runs_with_what_the_preview_resolved(
    app: FastAPI, client: AsyncClient
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    apply = ["negative_prompt", "steps", "guidance", "seed", "width", "height"]
    preview = (await _preview(client, artifact_id, revision_id, profile_id, apply)).json()
    resolved = preview["resolved"]["settings"]
    chat = await client.post("/api/chats", json={"title": "Same choices as a turn"})

    async with app.state.services.scheduler.lease("primary"):
        turn = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": PROMPT,
                "mode": "image",
                "profile_id": profile_id,
                "workflow_revision_id": revision_id,
                "preset_id": None,
                "settings": {
                    "negative_prompt": NEGATIVE,
                    "steps": 20,
                    "cfg": 7.0,
                    "seed": 12345,
                    "width": 512,
                    "height": 768,
                    "batch_size": 1,
                },
            },
        )
        assert turn.status_code == 202, turn.text
        with SessionLocal() as session:
            run = session.get(Run, turn.json()["run"]["id"])
            assert run is not None
            assert run.settings_json == resolved


def _runs(chat_id: str) -> list[Run]:
    with SessionLocal() as session:
        return list(
            session.scalars(
                select(Run)
                .join(WorkPlan, Run.work_plan_id == WorkPlan.id)
                .where(WorkPlan.chat_id == chat_id)
            )
        )


async def _new_chat(client: AsyncClient, title: str = "Remix") -> str:
    response = await client.post("/api/chats", json={"title": title})
    assert response.status_code in (200, 201), response.text
    return str(response.json()["id"])


async def _queue(
    client: AsyncClient,
    chat_id: str,
    artifact_id: str,
    preview: dict[str, Any],
    apply: list[str],
) -> Any:
    return await client.post(
        f"/api/chats/{chat_id}/remixes",
        json={
            "artifact_id": artifact_id,
            "workflow_revision_id": preview["workflow_revision_id"],
            "profile_id": preview["profile_id"],
            "apply": apply,
            "review_digest": preview["review_digest"],
        },
    )


async def test_a_remix_is_queued_as_it_was_previewed(app: FastAPI, client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    apply = ["negative_prompt", "steps", "guidance", "seed", "width", "height"]
    preview = (await _preview(client, artifact_id, revision_id, profile_id, apply)).json()
    chat_id = await _new_chat(client)

    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, apply)

        assert response.status_code == 202, response.text
        [run] = _runs(chat_id)
        assert run.settings_json == preview["resolved"]["settings"]
        assert run.settings_json["seed"] == 12345
        assert run.standalone_prompt == PROMPT
        receipt = run.provenance_json["remix"]
        assert receipt["source_artifact_id"] == artifact_id
        assert receipt["review_digest"] == preview["review_digest"]
        assert receipt["choices"] == {
            "workflow_revision_id": revision_id,
            "profile_id": profile_id,
        }
        applied = {claim["key"] for claim in receipt["claims"] if claim["applied"]}
        assert applied == set(apply)
        # Which settings, never what the file said: the run itself holds what was applied.
        text = json.dumps(receipt)
        assert PROMPT not in text and NEGATIVE not in text and "12345" not in text
    with SessionLocal() as session:
        row = session.get(Artifact, artifact_id)
        assert row is not None and row.metadata_json == {"uploaded": True}


async def test_a_model_s_saved_strength_and_seed_are_previewed_as_a_remix_runs_them(
    app: FastAPI, client: AsyncClient
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id = _profile("Ceramic model", denoise=0.6, seed=42)
    revision_id = _revision("Ceramic workflow")
    preview = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    settings = preview["resolved"]["settings"]
    # Words alone make a whole new picture, whatever strength the model saves.
    assert settings["denoise"] == 1
    assert settings["seed"] == 42
    chat_id = await _new_chat(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, [])
        assert response.status_code == 202, response.text
        [run] = _runs(chat_id)
        assert run.settings_json == settings


async def test_a_workspace_settings_recipe_never_enters_a_remix(
    app: FastAPI, client: AsyncClient
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    with SessionLocal() as session:
        session.add(
            WorkflowUseCasePreset(
                name="Everyday recipe",
                use_case="image_generation",
                settings_json={"steps": 40},
                is_default=True,
            )
        )
        session.commit()
    preview = (await _preview(client, artifact_id, revision_id, profile_id)).json()
    assert preview["resolved"]["settings"]["steps"] != 40

    async with app.state.services.scheduler.lease("primary"):
        remix_chat = await _new_chat(client)
        response = await _queue(client, remix_chat, artifact_id, preview, [])
        assert response.status_code == 202, response.text
        [remix] = _runs(remix_chat)
        # No seed was set, so one was drawn when the remix was accepted.
        drawn = {key: value for key, value in remix.settings_json.items() if key != "seed"}
        assert drawn == preview["resolved"]["settings"]
        # The recipe does reach an ordinary turn, so its absence above is the remix's doing.
        plain_chat = await _new_chat(client, "Plain")
        plain = await client.post(
            f"/api/chats/{plain_chat}/turns",
            json={
                "text": PROMPT,
                "mode": "image",
                "profile_id": profile_id,
                "workflow_revision_id": revision_id,
                "preset_id": None,
            },
        )
        assert plain.status_code == 202, plain.text
        [ordinary] = _runs(plain_chat)
        assert ordinary.settings_json["steps"] == 40


async def test_a_remix_is_refused_unless_it_is_what_was_previewed(
    app: FastAPI, client: AsyncClient
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    preview = (await _preview(client, artifact_id, revision_id, profile_id, ["steps"])).json()

    async with app.state.services.scheduler.lease("primary"):
        changed = await _queue(client, await _new_chat(client), artifact_id, preview, ["seed"])
        assert changed.status_code == 409
        assert changed.json()["code"] == "remix-review-changed"

        used = await _new_chat(client)
        with SessionLocal() as session:
            chat = session.get(Chat, used)
            assert chat is not None
            chat.generation_settings_json = {"steps": 3}
            session.commit()
        not_clean = await _queue(client, used, artifact_id, preview, ["steps"])
        assert not_clean.status_code == 409
        assert not_clean.json()["code"] == "remix-chat-not-clean"

        unusable = {**preview, "profile_id": "profile_missing"}
        refused = await _queue(client, await _new_chat(client), artifact_id, unusable, ["steps"])
        assert refused.status_code == 409
        assert refused.json()["code"] == "remix-unavailable"
        assert [item["code"] for item in refused.json()["refusals"]] == ["remix-model-unusable"]

    assert _counts()["Run"] == 0


async def test_a_prompt_that_asks_for_several_pictures_is_not_remixed(
    client: AsyncClient,
) -> None:
    text = f"two versions of {PROMPT}\nSteps: 20"
    artifact_id = await _uploaded(client, _png(("parameters", text)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert body["ready"] is False
    assert [item["code"] for item in body["refusals"]] == ["remix-several-pictures"]


async def test_controls_a_workflow_splits_or_shares_are_unresolved(client: AsyncClient) -> None:
    split = {
        "type": "object",
        "properties": {
            "workflow_steps_0a1b2c": {"type": "integer", "minimum": 1, "maximum": 200},
            "workflow_steps_3d4e5f": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "x-lm-atelier-graph-settings": {
            "version": 1,
            "bindings": [
                {"parameter": "workflow_steps_0a1b2c", "node_id": "3", "input_name": "steps"},
                {"parameter": "workflow_steps_3d4e5f", "node_id": "9", "input_name": "steps"},
            ],
        },
    }
    # One control kept from a split: the other pass's copy is not one this can set.
    single = {
        "type": "object",
        "properties": {
            "workflow_steps_0a1b2c": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "x-lm-atelier-graph-settings": {
            "version": 1,
            "bindings": [
                {"parameter": "workflow_steps_0a1b2c", "node_id": "3", "input_name": "steps"},
            ],
        },
    }
    profile_id = _profile("Ceramic model")
    split_revision = _revision("Split passes", input_schema=split)
    single_revision = _revision("One of a split", input_schema=single)
    shared_revision = _revision("Shared passes")
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, shared_revision)
        assert revision is not None
        revision.api_graph_json = {
            "3": {"class_type": "KSampler", "inputs": {"steps": "${steps}"}},
            "9": {"class_type": "KSampler", "inputs": {"steps": "${steps}"}},
        }
        session.commit()
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    for revision_id in (split_revision, single_revision, shared_revision):
        body = (await _preview(client, artifact_id, revision_id, profile_id)).json()
        assert body["ready"] is True, body
        assert _states(body)["steps"] == ("steps", "unresolved", "several_controls")


async def test_a_whole_number_suits_a_whole_number_setting_and_numbers_are_never_names(
    client: AsyncClient,
) -> None:
    whole = {
        "type": "object",
        "properties": {"cfg": {"type": "integer", "minimum": 1, "maximum": 30}},
    }
    fixed = {
        "type": "object",
        "properties": {"steps": {"type": "integer", "enum": [4, 8], "default": 4}},
    }
    profile_id = _profile("Ceramic model")
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    whole_body = (
        await _preview(
            client, artifact_id, _revision("Whole", input_schema=whole), profile_id, ["guidance"]
        )
    ).json()
    fixed_body = (
        await _preview(client, artifact_id, _revision("Fixed", input_schema=fixed), profile_id)
    ).json()

    assert _states(whole_body)["guidance"] == ("cfg", "supported", None)
    assert whole_body["resolved"]["settings"]["cfg"] == 7
    assert isinstance(whole_body["resolved"]["settings"]["cfg"], int)
    assert _states(fixed_body)["steps"] == ("steps", "incompatible", "value_refused")


async def test_a_remix_that_would_not_run_as_shown_is_never_committed(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id, revision_id = _profile("Ceramic model"), _revision("Ceramic workflow")
    preview = (await _preview(client, artifact_id, revision_id, profile_id, ["steps"])).json()
    real = preview_remix

    async def shown_otherwise(*args: Any, **kwargs: Any) -> Any:
        # Same digest, but what was shown no longer matches what would run.
        resolved = await real(*args, **kwargs)
        settings = {**resolved.settings, "steps": resolved.settings["steps"] + 1}
        return dataclasses.replace(resolved, settings=settings)

    monkeypatch.setattr(api_module, "preview_remix", shown_otherwise)
    chat_id = await _new_chat(client)
    async with app.state.services.scheduler.lease("primary"):
        response = await _queue(client, chat_id, artifact_id, preview, ["steps"])

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "remix-differs"
    assert _runs(chat_id) == []


def _with_graph(revision_id: str, graph: dict[str, Any]) -> None:
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.api_graph_json = graph
        session.commit()


async def test_a_seed_is_said_to_be_drawn_only_when_one_is(client: AsyncClient) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    drawn = _profile("Ceramic model")
    saved = _profile("Saved seed", seed=42)
    plain = _revision("Ceramic workflow")
    # Two samplers, each with its own seed: no one seed setting is offered.
    split = _revision(
        "Two seeds",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_seed_0a1b2c": {"type": "integer", "minimum": 0, "maximum": 99},
                "workflow_seed_3d4e5f": {"type": "integer", "minimum": 0, "maximum": 99},
            },
            "x-lm-atelier-graph-settings": {
                "version": 1,
                "bindings": [
                    {"parameter": "workflow_seed_0a1b2c", "node_id": "3", "input_name": "seed"},
                    {"parameter": "workflow_seed_3d4e5f", "node_id": "9", "input_name": "seed"},
                ],
            },
        },
    )

    cases = [(drawn, plain, True), (saved, plain, False), (drawn, split, False)]
    for profile_id, revision_id, expected in cases:
        resolved = (await _preview(client, artifact_id, revision_id, profile_id)).json()["resolved"]
        assert resolved["seed_drawn"] is expected, (profile_id, revision_id)
        assert ("seed" in resolved["settings"]) is (profile_id == saved)


async def test_a_workflow_that_makes_several_pictures_at_once_is_not_remixed(
    client: AsyncClient,
) -> None:
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))
    profile_id = _profile("Ceramic model")
    cases = {
        "Four at once": (4, False),
        "Set elsewhere": (["10", 0], False),
        "Asked here": ("${batch_size}", True),
        "One at once": (1, True),
    }
    for name, (count, ready) in cases.items():
        revision_id = _revision(name)
        _with_graph(
            revision_id,
            {"5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": count}}},
        )
        body = (await _preview(client, artifact_id, revision_id, profile_id)).json()
        assert body["ready"] is ready, name
        if not ready:
            assert [item["code"] for item in body["refusals"]] == ["remix-workflow-makes-several"]


async def test_a_copy_that_never_runs_is_no_second_control(client: AsyncClient) -> None:
    schema = {
        "type": "object",
        "properties": {"steps": {"type": "integer", "minimum": 1, "maximum": 200}},
        "x-lm-atelier-graph-settings": {
            "version": 1,
            "bindings": [{"parameter": "steps", "node_id": "3", "input_name": "steps"}],
            "fixed": [
                {
                    "node_id": "9",
                    "input_name": "steps",
                    "label": "Steps (KSampler)",
                    "reason": "disconnected",
                }
            ],
        },
    }
    profile_id = _profile("Ceramic model")
    revision_id = _revision("Left over", input_schema=schema)
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert _states(body)["steps"] == ("steps", "supported", None)


async def test_a_copy_set_by_another_node_is_a_second_control(client: AsyncClient) -> None:
    profile_id = _profile("Ceramic model")
    revision_id = _revision("Linked seed")
    _with_graph(
        revision_id,
        {
            "3": {"class_type": "KSampler", "inputs": {"seed": "${seed}"}},
            "9": {"class_type": "KSampler", "inputs": {"seed": ["10", 0]}},
        },
    )
    artifact_id = await _uploaded(client, _png(("parameters", SETTINGS)))

    body = (await _preview(client, artifact_id, revision_id, profile_id)).json()

    assert _states(body)["seed"] == ("seed", "unresolved", "several_controls")
