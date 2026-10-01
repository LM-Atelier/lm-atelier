"""Two generation choices checked against one frozen request, through the real endpoint."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import func, select

from local_lm.db import SessionLocal
from local_lm.models import (
    Chat,
    GenerationPreset,
    Job,
    Message,
    ModelInstall,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowRevision,
    WorkPlan,
    WorkStep,
)

PREFLIGHT = "/api/generation-experiments/preflight"
PROMPT = "A quiet harbor at dawn with three small boats"
NEGATIVE = "blurry, text"
COUNTED = (Chat, Message, WorkPlan, WorkStep, Run, Job, GenerationPreset, ModelProfile)


def _profile(name: str, *, role: str = "image", engine: str = "mock", **values: Any) -> str:
    with SessionLocal() as session:
        profile = ModelProfile(name=name, role=role, engine=engine, **values)
        session.add(profile)
        session.commit()
        return profile.id


def _revision(name: str, *, operation: str = "text_to_image", **values: Any) -> str:
    with SessionLocal() as session:
        family = WorkflowFamily(name=name)
        definition = WorkflowDefinition(
            family=family, variant_key="create", name=name, operation=operation
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            **{
                "api_graph_json": {},
                "input_schema_json": {},
                "dependencies_json": {},
                "trusted": True,
                **values,
            },
        )
        session.add_all([family, definition, revision])
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


def _arm(label: str, profile_id: str, revision_id: str, **settings: Any) -> dict[str, Any]:
    return {
        "label": label,
        "profile_id": profile_id,
        "workflow_revision_id": revision_id,
        "settings": settings,
    }


def _request(*arms: dict[str, Any], **values: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "Harbor comparison",
        "operation": "text_to_image",
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE,
        "geometry": {"mode": "size", "width": 512, "height": 384},
        "seed_policy": {"kind": "same_recorded_number", "seed": 41},
        "arms": list(arms),
    }
    body.update(values)
    return body


def _counts() -> dict[str, int]:
    with SessionLocal() as session:
        return {
            model.__name__: int(session.scalar(select(func.count()).select_from(model)) or 0)
            for model in COUNTED
        }


def _two_choices() -> tuple[dict[str, Any], dict[str, Any]]:
    first = _arm("Fewer steps", _profile("Harbor one"), _revision("Harbor one"), steps=8)
    second = _arm("More steps", _profile("Harbor two"), _revision("Harbor two"), steps=20)
    return first, second


async def test_a_compatible_comparison_is_resolved_and_writes_nothing(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    before = _counts()
    response = await client.post(PREFLIGHT, json=_request(first, second))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "compatible" and body["refusals"] == []
    assert len(body["preflight_sha256"]) == 64
    assert body["seed_equivalence"] == "none"
    for arm, steps in zip(body["arms"], (8, 20), strict=True):
        assert arm["outcome"] == "compatible"
        settings = arm["effective_settings"]
        assert settings["negative_prompt"] == NEGATIVE
        assert (settings["width"], settings["height"]) == (512, 384)
        assert (arm["width"], arm["height"]) == (512, 384)
        assert settings["steps"] == steps
        assert "seed" not in settings
        assert len(arm["snapshot_sha256"]) == 64
    assert [item["resource"] for item in body["estimate"]] == ["work_units", "output_bytes"]
    assert all(item["kind"] == "estimated" for item in body["estimate"])
    assert _counts() == before

    again = await client.post(PREFLIGHT, json=_request(first, second))
    assert again.json()["preflight_sha256"] == body["preflight_sha256"]


async def test_a_choice_resolves_to_the_settings_an_ordinary_turn_runs_with(
    app: FastAPI, client: AsyncClient
) -> None:
    """The preflight composes the turn's own steps; a turn with the same choices agrees."""

    first, second = _two_choices()
    body = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert body["outcome"] == "compatible", body
    arm = body["arms"][0]
    chat = await client.post("/api/chats", json={"title": "Same choices as a turn"})
    async with app.state.services.scheduler.lease("primary"):
        turn = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": PROMPT,
                "mode": "image",
                "profile_id": first["profile_id"],
                "workflow_revision_id": first["workflow_revision_id"],
                "preset_id": None,
                "settings": {
                    **first["settings"],
                    "negative_prompt": NEGATIVE,
                    "width": 512,
                    "height": 384,
                    "seed": 41,
                },
            },
        )
        assert turn.status_code == 202, turn.text
        with SessionLocal() as session:
            run = session.get(Run, turn.json()["run"]["id"])
            assert run is not None
            assert run.settings_json == {**arm["effective_settings"], "seed": 41}


async def test_saved_defaults_never_enter_a_choice(client: AsyncClient) -> None:
    first, second = _two_choices()
    plain = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    with SessionLocal() as session:
        session.add(
            GenerationPreset(
                name="Everyday image",
                role="image",
                is_default=True,
                settings_json={"steps": 3, "cfg": 1.5, "negative_prompt": "saved words"},
            )
        )
        session.commit()
    with_default = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert with_default["preflight_sha256"] == plain["preflight_sha256"]
    for before, after in zip(plain["arms"], with_default["arms"], strict=True):
        assert after["effective_settings"] == before["effective_settings"]
        assert after["effective_settings"]["cfg"] != 1.5
        assert after["effective_settings"]["negative_prompt"] == NEGATIVE


SETTING_NAMED = {
    "unknown-setting": "sparkle",
    "invalid-setting": "steps",
    "own-seed": "seed",
    "own-size": "width",
    "selection": "mask",
}


def _variant(case: str, profile_id: str, revision_id: str) -> tuple[dict[str, Any], str]:
    """One broken second choice and the refusal it must produce."""

    arm = _arm("Second", profile_id, revision_id)
    if case == "image-edit-workflow":
        arm["workflow_revision_id"] = _revision("Edit only", operation="image_to_image")
        return arm, "arm-operation-mismatch"
    if case == "unreviewed-workflow":
        arm["workflow_revision_id"] = _revision("Unreviewed", trusted=False)
        return arm, "arm-workflow-untrusted"
    if case == "dependencies-not-ready":
        arm["workflow_revision_id"] = _revision("Not ready", dependency_contract_sha256="a" * 64)
        return arm, "arm-activation-not-ready"
    if case == "missing-package":
        arm["workflow_revision_id"] = _revision(
            "Needs a package", dependencies_json={"custom_nodes": ["neutral-node-pack"]}
        )
        return arm, "arm-package-missing"
    if case == "video-model":
        arm["profile_id"] = _profile("Video model", role="video")
        return arm, "arm-profile-unavailable"
    if case == "missing-model":
        arm["profile_id"] = "profile_missing"
        return arm, "arm-profile-unavailable"
    if case == "missing-workflow":
        arm["workflow_revision_id"] = "wfrev_missing"
        return arm, "arm-workflow-unavailable"
    if case == "unknown-setting":
        arm["settings"] = {"sparkle": 3}
        return arm, "arm-setting-unsupported"
    if case == "invalid-setting":
        arm["settings"] = {"steps": -4}
        return arm, "arm-setting-invalid"
    if case == "own-seed":
        arm["settings"] = {"seed": 7}
        return arm, "common-input-overridden"
    if case == "own-size":
        arm["settings"] = {"width": 640}
        return arm, "common-input-overridden"
    if case == "selection":
        arm["settings"] = {"mask": {"kind": "rect"}}
        return arm, "arm-input-unsupported"
    raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "image-edit-workflow",
        "unreviewed-workflow",
        "dependencies-not-ready",
        "missing-package",
        "video-model",
        "missing-model",
        "missing-workflow",
        "unknown-setting",
        "invalid-setting",
        "own-seed",
        "own-size",
        "selection",
    ],
)
async def test_one_choice_that_cannot_run_refuses_the_comparison(
    client: AsyncClient, case: str
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second, code = _variant(case, _profile("Second model"), _revision("Second workflow"))
    before = _counts()
    response = await client.post(PREFLIGHT, json=_request(first, second))
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["outcome"] == "refused" and body["preflight_sha256"] is None
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [(code, 2)]
    assert body["refusals"][0]["setting"] == SETTING_NAMED.get(case)
    assert [arm["outcome"] for arm in body["arms"]] == ["compatible", "refused"]
    assert PROMPT not in response.text and NEGATIVE not in response.text.replace(
        f'"negative_prompt":"{NEGATIVE}"', ""
    )
    assert _counts() == before


async def test_a_refusal_names_a_setting_only_when_it_looks_like_one(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Other model"), _revision("Other workflow"))
    second["settings"] = {PROMPT: 1}
    body = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert body["refusals"][0]["code"] == "arm-setting-unsupported"
    assert body["refusals"][0]["setting"] is None
    assert PROMPT not in str(body["refusals"])


async def test_one_seed_is_offered_as_the_same_start_only_within_one_family(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    body = (
        await client.post(
            PREFLIGHT,
            json=_request(first, second, seed_policy={"kind": "fixed_numeric", "seed": 9}),
        )
    ).json()
    assert body["outcome"] == "refused"
    assert body["refusals"] == [
        {
            "code": "seed-family-unproven",
            "arm_ordinal": None,
            "setting": None,
            "alternative": {"seed_policy": "same_recorded_number", "profile_id": None},
            "message": body["refusals"][0]["message"],
        }
    ]
    assert body["seed_equivalence"] == "none"


async def test_two_identical_choices_with_one_seed_are_refused(client: AsyncClient) -> None:
    profile_id = _profile("Twin model")
    revision_id = _revision("Twin workflow")
    first = _arm("Left", profile_id, revision_id, steps=12)
    second = _arm("Right", profile_id, revision_id, steps=12)
    same = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert [item["code"] for item in same["refusals"]] == ["arms-identical"]
    independent = (
        await client.post(
            PREFLIGHT,
            json=_request(
                first, second, seed_policy={"kind": "independent_deterministic", "seed": 4}
            ),
        )
    ).json()
    assert independent["outcome"] == "compatible"


async def test_a_change_to_a_choice_changes_the_digest(client: AsyncClient) -> None:
    first, second = _two_choices()
    before = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    with SessionLocal() as session:
        profile = session.get(ModelProfile, first["profile_id"])
        assert profile is not None
        profile.load_settings_json = {**(profile.load_settings_json or {}), "steps": 30}
        session.commit()
    after = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert after["outcome"] == "compatible"
    assert after["preflight_sha256"] != before["preflight_sha256"]
    assert after["arms"][1]["snapshot_sha256"] == before["arms"][1]["snapshot_sha256"]


async def test_too_much_work_is_refused_and_heavy_work_asks_for_confirmation(
    app: FastAPI, client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _two_choices()
    work = 512 * 384 * (8 + 20)
    settings = app.state.services.settings
    monkeypatch.setattr(settings, "video_confirmation_work_units", work)
    asked = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert asked["outcome"] == "compatible" and asked["confirmation_required"] is True
    assert asked["estimate"][0]["value"] == work
    monkeypatch.setattr(settings, "max_media_plan_work_units", work - 1)
    refused = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert [item["code"] for item in refused["refusals"]] == ["experiment-too-large"]


async def test_a_shape_a_workflow_cannot_be_shown_to_make_is_refused(
    client: AsyncClient,
) -> None:
    first, second = _two_choices()
    body = (
        await client.post(
            PREFLIGHT,
            json=_request(first, second, geometry={"mode": "preset", "preset_id": "16:9"}),
        )
    ).json()
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [
        ("arm-geometry-unreachable", 1),
        ("arm-geometry-unreachable", 2),
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"arms": "one"},
        {"seed_policy": {"kind": "fixed_numeric"}},
        {"seed_policy": {"kind": "random_per_trial", "seed": 3}},
        {"seed_policy": {"kind": "same_recorded_number", "seed": -1}},
        {"operation": "image_to_image"},
        {"extra": True},
        {"labels": "duplicate"},
    ],
)
async def test_a_malformed_comparison_is_refused_before_it_is_resolved(
    client: AsyncClient, change: dict[str, Any]
) -> None:
    first, second = _two_choices()
    body = _request(first, second)
    if change == {"arms": "one"}:
        body["arms"] = [first]
    elif change == {"labels": "duplicate"}:
        body["arms"] = [first, {**copy.deepcopy(second), "label": first["label"].upper()}]
    else:
        body.update(change)
    before = _counts()
    response = await client.post(PREFLIGHT, json=body)
    assert response.status_code == 422, response.text
    assert _counts() == before


def _family_choice(label: str, family: str) -> dict[str, Any]:
    """A choice whose workflow declares the installed model it runs, of one named family."""

    with SessionLocal() as session:
        install = ModelInstall(
            name=f"{label} base",
            role="image",
            engine="mock",
            local_path=f"C:/managed/{label.casefold()}",
            manifest_json={"family": family},
            active=True,
        )
        session.add(install)
        session.commit()
        install_id = install.id
    profile_id = _profile(f"{label} model", model_install_id=install_id)
    revision_id = _revision(
        f"{label} workflow", dependencies_json={"model_install_ids": [install_id]}
    )
    return _arm(label, profile_id, revision_id)


@pytest.mark.parametrize(
    ("families", "accepted"),
    [(("Krea-2", "krea2"), True), (("Krea-2", "z-image-turbo"), False)],
)
async def test_one_fixed_seed_is_accepted_only_for_one_proven_family(
    client: AsyncClient, families: tuple[str, str], accepted: bool
) -> None:
    first = _family_choice("Left", families[0])
    second = _family_choice("Right", families[1])
    body = (
        await client.post(
            PREFLIGHT,
            json=_request(first, second, seed_policy={"kind": "fixed_numeric", "seed": 9}),
        )
    ).json()
    assert [arm["model_family"] for arm in body["arms"]] == [
        families[0].casefold(),
        families[1].casefold(),
    ]
    if accepted:
        assert body["outcome"] == "compatible" and body["seed_equivalence"] == "same_family"
    else:
        assert [item["code"] for item in body["refusals"]] == ["seed-family-unproven"]
        assert body["seed_equivalence"] == "none"


async def test_a_model_on_an_engine_that_is_not_configured_is_refused(
    client: AsyncClient,
) -> None:
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Other engine", engine="comfyui"), _revision("Other"))
    body = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert [(item["code"], item["arm_ordinal"]) for item in body["refusals"]] == [
        ("arm-engine-unavailable", 2)
    ]


def _bound_revision(bound_profile_id: str) -> str:
    """A picture workflow whose ready dependencies bind it to one model."""

    with SessionLocal() as session:
        family = WorkflowFamily(name="Bound pictures")
        definition = WorkflowDefinition(
            family=family, variant_key="create", name="Bound pictures", operation="text_to_image"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            dependency_contract_sha256="d" * 64,
            trusted=True,
        )
        session.add_all([family, definition, revision])
        session.flush()
        definition.current_revision_id = revision.id
        activation = WorkflowActivation(
            workflow_revision_id=revision.id,
            resolver_version="resolver-v1",
            dependency_contract_sha256=revision.dependency_contract_sha256,
            binding_sha256="e" * 64,
            state="ready",
            is_active=True,
            details_json={"launch_sha256": "f" * 64},
        )
        slot = WorkflowDependencySlot(
            workflow_revision_id=revision.id,
            name="primary",
            resource_kind="model_profile",
            required=True,
            satisfaction="all_of",
            requirements_json=[{"key": "default", "constraints": {}}],
            contract_sha256="1" * 64,
            ordinal=0,
        )
        session.add_all([activation, slot])
        session.flush()
        session.add(
            WorkflowDependencyBinding(
                workflow_revision_id=revision.id,
                workflow_activation_id=activation.id,
                workflow_dependency_slot_id=slot.id,
                requirement_key="default",
                model_profile_id=bound_profile_id,
                resource_identity_sha256="2" * 64,
            )
        )
        session.commit()
        return revision.id


async def test_a_workflow_bound_to_another_model_names_the_model_it_runs(
    client: AsyncClient,
) -> None:
    bound = _profile("Bound model")
    revision_id = _bound_revision(bound)
    first = _arm("First", _profile("Ready model"), _revision("Ready workflow"))
    second = _arm("Second", _profile("Wrong model"), revision_id)
    body = (await client.post(PREFLIGHT, json=_request(first, second))).json()
    assert body["refusals"] == [
        {
            "code": "arm-model-mismatch",
            "arm_ordinal": 2,
            "setting": None,
            "alternative": {"seed_policy": None, "profile_id": bound},
            "message": body["refusals"][0]["message"],
        }
    ]
    right = _arm("Second", bound, revision_id)
    accepted = (await client.post(PREFLIGHT, json=_request(first, right))).json()
    assert accepted["outcome"] == "compatible", accepted["refusals"]
    assert accepted["arms"][1]["workflow_activation_id"] is not None


async def test_a_choice_carries_the_trigger_words_its_turn_would_append(
    app: FastAPI, client: AsyncClient
) -> None:
    with SessionLocal() as session:
        install = ModelInstall(
            name="Worded base",
            role="image",
            engine="mock",
            local_path="C:/managed/worded",
            manifest_json={"trigger_words": ["harborlight"]},
            active=True,
        )
        session.add(install)
        session.commit()
        install_id = install.id
    worded = _arm(
        "Worded", _profile("Worded model", model_install_id=install_id), _revision("Worded")
    )
    plain = _arm("Plain", _profile("Plain model"), _revision("Plain"))
    body = (await client.post(PREFLIGHT, json=_request(worded, plain))).json()
    assert body["outcome"] == "compatible", body["refusals"]
    assert [arm["trigger_words_applied"] for arm in body["arms"]] == [["harborlight"], []]
    chat = await client.post("/api/chats", json={"title": "Worded turn"})
    async with app.state.services.scheduler.lease("primary"):
        turn = await client.post(
            f"/api/chats/{chat.json()['id']}/turns",
            json={
                "text": PROMPT,
                "mode": "image",
                "profile_id": worded["profile_id"],
                "workflow_revision_id": worded["workflow_revision_id"],
                "preset_id": None,
                "settings": {"negative_prompt": NEGATIVE, "width": 512, "height": 384, "seed": 41},
            },
        )
        assert turn.status_code == 202, turn.text
        recorded = turn.json()["run"]["provenance_json"]["auxiliary_assets"]
        assert recorded["trigger_words_applied"] == ["harborlight"]
