"""Admit each recipe use case through HTTP and explain incompatible choices."""

from io import BytesIO
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from PIL import Image
from sqlalchemy import select

from local_lm.db import SessionLocal
from local_lm.models import (
    ChatWorkflowUseCaseSelection,
    ModelProfile,
    Run,
    WorkflowActivation,
    WorkflowDefinition,
    WorkflowDependencyBinding,
    WorkflowDependencySlot,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
    WorkflowUseCasePreset,
)

CASES = [
    ("chat", "text", "text", False),
    ("image_generation", "text_to_image", "image", False),
    ("image_edit", "image_to_image", "image", True),
    ("image_inpaint", "image_to_image", "image", True),
    ("image_outpaint", "image_to_image", "image", True),
    ("image_upscale", "image_to_image", "image", True),
    ("video_generation", "text_to_video", "video", False),
    ("video_animate", "image_to_video", "video", True),
]


def _workflow(use_case: str, operation: str, role: str) -> tuple[str, str]:
    properties: dict[str, Any] = {
        "recipe_quality": {"type": "number", "default": 0.25, "minimum": 0, "maximum": 1}
    }
    graph: dict[str, Any] = {
        "source": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "output": {"class_type": "TestOutput", "inputs": {"quality": "${recipe_quality}"}},
    }
    if use_case == "image_inpaint":
        properties["mask"] = {"type": "object", "x-lm-atelier-kind": "mask"}
        graph["output"]["inputs"]["mask"] = "${mask}"
    if use_case == "image_outpaint":
        properties["outpaint_margins"] = {
            "type": "object",
            "x-lm-atelier-kind": "outpaint",
            "default": {"top": 0, "right": 0, "bottom": 0, "left": 0},
        }
        graph["pad"] = {
            "class_type": "ImagePadForOutpaint",
            "inputs": {"image": ["source", 0], "top": 0, "right": 0, "bottom": 0, "left": 0},
        }
    if use_case == "image_upscale":
        properties["upscale_factor"] = {
            "type": "number",
            "default": 2,
            "x-lm-atelier-kind": "upscale",
        }
        graph["scale"] = {
            "class_type": "ImageScaleBy",
            "inputs": {"image": ["source", 0], "scale_by": "${upscale_factor}"},
        }
    with SessionLocal() as session:
        family = WorkflowFamily(name="Example request workflow", use_case="Example request")
        definition = WorkflowDefinition(
            family=family, name="Example request", operation=operation, variant_key="request"
        )
        revision = WorkflowRevision(
            definition=definition,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json={"type": "object", "properties": properties},
            dependencies_json={},
            dependency_contract_sha256="a" * 64 if role == "chat" else None,
        )
        session.add_all(
            [
                family,
                definition,
                revision,
                WorkflowPreference(family=family, selector_capability=role),
            ]
        )
        session.flush()
        definition.current_revision_id = revision.id
        if role == "chat":
            profile = ModelProfile(name="Example chat", role="chat", engine="mock")
            activation = WorkflowActivation(
                workflow_revision_id=revision.id,
                resolver_version="resolver-v1",
                dependency_contract_sha256="a" * 64,
                binding_sha256="b" * 64,
                state="ready",
                is_active=True,
                details_json={"launch_sha256": "c" * 64},
            )
            slot = WorkflowDependencySlot(
                workflow_revision_id=revision.id,
                name="primary",
                resource_kind="model_profile",
                required=True,
                satisfaction="all_of",
                requirements_json=[{"key": "default"}],
                contract_sha256="d" * 64,
                ordinal=0,
            )
            session.add_all([profile, activation, slot])
            session.flush()
            session.add(
                WorkflowDependencyBinding(
                    workflow_revision_id=revision.id,
                    workflow_activation_id=activation.id,
                    workflow_dependency_slot_id=slot.id,
                    requirement_key="default",
                    model_profile_id=profile.id,
                    resource_identity_sha256="e" * 64,
                )
            )
        session.commit()
        return family.id, revision.id


async def _picture(client: AsyncClient) -> str:
    content = BytesIO()
    with Image.new("RGB", (64, 64), (90, 120, 150)) as picture:
        picture.save(content, format="PNG")
    response = await client.post(
        "/api/artifacts", files={"file": ("example.png", content.getvalue(), "image/png")}
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


@pytest.mark.parametrize("use_case,operation,mode,needs_source", CASES)
async def test_every_recipe_use_case_reaches_its_selected_revision(
    app: FastAPI, client: AsyncClient, use_case: str, operation: str, mode: str, needs_source: bool
) -> None:
    role = "chat" if mode == "text" else mode
    family_id, revision_id = _workflow(use_case, operation, role)
    chat = (await client.post("/api/chats", json={"title": "Recipe request"})).json()
    created = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Example request recipe",
            "use_case": use_case,
            "settings_json": {"recipe_quality": 0.75},
            "is_default": True,
        },
    )
    assert created.status_code == 201, created.text
    selected = await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/{role}",
        json={
            "mode": "family",
            "workflow_family_id": family_id,
        },
    )
    assert selected.status_code == 200, selected.text
    payload: dict[str, Any] = {"text": "A blue square", "mode": mode}
    if needs_source:
        payload["input_artifact_ids"] = [await _picture(client)]
    settings: dict[str, Any] = {}
    if use_case == "image_inpaint":
        settings["mask"] = {"artifact_id": await _picture(client)}
    if use_case == "image_outpaint":
        settings["outpaint_margins"] = {"right": 0.25}
    if use_case == "image_upscale":
        settings["upscale_factor"] = 2
    payload["settings"] = settings
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
        assert response.status_code == 202, response.text
        with SessionLocal() as session:
            run = session.get(Run, response.json()["run"]["id"])
            assert run is not None and run.operation == operation
            assert run.workflow_revision_id == revision_id
            assert run.settings_json["recipe_quality"] == 0.75
            receipt = run.provenance_json["workflow_use_case_preset"]
            assert receipt["use_case"] == use_case
            assert receipt["preset_id"] == created.json()["id"]
            assert receipt["workflow_revision_id"] == revision_id
            assert receipt["settings_json"] == {"recipe_quality": 0.75}


@pytest.mark.parametrize(
    "problem,code",
    [
        ("unsupported", "workflow-use-case-preset-setting-unsupported"),
        ("graphless", "workflow-use-case-preset-revision-required"),
        ("disabled", "workflow-use-case-preset-disabled"),
    ],
)
async def test_recipe_refusals_offer_a_specific_correction_without_creating_work(
    client: AsyncClient, problem: str, code: str
) -> None:
    mode = "text" if problem == "graphless" else "image"
    use_case = "chat" if problem == "graphless" else "image_generation"
    chat = (await client.post("/api/chats", json={"title": "Recipe refusal"})).json()
    with SessionLocal() as session:
        preset = WorkflowUseCasePreset(
            name="Example refused recipe",
            use_case=use_case,
            settings_json={"unsupported_control": 1} if problem == "unsupported" else {},
            enabled=problem != "disabled",
        )
        session.add(preset)
        session.flush()
        session.add(
            ChatWorkflowUseCaseSelection(
                chat_id=chat["id"],
                use_case=use_case,
                preset_id=preset.id,
            )
        )
        session.commit()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "A blue square",
            "mode": mode,
        },
    )
    assert response.status_code == 422, response.text
    assert response.json()["code"] == code, response.text
    assert "recipe" in response.json()["detail"].lower()
    assert "Automatic" in response.json()["detail"]
    with SessionLocal() as session:
        assert session.scalar(select(Run.id).where(Run.chat_id == chat["id"])) is None


async def test_unrelated_workflow_refusal_retains_its_existing_error(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Missing workflow"})).json()
    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "A blue square",
            "mode": "image",
            "workflow_revision_id": "missing-revision",
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "turn-invalid"


@pytest.mark.parametrize("recipe", [False, True])
@pytest.mark.parametrize("explicit_profile", [False, True])
async def test_recipe_preserves_unconfigured_profile_engine_refusal(
    client: AsyncClient, recipe: bool, explicit_profile: bool
) -> None:
    chat = (await client.post("/api/chats", json={"title": "Engine selection"})).json()
    with SessionLocal() as session:
        profile = session.scalar(
            select(ModelProfile).where(
                ModelProfile.role == "image", ModelProfile.is_default.is_(True)
            )
        )
        assert profile is not None
        profile.engine = "comfyui"
        profile_id = profile.id
        if recipe:
            preset = WorkflowUseCasePreset(
                name="Example image recipe", use_case="image_generation", settings_json={}
            )
            session.add(preset)
            session.flush()
            session.add(
                ChatWorkflowUseCaseSelection(
                    chat_id=chat["id"], use_case="image_generation", preset_id=preset.id
                )
            )
        session.commit()
    payload = {"text": "A blue square", "mode": "image"}
    if explicit_profile:
        payload["profile_id"] = profile_id
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=payload)
    assert response.status_code == 409, response.text
    assert response.json()["code"] == "engine-not-configured"
    assert "comfyui" in response.json()["detail"]
    with SessionLocal() as session:
        assert session.scalar(select(Run.id).where(Run.chat_id == chat["id"])) is None


@pytest.mark.parametrize("choice", ["family", "automatic", "automatic_unavailable"])
async def test_recipe_compatibility_preserves_explicit_family_authority(
    app: FastAPI, client: AsyncClient, choice: str
) -> None:
    selected_family, selected_revision = _workflow("image_generation", "text_to_image", "image")
    _, alternate_revision = _workflow("image_generation", "text_to_image", "image")
    with SessionLocal() as session:
        selected = session.get(WorkflowRevision, selected_revision)
        alternate = session.get(WorkflowRevision, alternate_revision)
        assert selected is not None and alternate is not None
        selected.input_schema_json = {
            "type": "object",
            "properties": {"recipe_quality": {"type": "number", "default": 0.25, "readOnly": True}},
        }
        if choice == "automatic_unavailable":
            alternate.input_schema_json = {}
        session.commit()
    chat = (await client.post("/api/chats", json={"title": "Explicit recipe workflow"})).json()
    created = await client.post(
        "/api/workflow-use-case-presets",
        json={
            "name": "Example quality recipe",
            "use_case": "image_generation",
            "settings_json": {"recipe_quality": 0.75},
            "is_default": True,
        },
    )
    assert created.status_code == 201, created.text
    selection = (
        {"mode": "family", "workflow_family_id": selected_family}
        if choice == "family"
        else {"mode": "automatic"}
    )
    chosen = await client.put(f"/api/chats/{chat['id']}/workflow-selections/image", json=selection)
    assert chosen.status_code == 200, chosen.text
    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat['id']}/turns",
            json={
                "text": "A blue square",
                "mode": "image",
            },
        )
        with SessionLocal() as session:
            runs = list(session.scalars(select(Run).where(Run.chat_id == chat["id"])))
            if choice == "automatic":
                assert response.status_code == 202, response.text
                assert len(runs) == 1 and runs[0].workflow_revision_id == alternate_revision
                assert runs[0].settings_json["recipe_quality"] == 0.75
            else:
                assert response.status_code == 422, response.text
                expected = (
                    "workflow-use-case-preset-setting-unavailable"
                    if choice == "family"
                    else "workflow-use-case-no-compatible-workflow"
                )
                assert response.json()["code"] == expected, response.text
                assert "Automatic" in response.json()["detail"]
                assert selected_family not in response.json()["detail"]
                assert not runs
