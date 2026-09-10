"""A workflow that never reads the description is refused before it runs."""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient
from sqlalchemy.orm import Session

from local_lm.config import Settings
from local_lm.db import SessionLocal
from local_lm.models import (
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowRevision,
)
from local_lm.prompt_binding import binds_prompt, ignores_the_description

pytestmark = pytest.mark.asyncio


def _graph(prompt_value: object) -> dict[str, Any]:
    return {
        "encode": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt_value}},
        "sampler": {"class_type": "KSampler", "inputs": {"positive": ["encode", 0]}},
    }


def test_a_graph_naming_the_placeholder_binds_the_description() -> None:
    assert binds_prompt(_graph("${prompt}")) is True


def test_a_graph_with_a_baked_in_prompt_does_not() -> None:
    assert binds_prompt(_graph("a post-rain dusk metropolis, anamorphic lens")) is False


def test_an_embedded_placeholder_is_not_a_binding() -> None:
    """The compiler substitutes a whole value or nothing.

    `ComfyUIAdapter._compile` takes `value[2:-1]` off a string it has required
    to begin `${` and end `}`, so `masterpiece, ${prompt}` looks for a
    parameter named `masterpiece, ${prompt` and, finding none, leaves the text
    exactly as written. Reading an embedded placeholder as a binding would let
    this check pass for a graph the description still cannot reach.
    """
    assert binds_prompt(_graph("masterpiece, ${prompt}")) is False
    assert binds_prompt(_graph("${prompt} in ink")) is False


def test_the_placeholder_is_found_wherever_it_sits() -> None:
    assert binds_prompt({"a": {"b": [{"c": ["${prompt}"]}]}}) is True
    assert binds_prompt({"a": [1, None, True, {"b": 2}]}) is False


def test_an_empty_graph_binds_nothing() -> None:
    assert binds_prompt({}) is False


@pytest.mark.parametrize("operation", ["text_to_image", "text_to_video"])
def test_a_description_only_operation_that_reads_no_description_is_ignored_input(
    operation: str,
) -> None:
    assert ignores_the_description("comfyui", operation, _graph("baked in")) is True
    assert ignores_the_description("comfyui", operation, _graph("${prompt}")) is False


@pytest.mark.parametrize("operation", ["image_to_image", "image_to_video"])
def test_an_operation_with_a_source_picture_is_left_alone(operation: str) -> None:
    """An upscale or a restoration works from the picture and needs no words."""
    assert ignores_the_description("comfyui", operation, _graph("baked in")) is False


def test_another_engine_binds_its_inputs_its_own_way() -> None:
    """The mock engine stands in for a backend rather than compiling a graph."""
    assert ignores_the_description("mock", "text_to_image", _graph("baked in")) is False


def _comfy_image_profile(session: Session) -> None:
    """Point the default image profile at an installed ComfyUI model.

    Selection refuses an unavailable profile before it ever looks at the
    revision, so without this the checks below would pass for the wrong reason.
    """
    install = ModelInstall(
        name="Installed image model",
        role="image",
        engine="comfyui",
        local_path="C:/managed/installed-image-model",
        manifest_json={},
        active=True,
    )
    session.add(install)
    session.flush()
    profile = session.query(ModelProfile).filter_by(role="image", is_default=True).one()
    profile.engine = "comfyui"
    profile.model_install_id = install.id


def _comfy_family(
    session: Session, *, name: str, prompt_value: object, operation: str = "text_to_image"
) -> tuple[str, str]:
    family = WorkflowFamily(name=name)
    definition = WorkflowDefinition(
        family=family,
        variant_key="create",
        name=f"{name} create",
        operation=operation,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        api_graph_json=_graph(prompt_value),
        input_schema_json={"type": "object", "properties": {}},
        dependencies_json={},
        trusted=True,
    )
    preference = WorkflowPreference(family=family, selector_capability="image")
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return family.id, revision.id


async def test_the_catalog_reports_a_workflow_that_ignores_the_description_unavailable(
    client: AsyncClient, settings: Settings
) -> None:
    """The tool says so before anyone picks it.

    Readiness is what decides whether a workflow is offered at all, so a
    workflow the description cannot reach belongs in the same unavailable list
    as one built for another engine.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        ignored, _ = _comfy_family(
            session, name="Baked-in cinematic", prompt_value="a post-rain dusk metropolis"
        )
        reads, _ = _comfy_family(session, name="Reads the description", prompt_value="${prompt}")
        session.commit()

    response = await client.get("/api/workflow-families?selector_capability=image")

    assert response.status_code == 200, response.text
    cards = {item["id"]: item for item in response.json()}
    refused = cards[ignored]["variants"][0]
    assert refused["readiness"] == "unavailable"
    assert refused["readiness_reason"] == "revision_ignores_the_description"
    # The control. Identical in every respect but the one value the check
    # reads, so a refusal that fired on the engine, the operation or the empty
    # schema would fail here rather than pass quietly.
    accepted = cards[reads]["variants"][0]
    assert accepted["readiness_reason"] != "revision_ignores_the_description"


async def test_a_turn_cannot_select_a_workflow_that_ignores_the_description(
    client: AsyncClient, settings: Settings
) -> None:
    """The refusal is on the turn, not only on a helper.

    A helper that answers correctly while nothing calls it leaves the defect
    exactly where it was: the turn accepted, the description shown in the
    conversation beside a picture made without it. This drives the real
    endpoint so that removing the call from selection fails here. Selection
    refuses before any run is created, so nothing is dispatched.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        family_id, _ = _comfy_family(
            session, name="Baked-in cinematic turn", prompt_value="a post-rain dusk metropolis"
        )
        session.commit()

    chat = (await client.post("/api/chats", json={"title": "Ignored description"})).json()
    await client.put(
        f"/api/chats/{chat['id']}/workflow-selections/image",
        json={"mode": "family", "workflow_family_id": family_id},
    )

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "a slow aerial drift over an autumn forest canopy at sunrise",
            "mode": "image",
        },
    )

    assert response.status_code >= 400, response.text
    assert "ignores_the_description" in response.text


async def test_a_project_pin_on_such_a_workflow_is_refused_too(
    client: AsyncClient, settings: Settings
) -> None:
    """The pin branch reaches execution by its own road.

    Its own docstring records that it once made only the first of the checks
    the other branches make, so a pin for another engine was selected and then
    failed during execution with an error that never mentioned the pin. Adding
    a check to selection and readiness and not to this one would rebuild
    exactly that hole, so this drives a pinned turn rather than reading the
    message table.
    """
    settings.media_engine = "comfyui"
    with SessionLocal() as session:
        _comfy_image_profile(session)
        _, revision_id = _comfy_family(
            session, name="Baked-in cinematic pin", prompt_value="a post-rain dusk metropolis"
        )
        session.commit()

    project = (await client.post("/api/projects", json={"name": "Pinned"})).json()
    pinned = await client.put(
        f"/api/projects/{project['id']}/workflow-selections/image",
        json={"mode": "revision", "workflow_revision_id": revision_id},
    )
    assert pinned.status_code == 200, pinned.text
    chat = (
        await client.post(
            "/api/chats", json={"title": "Pinned description", "project_id": project["id"]}
        )
    ).json()

    response = await client.post(
        f"/api/chats/{chat['id']}/turns",
        json={
            "text": "a slow aerial drift over an autumn forest canopy at sunrise",
            "mode": "image",
        },
    )

    assert response.status_code >= 400, response.text
    assert "never reads what you type" in response.text
