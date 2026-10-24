"""An edit that names no margins is never handed a workflow that paints past the edge.

Run without margins, an outpainter pads the picture by whatever its graph was
saved with, so an ordinary edit comes back on a larger canvas, or, padded by
nothing, unchanged. Only a turn that names margins, or a canvas to extend onto,
asks for one. A workflow someone names keeps running whatever the turn asks.

The other inputs a workflow declares follow the same rule. An inpainter run
without a selection, or an upscaler run for an ordinary edit, does what the
turn never asked, so only a turn that brings a selection for the workflow, or
asks for a larger picture, is given one. A turn that brings a selection or
margins looks first for a workflow that takes them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_automatic_image_edit_selection import (
    GLOBAL,
    LOCALIZED,
    _chat,
    _graphs,
    _node,
    _orchestrator,
)
from test_automatic_image_edit_selection import session as session
from test_outpaint_dispatch import DECLARED_MARGINS, _outpaint_graph, _outpainter
from test_studio_region_edit_turns import _mask, _png, _source

from local_lm.db import SessionLocal
from local_lm.domain import Operation
from local_lm.image_edit_kind import image_edit_kind
from local_lm.models import (
    Chat,
    ModelInstall,
    ModelProfile,
    WorkflowDefinition,
    WorkflowFamily,
    WorkflowPreference,
    WorkflowProfileCompatibility,
    WorkflowRevision,
)
from local_lm.outpaint_workflows import OUTPAINT_SETTING_KEY
from local_lm.schemas import SourceFitRequest, TurnRequest
from local_lm.workflow_selection import WorkflowFamilySelectionError

OUTPAINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {OUTPAINT_SETTING_KEY: DECLARED_MARGINS},
}
PLAIN_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}
MARGINS = {OUTPAINT_SETTING_KEY: {"right": 0.25}}
INPAINT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"mask": {"type": "object", "x-lm-atelier-kind": "mask"}},
}
UPSCALE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "upscale_factor": {
            "type": "number",
            "default": 2,
            "minimum": 1,
            "maximum": 8,
            "x-lm-atelier-kind": "upscale",
        }
    },
}


def _fill_graphs() -> tuple[dict[str, Any], dict[str, Any]]:
    """A Fill-style outpainter: the padded source conditions the sampler with the words."""

    ui: dict[str, list[Any]] = {
        "nodes": [
            _node(1, "LoadImage", "IMAGE", "MASK"),
            _node(2, "ImagePadForOutpaint", "IMAGE", "MASK"),
            _node(4, "CLIPTextEncode", "CONDITIONING"),
            _node(5, "InpaintModelConditioning", "CONDITIONING", "CONDITIONING", "LATENT"),
            _node(3, "Sampler", "LATENT"),
            _node(9, "SaveImage"),
        ],
        "links": [
            [10, 1, 0, 2, 0, "IMAGE"],
            [11, 2, 0, 5, 2, "IMAGE"],
            [12, 2, 1, 5, 3, "MASK"],
            [13, 4, 0, 5, 0, "CONDITIONING"],
            [14, 4, 0, 5, 1, "CONDITIONING"],
            [15, 5, 0, 3, 0, "CONDITIONING"],
            [16, 5, 1, 3, 1, "CONDITIONING"],
            [17, 5, 2, 3, 3, "LATENT"],
            [18, 3, 0, 9, 0, "LATENT"],
        ],
    }
    api: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "${input_image}"}},
        "2": {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": ["1", 0],
                "left": 400,
                "top": 0,
                "right": 400,
                "bottom": 400,
                "feathering": 24,
            },
        },
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "${prompt}"}},
        "5": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["4", 0],
                "pixels": ["2", 0],
                "mask": ["2", 1],
            },
        },
        "3": {
            "class_type": "Sampler",
            "inputs": {"positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2]},
        },
        "9": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
    }
    return ui, api


def _family(
    session: Session,
    name: str,
    graphs: tuple[dict[str, Any], dict[str, Any]],
    schema: dict[str, Any],
    *,
    is_default: bool,
    use_case: str,
) -> WorkflowRevision:
    family = WorkflowFamily(name=name, use_case=use_case)
    definition = WorkflowDefinition(
        family=family,
        variant_key="edit",
        name=f"{name} edit",
        operation=Operation.IMAGE_TO_IMAGE.value,
    )
    revision = WorkflowRevision(
        definition=definition,
        version=1,
        engine="comfyui",
        ui_graph_json=graphs[0],
        api_graph_json=graphs[1],
        input_schema_json=schema,
        trusted=True,
    )
    preference = WorkflowPreference(
        family=family, selector_capability="image", is_default=is_default
    )
    session.add_all([family, definition, revision, preference])
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _model(session: Session) -> tuple[ModelProfile, WorkflowFamily]:
    """One installed image model, mapped to its generated compatibility family."""

    install = ModelInstall(
        name="Neutral image model",
        role="image",
        engine="comfyui",
        local_path="neutral-model",
        active=True,
    )
    session.add(install)
    session.flush()
    profile = ModelProfile(
        name="Neutral image model", role="image", engine="comfyui", model_install_id=install.id
    )
    family = WorkflowFamily(name="Neutral image model", use_case="")
    session.add_all([profile, family])
    session.flush()
    session.add_all(
        [
            WorkflowPreference(family=family, selector_capability="image", is_default=False),
            WorkflowProfileCompatibility(
                workflow_family_id=family.id,
                model_profile_id=profile.id,
                source_fingerprint_sha256="a" * 64,
            ),
        ]
    )
    session.flush()
    return profile, family


def _model_workflow(
    session: Session,
    profile: ModelProfile,
    name: str,
    graphs: tuple[dict[str, Any], dict[str, Any]],
    schema: dict[str, Any],
    created: datetime,
) -> WorkflowRevision:
    """An edit workflow that names the model as its own, outside any family."""

    definition = WorkflowDefinition(
        name=name, operation=Operation.IMAGE_TO_IMAGE.value, created_at=created
    )
    session.add(definition)
    session.flush()
    revision = WorkflowRevision(
        workflow_id=definition.id,
        version=1,
        engine="comfyui",
        ui_graph_json=graphs[0],
        api_graph_json=graphs[1],
        input_schema_json=schema,
        dependencies_json={"model_install_ids": [profile.model_install_id]},
        trusted=True,
        created_at=created,
    )
    session.add(revision)
    session.flush()
    definition.current_revision_id = revision.id
    session.flush()
    return revision


def _execution(
    session: Session, chat: Chat, prompt: str, **request: Any
) -> tuple[ModelProfile | None, dict[str, Any], WorkflowRevision | None]:
    return _orchestrator()._execution_for_turn(
        session,
        chat,
        Operation.IMAGE_TO_IMAGE,
        prompt,
        TurnRequest(text=prompt, mode="image", **request),
    )


def _without_profile_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "local_lm.orchestrator.ConversationOrchestrator._profile_for_operation",
        Mock(side_effect=AssertionError("the profile fallback is not where this is decided")),
    )


@pytest.mark.parametrize("prompt", [LOCALIZED, GLOBAL], ids=["localized", "global"])
def test_auto_passes_over_an_outpainter_for_an_edit_that_names_no_margins(
    session: Session, prompt: str
) -> None:
    ui, api = _fill_graphs()
    # The outpainter is an instruction edit by structure, so nothing else turns it away.
    assert image_edit_kind("image_to_image", ui, api, OUTPAINT_SCHEMA) == "instruction"
    outpainter = _family(session, "Fill", (ui, api), OUTPAINT_SCHEMA, is_default=True, use_case="")
    plain = _family(
        session,
        "Plain",
        _graphs(instruction=True),
        PLAIN_SCHEMA,
        is_default=False,
        use_case="studio photos",
    )
    chat = _chat(session, "automatic")

    assert _execution(session, chat, prompt)[2] == plain
    assert _execution(session, chat, prompt, settings=MARGINS)[2] == outpainter
    extend = SourceFitRequest(mode="extend", width=96, height=48)
    assert _execution(session, chat, prompt, source_fit=extend)[2] == outpainter
    # A crop keeps the source's own canvas, so it is an ordinary edit here.
    crop = SourceFitRequest(mode="crop", width=96, height=48)
    assert _execution(session, chat, prompt, source_fit=crop)[2] == plain


def test_auto_finds_nothing_for_a_localized_edit_when_only_an_outpainter_is_installed(
    session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _family(session, "Fill", _fill_graphs(), OUTPAINT_SCHEMA, is_default=True, use_case="")
    _without_profile_fallback(monkeypatch)

    with pytest.raises(WorkflowFamilySelectionError) as raised:
        _execution(session, _chat(session, "automatic"), LOCALIZED)

    assert raised.value.reason == "no_ready_workflow"


@pytest.mark.parametrize("mode", ["automatic", "explicit"])
def test_a_compatibility_family_resolves_to_its_models_edit_workflow_rather_than_its_outpainter(
    session: Session, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    profile, family = _model(session)
    plain = _model_workflow(
        session,
        profile,
        "Plain",
        _graphs(instruction=True),
        PLAIN_SCHEMA,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    outpainter = _model_workflow(
        session, profile, "Fill", _fill_graphs(), OUTPAINT_SCHEMA, datetime(2026, 1, 2, tzinfo=UTC)
    )
    chat = (
        _chat(session, "automatic") if mode == "automatic" else _chat(session, "family", family.id)
    )
    _without_profile_fallback(monkeypatch)

    _profile, selection, revision = _execution(session, chat, GLOBAL)
    assert revision == plain
    assert selection["workflow_compatibility"] is True
    assert selection["workflow_family_id"] == family.id
    assert _execution(session, chat, GLOBAL, settings=MARGINS)[2] == outpainter


@pytest.mark.parametrize("route", ["legacy-profile", "compatibility-family"])
def test_a_chosen_model_with_only_an_outpainter_gets_no_workflow_for_an_edit_that_names_no_margins(
    session: Session, route: str
) -> None:
    profile, family = _model(session)
    outpainter = _model_workflow(
        session, profile, "Fill", _fill_graphs(), OUTPAINT_SCHEMA, datetime(2026, 1, 2, tzinfo=UTC)
    )
    if route == "legacy-profile":
        chat = Chat(title="Edits", active_image_profile_id=profile.id)
        session.add(chat)
        session.flush()
    else:
        chat = _chat(session, "family", family.id)

    chosen, selection, revision = _execution(session, chat, GLOBAL)
    assert chosen == profile
    assert selection["mode"] == "explicit"
    assert revision is None
    assert _execution(session, chat, GLOBAL, settings=MARGINS)[2] == outpainter


def test_a_named_outpainter_still_runs_an_edit_that_names_no_margins(session: Session) -> None:
    """A choice someone made is kept; only the app's own choice passes the outpainter by."""

    outpainter = _family(
        session, "Fill", _fill_graphs(), OUTPAINT_SCHEMA, is_default=True, use_case=""
    )
    _family(
        session,
        "Plain",
        _graphs(instruction=True),
        PLAIN_SCHEMA,
        is_default=False,
        use_case="studio photos",
    )

    named = _execution(
        session, _chat(session, "automatic"), GLOBAL, workflow_revision_id=outpainter.id
    )
    assert named[2] == outpainter
    family_id = outpainter.definition.family_id
    assert _execution(session, _chat(session, "family", family_id), GLOBAL)[2] == outpainter
    # With no choice of its own, a chat takes the default family, which is this one.
    unset = Chat(title="Edits")
    session.add(unset)
    session.flush()
    assert _execution(session, unset, GLOBAL)[2] == outpainter


async def _import_model(client: AsyncClient, tmp_path: Path) -> str:
    model_dir = tmp_path / "neutral-image-model"
    model_dir.mkdir()
    (model_dir / "neutral-image-model.safetensors").write_bytes(b"safe")
    imported = await client.post(
        "/api/models/import",
        json={
            "name": "neutral-image-model",
            "role": "image",
            "engine": "mock",
            "local_path": str(model_dir),
        },
    )
    assert imported.status_code in {200, 201}, imported.text
    return str(imported.json()["id"])


def _mock_workflow(
    name: str, install_id: str, graph: dict[str, Any], schema: dict[str, Any], created: datetime
) -> str:
    with SessionLocal() as db:
        definition = WorkflowDefinition(
            name=name, operation=Operation.IMAGE_TO_IMAGE.value, created_at=created
        )
        db.add(definition)
        db.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            trusted=True,
            api_graph_json=graph,
            input_schema_json=schema,
            dependencies_json={"model_install_ids": [install_id]},
            created_at=created,
        )
        db.add(revision)
        db.flush()
        definition.current_revision_id = revision.id
        db.commit()
        return revision.id


def _clear_seeded_edit_workflows() -> None:
    with SessionLocal() as db:
        for definition in db.scalars(
            select(WorkflowDefinition).where(
                WorkflowDefinition.operation == Operation.IMAGE_TO_IMAGE.value
            )
        ):
            definition.current_revision_id = None
        db.commit()


async def _upload(client: AsyncClient, name: str, content: bytes) -> str:
    uploaded = await client.post("/api/artifacts", files={"file": (name, content, "image/png")})
    assert uploaded.status_code == 201, uploaded.text
    return str(uploaded.json()["id"])


def _removal(source_id: str, mask_id: str) -> dict[str, Any]:
    """Remove as the studio sends it: the words, the marked part, blended back."""

    return {
        "text": (
            "Remove the blue square. Fill the space it leaves to match what surrounds it, "
            "and leave everything else unchanged."
        ),
        "mode": "image",
        "input_artifact_ids": [source_id],
        "settings": {
            "mask": {"artifact_id": mask_id, "feather_px": 4, "invert": False, "apply": "blend"}
        },
    }


def _extension(source_id: str) -> dict[str, Any]:
    return {
        "text": "Extend the picture to the right.",
        "mode": "image",
        "input_artifact_ids": [source_id],
        "settings": MARGINS,
    }


async def test_a_removal_runs_the_models_edit_workflow_rather_than_its_outpainter(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    """The app's own choice for a removal passes over the model's outpainter."""

    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    editor = _mock_workflow(
        "Neutral editor",
        install_id,
        {"nodes": [{"inputs": {"image": "${input_image}"}}]},
        PLAIN_SCHEMA,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    outpainter = _mock_workflow(
        "Neutral outpainter",
        install_id,
        _outpaint_graph(),
        OUTPAINT_SCHEMA,
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    source_id = await _upload(client, "source.png", _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        chat = (await client.post("/api/chats", json={"title": "Remove"})).json()
        removed = await client.post(
            f"/api/chats/{chat['id']}/turns", json=_removal(source_id, mask_id)
        )
        assert removed.status_code == 202, removed.text
        assert removed.json()["run"]["workflow_revision_id"] == editor
        # Margins named, the same app choice still finds the outpainter.
        other = (await client.post("/api/chats", json={"title": "Extend"})).json()
        extended = await client.post(f"/api/chats/{other['id']}/turns", json=_extension(source_id))
        assert extended.status_code == 202, extended.text
        assert extended.json()["run"]["workflow_revision_id"] == outpainter


async def test_with_only_an_outpainter_installed_a_removal_is_refused_before_anything_runs(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    outpainter = _mock_workflow(
        "Neutral outpainter",
        install_id,
        _outpaint_graph(),
        OUTPAINT_SCHEMA,
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    source_id = await _upload(client, "source.png", _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        chat = (await client.post("/api/chats", json={"title": "Remove"})).json()
        removed = await client.post(
            f"/api/chats/{chat['id']}/turns", json=_removal(source_id, mask_id)
        )
        assert removed.status_code == 422, removed.text
        assert "No ready workflow matches" in removed.json()["detail"]
        other = (await client.post("/api/chats", json={"title": "Extend"})).json()
        extended = await client.post(f"/api/chats/{other['id']}/turns", json=_extension(source_id))
        assert extended.status_code == 202, extended.text
        assert extended.json()["run"]["workflow_revision_id"] == outpainter


async def test_a_selection_is_never_placed_back_through_a_workflow_that_extends_the_picture(
    app: FastAPI, client: AsyncClient
) -> None:
    """Named outright, an outpainter still cannot hold a blended selection."""

    revision_id, source_id, chat_id = await _outpainter(client, _outpaint_graph(), _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        response = await client.post(
            f"/api/chats/{chat_id}/turns",
            json={**_removal(source_id, mask_id), "workflow_revision_id": revision_id},
        )

    assert response.status_code == 422, response.text
    assert "cannot be placed back through a selection" in response.json()["detail"]


_EDIT_GRAPH: dict[str, Any] = {"nodes": [{"inputs": {"image": "${input_image}"}}]}
_INPAINT_GRAPH: dict[str, Any] = {
    "nodes": [{"inputs": {"image": "${input_image}", "mask": "${mask}"}}]
}
_UPSCALE_GRAPH: dict[str, Any] = {
    "nodes": [{"inputs": {"image": "${input_image}", "scale_by": "${upscale_factor}"}}]
}
_A_SELECTION = {"mask": {"artifact_id": f"sha256:{'0' * 64}", "feather_px": 0, "invert": False}}


def _instruction(source_id: str) -> dict[str, Any]:
    return {
        "text": "Make the sky a deeper blue.",
        "mode": "image",
        "input_artifact_ids": [source_id],
    }


def _brushed(source_id: str, mask_id: str) -> dict[str, Any]:
    """A selection drawn with a brush, handed to the workflow's own mask input."""

    return {
        "text": "Paint the marked part green.",
        "mode": "image",
        "input_artifact_ids": [source_id],
        "settings": {"mask": {"artifact_id": mask_id, "feather_px": 4, "invert": False}},
    }


async def _chosen(client: AsyncClient, body: dict[str, Any]) -> str:
    """The workflow the app picks for a turn that names none."""

    chat = (await client.post("/api/chats", json={"title": "Studio"})).json()
    response = await client.post(f"/api/chats/{chat['id']}/turns", json=body)
    assert response.status_code == 202, response.text
    return str(response.json()["run"]["workflow_revision_id"])


async def test_an_edit_without_a_selection_passes_over_a_newer_inpainter(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    """Run without a selection, an inpainter has nothing to give its mask input."""

    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    editor = _mock_workflow(
        "Neutral editor", install_id, _EDIT_GRAPH, PLAIN_SCHEMA, datetime(2026, 1, 1, tzinfo=UTC)
    )
    inpainter = _mock_workflow(
        "Neutral inpainter",
        install_id,
        _INPAINT_GRAPH,
        INPAINT_SCHEMA,
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    source_id = await _upload(client, "source.png", _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        assert await _chosen(client, _instruction(source_id)) == editor
        # Remove keeps its selection out of the workflow and places it back after.
        assert await _chosen(client, _removal(source_id, mask_id)) == editor
        assert await _chosen(client, _brushed(source_id, mask_id)) == inpainter


async def test_a_selection_for_the_workflow_finds_an_inpainter_older_than_the_editor(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    """The newest editor has nowhere to put the selection, so it would refuse the turn."""

    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    inpainter = _mock_workflow(
        "Neutral inpainter",
        install_id,
        _INPAINT_GRAPH,
        INPAINT_SCHEMA,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    editor = _mock_workflow(
        "Neutral editor", install_id, _EDIT_GRAPH, PLAIN_SCHEMA, datetime(2026, 1, 2, tzinfo=UTC)
    )
    source_id = await _upload(client, "source.png", _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        assert await _chosen(client, _brushed(source_id, mask_id)) == inpainter
        assert await _chosen(client, _instruction(source_id)) == editor


async def test_an_edit_that_asks_for_no_enlargement_passes_over_a_newer_upscaler(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    """Chosen for an instruction, an upscaler hands the picture back at twice its size."""

    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    editor = _mock_workflow(
        "Neutral editor", install_id, _EDIT_GRAPH, PLAIN_SCHEMA, datetime(2026, 1, 1, tzinfo=UTC)
    )
    upscaler = _mock_workflow(
        "Neutral upscaler",
        install_id,
        _UPSCALE_GRAPH,
        UPSCALE_SCHEMA,
        datetime(2026, 1, 2, tzinfo=UTC),
    )
    source_id = await _upload(client, "source.png", _png(_source()))
    mask_id = await _upload(client, "selection.png", _mask())

    async with app.state.services.scheduler.lease("primary"):
        assert await _chosen(client, _instruction(source_id)) == editor
        assert await _chosen(client, _removal(source_id, mask_id)) == editor
        # Asked for a larger picture, the same choice still finds the upscaler.
        enlarge = {**_instruction(source_id), "text": "Enlarge the picture.", "upscale": True}
        assert await _chosen(client, enlarge) == upscaler


async def test_margins_find_an_outpainter_older_than_the_editor(
    app: FastAPI, client: AsyncClient, tmp_path: Path
) -> None:
    """The newest editor cannot paint past the edge, so it would refuse the margins."""

    install_id = await _import_model(client, tmp_path)
    _clear_seeded_edit_workflows()
    outpainter = _mock_workflow(
        "Neutral outpainter",
        install_id,
        _outpaint_graph(),
        OUTPAINT_SCHEMA,
        datetime(2026, 1, 1, tzinfo=UTC),
    )
    editor = _mock_workflow(
        "Neutral editor", install_id, _EDIT_GRAPH, PLAIN_SCHEMA, datetime(2026, 1, 2, tzinfo=UTC)
    )
    source_id = await _upload(client, "source.png", _png(_source()))

    async with app.state.services.scheduler.lease("primary"):
        assert await _chosen(client, _extension(source_id)) == outpainter
        assert await _chosen(client, _instruction(source_id)) == editor


@pytest.mark.parametrize("schema", [INPAINT_SCHEMA, UPSCALE_SCHEMA], ids=["mask", "upscale"])
def test_auto_passes_over_a_default_workflow_whose_input_the_edit_leaves_empty(
    session: Session, schema: dict[str, Any]
) -> None:
    specialist = _family(
        session, "Specialist", _graphs(instruction=True), schema, is_default=True, use_case=""
    )
    # A use case makes the family one Auto may choose; it matches nothing in the words.
    plain = _family(
        session,
        "Plain",
        _graphs(instruction=True),
        PLAIN_SCHEMA,
        is_default=False,
        use_case="studio photos",
    )
    chat = _chat(session, "automatic")

    _profile, selection, revision = _execution(session, chat, GLOBAL)
    assert revision == plain
    assert selection["workflow_family_id"] == plain.definition.family_id
    # A choice someone made is kept: the specialist still runs when its family is named.
    family_id = specialist.definition.family_id
    assert _execution(session, _chat(session, "family", family_id), GLOBAL)[2] == specialist


def test_auto_puts_an_inpainter_first_for_a_selection_it_is_handed(session: Session) -> None:
    inpainter = _family(
        session,
        "Inpainter",
        _graphs(instruction=True),
        INPAINT_SCHEMA,
        is_default=False,
        use_case="studio photos",
    )
    plain = _family(
        session, "Plain", _graphs(instruction=True), PLAIN_SCHEMA, is_default=True, use_case=""
    )
    chat = _chat(session, "automatic")

    _profile, selection, revision = _execution(session, chat, GLOBAL, settings=_A_SELECTION)
    assert revision == inpainter
    assert selection["workflow_family_id"] == inpainter.definition.family_id
    # A selection placed back after the edit never reaches the workflow.
    blend = {"mask": {**_A_SELECTION["mask"], "apply": "blend"}}
    assert _execution(session, chat, GLOBAL, settings=blend)[2] == plain
    assert _execution(session, chat, GLOBAL)[2] == plain
