"""The studio offers a tool only on a workflow its run would accept.

A tool offered on a workflow the run then refuses is found out only after a
selection has been drawn and the edit accepted. The run refuses a ComfyUI
workflow nobody has reviewed and one whose node packages are not installed, so
neither may make a tool look ready, and the tool says that finishing the setup,
not installing another, is what it waits for.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from workflow_fixtures import seed_workflow_trust

from local_lm.db import SessionLocal
from local_lm.matting_workflows import MATTING_SCHEMA_KIND, MATTING_SETTING_KEY
from local_lm.models import WorkflowDefinition, WorkflowRevision
from local_lm.studio_masks import MASK_SCHEMA_KIND, MASK_SETTING_KEY
from local_lm.workflow_node_dependencies import CUSTOM_NODE_KEY

WAITING = (
    "An installed workflow can do this once it has been reviewed and has the node packages "
    "it needs."
)
INPAINT = {MASK_SETTING_KEY: {"type": "object", "x-lm-atelier-kind": MASK_SCHEMA_KIND}}
MATTING = {
    MATTING_SETTING_KEY: {
        "type": "boolean",
        "readOnly": True,
        "x-lm-atelier-kind": MATTING_SCHEMA_KIND,
    }
}


@pytest.fixture(autouse=True)
def _comfyui_engine(app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
    # Review is asked of ComfyUI workflows only. Nothing here starts a job, so
    # no worker is started and no runtime is set up.
    monkeypatch.setattr(app.state.services.engines.settings, "media_engine", "comfyui")


def _edit_workflow(
    name: str,
    properties: dict[str, Any],
    *,
    dependencies: dict[str, Any] | None = None,
) -> str:
    """An installed edit workflow, as an import leaves one: not yet reviewed."""
    with SessionLocal() as session:
        definition = WorkflowDefinition(name=name, operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="comfyui",
            trusted=False,
            api_graph_json={"save": {"class_type": "SaveImage", "inputs": {}}},
            input_schema_json={"type": "object", "properties": properties},
            dependencies_json=dependencies or {},
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        return revision.id


async def _tools(client: AsyncClient) -> dict[str, dict[str, Any]]:
    response = await client.get("/api/studio/capabilities")
    assert response.status_code == 200, response.text
    return {tool["kind"]: tool for tool in response.json()["tools"]}


async def test_nothing_installed_asks_for_an_install(client: AsyncClient) -> None:
    tools = await _tools(client)

    assert tools["brush"]["available"] is False
    assert tools["brush"]["reason"] == "Install an inpainting workflow to edit part of a picture."


async def test_a_workflow_waiting_for_review_offers_no_tool_until_it_is_reviewed(
    client: AsyncClient,
) -> None:
    revision_id = _edit_workflow("Neutral inpainter", INPAINT)

    waiting = await _tools(client)
    assert waiting["brush"]["available"] is False
    assert waiting["instruct"]["available"] is False
    assert waiting["brush"]["reason"] == WAITING
    assert waiting["instruct"]["reason"] == WAITING
    # Nothing installed can enlarge a picture, so that tool still asks for an install.
    assert waiting["enhance"]["reason"] == "Install an upscaling workflow to enlarge a picture."

    seed_workflow_trust(revision_id)
    reviewed = await _tools(client)
    assert reviewed["brush"]["available"] is True
    assert reviewed["instruct"]["available"] is True


async def test_a_cutout_runs_on_a_reviewed_workflow_even_when_an_unreviewed_one_comes_first(
    client: AsyncClient,
) -> None:
    _edit_workflow("A neutral cutout", MATTING)
    reviewed = _edit_workflow("B neutral cutout", MATTING)
    seed_workflow_trust(reviewed)

    tools = await _tools(client)

    assert tools["isolate"]["available"] is True
    assert tools["isolate"]["workflow_revision_id"] == reviewed


async def test_a_workflow_whose_node_packages_are_not_installed_offers_no_tool(
    client: AsyncClient,
) -> None:
    revision_id = _edit_workflow(
        "Neutral inpainter with a missing package",
        INPAINT,
        dependencies={CUSTOM_NODE_KEY: [{"name": "neutral-missing-nodes"}]},
    )
    seed_workflow_trust(revision_id)

    tools = await _tools(client)

    assert tools["brush"]["available"] is False
    assert tools["instruct"]["available"] is False
    assert tools["brush"]["reason"] == WAITING
