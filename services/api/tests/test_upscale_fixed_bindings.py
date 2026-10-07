"""Keep fixed enlargement values executable without offering an override."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status
from test_upscale_preview_api import _selection

from local_lm.adapters.base import MediaEvent, MediaRequest
from local_lm.adapters.comfyui import ComfyUIAdapter
from local_lm.adapters.mock import MockMediaAdapter
from local_lm.db import SessionLocal
from local_lm.models import WorkflowRevision
from local_lm.upscale_workflows import upscale_setting_schema


async def _fixed_selection(
    client: AsyncClient, declaration: str
) -> tuple[str, str, dict[str, Any]]:
    chat_id, revision_id, payload = await _selection(client, adjustable=True)
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "upscale_factor": {
                    "type": "number",
                    "readOnly": True,
                    declaration: [4] if declaration == "enum" else 4,
                    "x-lm-atelier-kind": "upscale",
                }
            },
        }
        session.commit()
    return chat_id, revision_id, payload


@pytest.mark.parametrize("declaration", ["const", "default", "enum"])
async def test_a_fixed_bound_factor_is_previewed_without_a_control(
    client: AsyncClient, declaration: str
) -> None:
    chat_id, _, payload = await _fixed_selection(client, declaration)

    response = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)

    assert response.status_code == 200, response.text
    assert response.json()["factor"] is None
    assert response.json()["fixed_factor"] == 4


@pytest.mark.parametrize("declaration", ["const", "default", "enum"])
async def test_a_fixed_bound_factor_reaches_dispatch_without_a_turn_override(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch, declaration: str
) -> None:
    chat_id, revision_id, payload = await _fixed_selection(client, declaration)
    captured: list[MediaRequest] = []
    original = MockMediaAdapter.generate

    async def generate(self: MockMediaAdapter, request: MediaRequest) -> AsyncIterator[MediaEvent]:
        captured.append(request)
        async for event in original(self, request):
            yield event

    monkeypatch.setattr(MockMediaAdapter, "generate", generate)
    response = await client.post(
        f"/api/chats/{chat_id}/turns", json={**payload, "workflow_revision_id": revision_id}
    )
    assert response.status_code == 202, response.text
    run = response.json()["run"]
    assert "upscale_factor" not in run["settings_json"]

    async def read() -> dict[str, Any]:
        result = await client.get(f"/api/runs/{run['id']}")
        assert result.status_code == 200, result.text
        return dict(result.json())

    await wait_for_terminal_status(read, what="Fixed enlargement")
    assert len(captured) == 1
    graph = ComfyUIAdapter._compile(captured[0].workflow, captured[0].parameters)
    assert graph["scale"]["inputs"]["scale_by"] == 4
    assert "upscale_factor" not in (await read())["settings_json"]


@pytest.mark.parametrize("declaration", ["const", "default", "enum"])
def test_a_catalog_declaration_keeps_its_bound_fixed_value(declaration: str) -> None:
    graph = {"scale": {"class_type": "ImageScaleBy", "inputs": {"scale_by": "${upscale_factor}"}}}
    field = {"type": "number", "readOnly": True, declaration: [4] if declaration == "enum" else 4}

    result = upscale_setting_schema(graph, {"properties": {"upscale_factor": field}})

    assert result == {**field, "x-lm-atelier-kind": "upscale"}


@pytest.mark.parametrize(
    "fixed",
    [
        {},
        {"const": 0},
        {"const": True},
        {"const": "4"},
        {"const": 4, "minimum": 5},
        {"const": 4, "maximum": 3},
        {"const": 4, "enum": [3]},
        {"const": 4, "multipleOf": 3},
        {"type": "integer", "default": 4.5},
    ],
)
async def test_a_fixed_binding_without_a_legal_value_cannot_be_previewed(
    client: AsyncClient, fixed: dict[str, Any]
) -> None:
    chat_id, revision_id, payload = await _fixed_selection(client, "const")
    with SessionLocal() as session:
        revision = session.get(WorkflowRevision, revision_id)
        assert revision is not None
        revision.input_schema_json = {
            "type": "object",
            "properties": {
                "upscale_factor": {
                    "type": "number",
                    "readOnly": True,
                    "x-lm-atelier-kind": "upscale",
                    **fixed,
                }
            },
        }
        session.commit()

    response = await client.post(f"/api/chats/{chat_id}/upscale/preview", json=payload)

    assert response.status_code == 409, response.text
    assert response.json()["code"] == "upscale-preview-unavailable"
