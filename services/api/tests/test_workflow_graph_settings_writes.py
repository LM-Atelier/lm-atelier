from copy import deepcopy
from typing import Any

import pytest
from httpx2 import AsyncClient


@pytest.mark.parametrize("target", ["create", "revision"])
@pytest.mark.parametrize("damage", ["binding", "availability"])
async def test_workflow_writes_refuse_settings_that_disagree_with_the_executable_graph(
    client: AsyncClient, target: str, damage: str
) -> None:
    payload: dict[str, Any] = {
        "name": "Declared graph controls",
        "operation": "text_to_image",
        "engine": "comfyui",
        "api_graph": {
            "1": {
                "class_type": "Source",
                "inputs": {
                    "seed": "${seed}",
                    "sampler": "${sampler}",
                },
            }
        },
        "input_schema": {
            "type": "object",
            "properties": {
                "seed": {"type": "integer", "default": 42},
                "sampler": {"type": "string", "default": "euler"},
            },
            "x-lm-atelier-graph-settings": {
                "version": 1,
                "bindings": [
                    {"node_id": "1", "input_name": "seed", "parameter": "seed"},
                ],
            },
        },
    }
    created = await client.post("/api/workflows", json=payload)
    assert created.status_code == 201, created.text
    original = created.json()
    changed = deepcopy(payload)
    if damage == "binding":
        changed["api_graph"]["1"]["inputs"]["seed"] = 99
    else:
        changed["input_schema"]["x-lm-atelier-graph-settings"]["unbound_parameters"] = ["sampler"]
    if target == "revision":
        path = f"/api/workflows/{original['id']}/revisions"
        changed = {
            key: value for key, value in changed.items() if key in {"api_graph", "input_schema"}
        }
    else:
        path = "/api/workflows"
    refused = await client.post(path, json=changed)
    assert refused.status_code == 422, "A settings record that contradicts execution was persisted"
    expected_code = "workflow-revision-invalid" if target == "revision" else "workflow-invalid"
    assert refused.json()["code"] == expected_code
    listed = [
        item
        for item in (await client.get("/api/workflows")).json()
        if item["name"] == "Declared graph controls"
    ]
    assert len(listed) == 1
    assert listed[0]["current_revision_id"] == original["current_revision_id"]
    assert len(listed[0]["revisions"]) == 1
