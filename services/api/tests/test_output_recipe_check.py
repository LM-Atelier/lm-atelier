"""Checking a generation record against this installation, through the real route."""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
from httpx2 import AsyncClient
from run_waits import wait_for_terminal_status

from local_lm.db import SessionLocal
from local_lm.models import (
    ModelAssetInstall,
    ModelComponentManifest,
    ModelInstall,
    WorkflowDefinition,
    WorkflowRevision,
)
from local_lm.output_recipe_v1 import canonical_bytes, record_digest, seal_output_recipe

pytestmark = pytest.mark.asyncio

_PROMPT = "a ceramic cup on a wooden table"


def _record(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "schema": "lm-atelier-output-recipe-v1",
        "version": 1,
        "exported_by": {"application": "LM Atelier", "version": "9.9.9"},
        "output": {
            "sha256": "a" * 64,
            "size_bytes": 10,
            "media_type": "image/png",
            "kind": "image",
            "index": 0,
            "count": 1,
            "engine": "comfyui",
            "node_id": "9",
            "collection": "images",
            "raster": None,
        },
        "operation": "image_to_image",
        "prompt": {
            "included": True,
            "positive": _PROMPT,
            "negative": None,
            "omitted_reason": None,
        },
        "seed": {"value": 7, "binding": "bound"},
        "settings": {"bound": {}, "unbound": {}},
        "inputs": [],
        "workflow": None,
        "model": None,
        "loras": [],
        "removed": [],
        "not_recorded": ["executed_graph_sha256", "runtime_version"],
        "reproducibility": {"status": "incomplete", "missing": ["frozen_snapshot_absent"]},
    }
    payload.update(overrides)
    return seal_output_recipe(payload)


def _workflow(identity: str) -> dict[str, Any]:
    return {
        "engine": "comfyui",
        "operation": "image_to_image",
        "artifact_sha256": identity,
        "verified": True,
        "graph_source": "live",
        "contract_version": 1,
        "dependency_contract_sha256": None,
        "binding_sha256": None,
    }


async def _check(client: AsyncClient, content: bytes) -> Any:
    return await client.post(
        "/api/output-recipes/check",
        content=content,
        headers={"content-type": "application/json"},
    )


def _states(report: dict[str, Any]) -> dict[tuple[str, str], str]:
    return {(item["kind"], item["sha256"]): item["state"] for item in report["requirements"]}


async def test_a_record_this_app_wrote_checks_against_this_app(client: AsyncClient) -> None:
    chat = (await client.post("/api/chats", json={"title": "Check"})).json()
    turn = await client.post(
        f"/api/chats/{chat['id']}/turns", json={"text": _PROMPT, "mode": "image"}
    )
    run_id = turn.json()["run"]["id"]

    async def read() -> dict[str, Any]:
        return cast(dict[str, Any], (await client.get(f"/api/runs/{run_id}")).json())

    run = await wait_for_terminal_status(read, what=run_id, expected="complete")
    artifact_id = run["provenance_json"]["outputs"][0]["artifact_id"]
    downloaded = await client.get(
        f"/api/runs/{run_id}/outputs/{artifact_id}/recipe", params={"prompts": "include"}
    )
    assert downloaded.status_code == 200, downloaded.text

    response = await _check(client, downloaded.content)

    assert response.status_code == 200, response.text
    report = response.json()
    assert report["digest"] == downloaded.headers["x-output-recipe-digest"]
    assert [item["kind"] for item in report["requirements"]] == ["workflow"]
    assert report["requirements"][0]["state"] == "present"
    assert report["all_present"] is True
    assert _PROMPT not in response.text
    assert response.headers["cache-control"] == "no-store"


async def test_each_requirement_is_matched_by_its_exact_identity(client: AsyncClient) -> None:
    """Installed and active is present; installed but not ready is inactive; else missing."""

    ready, idle, absent = "1" * 64, "2" * 64, "3" * 64
    legacy = "4" * 64
    with SessionLocal() as session:
        active = ModelInstall(
            name="Active model",
            role="image",
            engine="comfyui",
            local_path="C:/managed/active-model",
            manifest_json={"expected_sha256": {"legacy.safetensors": legacy}},
            active=True,
        )
        inactive = ModelInstall(
            name="Inactive model",
            role="image",
            engine="comfyui",
            local_path="C:/managed/inactive-model",
            manifest_json={},
            active=False,
        )
        session.add_all([active, inactive])
        session.flush()
        session.add_all(
            [
                ModelComponentManifest(
                    model_install_id=active.id,
                    kind="checkpoint",
                    relative_path="ready.safetensors",
                    target_folder="checkpoints",
                    sha256=ready,
                ),
                ModelComponentManifest(
                    model_install_id=inactive.id,
                    kind="checkpoint",
                    relative_path="idle.safetensors",
                    target_folder="checkpoints",
                    sha256=idle,
                ),
                ModelAssetInstall(
                    name="Ready LoRA",
                    kind="lora",
                    local_path="C:/managed/ready-lora",
                    manifest_json={"sha256": ready},
                    active=True,
                ),
                ModelAssetInstall(
                    name="Idle LoRA",
                    kind="lora",
                    local_path="C:/managed/idle-lora",
                    manifest_json={"sha256": idle},
                    active=False,
                ),
            ]
        )
        definition = WorkflowDefinition(name="Untrusted edit", operation="image_to_image")
        session.add(definition)
        session.flush()
        session.add(
            WorkflowRevision(
                definition=definition,
                version=1,
                engine="comfyui",
                api_graph_json={},
                input_schema_json={},
                dependencies_json={},
                artifact_sha256=idle,
                trusted=False,
            )
        )
        session.commit()

    lora = {"model_strength": 1.0, "clip_strength": 1.0, "enabled": True}
    content = _record(
        workflow=_workflow(idle),
        model={
            "files": {
                "a.safetensors": ready,
                "b.safetensors": idle,
                "c.safetensors": absent,
                "d.safetensors": legacy,
            },
            "provider": None,
            "remote_id": None,
            "revision": None,
            "content_rating": None,
        },
        loras=[
            {"sha256": ready, "position": 0, **lora},
            {"sha256": idle, "position": 1, **lora},
            {"sha256": absent, "position": 2, **lora},
        ],
        inputs=[{"sha256": absent, "role": "source", "size_bytes": 10, "media_type": "image/png"}],
    )

    response = await _check(client, content)

    assert response.status_code == 200, response.text
    report = response.json()
    assert _states(report) == {
        ("workflow", idle): "inactive",
        ("model_file", ready): "present",
        ("model_file", idle): "inactive",
        ("model_file", absent): "missing",
        ("model_file", legacy): "present",
        ("lora", ready): "present",
        ("lora", idle): "inactive",
        ("lora", absent): "missing",
        ("input", absent): "missing",
    }
    assert report["all_present"] is False
    assert report["requirements"][-1]["role"] == "source"


async def test_a_changed_record_is_refused_without_echoing_it(client: AsyncClient) -> None:
    content = _record().replace(b'"value":7', b'"value":8')

    response = await _check(client, content)

    assert response.status_code == 422
    assert response.json()["code"] == "output-recipe-unreadable"
    assert _PROMPT not in response.text


async def test_a_number_too_long_to_read_is_refused_as_unreadable(client: AsyncClient) -> None:
    response = await _check(client, b'{"version":' + b"7" * 5000 + b"}")

    assert response.status_code == 422
    assert response.json()["code"] == "output-recipe-unreadable"


async def test_a_record_with_an_integer_too_large_for_a_float_is_checked(
    client: AsyncClient,
) -> None:
    payload = json.loads(_record())
    del payload["digest"]
    payload["settings"]["bound"]["steps"] = 10**400
    # Sealed by hand: the reader is what this checks, so it is not used to write.
    content = canonical_bytes({**payload, "digest": record_digest(payload)})

    response = await _check(client, content)

    assert response.status_code == 200, response.text


async def test_a_file_too_large_to_be_a_record_is_refused(client: AsyncClient) -> None:
    response = await _check(client, b" " * (256 * 1024 + 1))

    assert response.status_code == 413
    assert response.json()["code"] == "output-recipe-too-large"
