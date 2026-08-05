"""The prepare endpoint: a durable job that refuses honestly."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient

from local_lm.config import Settings

pytestmark = pytest.mark.asyncio


def _workflow() -> dict[str, object]:
    return {
        "version": 0.4,
        "nodes": [
            {
                "id": 1,
                "type": "ExampleNode",
                "properties": {"cnr_id": "example-pack", "ver": "1.2.3"},
                "widgets_values": [],
            }
        ],
        "links": [],
        "definitions": {"subgraphs": []},
    }


async def test_an_unconfigured_runtime_refuses_before_queueing(client: AsyncClient) -> None:
    """A job that must fail on its first step is worse than a typed 422."""

    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": _workflow()},
    )

    assert response.status_code == 422
    payload = response.json()
    assert payload["code"] == "managed_runtime_unavailable"
    jobs = (await client.get("/api/jobs")).json()
    assert all(job["kind"] != "registry_prepare" for job in jobs)


async def test_a_configured_runtime_queues_and_fails_closed_on_the_probe(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Until a verified target probe exists, preparation refuses with its code.

    The job still runs end to end as a job: queued, leased, failed with the
    stable code recorded - never a silent death.
    """

    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "ComfyUI")

    from local_lm import api as api_module
    from local_lm.workflow_package_preparation import WorkflowPackagePreparationError

    async def refusing_preparation(*args: object, **kwargs: object) -> object:
        raise WorkflowPackagePreparationError(
            "interpreter_probe_unavailable", "no verified probe yet"
        )

    monkeypatch.setattr(api_module, "prepare_workflow_package", refusing_preparation)

    response = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": _workflow()},
    )

    assert response.status_code == 202
    job_id = response.json()["id"]
    assert response.json()["kind"] == "registry_prepare"

    import asyncio

    job = None
    for _ in range(200):
        jobs = (await client.get("/api/jobs")).json()
        job = next(item for item in jobs if item["id"] == job_id)
        if job["status"] in {"failed", "complete", "cancelled"}:
            break
        await asyncio.sleep(0.05)
    assert job is not None
    assert job["status"] == "failed"
    assert job["payload_json"]["error_code"] == "interpreter_probe_unavailable"
    assert job["error"]


async def test_cancel_stops_a_blocking_preparation_for_good(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Cancellation reaches the live task, and nothing overwrites CANCELLED.

    The fake preparation blocks indefinitely unless cancelled; if the task
    survived the endpoint call, the completion write below the block would
    turn the cancelled row COMPLETE - the exact overwrite this forbids.
    """

    import asyncio

    executable = tmp_path / "python.exe"
    executable.write_bytes(b"")
    monkeypatch.setattr(settings, "comfy_executable", executable)
    monkeypatch.setattr(settings, "comfy_directory", tmp_path / "ComfyUI")

    from types import SimpleNamespace

    from local_lm import api as api_module

    entered = asyncio.Event()
    resumed = asyncio.Event()

    async def blocking_preparation(*args: object, **kwargs: object) -> object:
        entered.set()
        await asyncio.sleep(3600)
        resumed.set()
        return SimpleNamespace(
            install_id="never",
            installed_path="never",
            wheel_environment_path="never",
            archive_sha256="never",
            manifest_sha256="never",
            wheel_closure_sha256="never",
            wheel_environment_sha256="never",
            reused_wheel_environment=False,
        )

    monkeypatch.setattr(api_module, "prepare_workflow_package", blocking_preparation)

    queued = await client.post(
        "/api/workflows/packages/prepare",
        json={"package_id": "example-pack", "version": "1.2.3", "ui_graph": _workflow()},
    )
    assert queued.status_code == 202
    job_id = queued.json()["id"]
    await asyncio.wait_for(entered.wait(), timeout=5)

    cancelled = await client.post(f"/api/jobs/{job_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"

    # The task never resumes past its block, and the row stays cancelled.
    await asyncio.sleep(0.2)
    assert not resumed.is_set()
    jobs = (await client.get("/api/jobs")).json()
    job = next(item for item in jobs if item["id"] == job_id)
    assert job["status"] == "cancelled"


async def test_an_unknown_workflow_revision_refuses_before_queueing(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "example-pack",
            "version": "1.2.3",
            "ui_graph": _workflow(),
            "workflow_revision_id": "rev_missing",
        },
    )

    assert response.status_code == 404
    assert response.json()["code"] == "workflow-revision-not-found"


async def test_a_caller_graph_cannot_be_bound_to_another_revision(
    client: AsyncClient,
) -> None:
    """The proof subject is the stored workflow, never the submitted one.

    Re-analyzing a submitted graph proves that graph is internally consistent.
    It does not bind it to anything this machine saved, so a caller that could
    name one revision while sending a different graph could choose the subject
    of its own proof.
    """
    from local_lm.db import SessionLocal
    from local_lm.models import WorkflowDefinition, WorkflowRevision

    stored = dict(_workflow())
    with SessionLocal() as session:
        definition = WorkflowDefinition(name="Stored", operation="image_to_image")
        session.add(definition)
        session.flush()
        revision = WorkflowRevision(
            workflow_id=definition.id,
            version=1,
            engine="mock",
            ui_graph_json=stored,
            api_graph_json={},
            input_schema_json={},
            dependencies_json={},
            trusted=True,
        )
        session.add(revision)
        session.flush()
        definition.current_revision_id = revision.id
        session.commit()
        revision_id = revision.id

    # The same node type, attributed to a different package. Comparing only the
    # required node names would let this through, and the caller would prepare
    # one package while binding the proof to a graph that names another.
    forged = dict(_workflow())
    forged["nodes"] = [
        {
            "id": 1,
            "type": "ExampleNode",
            "properties": {"cnr_id": "other-pack", "ver": "9.9.9"},
            "widgets_values": [],
        }
    ]

    response = await client.post(
        "/api/workflows/packages/prepare",
        json={
            "package_id": "example-pack",
            "version": "1.2.3",
            "ui_graph": forged,
            "workflow_revision_id": revision_id,
        },
    )

    assert response.status_code == 422
    assert response.json()["code"] == "workflow-graph-mismatch"
