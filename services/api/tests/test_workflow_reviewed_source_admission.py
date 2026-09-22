from __future__ import annotations

import sys
import traceback
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from httpx2 import AsyncClient
from sqlalchemy import delete, select
from test_workflow_package_import_endpoint import _ui_graph
from test_workflow_reviewed_package_plan import _close, _inputs
from test_workflow_revision_review import reviewed_runtime as reviewed_runtime
from test_workflow_source_completion import _approve
from test_workflow_source_completion import source_runtime as source_runtime

from local_lm import comfy_registry_interpreter as interpreter
from local_lm import models
from local_lm import workflow_package_extension_preflight as preflight
from local_lm import workflow_review_runtime as reviews
from local_lm import workflow_runtime_targets as targets
from local_lm import workflow_source_completion as completion
from local_lm import workflow_source_runtime as runtime
from local_lm.comfy_registry_downloads import ComfyRegistryArchiveDownloader
from local_lm.db import SessionLocal
from local_lm.workflow_package_preparation import PreparationContext
from local_lm.workflow_revision_reviews import review_is_current
from local_lm.workflow_source_extensions import (
    ExtensionPreparationServices,
    prepare_workflow_source_extensions,
)

URL = "/api/workflows/packages/install-plans"


@pytest.fixture
async def source(
    app: FastAPI,
    client: AsyncClient,
    source_runtime: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> AsyncIterator[dict[str, Any]]:
    services = app.state.services
    services.settings.comfy_executable = Path(sys.executable)
    services.settings.custom_node_dir.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as session:
        inputs = _inputs(
            (session, services.artifacts),
            tmp_path,
            commit=False,
            inactive=getattr(request, "param", "active") == "inactive",
        )
    args = inputs["plan_arguments"]
    failures: list[Any] = []
    original_complete = completion.complete_workflow_source

    async def complete(*values: Any, **kwargs: Any) -> Any:
        try:
            return await original_complete(*values, **kwargs)
        except Exception as exc:
            failures.append(
                (
                    type(exc).__name__,
                    getattr(exc, "code", None),
                    [
                        (frame.name, frame.lineno)
                        for frame in traceback.extract_tb(exc.__traceback__)
                    ],
                )
            )
            raise

    monkeypatch.setattr(completion, "complete_workflow_source", complete)

    async def close() -> None:
        pass

    monkeypatch.setattr(
        preflight,
        "ComfyRegistryClient",
        lambda: SimpleNamespace(resolve=args["registry_client"].resolve, close=close),
    )
    monkeypatch.setattr(
        preflight,
        "ComfyRegistryWheelProjectClient",
        lambda: SimpleNamespace(fetch=args["project_client"].fetch, close=close),
    )
    monkeypatch.setattr(
        preflight,
        "ComfyRegistryWheelMetadataClient",
        lambda: SimpleNamespace(fetch=args["metadata_client"].fetch, close=close),
    )
    monkeypatch.setattr(
        preflight,
        "ComfyRegistryArchiveDownloader",
        lambda: ComfyRegistryArchiveDownloader(
            transport=args["archive_downloader"]._client._transport
        ),
    )
    monkeypatch.setattr(targets, "probe_comfy_registry_runtime_target", args["interpreter_probe"])
    monkeypatch.setattr(
        interpreter, "probe_comfy_registry_runtime_target", args["interpreter_probe"]
    )

    async def prepare(*values: Any, **kwargs: Any) -> Any:
        prepared = await prepare_workflow_source_extensions(
            *values,
            **kwargs,
            services=ExtensionPreparationServices(
                args["interpreter_probe"],
                args["registry_client"],
                args["project_client"],
                args["metadata_client"],
                args["archive_downloader"],
                inputs["downloader"],
            ),
        )
        for kind in inputs["selected"].node_types:
            source_runtime[kind] = {
                "python_module": "custom_nodes.neutral",
                "input": {"required": {}},
                "output": [],
            }
        return prepared

    async def inventory() -> frozenset[str]:
        return frozenset(source_runtime)

    monkeypatch.setattr(runtime, "prepare_workflow_source_extensions", prepare)
    monkeypatch.setattr(services.processes, "comfy_node_inventory", inventory)
    graph = _ui_graph()
    selected = inputs["selected"]
    for kind in selected.node_types:
        graph["nodes"].append(
            dict(
                id=max(n["id"] for n in graph["nodes"]) + 1,
                type=kind,
                mode=0,
                inputs=[],
                outputs=[],
                widgets_values=[],
                properties={"cnr_id": selected.package_id, "ver": selected.declared_version},
            )
        )
    payload = dict(
        name="Neutral reviewed extension",
        operation="text_to_image",
        ui_graph=graph,
        dependencies={"version": 1, "slots": []},
        selections=[],
    )
    try:
        yield dict(
            inputs=inputs,
            payload=payload,
            info=source_runtime,
            services=services,
            failures=failures,
        )
    finally:
        await _close(inputs)


@pytest.mark.parametrize(
    ("source", "change"),
    [("active", "none"), ("inactive", "none"), ("active", "files"), ("active", "authority")],
    indirect=["source"],
)
async def test_public_preview_and_accept_complete_reviewed_source_extensions(
    source: dict[str, Any],
    client: AsyncClient,
    app: FastAPI,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    response = await client.post(URL, json=source["payload"])
    assert response.status_code == 201, response.text
    plan = response.json()
    assert plan["can_accept"] and not plan["blockers"], plan
    selected = source["inputs"]["selected"]
    extension = plan["extension_execution"]["plans"][selected.package_id]
    assert extension["version"] == 2
    expected_wheels = (
        source["inputs"]["remote_bytes"]
        if extension["closure"]["manifest"]["reviewed_local"]
        else 0
    )
    assert plan["total_download_bytes"] == extension["archive_bytes"] + expected_wheels
    refreshed = await client.get(f"{URL}/{plan['id']}")
    assert refreshed.status_code == 200 and refreshed.json() == plan, refreshed.text
    with SessionLocal() as session:
        assert session.scalar(select(models.ComfyRegistryInstall)) is None
    context = PreparationContext.from_settings(source["services"].settings)
    assert context.source_store is not None
    validated = []

    async def validate(graph: dict[str, Any]) -> list[str]:
        validated.append(graph)
        if change == "files":
            path = next(context.custom_node_root.glob("*/__init__.py"))
            path.write_text("raise RuntimeError('Changed neutral code')", encoding="utf-8")
        return []

    original_refresh = reviews.VerifiedReviewedPackages.refresh

    async def refresh(self: reviews.VerifiedReviewedPackages) -> reviews.VerifiedReviewedPackages:
        result = await original_refresh(self)
        if change == "authority":
            with SessionLocal() as session:
                session.execute(delete(models.ComfyRegistrySourceArtifactReview))
                session.commit()
        return result

    monkeypatch.setattr(reviews.VerifiedReviewedPackages, "refresh", refresh)
    monkeypatch.setattr(source["services"].engines.media, "validate_workflow", validate)
    offer_id = await _approve(client, app, plan["id"])
    assert len(validated) == 1
    with SessionLocal() as session:
        offer = session.get(models.WorkflowInstallOffer, offer_id)
        assert offer is not None
        if change != "none":
            assert offer.status == "queued" and offer.completion_error_code
            revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
            assert revision is not None and not revision.trusted
            assert session.get(models.WorkflowRevisionReview, revision.id) is None
            install = session.scalar(select(models.ComfyRegistryInstall))
            assert install is not None and not install.active
            job = session.scalar(select(models.Job).where(models.Job.kind == "workflow_install"))
            assert job is not None and job.status == "failed"
            return
        assert offer.status == "completed", (offer.completion_error_code, source["failures"])
        revision = session.get(models.WorkflowRevision, offer.workflow_revision_id)
        assert revision is not None and revision.trusted
        definition = session.get(models.WorkflowDefinition, revision.workflow_id)
        assert definition is not None and review_is_current(session, definition, revision)
        install = session.scalar(select(models.ComfyRegistryInstall))
        assert install is not None and install.trusted and install.active
        assert install.wheel_closure_sha256 == extension["closure"]["closure_sha256"]
        assert install.review_json["reviewed_wheel_closure"]
        job = session.scalar(select(models.Job).where(models.Job.kind == "workflow_install"))
        assert job is not None and job.status == "complete"


async def test_public_preview_refuses_a_revoked_source_review(
    source: dict[str, Any],
    client: AsyncClient,
) -> None:
    with SessionLocal() as session:
        session.execute(delete(models.ComfyRegistrySourceArtifactReview))
        session.commit()
    response = await client.post(URL, json=source["payload"])
    assert response.status_code == 201, response.text
    plan = response.json()
    assert not plan["can_accept"] and plan["extension_execution"]["errors"]
    with SessionLocal() as session:
        assert session.scalar(select(models.ComfyRegistryInstall)) is None
